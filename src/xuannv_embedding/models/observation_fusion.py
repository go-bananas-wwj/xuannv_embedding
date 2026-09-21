"""Month-bound, permutation-invariant residual fusion for native-grid observations."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


def fuse_observations(
    base: torch.Tensor,
    frames: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    assigned_months: dict[str, torch.Tensor],
    output_months: torch.Tensor,
    encoders: nn.ModuleDict,
    fusion: nn.Module,
) -> torch.Tensor:
    if set(frames) != set(masks) or set(frames) != set(assigned_months):
        raise ValueError("Observation frames, masks and months must have identical sources")
    batch_size, month_count, channels, height, width = base.shape
    feature_sum = torch.zeros_like(base)
    support = base.new_zeros(batch_size, month_count, 1, height, width)
    for source in sorted(frames):
        images, valid, months = frames[source], masks[source], assigned_months[source]
        if source not in encoders:
            raise ValueError(f"Unregistered high-resolution observation source: {source}")
        if images.ndim != 5 or valid.shape != (*images.shape[:2], 1, *images.shape[-2:]):
            raise ValueError("Observation frames require B,K,C,H,W and masks B,K,1,H,W")
        if images.shape[0] != batch_size or months.shape != images.shape[:2]:
            raise ValueError("Observation month or batch shape mismatch")
        if not torch.isfinite(images).all() or not torch.isfinite(valid).all():
            raise ValueError("Nonfinite observation input")
        if bool(((valid < 0) | (valid > 1)).any()):
            raise ValueError("Observation masks must be in [0,1]")
        active = valid.flatten(2).any(dim=-1)
        belongs = months[:, :, None] == output_months[:, None, :]
        if bool((active & ~belongs.any(dim=-1)).any()):
            raise ValueError("Active observation must have a selected output month")
        if not bool(active.any()):
            continue
        selected_images = images[active] * valid[active]
        selected_features = encoders[source](
            selected_images, target_size=(height, width), encode_before_resize=True
        )
        features = base.new_zeros(*images.shape[:2], channels, height, width)
        features[active] = selected_features.to(base.dtype)
        resized_masks = (
            functional.interpolate(valid.flatten(0, 1), size=(height, width), mode="nearest")
            .view(*images.shape[:2], 1, height, width)
            .to(base.dtype)
        )
        for month_index in range(month_count):
            weights = resized_masks * belongs[:, :, month_index, None, None, None]
            feature_sum[:, month_index] += (features * weights).sum(dim=1)
            support[:, month_index] += weights.sum(dim=1)
    available = (support > 0).to(base.dtype)
    features = feature_sum / support.clamp(min=1)
    fused = fusion(
        base.flatten(0, 1),
        features.flatten(0, 1),
        available.flatten(0, 1),
    ).view_as(base)
    if getattr(fusion, "residual_mode", False):
        return fused
    return base + available * fused
