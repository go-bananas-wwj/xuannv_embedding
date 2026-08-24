from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class InputMaskingConfig:
    """Training-only hard masking for multimodal monthly reconstruction."""

    enabled: bool = False
    drop_availability_masks: bool = True
    modality_dropout_probs: dict[str, float] = field(default_factory=dict)
    month_dropout_prob: float = 0.0
    max_months_per_sample: int = 1
    spatial_block_prob: float = 0.0
    spatial_block_size: int = 16
    spatial_block_ratio: float = 0.15

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "InputMaskingConfig":
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            drop_availability_masks=bool(raw.get("drop_availability_masks", True)),
            modality_dropout_probs={
                str(k): float(v) for k, v in raw.get("modality_dropout_probs", {}).items()
            },
            month_dropout_prob=float(raw.get("month_dropout_prob", 0.0)),
            max_months_per_sample=int(raw.get("max_months_per_sample", 1)),
            spatial_block_prob=float(raw.get("spatial_block_prob", 0.0)),
            spatial_block_size=int(raw.get("spatial_block_size", 16)),
            spatial_block_ratio=float(raw.get("spatial_block_ratio", 0.15)),
        )


def _metric(value: float, reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(float(value), dtype=torch.float32, device=reference.device)


def _drop_temporal_source(
    frames: torch.Tensor,
    masks: torch.Tensor,
    prob: float,
    drop_masks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prob <= 0.0 or frames.shape[0] == 0:
        return frames, masks, _metric(0.0, frames)

    batch_size = frames.shape[0]
    if prob >= 1.0:
        keep = torch.zeros(batch_size, device=frames.device, dtype=frames.dtype)
    else:
        keep = (torch.rand(batch_size, device=frames.device) >= prob).to(frames.dtype)
    dropped = 1.0 - keep
    frames = frames * keep[:, None, None, None, None]
    if drop_masks:
        masks = masks * keep[:, None]
    return frames, masks, dropped.mean().detach()


def _drop_highres_source(
    frames: torch.Tensor,
    masks: torch.Tensor,
    prob: float,
    drop_masks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prob <= 0.0 or frames.shape[0] == 0:
        return frames, masks, _metric(0.0, frames)

    batch_size = frames.shape[0]
    if prob >= 1.0:
        keep = torch.zeros(batch_size, device=frames.device, dtype=frames.dtype)
    else:
        keep = (torch.rand(batch_size, device=frames.device) >= prob).to(frames.dtype)
    dropped = 1.0 - keep
    frames = frames * keep[:, None, None, None]
    if drop_masks:
        masks = masks * keep[:, None, None, None]
    return frames, masks, dropped.mean().detach()


def _drop_months(
    source_frames: dict[str, torch.Tensor],
    source_masks: dict[str, torch.Tensor],
    prob: float,
    max_months_per_sample: int,
    drop_masks: bool = True,
) -> dict[str, torch.Tensor]:
    stats: dict[str, torch.Tensor] = {}
    if prob <= 0.0 or max_months_per_sample <= 0:
        return stats

    temporal_sources = [
        source
        for source, frames in source_frames.items()
        if frames.dim() == 5 and frames.shape[1] > 0
    ]
    if not temporal_sources:
        return stats

    ref = source_frames[temporal_sources[0]]
    batch_size, num_months = ref.shape[:2]
    device = ref.device
    drop = torch.zeros(batch_size, num_months, device=device, dtype=ref.dtype)
    max_months = min(int(max_months_per_sample), num_months)
    if max_months <= 0:
        return stats

    active = torch.rand(batch_size, device=device) < prob
    for b in torch.where(active)[0].tolist():
        count = torch.randint(1, max_months + 1, (1,), device=device).item()
        idx = torch.randperm(num_months, device=device)[:count]
        drop[b, idx] = 1.0

    keep = 1.0 - drop
    for source in temporal_sources:
        source_frames[source] = source_frames[source] * keep[:, :, None, None, None]
        if drop_masks:
            source_masks[source] = source_masks[source] * keep

    stats["masking_month_drop_ratio"] = drop.mean().detach()
    return stats


def _make_spatial_keep_mask(
    batch_size: int,
    height: int,
    width: int,
    block_size: int,
    block_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    block_size = max(1, int(block_size))
    grid_h = max(1, (height + block_size - 1) // block_size)
    grid_w = max(1, (width + block_size - 1) // block_size)
    drop_grid = (torch.rand(batch_size, 1, grid_h, grid_w, device=device) < block_ratio).to(dtype)
    drop = F.interpolate(drop_grid, size=(height, width), mode="nearest")
    return 1.0 - drop


def _drop_spatial_blocks(
    source_frames: dict[str, torch.Tensor],
    highres_frames: dict[str, torch.Tensor],
    highres_masks: dict[str, torch.Tensor],
    prob: float,
    block_size: int,
    block_ratio: float,
    drop_highres_masks: bool = True,
) -> dict[str, torch.Tensor]:
    stats: dict[str, torch.Tensor] = {}
    if prob <= 0.0 or block_ratio <= 0.0:
        return stats

    all_sources = [
        (source, frames)
        for source, frames in source_frames.items()
        if frames.dim() == 5 and frames.shape[1] > 0
    ]
    if not all_sources:
        return stats

    ref = all_sources[0][1]
    batch_size = ref.shape[0]
    device = ref.device
    active = (torch.rand(batch_size, device=device) < prob).to(ref.dtype)
    if float(active.sum().item()) <= 0:
        stats["masking_spatial_drop_ratio"] = _metric(0.0, ref)
        return stats

    keep_masks: dict[tuple[int, int], torch.Tensor] = {}
    dropped_sum = _metric(0.0, ref)
    dropped_count = 0

    for source, frames in all_sources:
        height, width = frames.shape[-2:]
        key = (height, width)
        if key not in keep_masks:
            keep = _make_spatial_keep_mask(
                batch_size=batch_size,
                height=height,
                width=width,
                block_size=block_size,
                block_ratio=block_ratio,
                device=device,
                dtype=frames.dtype,
            )
            keep = active[:, None, None, None] * keep + (1.0 - active[:, None, None, None])
            keep_masks[key] = keep
        keep = keep_masks[key]
        source_frames[source] = frames * keep[:, None]
        dropped_sum = dropped_sum + (1.0 - keep).mean().detach()
        dropped_count += 1

    for source, frames in list(highres_frames.items()):
        height, width = frames.shape[-2:]
        key = (height, width)
        if key not in keep_masks:
            keep = _make_spatial_keep_mask(
                batch_size=batch_size,
                height=height,
                width=width,
                block_size=block_size,
                block_ratio=block_ratio,
                device=frames.device,
                dtype=frames.dtype,
            )
            keep = active[:, None, None, None] * keep + (1.0 - active[:, None, None, None])
            keep_masks[key] = keep
        keep = keep_masks[key]
        highres_frames[source] = frames * keep
        if drop_highres_masks and source in highres_masks:
            highres_masks[source] = highres_masks[source] * F.interpolate(
                keep,
                size=highres_masks[source].shape[-2:],
                mode="nearest",
            )
        dropped_sum = dropped_sum + (1.0 - keep).mean().detach()
        dropped_count += 1

    if dropped_count > 0:
        stats["masking_spatial_drop_ratio"] = dropped_sum / dropped_count
    return stats


def apply_input_masking(
    prepared: dict[str, Any],
    config: InputMaskingConfig | dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply training-only hard masking after targets have been prepared."""
    cfg = config if isinstance(config, InputMaskingConfig) else InputMaskingConfig.from_dict(config)
    if not cfg.enabled:
        return prepared

    source_frames = prepared["source_frames"]
    source_masks = prepared["source_masks"]
    highres_frames = prepared.get("highres_frames", {})
    highres_masks = prepared.get("highres_masks", {})
    stats: dict[str, torch.Tensor] = {}

    for source, prob in cfg.modality_dropout_probs.items():
        if source in source_frames:
            frames, masks, dropped = _drop_temporal_source(
                source_frames[source],
                source_masks[source],
                prob,
                drop_masks=cfg.drop_availability_masks,
            )
            source_frames[source] = frames
            source_masks[source] = masks
            stats[f"masking_modality_drop_{source}"] = dropped
        elif source in highres_frames:
            frames, masks, dropped = _drop_highres_source(
                highres_frames[source],
                highres_masks[source],
                prob,
                drop_masks=cfg.drop_availability_masks,
            )
            highres_frames[source] = frames
            highres_masks[source] = masks
            stats[f"masking_modality_drop_{source}"] = dropped

    stats.update(
        _drop_months(
            source_frames,
            source_masks,
            cfg.month_dropout_prob,
            cfg.max_months_per_sample,
            drop_masks=cfg.drop_availability_masks,
        )
    )
    stats.update(
        _drop_spatial_blocks(
            source_frames,
            highres_frames,
            highres_masks,
            cfg.spatial_block_prob,
            cfg.spatial_block_size,
            cfg.spatial_block_ratio,
            drop_highres_masks=cfg.drop_availability_masks,
        )
    )
    prepared["masking_stats"] = stats
    return prepared
