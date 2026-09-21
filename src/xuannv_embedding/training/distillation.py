"""Frozen-teacher latent prediction for the annual multi-view objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_latent_prediction_loss(
    student_map: torch.Tensor,
    teacher_map: torch.Tensor,
    validity_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compare student and frozen-teacher maps only where the target is valid.

    Both maps use ``[B, M, D, H, W]``. The teacher is detached inside this
    function so callers cannot accidentally update the frozen target network.
    Cosine distance is used because the model's public bottleneck lives on the
    unit sphere.
    """
    if student_map.shape != teacher_map.shape or student_map.ndim != 5:
        raise ValueError("student_map 与 teacher_map 必须同为 [B,M,D,H,W] 且形状一致")
    student = F.normalize(student_map.float(), dim=2)
    teacher = F.normalize(teacher_map.detach().float(), dim=2)
    error = 1.0 - (student * teacher).sum(dim=2)
    if validity_mask is None:
        valid = torch.ones_like(error)
    else:
        valid = validity_mask.to(device=error.device, dtype=error.dtype)
        if valid.ndim == 5:
            valid = valid.squeeze(2)
        if valid.ndim == 2:
            valid = valid[:, :, None, None]
        try:
            valid = valid.expand_as(error)
        except RuntimeError as exc:
            raise ValueError("validity_mask 必须可广播为 [B,M,H,W]") from exc
    return torch.where(valid > 0, error * valid, 0.0).sum() / valid.sum().clamp(min=1.0)


def freeze_teacher(model: torch.nn.Module) -> torch.nn.Module:
    """Put a teacher in deterministic inference mode and disable gradients."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
