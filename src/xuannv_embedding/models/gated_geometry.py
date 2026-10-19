"""Optional spatial context and local feature-geometry retention."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from xuannv_embedding.models.time_modulated import TimeModulatedFusion


class GatedSpatialContext(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        if type(width) is not int or width < 8:
            raise ValueError("Invalid spatial context width")
        self.layers = nn.Sequential(
            nn.Conv2d(3 * width + 2, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2),
            nn.GroupNorm(1, width),
            nn.GELU(),
        )
        self.project = nn.Conv2d(width, width, 1)
        self.gate = nn.Conv2d(3 * width + 2, width, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2)

    def _mix(self, x):
        return self.project(self.layers(x)) * self.gate(x).sigmoid()

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self._mix, x, use_reentrant=False)
        return self._mix(x)


class GatedGeometryFusion(TimeModulatedFusion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base.local_mixer = GatedSpatialContext(kwargs.get("width", 128))
        self.collect_local_geometry = True


def relational_geometry_loss(
    student,
    teacher,
    domain,
    labels,
    *,
    step,
    samples,
    same_class_weight,
    seed,
):
    """Match masked cosine relations, emphasizing variation inside known WC classes.

    This is a cosine-Gram objective inspired by relational distillation, not the
    original RKD distance-wise/angle-wise objective. No primary reference is read.
    """
    if (
        student.ndim != 4
        or student.shape[1] != 64
        or teacher.ndim != 4
        or teacher.shape[:2] != (len(student), 1024)
        or domain.shape != (len(student), 1, *teacher.shape[-2:])
        or domain.dtype != torch.bool
        or labels.shape != (len(student), *student.shape[-2:])
        or labels.dtype != torch.long
        or (labels < 0).any()
        or (labels > 11).any()
        or type(step) is not int
        or step < 0
        or type(samples) is not int
        or not 2 <= samples <= 4096
        or type(seed) is not int
        or not 0 <= seed < 2**32
        or type(same_class_weight) not in {int, float}
        or not math.isfinite(same_class_weight)
        or not 1 <= same_class_weight <= 4
    ):
        raise ValueError("Invalid local geometry or weak-label contract")
    with torch.autocast(device_type=student.device.type, enabled=False):
        spatial = teacher.shape[-2:]
        s = F.interpolate(student.float(), size=spatial, mode="bilinear", align_corners=False)
        yy = (
            (torch.arange(spatial[0], device=labels.device).float() + 0.5)
            * labels.shape[-2]
            / spatial[0]
        ).long()
        xx = (
            (torch.arange(spatial[1], device=labels.device).float() + 0.5)
            * labels.shape[-1]
            / spatial[1]
        ).long()
        y = labels.index_select(1, yy).index_select(2, xx).flatten()
        s = s.permute(0, 2, 3, 1).reshape(-1, 64)
        t = teacher.detach().float().permute(0, 2, 3, 1).reshape(-1, 1024)
        valid = domain.flatten()
        generator = torch.Generator(device="cpu").manual_seed((seed + step) % 2**32)
        ids = torch.randperm(len(s), generator=generator)[: min(samples, len(s))].to(s.device)
        s, t, y, valid = s[ids], t[ids], y[ids], valid[ids]
        s = torch.where(valid[:, None], s, 0)
        t = torch.where(valid[:, None], t, 0)
        if not torch.isfinite(s).all() or not torch.isfinite(t).all():
            raise ValueError("Observed geometry vectors must be finite")
        s = F.normalize(s, dim=1, eps=1e-6)
        t = F.normalize(t, dim=1, eps=1e-6)
        weight = torch.where((y[:, None] == y[None, :]) & (y[:, None] > 0), same_class_weight, 1.0)
        mask = (valid[:, None] & valid[None, :]) & ~torch.eye(
            len(s), device=s.device, dtype=torch.bool
        )
        weight = torch.where(mask, weight, 0)
        error = (s @ s.T - t @ t.T).square()
        return (error * weight).sum() / weight.sum().clamp_min(1)
