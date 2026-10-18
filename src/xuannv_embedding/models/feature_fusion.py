"""Small shared adapter of frozen public, spectral and local-imagery features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from xuannv_embedding.models.incremental_highres import HighResResidual


@dataclass
class FeatureFusionOutput:
    embedding: torch.Tensor
    public_embedding: torch.Tensor
    worldcover_logits: torch.Tensor
    spectral_reconstruction: torch.Tensor
    semantic_reconstruction: torch.Tensor
    semantic_mask: torch.Tensor
    observation_mask: torch.Tensor


class FrozenFeatureFusion(nn.Module):
    """Keep the public vector, adapt it from observations, add an optional local residual.

    Frozen inputs are geographically corresponding NCHW grids. The eleven official
    WorldCover classes use indices 1..11; index zero is ignored by the weak loss.
    """

    def __init__(
        self, *, width: int = 128, residual_scale: float = 0.25, spectral_channels: int = 95
    ):
        super().__init__()
        if (
            type(width) is not int
            or width < 8
            or type(spectral_channels) is not int
            or spectral_channels < 1
            or type(residual_scale) not in {int, float}
            or not math.isfinite(residual_scale)
            or not 0 < residual_scale <= 1
        ):
            raise ValueError("Invalid frozen-feature fusion dimensions or residual scale")
        self.width, self.residual_scale = width, residual_scale
        self.spectral_channels = spectral_channels
        for name, channels in [("public", 64), ("spectral", spectral_channels), ("semantic", 64)]:
            self.register_buffer(name + "_mean", torch.zeros(1, channels, 1, 1))
            self.register_buffer(name + "_std", torch.ones(1, channels, 1, 1))
        self.public_adapter = nn.Sequential(
            nn.Conv2d(64 + spectral_channels, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv2d(width, 64, 1),
        )
        nn.init.zeros_(self.public_adapter[-1].weight)
        nn.init.zeros_(self.public_adapter[-1].bias)
        self.highres = HighResResidual(64, 64, native=True)
        self.worldcover_head = nn.Conv2d(64, 12, 1)
        self.spectral_head = nn.Conv2d(64, spectral_channels, 1)
        self.semantic_head = nn.Conv2d(64, 64, 1)

    def set_statistics(self, name, mean, std):
        if name not in {"public", "spectral", "semantic"}:
            raise ValueError("Unknown frozen feature source")
        target = getattr(self, name + "_mean")
        mean, std = [
            torch.as_tensor(v, dtype=torch.float32).reshape(1, -1, 1, 1) for v in [mean, std]
        ]
        if (
            mean.shape != target.shape
            or std.shape != target.shape
            or (not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any())
        ):
            raise ValueError("Finite per-source statistics and positive deviations are required")
        with torch.no_grad():
            target.copy_(mean)
            getattr(self, name + "_std").copy_(std)

    def standardize(self, value, name, mask=None):
        result = (
            (value.detach().float() - getattr(self, name + "_mean")) / getattr(self, name + "_std")
        ).clamp(-6, 6)
        return torch.where(mask, result, 0) if mask is not None else result

    def forward(
        self, public: torch.Tensor, spectral: torch.Tensor, semantic: torch.Tensor | None = None
    ) -> FeatureFusionOutput:
        if (
            public.ndim != 4
            or public.shape[1] != 64
            or spectral.shape != (public.shape[0], self.spectral_channels, *public.shape[-2:])
            or (semantic is not None and semantic.shape != public.shape)
        ):
            raise ValueError("Frozen fusion inputs require aligned 64/95/64-channel maps")
        public, spectral = public.detach(), spectral.detach()
        source_mask = spectral.abs().sum(1, keepdim=True) > 1e-8
        public_mask = public.abs().sum(1, keepdim=True) > 1e-8
        if semantic is None:
            semantic = torch.zeros_like(public)
        semantic = semantic.detach()
        semantic_mask = semantic.abs().sum(1, keepdim=True) > 1e-8
        features = torch.cat(
            [
                self.standardize(public, "public", public_mask),
                self.standardize(spectral, "spectral", source_mask),
            ],
            1,
        )
        adapted = public + self.residual_scale * self.public_adapter(features).tanh()
        local = self.standardize(semantic, "semantic", semantic_mask)
        embedding = self.highres(adapted[:, None], local, semantic_mask.float())[:, 0]
        return FeatureFusionOutput(
            embedding,
            adapted,
            self.worldcover_head(embedding),
            self.spectral_head(embedding),
            self.semantic_head(embedding),
            semantic_mask,
            # Full/no-imagery fits receive the same weak-label exposure. Optional
            # imagery may add reconstruction targets, never extra labelled pixels.
            source_mask,
        )


def _masked_mean(values, mask):
    mask = torch.broadcast_to(mask, values.shape)
    return torch.where(mask, values, 0).sum() / mask.sum().clamp_min(1)


def feature_fusion_loss(
    output,
    public_teacher,
    spectral_teacher,
    semantic_teacher,
    labels,
    *,
    spectral_mask,
    class_weights,
    spectral_weight=0.2,
    semantic_weight=0.1,
    preservation_weight=0.02,
):
    """Real weak-map CE plus normalized observation reconstruction and base anchoring.

    Spectral/semantic teachers are independently standardized frozen observations;
    public_teacher remains the original unnormalized public vector.
    """
    if (
        labels.shape != output.embedding.shape[:1] + output.embedding.shape[-2:]
        or labels.dtype != torch.long
        or (labels < 0).any()
        or (labels >= 12).any()
        or spectral_teacher.shape != output.spectral_reconstruction.shape
        or spectral_mask.shape != spectral_teacher.shape
        or spectral_mask.dtype != torch.bool
        or class_weights.shape != (12,)
        or not torch.isfinite(class_weights).all()
        or (class_weights < 0).any()
        or any(
            type(v) not in {int, float} or not math.isfinite(v) or v < 0
            for v in [spectral_weight, semantic_weight, preservation_weight]
        )
    ):
        raise ValueError("Invalid frozen fusion weak labels, masks, targets or loss weights")
    valid = (labels > 0) & output.observation_mask[:, 0]
    target = torch.where(valid, labels, 0)
    weights = class_weights.to(device=labels.device, dtype=torch.float32)
    ce = F.cross_entropy(
        output.worldcover_logits.float(), target, weight=weights, ignore_index=0, reduction="sum"
    )
    weak = ce / torch.where(valid, weights[target], 0).sum().clamp_min(1)
    spectral = _masked_mean(
        (output.spectral_reconstruction.float() - spectral_teacher.detach().float()).square(),
        spectral_mask & output.observation_mask,
    )
    semantic = output.embedding.sum() * 0
    if semantic_teacher is not None:
        if semantic_teacher.shape != output.semantic_reconstruction.shape:
            raise ValueError("Semantic teacher shape differs")
        semantic = _masked_mean(
            (output.semantic_reconstruction.float() - semantic_teacher.detach().float()).square(),
            output.semantic_mask,
        )
    public = public_teacher.detach().float()
    if public.shape != output.embedding.shape:
        raise ValueError("Public teacher shape differs")
    anchor_mask = (public.abs().sum(1) > 1e-8) & output.observation_mask[:, 0]
    preservation = _masked_mean(
        (1 - F.cosine_similarity(output.public_embedding.float(), public, dim=1)).clamp_min(0),
        anchor_mask,
    )
    return {
        "loss": weak
        + spectral_weight * spectral
        + semantic_weight * semantic
        + preservation_weight * preservation,
        "worldcover": weak,
        "spectral": spectral,
        "semantic": semantic,
        "preservation": preservation,
    }
