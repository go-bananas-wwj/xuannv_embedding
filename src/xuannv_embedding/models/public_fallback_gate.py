"""Content-dependent mixtures with an explicit genuine-public expert."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from xuannv_embedding.models.content_adaptive_residual import (
    AdaptiveResidualOutput,
    ContentAdaptiveResidual,
)


@dataclass
class PublicFallbackOutput(AdaptiveResidualOutput):
    expert_weights: torch.Tensor


class PublicFallbackGate(ContentAdaptiveResidual):
    def __init__(self, *, width: int = 32, initial_weights=(0.05, 0.8075, 0.1425)) -> None:
        if (
            not isinstance(initial_weights, (list, tuple))
            or len(initial_weights) != 3
            or any(
                type(value) not in {int, float} or not math.isfinite(value) or value <= 0
                for value in initial_weights
            )
            or not math.isclose(sum(initial_weights), 1, rel_tol=0, abs_tol=1e-8)
        ):
            raise ValueError("Three positive initial expert weights must sum to one")
        super().__init__(width=width)
        self.register_buffer(
            "initial_weights",
            torch.tensor(initial_weights, dtype=torch.float32).view(1, 3, 1, 1, 1),
        )
        self.body = nn.Sequential(
            nn.Conv2d(638, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
        )
        self.project = nn.Conv2d(width, 192, 1)
        nn.init.zeros_(self.project.weight)
        with torch.no_grad():
            self.project.bias.copy_(self.initial_weights.flatten().log().repeat_interleave(64))
        for head in [self.worldcover_head, self.area_head, self.geometry_head]:
            head.requires_grad_(True)

    def forward(
        self, public, semantic, geometry, spectral, spectral_mask, foundation, *, heads_only=False
    ):
        if not self.teacher_ready.item():
            raise ValueError("Verified source statistics and initial weak heads are required")
        values = [public, semantic, geometry, foundation, spectral]
        if (
            type(heads_only) is not bool
            or any(not torch.is_tensor(value) or not value.is_floating_point() for value in values)
            or not torch.is_tensor(spectral_mask)
            or public.ndim != 4
            or public.shape[1] != 64
            or min(public.shape) < 1
            or any(value.shape != public.shape for value in values[1:4])
            or spectral.shape != (len(public), 95, *public.shape[-2:])
            or spectral_mask.shape != spectral.shape
            or spectral_mask.dtype != torch.bool
            or any(value.device != public.device for value in [*values, spectral_mask])
        ):
            raise ValueError("Public fallback source geometry or dtype differs")
        if any(not torch.isfinite(value).all() for value in values[:4]):
            raise ValueError("Nonfinite public fallback source")
        if not torch.isfinite(torch.where(spectral_mask, spectral, 0)).all():
            raise ValueError("Nonfinite observed public physical data")
        with torch.autocast(device_type=public.device.type, enabled=False):
            public, semantic, geometry, foundation = [
                value.detach().float() for value in values[:4]
            ]
            visible = public.abs().sum(1, keepdim=True) > 1e-8
            base = torch.where(
                visible, ((public - self.public_mean) / self.public_std).clamp(-6, 6), 0
            )
            sources = [foundation, semantic, geometry]
            initial = sum(self.initial_weights[:, i] * value for i, value in enumerate(sources))
            if not heads_only:
                first, second, third = [(value - public) / self.public_std for value in sources]
                mask = spectral_mask.detach()
                spectral = torch.where(mask, spectral.detach().float(), 0)
                physical = torch.where(
                    mask,
                    ((spectral - self.spectral_mean) / self.spectral_std).clamp(-6, 6),
                    0,
                )
                conditioned = torch.cat(
                    [
                        base,
                        first,
                        second,
                        third,
                        (second - third).abs(),
                        (first - second).abs(),
                        (first - third).abs(),
                        physical,
                        mask.float(),
                    ],
                    1,
                )
                if not torch.isfinite(conditioned).all():
                    raise ValueError("Nonfinite standardized public fallback input")
        if heads_only:
            weights = self.initial_weights.expand(len(public), 3, 64, *public.shape[-2:])
        else:
            hidden = self.body(conditioned)
            with torch.autocast(device_type=public.device.type, enabled=False):
                logits = F.conv2d(
                    hidden.float(), self.project.weight.float(), self.project.bias.float()
                )
                weights = logits.reshape(len(public), 3, 64, *public.shape[-2:]).softmax(1)
        with torch.autocast(device_type=public.device.type, enabled=False):
            embedding = sum(weights[:, i] * value for i, value in enumerate(sources))
            white = base + (embedding - public) / self.public_std

            def logits(head):
                return F.conv2d(white, head.weight.float(), head.bias.float())

            return PublicFallbackOutput(
                embedding,
                weights[:, 2],
                (embedding - initial) / self.public_std,
                logits(self.worldcover_head),
                logits(self.area_head),
                logits(self.geometry_head),
                weights,
            )
