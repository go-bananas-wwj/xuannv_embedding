"""Bounded content-dependent corrections with frozen, provenance-bound weak heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AdaptiveResidualOutput:
    embedding: torch.Tensor
    alpha: torch.Tensor
    standardized_change: torch.Tensor
    worldcover_logits: torch.Tensor
    area_logits: torch.Tensor
    geometry_logits: torch.Tensor


class ContentAdaptiveResidual(nn.Module):
    def __init__(self, *, width: int = 32, initial_alpha: float = 0.15) -> None:
        super().__init__()
        if (
            type(width) is not int
            or width < 8
            or type(initial_alpha) not in {int, float}
            or not math.isfinite(initial_alpha)
            or not 0 < initial_alpha < 1
        ):
            raise ValueError("Invalid adaptive residual width or initial coefficient")
        for name, channels in [("public", 64), ("spectral", 95)]:
            self.register_buffer(name + "_mean", torch.zeros(1, channels, 1, 1))
            self.register_buffer(name + "_std", torch.ones(1, channels, 1, 1))
        self.register_buffer("teacher_ready", torch.tensor(False))
        self.worldcover_head = nn.Conv2d(64, 12, 1)
        self.area_head = nn.Conv2d(64, 7, 1)
        self.geometry_head = nn.Conv2d(64, 4, 1)
        for head in [self.worldcover_head, self.area_head, self.geometry_head]:
            head.requires_grad_(False)
        self.body = nn.Sequential(
            nn.Conv2d(446, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
        )
        self.project = nn.Conv2d(width, 64, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.constant_(self.project.bias, math.log(initial_alpha / (1 - initial_alpha)))

    def set_teacher(self, values: dict[str, torch.Tensor]) -> None:
        targets = {
            name: getattr(self, name)
            for name in ["public_mean", "public_std", "spectral_mean", "spectral_std"]
        }
        for name in ["worldcover", "area", "geometry"]:
            head = getattr(self, name + "_head")
            targets[name + "_head.weight"] = head.weight
            targets[name + "_head.bias"] = head.bias
        if not isinstance(values, dict) or set(values) != set(targets):
            raise ValueError("Frozen teacher fields differ")
        for name, value in values.items():
            if (
                not torch.is_tensor(value)
                or not value.is_floating_point()
                or value.shape != targets[name].shape
                or not torch.isfinite(value).all()
                or (name.endswith("_std") and (value <= 0).any())
            ):
                raise ValueError("Invalid frozen teacher tensor")
        with torch.no_grad():
            for name, target in targets.items():
                target.copy_(values[name])
            self.teacher_ready.fill_(True)

    def forward(self, public, semantic, geometry, spectral, spectral_mask):
        if not self.teacher_ready.item():
            raise ValueError("A verified frozen teacher must be installed")
        values = [public, semantic, geometry, spectral]
        if (
            any(not torch.is_tensor(value) or not value.is_floating_point() for value in values)
            or not torch.is_tensor(spectral_mask)
            or public.ndim != 4
            or public.shape[1] != 64
            or min(public.shape) < 1
            or semantic.shape != public.shape
            or geometry.shape != public.shape
            or spectral.shape != (len(public), 95, *public.shape[-2:])
            or spectral_mask.shape != spectral.shape
            or spectral_mask.dtype != torch.bool
            or any(value.device != public.device for value in [*values, spectral_mask])
        ):
            raise ValueError("Adaptive residual source geometry or dtype differs")
        if any(not torch.isfinite(value).all() for value in [public, semantic, geometry]):
            raise ValueError("Nonfinite observed source features")
        if not torch.isfinite(torch.where(spectral_mask, spectral, 0)).all():
            raise ValueError("Nonfinite observed physical features")
        with torch.autocast(device_type=public.device.type, enabled=False):
            public, semantic, geometry = [value.detach().float() for value in values[:3]]
            mask = spectral_mask.detach()
            spectral = torch.where(mask, spectral.detach().float(), 0)
            visible = public.abs().sum(1, keepdim=True) > 1e-8
            base = torch.where(
                visible, ((public - self.public_mean) / self.public_std).clamp(-6, 6), 0
            )
            first = (semantic - public) / self.public_std
            second = (geometry - public) / self.public_std
            physical = torch.where(
                mask,
                ((spectral - self.spectral_mean) / self.spectral_std).clamp(-6, 6),
                0,
            )
            conditioned = torch.cat(
                [base, first, second, (first - second).abs(), physical, mask.float()], 1
            )
            if not torch.isfinite(conditioned).all():
                raise ValueError("Nonfinite standardized gate input")
        hidden = self.body(conditioned)
        with torch.autocast(device_type=public.device.type, enabled=False):
            alpha = F.conv2d(
                hidden.float(), self.project.weight.float(), self.project.bias.float()
            ).sigmoid()
            embedding = (1 - alpha) * semantic + alpha * geometry
            white = base + (embedding - public) / self.public_std

            def logits(head):
                return F.conv2d(white, head.weight.float(), head.bias.float())

            return AdaptiveResidualOutput(
                embedding,
                alpha,
                (embedding - semantic) / self.public_std,
                logits(self.worldcover_head),
                logits(self.area_head),
                logits(self.geometry_head),
            )


def adaptive_residual_loss(output, labels, domain, class_weights, *, preservation_weight):
    if (
        not isinstance(labels, dict)
        or not isinstance(class_weights, dict)
        or not torch.is_tensor(domain)
        or set(labels) != {"worldcover", "area", "geometry"}
        or set(class_weights) != set(labels)
        or domain.dtype != torch.bool
        or domain.shape != (len(output.embedding), *output.embedding.shape[-2:])
        or type(preservation_weight) not in {int, float}
        or not math.isfinite(preservation_weight)
        or preservation_weight < 0
    ):
        raise ValueError("Invalid adaptive weak-loss contract")
    with torch.autocast(device_type=output.embedding.device.type, enabled=False):
        losses = {}
        for name in ["worldcover", "area", "geometry"]:
            logits = getattr(output, name + "_logits")
            y, weights = labels[name], class_weights[name]
            if (
                not torch.is_tensor(y)
                or not torch.is_tensor(weights)
                or y.shape != domain.shape
                or y.dtype != torch.long
                or (y < 0).any()
                or (y >= logits.shape[1]).any()
                or weights.shape != (logits.shape[1],)
                or not torch.isfinite(weights).all()
                or (weights < 0).any()
            ):
                raise ValueError("Invalid known weak labels or class weights")
            valid = domain & (y > 0)
            y = torch.where(valid, y, 0)
            weights = weights.to(logits.device).float()
            ce = F.cross_entropy(logits.float(), y, weight=weights, ignore_index=0, reduction="sum")
            losses[name] = ce / torch.where(valid, weights[y], 0).sum().clamp_min(1)
        change = torch.where(domain[:, None], output.standardized_change.float(), 0)
        losses["preservation"] = change.square().sum() / (
            domain.sum() * output.embedding.shape[1]
        ).clamp_min(1)
        losses["loss"] = (
            losses["worldcover"]
            + losses["area"]
            + 0.5 * losses["geometry"]
            + preservation_weight * losses["preservation"]
        )
    return losses
