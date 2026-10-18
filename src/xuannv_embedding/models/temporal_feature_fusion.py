"""Calendar-aware public reconstruction with a masked local-visual residual."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

SLICES = ((0, 11), (11, 13), (13, 19))


def block(incoming: int, outgoing: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(incoming, outgoing, 3, padding=1),
        nn.GroupNorm(1, outgoing),
        nn.GELU(),
        nn.Conv2d(outgoing, outgoing, 3, padding=1),
        nn.GroupNorm(1, outgoing),
        nn.GELU(),
    )


class SpatialContext(nn.Module):
    def __init__(self, channels: int, width: int):
        super().__init__()
        self.first = block(channels, width)
        self.second = block(width, 2 * width)
        self.bottom = block(2 * width, 4 * width)
        self.up_second = block(6 * width, 2 * width)
        self.up_first = block(3 * width, width)
        self.output = nn.Conv2d(width, 64, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.first(value)
        second = self.second(F.avg_pool2d(first, 2))
        bottom = self.bottom(F.avg_pool2d(second, 2))
        value = self.up_second(
            torch.cat(
                [
                    F.interpolate(
                        bottom, size=second.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    second,
                ],
                1,
            )
        )
        value = self.up_first(
            torch.cat(
                [
                    F.interpolate(
                        value, size=first.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    first,
                ],
                1,
            )
        )
        return self.output(value)


@dataclass
class TemporalFeatureOutput:
    monthly_embedding: torch.Tensor
    public_monthly_embedding: torch.Tensor
    embedding: torch.Tensor
    worldcover_logits: torch.Tensor
    observation_reconstruction: torch.Tensor
    public_reconstruction: torch.Tensor
    local_reconstruction: torch.Tensor
    public_mask: torch.Tensor
    local_mask: torch.Tensor


class TemporalFeatureFusion(nn.Module):
    def __init__(self, *, width: int = 32, residual_scale: float = 0.5):
        super().__init__()
        if (
            type(width) is not int
            or width < 8
            or type(residual_scale) not in {int, float}
            or not math.isfinite(residual_scale)
            or not 0 < residual_scale <= 1
        ):
            raise ValueError("Invalid temporal feature model width or residual scale")
        self.residual_scale = residual_scale
        self.sensor_stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(stop - start + 1, width, 1), nn.GroupNorm(1, width), nn.GELU()
                )
                for start, stop in SLICES
            ]
        )
        self.frame_projection = nn.Conv2d(3 * width + 3, 64, 1)
        self.context = SpatialContext(5 * 64 + 64 + 5 + 3 + 2, width)
        self.monthly_projection = nn.Sequential(
            nn.Conv2d(132, 128, 1), nn.GroupNorm(1, 128), nn.GELU(), nn.Conv2d(128, 64, 1)
        )
        self.local_stem = block(64, 64)
        self.local_gate = nn.Conv2d(129, 1, 1)
        self.local_correction = nn.Conv2d(128, 64, 1)
        nn.init.zeros_(self.local_correction.weight)
        nn.init.zeros_(self.local_correction.bias)
        self.worldcover_head = nn.Conv2d(64, 12, 1)
        self.observation_head = nn.Conv2d(64, 19, 1)
        self.public_head = nn.Conv2d(64, 64, 1)
        self.local_head = nn.Conv2d(64, 64, 1)
        for name, channels in [("public", 64), ("local", 64), ("terrain", 5)]:
            self.register_buffer(name + "_mean", torch.zeros(1, channels, 1, 1))
            self.register_buffer(name + "_std", torch.ones(1, channels, 1, 1))

    def set_statistics(self, name: str, mean, std) -> None:
        if name not in {"public", "local", "terrain"}:
            raise ValueError("Unknown temporal static source")
        target = getattr(self, name + "_mean")
        values = [torch.as_tensor(a, dtype=torch.float32).reshape(1, -1, 1, 1) for a in [mean, std]]
        if (
            any(a.shape != target.shape or not torch.isfinite(a).all() for a in values)
            or (values[1] <= 0).any()
        ):
            raise ValueError("Invalid temporal static source statistics")
        with torch.no_grad():
            target.copy_(values[0])
            getattr(self, name + "_std").copy_(values[1])

    def standardize(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return (
            (value.detach().float() - getattr(self, name + "_mean")) / getattr(self, name + "_std")
        ).clamp(-6, 6)

    def forward(self, frames, masks, months, public, terrain, local=None) -> TemporalFeatureOutput:
        if (
            frames.ndim != 5
            or frames.shape[1:3] != (12, 19)
            or masks.shape != (frames.shape[0], 12, 3, *frames.shape[-2:])
            or masks.dtype != torch.bool
        ):
            raise ValueError("Temporal inputs require B12x19 frames and B12x3 boolean masks")
        b, t, _, h, w = frames.shape
        if (
            min(h, w) < 4
            or public.shape != (b, 64, h, w)
            or terrain.shape != (b, 5, h, w)
            or (local is not None and local.shape != public.shape)
        ):
            raise ValueError("Temporal public, local and terrain grids must coincide")
        if months.shape == (t,):
            months = months[None].expand(b, -1)
        if (
            months.shape != (b, t)
            or months.dtype not in [torch.int32, torch.int64]
            or (months < 190001).any()
            or ((months % 100 < 1) | (months % 100 > 12)).any()
        ):
            raise ValueError("Invalid temporal calendar timestamps")
        phase = (months % 100 - 1).float() * (2 * math.pi / 12)
        calendar = torch.stack([phase.sin(), phase.cos(), (2 * phase).sin(), (2 * phase).cos()], 2)
        basis = torch.cat([torch.ones_like(calendar[:, :, :1]), calendar], 2)
        frames = frames.detach()
        encoded = []
        for j, ((start, stop), stem) in enumerate(zip(SLICES, self.sensor_stems)):
            mask = masks[:, :, j : j + 1]
            value = torch.where(mask, frames[:, :, start:stop], 0)
            value = torch.cat([value, mask.to(value.dtype)], 2).flatten(0, 1)
            encoded.append(stem(value).reshape(b, t, -1, h, w) * mask)
        features = self.frame_projection(
            torch.cat([*encoded, masks.to(frames.dtype)], 2).flatten(0, 1)
        ).reshape(b, t, 64, h, w)
        observed = masks.any(2)
        denominator = observed.sum(1).clamp_min(1)[:, None]
        summaries = [
            (features * observed[:, :, None] * basis[:, :, k, None, None, None]).sum(1)
            / denominator
            for k in range(5)
        ]
        public_mask = public.detach().abs().sum(1, keepdim=True) > 1e-8
        terrain_mask = terrain.detach().abs().sum(1, keepdim=True) > 1e-8
        clean_public = torch.where(public_mask, self.standardize(public, "public"), 0)
        clean_terrain = torch.where(terrain_mask, self.standardize(terrain, "terrain"), 0)
        context = self.context(
            torch.cat(
                [
                    *summaries,
                    clean_public,
                    clean_terrain,
                    masks.float().mean(1),
                    public_mask.float(),
                    terrain_mask.float(),
                ],
                1,
            )
        )
        repeated_context = context[:, None].expand(-1, t, -1, -1, -1)
        dates = calendar[:, :, :, None, None].expand(-1, -1, -1, h, w)
        public_monthly = self.monthly_projection(
            torch.cat([features, repeated_context, dates], 2).flatten(0, 1)
        ).reshape(b, t, 64, h, w)
        local = torch.zeros_like(public) if local is None else local.detach()
        local_mask = local.abs().sum(1, keepdim=True) > 1e-8
        local_features = (
            self.local_stem(torch.where(local_mask, self.standardize(local, "local"), 0))
            * local_mask
        )
        gate = self.local_gate(
            torch.cat([public_monthly.mean(1), local_features, local_mask.float()], 1)
        ).sigmoid()
        repeated_local = local_features[:, None].expand(-1, t, -1, -1, -1)
        delta = self.local_correction(
            torch.cat([public_monthly, repeated_local], 2).flatten(0, 1)
        ).reshape(b, t, 64, h, w)
        monthly = (
            public_monthly
            + self.residual_scale * delta.tanh() * gate[:, None] * local_mask[:, None]
        )
        embedding = monthly.mean(1)
        return TemporalFeatureOutput(
            monthly,
            public_monthly,
            embedding,
            self.worldcover_head(embedding),
            self.observation_head(monthly.flatten(0, 1)).reshape(b, t, 19, h, w),
            self.public_head(embedding),
            self.local_head(embedding),
            public_mask,
            local_mask,
        )


def _reconstruction(prediction, target, mask):
    valid = torch.broadcast_to(mask, prediction.shape)
    prediction = torch.where(valid, prediction.float(), 0)
    target = torch.where(valid, target.detach().float(), 0)
    return (prediction - target).square().sum() / valid.sum().clamp_min(1)


def temporal_feature_loss(
    output,
    frames,
    masks,
    public,
    local,
    labels,
    class_weights,
    model,
    *,
    worldcover_weight=0.45,
    observation_weight=1.0,
    visual_weight=0.1,
    preservation_weight=0.02,
):
    if (
        labels.shape != output.embedding.shape[:1] + output.embedding.shape[-2:]
        or labels.dtype != torch.long
        or (labels < 0).any()
        or (labels > 11).any()
        or class_weights.shape != (12,)
        or not torch.isfinite(class_weights).all()
        or (class_weights < 0).any()
        or any(
            type(v) not in {int, float} or not math.isfinite(v) or v < 0
            for v in [worldcover_weight, observation_weight, visual_weight, preservation_weight]
        )
    ):
        raise ValueError("Invalid temporal weak targets or loss weights")
    domain = masks.any((1, 2)) & (labels > 0)
    target = torch.where(domain, labels, 0)
    weights = class_weights.to(device=labels.device, dtype=torch.float32)
    weak = F.cross_entropy(
        output.worldcover_logits.float(), target, weight=weights, ignore_index=0, reduction="sum"
    ) / torch.where(domain, weights[target], 0).sum().clamp_min(1)
    observation = sum(
        weight
        * _reconstruction(
            output.observation_reconstruction[:, :, start:stop],
            frames[:, :, start:stop],
            masks[:, :, j : j + 1],
        )
        for j, ((start, stop), weight) in enumerate(zip(SLICES, [0.4, 0.125, 0.225]))
    )
    public_mask = public.detach().abs().sum(1, keepdim=True) > 1e-8
    public_loss = _reconstruction(
        output.public_reconstruction, model.standardize(public, "public"), public_mask
    )
    local_loss = output.embedding.sum() * 0
    if local is not None:
        local_mask = local.detach().abs().sum(1, keepdim=True) > 1e-8
        local_loss = _reconstruction(
            output.local_reconstruction, model.standardize(local, "local"), local_mask
        )
    preservation = (
        (output.monthly_embedding - output.public_monthly_embedding).float().square().mean()
    )
    return {
        "loss": worldcover_weight * weak
        + observation_weight * observation
        + visual_weight * (public_loss + local_loss)
        + preservation_weight * preservation,
        "worldcover": weak,
        "observation": observation,
        "public_visual": public_loss,
        "local_visual": local_loss,
        "preservation": preservation,
    }
