"""Annual native-scale encoders, conservative geometric alignment and a single bottleneck."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from xuannv_embedding.models.blocks import STPEncoder
from xuannv_embedding.models.bottleneck import VMFBottleneck
from xuannv_embedding.models.model import AEFOutput


def overlap_weights(input_size: int, output_size: int, reference: torch.Tensor) -> torch.Tensor:
    """Exact pixel-footprint overlaps for coextensive, axis-aligned grids."""
    edges = torch.arange(output_size + 1, device=reference.device, dtype=torch.float32)
    edges = edges * (input_size / output_size)
    pixels = torch.arange(input_size, device=reference.device, dtype=torch.float32)
    overlap = torch.minimum(edges[1:, None], pixels[None] + 1)
    overlap = overlap - torch.maximum(edges[:-1, None], pixels[None])
    return overlap.clamp(min=0) / (input_size / output_size)


def area_resample(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Area-integrate features, including the noninteger 640 to 256 ratio."""
    if values.shape[-2:] == size:
        return values
    rows = overlap_weights(values.shape[-2], size[0], values)
    columns = overlap_weights(values.shape[-1], size[1], values)
    with torch.autocast(device_type=values.device.type, enabled=False):
        return torch.matmul(torch.matmul(rows, values.float()), columns.t())


def masked_resample(
    values: torch.Tensor, mask: torch.Tensor, size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample valid values and support separately, never average nodata into data."""
    clean = torch.where(mask > 0, values, 0.0)
    support = area_resample(mask.float(), size)
    numerator = area_resample(clean * mask, size)
    return numerator / support.clamp(min=1e-6), support


def seasonal_features(timestamps: torch.Tensor) -> torch.Tensor:
    months = timestamps.remainder(100)
    if bool(((months < 1) | (months > 12)).any()):
        raise ValueError("Annual timestamps require valid YYYYMM months")
    phase = (months.float() - 1) * (2 * math.pi / 12)
    return torch.stack((phase.sin(), phase.cos()), dim=-1)


class NativeEncoder(nn.Module):
    """Local context at the sensor's native scale, without a second L2 bottleneck."""

    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        groups = math.gcd(8, width)
        self.stem = nn.Conv2d(channels, width, 3, padding=1)
        self.refine = nn.Sequential(
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width, 1),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.stem(torch.where(mask > 0, values, 0.0))
        return (features + self.refine(features)) * mask


class QualityPool(nn.Module):
    """Permutation-invariant content/date pooling with hard per-pixel support."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.score = nn.Conv2d(width, 1, 1)
        self.date = nn.Linear(2, 1, bias=False)

    def weights(
        self, features: torch.Tensor, support: torch.Tensor, dates: torch.Tensor
    ) -> torch.Tensor:
        logits = self.score(features).float() + self.date(dates.float())[:, :, None, None]
        return (4 * logits.tanh()).exp() * support.float()

    def forward(
        self, features: torch.Tensor, support: torch.Tensor, dates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, observations, channels, height, width = features.shape
        weights = self.weights(features.flatten(0, 1), support.flatten(0, 1), dates.flatten(0, 1))
        weights = weights.view(batch_size, observations, 1, height, width)
        denominator = weights.sum(dim=1)
        pooled = (features.float() * weights).sum(dim=1) / denominator.clamp(min=1e-6)
        return pooled, support.amax(dim=1)


class DateDecoder(nn.Module):
    """Read a target source/date exclusively from the public embedding."""

    def __init__(self, embed_dim: int, channels: int) -> None:
        super().__init__()
        self.date = nn.Linear(2, embed_dim)
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, channels, 1),
        )

    def forward(
        self, embedding: torch.Tensor, date: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        features = area_resample(embedding, size)
        return self.head(features + self.date(date.float())[:, :, None, None])


@dataclass
class AnnualOutput(AEFOutput):
    highres_reconstructions: list[list[torch.Tensor]] = field(default_factory=list)
    support: dict[str, torch.Tensor] = field(default_factory=dict)


class Annual5mModel(nn.Module):
    """Low-resolution annual STP + native MS/PAN + gated joint 5 m decoding."""

    def __init__(
        self,
        sensor_channels: dict[str, int],
        source_roles: dict[str, str],
        pan_sources: tuple[str, ...],
        target_sources: dict[str, str],
        *,
        embed_dim: int = 64,
        feature_dim: int = 128,
        stem_dim: int = 32,
        stp: dict[str, Any],
        gradient_checkpointing: bool = False,
        backbone: str = "stp",
    ) -> None:
        super().__init__()
        self.sensor_channels = dict(sensor_channels)
        self.source_roles = dict(source_roles)
        self.pan_sources = set(pan_sources)
        self.target_sources = dict(target_sources)
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.backbone = backbone
        self.temporal_sources = tuple(
            name for name in sensor_channels if source_roles[name] == "temporal"
        )
        self.highres_sources = tuple(
            name for name in sensor_channels if source_roles[name] == "highres"
        )
        self.temporal_encoders = nn.ModuleDict(
            {name: NativeEncoder(sensor_channels[name], stem_dim) for name in self.temporal_sources}
        )
        self.source_weights = nn.Parameter(torch.zeros(len(self.temporal_sources)))
        self.temporal_encoder = (
            STPEncoder(
                input_channels=stem_dim,
                space_dim=stp["space_dim"],
                time_dim=stp["time_dim"],
                precision_dim=feature_dim,
                num_blocks=stp["num_blocks"],
                num_heads=stp["num_heads"],
                precision_scale=1,
                gradient_checkpointing=gradient_checkpointing,
                time_attention_mode=stp["time_attention_mode"],
            )
            if backbone == "stp"
            else NativeEncoder(stem_dim, feature_dim)
        )
        self.lowres_pool = QualityPool(feature_dim)
        self.highres_encoders = nn.ModuleDict(
            {
                name: NativeEncoder(
                    sensor_channels[name],
                    min(32, feature_dim) if name in self.pan_sources else feature_dim,
                )
                for name in self.highres_sources
            }
        )
        self.highres_projections = nn.ModuleDict(
            {name: nn.Conv2d(min(32, feature_dim), feature_dim, 1) for name in self.pan_sources}
        )
        self.highres_pools = nn.ModuleDict(
            {name: QualityPool(feature_dim) for name in self.highres_sources}
        )
        self.joint = nn.Sequential(
            nn.Conv2d(feature_dim * 3 + 2, feature_dim, 1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )
        self.gate = nn.Conv2d(feature_dim * 3 + 2, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.2)
        self.bottleneck = VMFBottleneck(feature_dim, embed_dim)
        self.bottleneck.log_kappa.requires_grad_(False)
        self.decoders = nn.ModuleDict(
            {
                name: DateDecoder(embed_dim, sensor_channels[source])
                for name, source in target_sources.items()
            }
        )
        self.highres_decoders = nn.ModuleDict(
            {name: DateDecoder(embed_dim, sensor_channels[name]) for name in self.highres_sources}
        )

    def forward(
        self,
        source_frames: dict[str, torch.Tensor],
        source_masks: dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        highres_frames=None,
        highres_masks=None,
        *,
        source_pixel_masks: dict[str, torch.Tensor],
        highres_observations: list[list[dict]] | None = None,
        reconstruction_requests: list[list[dict]] | None = None,
        decode: bool = True,
    ) -> AnnualOutput:
        if highres_frames or highres_masks:
            raise ValueError("Annual model requires native highres_observations")
        first = source_frames[self.temporal_sources[0]]
        batch_size, time_count = first.shape[:2]
        if timestamps.shape != (batch_size, time_count) or time_count != 12:
            raise ValueError("Annual input requires twelve YYYYMM slots")
        expected = timestamps[:, :1] // 100 * 100 + torch.arange(1, 13, device=timestamps.device)
        if not torch.equal(timestamps, expected):
            raise ValueError("Annual timestamps must be January through December of one year")
        native_size = max(
            (tuple(source_frames[name].shape[-2:]) for name in self.temporal_sources),
            key=lambda shape: shape[0] * shape[1],
        )
        output_size = (native_size[0] * 2, native_size[1] * 2)
        numerator = None
        denominator = first.new_zeros(batch_size * time_count, 1, *native_size)
        supports = []
        for index, source in enumerate(self.temporal_sources):
            values = source_frames[source]
            mask = (
                source_pixel_masks[source][:, :, None].float()
                * source_masks[source][:, :, None, None, None]
            )
            features = self.temporal_encoders[source](values.flatten(0, 1), mask.flatten(0, 1))
            features, support = masked_resample(features, mask.flatten(0, 1), native_size)
            weight = self.source_weights[index].sigmoid() * support
            contribution = features * weight
            numerator = contribution if numerator is None else numerator + contribution
            denominator = denominator + weight
            supports.append(support.view(batch_size, time_count, 1, *native_size))
        fused = numerator / denominator.clamp(min=1e-6)
        support = torch.stack(supports).amax(dim=0)
        dates = seasonal_features(timestamps)
        if self.backbone == "stp":
            features, _ = self.temporal_encoder(
                fused.view(batch_size, time_count, -1, *native_size).permute(0, 1, 3, 4, 2),
                timestamps,
                mask=support.flatten(2).any(dim=2),
            )
            features = features.permute(0, 1, 4, 2, 3)
        else:
            features = self.temporal_encoder(fused, support.flatten(0, 1))
            features = features.view(batch_size, time_count, self.feature_dim, *native_size)
        annual, low_support = self.lowres_pool(features, support, dates)
        low5 = functional.interpolate(
            annual, size=output_size, mode="bilinear", align_corners=False
        )
        validity = functional.interpolate(low_support, size=output_size, mode="nearest") > 0
        highres_observations = highres_observations or [[] for _ in range(batch_size)]
        if len(highres_observations) != batch_size:
            raise ValueError("Annual observation batch mismatch")
        multispectral, pan, ms_support, pan_support = self._highres(highres_observations, low5)
        combined = torch.cat((low5, multispectral, pan, ms_support, pan_support), dim=1)
        available = torch.maximum(ms_support, pan_support)
        refined = low5 + available * self.gate(combined).sigmoid() * self.joint(combined)
        embedding = self.bottleneck(refined)
        scene = (embedding * validity).sum(dim=(-2, -1)) / validity.sum(dim=(-2, -1)).clamp(min=1)
        scene = functional.normalize(scene, dim=1)
        reconstructions = {}
        highres_reconstructions = [[] for _ in range(batch_size)]
        if decode:
            for name, source in self.target_sources.items():
                size = tuple(source_frames[source].shape[-2:])
                reconstructions[name] = torch.stack(
                    [
                        self.decoders[name](embedding, dates[:, month], size)
                        for month in range(time_count)
                    ],
                    dim=1,
                )
            for sample_index, requests in enumerate(
                reconstruction_requests or highres_observations
            ):
                for request in requests:
                    source = request["source"]
                    prediction = self.highres_decoders[source](
                        embedding[sample_index : sample_index + 1],
                        request["date_features"][None],
                        output_size,
                    )
                    highres_reconstructions[sample_index].append(prediction[0])
        return AnnualOutput(
            embedding_map=embedding[:, None],
            embedding=scene[:, None],
            reconstructions=reconstructions,
            validity_mask=validity[:, None],
            highres_reconstructions=highres_reconstructions,
            support={"lowres": low_support, "ms5m": ms_support, "pan2m": pan_support},
        )

    def _highres(self, observations: list[list[dict]], base: torch.Tensor):
        branch_features = {name: [] for name in ("ms", "pan")}
        branch_supports = {name: [] for name in ("ms", "pan")}
        for sample_index, sample in enumerate(observations):
            sums = {
                name: torch.zeros_like(base[sample_index : sample_index + 1])
                for name in branch_features
            }
            supports = {name: base.new_zeros(1, 1, *base.shape[-2:]) for name in branch_features}
            source_sums, source_weights, source_supports = {}, {}, {}
            for observation in sample:
                source = observation["source"]
                if source not in self.highres_encoders:
                    raise ValueError(f"Unknown annual highres source: {source}")
                mask = observation["mask"][None, None].float()
                features = self.highres_encoders[source](observation["values"][None], mask)
                features, support = masked_resample(features, mask, tuple(base.shape[-2:]))
                if source in self.pan_sources:
                    features = self.highres_projections[source](features) * (support > 0)
                support = support * observation["quality_weight"]
                weights = self.highres_pools[source].weights(
                    features, support, observation["date_features"][None]
                )
                source_sums[source] = source_sums.get(source, 0) + features * weights
                source_weights[source] = source_weights.get(source, 0) + weights
                source_supports[source] = torch.maximum(
                    source_supports.get(source, torch.zeros_like(support)), support
                )
            for source, feature_sum in source_sums.items():
                branch = "pan" if source in self.pan_sources else "ms"
                confidence = source_supports[source]
                sums[branch] = (
                    sums[branch] + feature_sum / source_weights[source].clamp(min=1e-6) * confidence
                )
                supports[branch] = supports[branch] + confidence
            for branch in branch_features:
                branch_features[branch].append(sums[branch] / supports[branch].clamp(min=1e-6))
                branch_supports[branch].append(supports[branch].clamp(max=1))
        return (
            torch.cat(branch_features["ms"]),
            torch.cat(branch_features["pan"]),
            torch.cat(branch_supports["ms"]),
            torch.cat(branch_supports["pan"]),
        )
