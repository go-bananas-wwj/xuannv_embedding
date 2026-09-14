"""Xuannv V2 interval-query, multi-product, multi-resolution embedding model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from xuannv_embedding.config import V2Config
from xuannv_embedding.data.contracts import ProductSpec
from xuannv_embedding.models.bottleneck import VMFBottleneck
from xuannv_embedding.models.interval_fusion import (
    IntervalAttention,
    ProductGatedFusion,
    TemporalMode,
    interval_observation_mask,
    limit_observations,
)
from xuannv_embedding.models.multires_adapters import (
    DenseProductAdapter,
    MultiResolutionProductAdapter,
)


def _compatible_heads(dim: int, requested: int) -> int:
    return next(heads for heads in range(min(dim, requested), 0, -1) if dim % heads == 0)


@dataclass
class XuannvV2Output:
    embedding_map: torch.Tensor
    embedding: torch.Tensor
    reconstructions: dict[str, torch.Tensor]
    highres_detail_stats: dict[str, torch.Tensor]
    observation_selection: dict[str, torch.Tensor]


class _TransformerCore(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.num_heads = _compatible_heads(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, self.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self, values: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = self.norm1(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        values = values + attended
        return values + self.mlp(self.norm2(values))


class _FactorizedSTPBlock(nn.Module):
    """Large-capacity STP block with bounded spatial attention tokens."""

    def __init__(
        self,
        precision_dim: int,
        spatial_dim: int,
        temporal_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.precision_dim = precision_dim
        self.spatial_in = nn.Linear(precision_dim, spatial_dim)
        self.spatial_core = _TransformerCore(spatial_dim, num_heads, mlp_ratio=6)
        self.spatial_out = nn.Linear(spatial_dim, precision_dim)
        self.temporal_in = nn.Linear(precision_dim, temporal_dim)
        self.temporal_core = _TransformerCore(temporal_dim, num_heads, mlp_ratio=4)
        self.temporal_out = nn.Linear(temporal_dim, precision_dim)
        self.local = nn.Sequential(
            nn.GroupNorm(_compatible_heads(precision_dim, 8), precision_dim),
            nn.Conv2d(precision_dim, precision_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(precision_dim * 4, precision_dim, kernel_size=1),
        )

    def forward(
        self,
        values: torch.Tensor,
        output_intervals: torch.Tensor,
        temporal_mode: TemporalMode,
    ) -> torch.Tensor:
        batch, outputs, channels, height, width = values.shape
        flat = values.reshape(batch * outputs, channels, height, width)
        pooled = F.adaptive_avg_pool2d(flat, (4, 4))
        spatial_tokens = pooled.flatten(2).transpose(1, 2)
        spatial_tokens = self.spatial_out(self.spatial_core(self.spatial_in(spatial_tokens)))
        spatial = spatial_tokens.transpose(1, 2).reshape(batch * outputs, channels, 4, 4)
        spatial = F.interpolate(spatial, size=(height, width), mode="bilinear", align_corners=False)
        values = values + spatial.view(batch, outputs, channels, height, width)

        temporal_tokens = values.mean(dim=(-2, -1))
        temporal_mask: torch.Tensor | None = None
        if temporal_mode == "causal_window":
            ends = output_intervals[..., 1]
            temporal_mask = ends[:, None, :] > ends[:, :, None]
        elif temporal_mode == "within_period":
            temporal_mask = ~torch.eye(outputs, device=values.device, dtype=torch.bool).expand(
                batch, -1, -1
            )
        if temporal_mask is not None:
            temporal_mask = temporal_mask.repeat_interleave(self.temporal_core.num_heads, dim=0)
        temporal = self.temporal_out(
            self.temporal_core(self.temporal_in(temporal_tokens), temporal_mask)
        )
        values = values + temporal[:, :, :, None, None]
        local = self.local(values.reshape(batch * outputs, channels, height, width))
        return values + local.view(batch, outputs, channels, height, width)


class XuannvV2Model(nn.Module):
    """V2 model with independent product timelines and native high-resolution learning."""

    def __init__(
        self,
        products: dict[str, ProductSpec],
        *,
        embedding_dim: int,
        stem_dim: int,
        spatial_dim: int,
        temporal_dim: int,
        precision_dim: int,
        num_blocks: int,
        num_heads: int,
        temporal_mode: TemporalMode,
        dense_lookback_days: int = 365,
        highres_structure_days: int = 730,
        highres_appearance_days: int = 90,
        highres_structure_max_observations: int = 8,
        highres_appearance_max_observations: int = 4,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.products = dict(products)
        self.temporal_mode = temporal_mode
        self.dense_lookback_days = dense_lookback_days
        self.highres_structure_days = highres_structure_days
        self.highres_appearance_days = highres_appearance_days
        self.highres_structure_max_observations = highres_structure_max_observations
        self.highres_appearance_max_observations = highres_appearance_max_observations
        self.gradient_checkpointing = gradient_checkpointing
        dense = {name: spec for name, spec in products.items() if spec.role == "dense"}
        highres = {name: spec for name, spec in products.items() if spec.role == "highres"}
        if not dense:
            raise ValueError("XuannvV2Model 至少需要一个 dense 产品")
        attention_heads = _compatible_heads(stem_dim, num_heads)
        self.dense_adapters = nn.ModuleDict(
            {name: DenseProductAdapter(spec, stem_dim) for name, spec in dense.items()}
        )
        self.dense_attention = nn.ModuleDict(
            {name: IntervalAttention(stem_dim, attention_heads) for name in dense}
        )
        self.dense_fusion = ProductGatedFusion(list(dense), stem_dim)
        self.highres_adapters = nn.ModuleDict(
            {name: MultiResolutionProductAdapter(spec, stem_dim) for name, spec in highres.items()}
        )
        self.highres_structure_attention = nn.ModuleDict(
            {name: IntervalAttention(stem_dim, attention_heads) for name in highres}
        )
        self.highres_appearance_attention = nn.ModuleDict(
            {name: IntervalAttention(stem_dim, attention_heads) for name in highres}
        )
        self.highres_structure_fusion = (
            ProductGatedFusion(list(highres), stem_dim) if highres else None
        )
        self.highres_appearance_fusion = (
            ProductGatedFusion(list(highres), stem_dim) if highres else None
        )
        self.highres_merge = nn.Sequential(
            nn.Conv2d(stem_dim * 4, stem_dim, kernel_size=1),
            nn.GroupNorm(_compatible_heads(stem_dim, 8), stem_dim),
            nn.GELU(),
        )
        self.highres_gate = nn.Conv2d(stem_dim * 4, stem_dim, kernel_size=1)
        self.precision_projection = nn.Conv2d(stem_dim, precision_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _FactorizedSTPBlock(precision_dim, spatial_dim, temporal_dim, num_heads)
                for _ in range(num_blocks)
            ]
        )
        self.pre_bottleneck = nn.Sequential(
            nn.GroupNorm(_compatible_heads(precision_dim, 8), precision_dim),
            nn.Conv2d(precision_dim, precision_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.bottleneck = VMFBottleneck(precision_dim, embedding_dim)
        self.reconstruction_heads = nn.ModuleDict(
            {
                name: nn.Conv2d(precision_dim, len(spec.bands), kernel_size=1)
                for name, spec in dense.items()
            }
        )
        self.detail_heads = nn.ModuleDict(
            {name: nn.Conv2d(precision_dim, 3, kernel_size=1) for name in highres}
        )

    @classmethod
    def from_config(cls, config: V2Config) -> "XuannvV2Model":
        products = {
            product_id: product.to_product_spec(product_id)
            for product_id, product in config.products.items()
            if product.role in {"dense", "highres"}
        }
        return cls(
            products,
            embedding_dim=config.model.embedding_dim,
            stem_dim=config.model.stem_dim,
            spatial_dim=config.model.spatial_dim,
            temporal_dim=config.model.temporal_dim,
            precision_dim=config.model.precision_dim,
            num_blocks=config.model.num_blocks,
            num_heads=config.model.num_heads,
            temporal_mode=config.temporal.mode,
            dense_lookback_days=config.temporal.dense_lookback_days,
            highres_structure_days=config.temporal.highres_structure_days,
            highres_appearance_days=config.temporal.highres_appearance_days,
            highres_structure_max_observations=(config.temporal.highres_structure_max_observations),
            highres_appearance_max_observations=(
                config.temporal.highres_appearance_max_observations
            ),
            gradient_checkpointing=config.model.gradient_checkpointing,
        )

    def _dense_features(
        self,
        source_frames: dict[str, torch.Tensor],
        source_pixel_masks: dict[str, torch.Tensor],
        source_observation_masks: dict[str, torch.Tensor],
        source_time_bounds: dict[str, torch.Tensor],
        source_available_at: dict[str, torch.Tensor],
        output_intervals: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        summaries: dict[str, torch.Tensor] = {}
        availability: dict[str, torch.Tensor] = {}
        selections: dict[str, torch.Tensor] = {}
        for product_id, adapter in self.dense_adapters.items():
            required = (
                source_frames,
                source_pixel_masks,
                source_observation_masks,
                source_time_bounds,
                source_available_at,
            )
            if any(product_id not in values for values in required):
                raise KeyError(f"dense batch 缺少产品字段: {product_id}")
            encoded = adapter(
                source_frames[product_id],
                source_pixel_masks[product_id],
                source_observation_masks[product_id],
                output_size,
            )
            selected = interval_observation_mask(
                source_time_bounds[product_id],
                source_available_at[product_id],
                source_observation_masks[product_id],
                output_intervals,
                mode=self.temporal_mode,
                window_days=self.dense_lookback_days,
            )
            selections[product_id] = selected
            summaries[product_id], availability[product_id] = self.dense_attention[product_id](
                encoded,
                source_time_bounds[product_id],
                output_intervals,
                selected,
            )
        return self.dense_fusion(summaries, availability), selections

    def _highres_features(
        self,
        highres_frames: dict[str, torch.Tensor] | None,
        highres_masks: dict[str, torch.Tensor] | None,
        highres_acquired_at: dict[str, torch.Tensor] | None,
        highres_available_at: dict[str, torch.Tensor] | None,
        highres_geotransforms: dict[str, torch.Tensor] | None,
        output_geotransforms: torch.Tensor | None,
        output_intervals: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]] | None:
        if not highres_frames:
            return None
        if any(
            value is None
            for value in (
                highres_masks,
                highres_acquired_at,
                highres_available_at,
                highres_geotransforms,
            )
        ):
            raise ValueError("高分输入必须同时提供 masks、时间和 geotransforms")
        assert highres_masks is not None
        assert highres_acquired_at is not None
        assert highres_available_at is not None
        assert highres_geotransforms is not None
        structure_maps: dict[str, torch.Tensor] = {}
        appearance_maps: dict[str, torch.Tensor] = {}
        structure_available: dict[str, torch.Tensor] = {}
        appearance_available: dict[str, torch.Tensor] = {}
        selections: dict[str, torch.Tensor] = {}
        for product_id, frames in highres_frames.items():
            if product_id not in self.highres_adapters:
                raise KeyError(f"未知 highres product: {product_id}")
            encoded, _, quality = self.highres_adapters[product_id](
                frames,
                highres_masks[product_id],
                highres_geotransforms[product_id],
                output_size,
                output_geotransforms,
            )
            times = highres_acquired_at[product_id]
            bounds = torch.stack((times, times), dim=-1)
            observation_mask = quality > 0
            structure_selected = interval_observation_mask(
                bounds,
                highres_available_at[product_id],
                observation_mask,
                output_intervals,
                mode=self.temporal_mode,
                window_days=self.highres_structure_days,
            )
            appearance_selected = interval_observation_mask(
                bounds,
                highres_available_at[product_id],
                observation_mask,
                output_intervals,
                mode=self.temporal_mode,
                window_days=self.highres_appearance_days,
            )
            structure_selected = limit_observations(
                structure_selected,
                times,
                output_intervals,
                quality,
                self.highres_structure_max_observations,
            )
            appearance_selected = limit_observations(
                appearance_selected,
                times,
                output_intervals,
                quality,
                self.highres_appearance_max_observations,
            )
            selections[product_id] = structure_selected | appearance_selected
            structure_maps[product_id], structure_available[product_id] = (
                self.highres_structure_attention[product_id](
                    encoded, bounds, output_intervals, structure_selected
                )
            )
            appearance_maps[product_id], appearance_available[product_id] = (
                self.highres_appearance_attention[product_id](
                    encoded, bounds, output_intervals, appearance_selected
                )
            )
        assert self.highres_structure_fusion is not None
        assert self.highres_appearance_fusion is not None
        return (
            self.highres_structure_fusion(structure_maps, structure_available),
            self.highres_appearance_fusion(appearance_maps, appearance_available),
            selections,
        )

    def forward(
        self,
        source_frames: dict[str, torch.Tensor],
        source_pixel_masks: dict[str, torch.Tensor],
        source_observation_masks: dict[str, torch.Tensor],
        source_time_bounds: dict[str, torch.Tensor],
        source_available_at: dict[str, torch.Tensor],
        output_intervals: torch.Tensor,
        highres_frames: dict[str, torch.Tensor] | None = None,
        highres_masks: dict[str, torch.Tensor] | None = None,
        highres_acquired_at: dict[str, torch.Tensor] | None = None,
        highres_available_at: dict[str, torch.Tensor] | None = None,
        highres_geotransforms: dict[str, torch.Tensor] | None = None,
        output_geotransforms: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> XuannvV2Output:
        if output_intervals.ndim != 3 or output_intervals.shape[-1] != 2:
            raise ValueError("output_intervals 必须为 [B,M,2]")
        if output_size is None:
            first = next(iter(source_frames.values()))
            output_size = (first.shape[-2], first.shape[-1])
        fused, observation_selection = self._dense_features(
            source_frames,
            source_pixel_masks,
            source_observation_masks,
            source_time_bounds,
            source_available_at,
            output_intervals,
            output_size,
        )
        highres = self._highres_features(
            highres_frames,
            highres_masks,
            highres_acquired_at,
            highres_available_at,
            highres_geotransforms,
            output_geotransforms,
            output_intervals,
            output_size,
        )
        batch, outputs, channels, height, width = fused.shape
        if highres is not None:
            structure, appearance, highres_selection = highres
            observation_selection.update(highres_selection)
            combined = torch.cat(
                (fused, structure, appearance, (structure - appearance).abs()), dim=2
            )
            flat = combined.reshape(batch * outputs, channels * 4, height, width)
            candidate = self.highres_merge(flat)
            gate = torch.sigmoid(self.highres_gate(flat))
            fused = fused + (gate * candidate).view(batch, outputs, channels, height, width)
        precision = self.precision_projection(
            fused.reshape(batch * outputs, channels, height, width)
        ).view(batch, outputs, -1, height, width)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                precision = checkpoint(
                    lambda values, current_block=block: current_block(
                        values, output_intervals, self.temporal_mode
                    ),
                    precision,
                    use_reentrant=False,
                )
            else:
                precision = block(precision, output_intervals, self.temporal_mode)
        flat_precision = precision.reshape(batch * outputs, precision.shape[2], height, width)
        pre_bottleneck = self.pre_bottleneck(flat_precision)
        embedding_map = self.bottleneck(pre_bottleneck).view(batch, outputs, -1, height, width)
        detail = {
            product_id: head(flat_precision).view(batch, outputs, 3, height, width)
            for product_id, head in self.detail_heads.items()
        }
        reconstructions = {
            product_id: head(flat_precision).view(
                batch, outputs, len(self.products[product_id].bands), height, width
            )
            for product_id, head in self.reconstruction_heads.items()
        }
        return XuannvV2Output(
            embedding_map=embedding_map,
            embedding=embedding_map.mean(dim=(-2, -1)),
            reconstructions=reconstructions,
            highres_detail_stats=detail,
            observation_selection=observation_selection,
        )
