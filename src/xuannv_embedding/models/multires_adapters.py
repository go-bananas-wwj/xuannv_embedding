"""Product-specific dense and native-resolution high-resolution adapters."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from xuannv_embedding.data.contracts import ProductSpec


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DenseProductAdapter(nn.Module):
    """Encode on each product's stored grid before feature-space alignment."""

    def __init__(self, product: ProductSpec, dim: int) -> None:
        super().__init__()
        self.product = product
        self.native_stem = nn.Sequential(
            nn.Conv2d(len(product.bands), dim, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(dim), dim),
            nn.GELU(),
        )
        self.gsd_embedding = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        precision_names = {"exact": 0, "day": 1, "month": 2, "static": 3}
        self.precision_index = precision_names[product.time_precision]
        self.precision_embedding = nn.Embedding(4, dim)
        self.missing_embedding = nn.Embedding(2, dim)

    def forward(
        self,
        frames: torch.Tensor,
        pixel_masks: torch.Tensor,
        observation_masks: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, observations, channels, height, width = frames.shape
        if channels != len(self.product.bands):
            raise ValueError(f"{self.product.product_id} 输入通道不符合 ProductSpec")
        masked = frames * pixel_masks.to(frames.dtype)
        features = self.native_stem(masked.reshape(batch * observations, channels, height, width))
        if features.shape[-2:] != target_size:
            features = F.interpolate(
                features, size=target_size, mode="bilinear", align_corners=False
            )
        native_mean = sum(self.product.native_gsd_m) / len(self.product.native_gsd_m)
        metadata = features.new_tensor([math.log(native_mean), math.log(self.product.stored_gsd_m)])
        metadata_embedding = self.gsd_embedding(metadata).view(1, 1, -1, 1, 1)
        precision = self.precision_embedding(
            torch.tensor(self.precision_index, device=frames.device)
        ).view(1, 1, -1, 1, 1)
        present = observation_masks.long().clamp(0, 1)
        missing = self.missing_embedding(present).view(batch, observations, -1, 1, 1)
        features = features.view(batch, observations, -1, *target_size)
        return (features + metadata_embedding + precision + missing) * observation_masks[
            :, :, None, None, None
        ].to(features.dtype)


class _AntiAliasedDownsample(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.blur = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.GroupNorm(_groups(dim), dim)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(self.blur(values))))


class MultiResolutionProductAdapter(nn.Module):
    """Learn at native GSD first, then aggregate features—not pixels—to 10 m."""

    def __init__(self, product: ProductSpec, dim: int) -> None:
        super().__init__()
        self.product = product
        self.native_stem = nn.Sequential(
            nn.Conv2d(len(product.bands), dim, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(dim), dim),
            nn.GELU(),
        )
        blocks = []
        feature_gsd = product.stored_gsd_m
        while feature_gsd * 2 <= 5.0:
            blocks.append(_AntiAliasedDownsample(dim))
            feature_gsd *= 2
        self.downsample_blocks = nn.ModuleList(blocks)
        self.feature_gsd_m = feature_gsd
        self.product_embedding = nn.Parameter(torch.empty(dim).normal_(std=0.02))

    def forward(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        geotransforms: torch.Tensor,
        target_size: tuple[int, int],
        output_geotransforms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, observations, channels, height, width = frames.shape
        if geotransforms.shape != (batch, observations, 6):
            raise ValueError(f"{self.product.product_id} geotransforms 必须为 [B,N,6]")
        if channels != len(self.product.bands):
            raise ValueError(f"{self.product.product_id} 输入通道不符合 ProductSpec")
        actual_gsd = geotransforms[..., 0].abs()
        if not torch.allclose(
            actual_gsd,
            torch.full_like(actual_gsd, self.product.stored_gsd_m),
            rtol=0.05,
            atol=0.05,
        ):
            raise ValueError(f"{self.product.product_id} affine GSD 与 ProductSpec 不一致")
        values = (frames * masks.to(frames.dtype)).reshape(
            batch * observations, channels, height, width
        )
        features = self.native_stem(values)
        for block in self.downsample_blocks:
            features = block(features)
        if output_geotransforms is None:
            features = F.adaptive_avg_pool2d(features, target_size)
            aligned_masks = F.adaptive_avg_pool2d(
                masks.reshape(batch * observations, 1, height, width).float(), target_size
            )
        else:
            if output_geotransforms.shape != (batch, 6):
                raise ValueError("output_geotransforms 必须为 [B,6]")
            factor = 2 ** len(self.downsample_blocks)
            feature_transforms = geotransforms.clone()
            feature_transforms[..., 0] *= factor
            feature_transforms[..., 1] *= factor
            feature_transforms[..., 3] *= factor
            feature_transforms[..., 4] *= factor
            features = _affine_resample(
                features,
                feature_transforms.reshape(batch * observations, 6),
                output_geotransforms[:, None, :]
                .expand(-1, observations, -1)
                .reshape(batch * observations, 6),
                target_size,
            )
            aligned_masks = _affine_resample(
                masks.reshape(batch * observations, 1, height, width).float(),
                geotransforms.reshape(batch * observations, 6),
                output_geotransforms[:, None, :]
                .expand(-1, observations, -1)
                .reshape(batch * observations, 6),
                target_size,
            ).clamp(0, 1)
        features = features + self.product_embedding.view(1, -1, 1, 1)
        quality = aligned_masks.mean(dim=(-3, -2, -1)).view(batch, observations)
        features = features.view(batch, observations, -1, *target_size)
        aligned_masks = aligned_masks.view(batch, observations, 1, *target_size)
        features = features * aligned_masks
        return features, aligned_masks, quality


def _affine_resample(
    values: torch.Tensor,
    source_transforms: torch.Tensor,
    target_transforms: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Map target pixel centers through real affine transforms into a source grid."""
    target_height, target_width = target_size
    row, column = torch.meshgrid(
        torch.arange(target_height, device=values.device, dtype=values.dtype) + 0.5,
        torch.arange(target_width, device=values.device, dtype=values.dtype) + 0.5,
        indexing="ij",
    )
    grids = []
    for index in range(values.shape[0]):
        target = target_transforms[index].to(device=values.device, dtype=values.dtype)
        source = source_transforms[index].to(device=values.device, dtype=values.dtype)
        world_x = target[0] * column + target[1] * row + target[2]
        world_y = target[3] * column + target[4] * row + target[5]
        matrix = torch.stack((source[[0, 1]], source[[3, 4]]))
        inverse = torch.linalg.inv(matrix)
        coordinates = torch.stack((world_x - source[2], world_y - source[5]), dim=-1)
        source_centers = torch.einsum("ij,hwj->hwi", inverse, coordinates)
        normalized_x = 2 * source_centers[..., 0] / values.shape[-1] - 1
        normalized_y = 2 * source_centers[..., 1] / values.shape[-2] - 1
        grids.append(torch.stack((normalized_x, normalized_y), dim=-1))
    grid = torch.stack(grids, dim=0)
    return F.grid_sample(values, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
