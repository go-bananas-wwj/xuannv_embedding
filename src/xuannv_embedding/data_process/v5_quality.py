"""Non-destructive cloud QA and conservative geographic mask transfer."""

from __future__ import annotations

import math

import numpy as np
from rasterio.warp import Resampling, reproject


def quality_masks(
    classes: np.ndarray,
    data_valid: np.ndarray,
    *,
    gsd: float,
    buffer_m: float = 30,
    strict_threshold: float = 0.60,
) -> dict:
    labels = np.asarray(classes)
    valid = np.asarray(data_valid, dtype=bool)
    if labels.shape != valid.shape or labels.ndim != 2 or not labels.size:
        raise ValueError("cloud and data masks must share a nonempty native grid")
    if not set(np.unique(labels)).issubset({0, 1, 2, 3}):
        raise ValueError("invalid cloud class")
    if gsd <= 0 or buffer_m < 0 or not 0 <= strict_threshold <= 1:
        raise ValueError("invalid cloud buffer or threshold")
    # Clouds on NoData are not evidence of neighboring cloud: exclude before dilation.
    cloud = (labels != 0) & valid
    radius = math.ceil(buffer_m / gsd)
    buffered = cloud.copy()
    if radius:
        padded = np.pad(cloud, radius, constant_values=False)
        for y in range(2 * radius + 1):
            for x in range(2 * radius + 1):
                buffered |= padded[y : y + cloud.shape[0], x : x + cloud.shape[1]]
    final = valid & ~buffered
    fraction = float(final.mean())
    return {
        "before_buffer": valid & ~cloud,
        "cloud_buffered": buffered,
        "data_valid": valid.copy(),
        "valid": final,
        "clear_fraction": fraction,
        "strict_scene_qualified": fraction >= strict_threshold,
        "available": bool(final.any()),
        "buffer_pixels": radius,
    }


def transfer_invalid(
    invalid: np.ndarray, *, src_transform, src_crs, dst_transform, dst_crs, shape: tuple[int, int]
) -> np.ndarray:
    """Use any overlapping invalid source cell; uncovered destination cells stay invalid."""
    destination = np.full(shape, 255, dtype=np.uint8)
    reproject(
        source=np.asarray(invalid, dtype=np.uint8),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=None,
        dst_nodata=255,
        resampling=Resampling.max,
    )
    return destination != 0
