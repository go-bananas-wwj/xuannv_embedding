"""按区域和物理 source 计算 GeoTIFF 波段统计量。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio


def _valid_mask(values: np.ndarray, nodata: float | None) -> npt.NDArray[np.bool_]:
    mask = np.isfinite(values)
    if nodata is not None and not math.isnan(nodata):
        mask &= values != nodata
    return mask


class _WelfordAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        values = values.astype(np.float64, copy=False)
        count = int(values.size)
        previous_count = self.count
        batch_mean = float(values.mean())
        delta = batch_mean - self.mean
        self.count += count
        self.mean += delta * count / self.count
        batch_m2 = float(((values - batch_mean) ** 2).sum())
        self.m2 += batch_m2 + delta * delta * previous_count * count / self.count

    def std(self) -> float:
        return math.sqrt(self.m2 / self.count) if self.count else float("nan")


def _collect_tif_files(source_dir: Path, max_patches: int | None) -> list[Path]:
    files = sorted(
        path
        for path in source_dir.glob("*.tif")
        if path.is_file() and not path.stem.endswith("_mask")
    )
    return files[:max_patches] if max_patches is not None and max_patches > 0 else files


def compute_statistics(
    processed_dir: Path,
    source: str,
    max_patches: int | None = None,
    source_dirs: dict[str, str] | None = None,
    files: list[Path] | None = None,
) -> dict[str, list[float] | list[int] | int | str]:
    """使用 Welford 在线算法计算有效像素的逐波段 mean/std。"""
    relative_dir = source_dirs.get(source) if source_dirs else None
    if relative_dir is None:
        relative_dir = f"patches/{source}"
        if not (processed_dir / relative_dir).exists():
            relative_dir = source
    if files is None:
        files = _collect_tif_files(processed_dir / relative_dir, max_patches)
    if not files:
        return {"mean": [], "std": [], "count": 0, "num_files": 0, "source": source}

    accumulators: list[_WelfordAccumulator] | None = None
    successful = 0
    for path in files:
        with rasterio.open(path) as raster:
            data = raster.read(out_dtype=np.float64)
            nodata = raster.nodata
        if accumulators is None:
            accumulators = [_WelfordAccumulator() for _ in range(data.shape[0])]
        if data.shape[0] != len(accumulators):
            raise ValueError(f"波段数量冲突: {path}={data.shape[0]}, expected={len(accumulators)}")
        for accumulator, band in zip(accumulators, data, strict=True):
            accumulator.update(band[_valid_mask(band, nodata)])
        successful += 1

    assert accumulators is not None
    band_counts = [item.count for item in accumulators]
    return {
        "mean": [item.mean for item in accumulators],
        "std": [item.std() for item in accumulators],
        "count": sum(band_counts),
        "band_counts": band_counts,
        "num_files": successful,
        "source": source,
    }
