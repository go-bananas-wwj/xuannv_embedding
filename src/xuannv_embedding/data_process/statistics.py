"""按区域和物理 source 计算 GeoTIFF 波段统计量。"""

from __future__ import annotations

import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import rasterio
from rasterio.io import MemoryFile

from xuannv_embedding.data.contracts import ProductSpec


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


def compute_v2_archive_statistics(
    product: ProductSpec,
    split_path: Path,
    member_index_path: Path,
    *,
    max_observations: int | None = None,
) -> dict[str, object]:
    """Compute train-only statistics directly from immutable ZIP members in stored units."""
    split = pq.read_table(split_path, columns=["patch_id", "split"])
    train_ids = {str(row["patch_id"]) for row in split.to_pylist() if str(row["split"]) == "train"}
    if not train_ids:
        raise ValueError("V2 statistics 缺少 train split")
    accumulators = [_WelfordAccumulator() for _ in product.bands]
    observations = 0
    parquet = pq.ParquetFile(member_index_path)
    active_path: Path | None = None
    active_archive: ZipFile | None = None
    try:
        for batch in parquet.iter_batches(
            columns=["patch_id", "product_id", "archive_path", "member_name"],
            batch_size=8192,
        ):
            for row in batch.to_pylist():
                if row["product_id"] != product.product_id or row["patch_id"] not in train_ids:
                    continue
                path = Path(row["archive_path"])
                if path != active_path:
                    if active_archive is not None:
                        active_archive.close()
                    active_archive = ZipFile(path)
                    active_path = path
                assert active_archive is not None
                payload = active_archive.read(str(row["member_name"]))
                with MemoryFile(payload) as memory:
                    with memory.open() as dataset:
                        values = dataset.read()
                        nodata = dataset.nodata
                if values.shape[0] != len(accumulators):
                    raise ValueError(
                        f"波段数量冲突: {path}!{row['member_name']}={values.shape[0]}, "
                        f"expected={len(accumulators)}"
                    )
                valid = np.isfinite(values).all(axis=0)
                if nodata is not None and not math.isnan(nodata):
                    valid &= (values != nodata).all(axis=0)
                valid &= ~(values == 0).all(axis=0)
                valid &= ~(values == -32768).any(axis=0)
                for accumulator, band in zip(accumulators, values, strict=True):
                    accumulator.update(band[valid])
                observations += 1
                if max_observations is not None and observations >= max_observations:
                    break
            if max_observations is not None and observations >= max_observations:
                break
    finally:
        if active_archive is not None:
            active_archive.close()
    if observations == 0:
        raise ValueError(f"V2 statistics 未找到 train observations: {product.product_id}")
    return {
        "schema_version": "xuannv_v2_band_statistics_v1",
        "product_id": product.product_id,
        "bands": list(product.bands),
        "mean": [item.mean for item in accumulators],
        "std": [item.std() for item in accumulators],
        "band_counts": [item.count for item in accumulators],
        "num_observations": observations,
        "max_observations": max_observations,
        "complete_training_split": max_observations is None,
        "split": "train",
        "representation": "stored_dn",
        "scaling_applied": False,
    }


def compute_v2_highres_statistics(
    product: ProductSpec,
    split_path: Path,
    patch_index_path: Path,
    *,
    max_observations: int | None = None,
) -> dict[str, object]:
    """Compute train-patch high-resolution statistics without downloading or inventing QA."""
    from xuannv_embedding.data.v2_dataset import _read_highres_patch

    split = pq.read_table(split_path, columns=["patch_id", "split"])
    train_ids = {str(row["patch_id"]) for row in split.to_pylist() if row["split"] == "train"}
    rows = [
        row
        for row in pq.read_table(patch_index_path).to_pylist()
        if str(row["patch_id"]) in train_ids
    ]
    rows.sort(key=lambda row: (row["patch_id"], row["acquired_at"], row["scene_id"]))
    if max_observations is not None:
        rows = rows[:max_observations]
    accumulators = [_WelfordAccumulator() for _ in product.bands]
    for row in rows:
        frame, mask, _ = _read_highres_patch(
            row, bands=len(product.bands), stored_gsd_m=product.stored_gsd_m
        )
        valid = mask[0].numpy() > 0
        for accumulator, band in zip(accumulators, frame.numpy(), strict=True):
            accumulator.update(band[valid])
    if not rows or any(item.count == 0 for item in accumulators):
        raise ValueError(
            f"V2 highres statistics 未找到有效 train observations: {product.product_id}"
        )
    return {
        "schema_version": "xuannv_v2_band_statistics_v1",
        "product_id": product.product_id,
        "bands": list(product.bands),
        "mean": [item.mean for item in accumulators],
        "std": [item.std() for item in accumulators],
        "band_counts": [item.count for item in accumulators],
        "num_observations": len(rows),
        "max_observations": max_observations,
        "complete_training_split": max_observations is None,
        "split": "train",
        "representation": "stored_dn",
        "scaling_applied": False,
    }
