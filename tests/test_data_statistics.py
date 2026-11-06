from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from xuannv_embedding.data.contracts import ProductSpec
from xuannv_embedding.data_process.statistics import (
    compute_statistics,
    compute_v2_archive_statistics,
)


def _write_raster(path: Path, values: np.ndarray, nodata: float = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[1],
        width=values.shape[2],
        count=values.shape[0],
        dtype="float32",
        crs="EPSG:32650",
        transform=from_origin(0, 20, 10, 10),
        nodata=nodata,
    ) as destination:
        destination.write(values.astype(np.float32))


def test_statistics_are_per_band_and_ignore_nodata(tmp_path: Path) -> None:
    source_dir = tmp_path / "patches" / "physical_source"
    _write_raster(
        source_dir / "a.tif",
        np.asarray([[[1, 2], [3, -9999]], [[10, 20], [30, 40]]], dtype=np.float32),
    )
    result = compute_statistics(tmp_path, "physical_source")
    assert result["mean"] == pytest.approx([2.0, 25.0])
    assert result["std"] == pytest.approx([np.std([1, 2, 3]), np.std([10, 20, 30, 40])])
    assert result["band_counts"] == [3, 4]


def test_statistics_use_explicit_source_mapping(tmp_path: Path) -> None:
    _write_raster(tmp_path / "regional" / "optical" / "a.tif", np.ones((1, 2, 2)))
    result = compute_statistics(
        tmp_path,
        "highres_optical",
        source_dirs={"highres_optical": "regional/optical"},
    )
    assert result["source"] == "highres_optical"
    assert result["mean"] == [1.0]
    assert result["count"] == 4


def _memory_tiff(value: int) -> bytes:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="uint16",
            crs="EPSG:32650",
            transform=from_origin(0, 20, 10, 10),
        ) as destination:
            destination.write(np.full((1, 2, 2), value, dtype=np.uint16))
        return memory.read()


def test_v2_statistics_use_only_train_and_keep_stored_dn(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("source/train.tif", _memory_tiff(10_000))
        archive.writestr("source/test.tif", _memory_tiff(7))
    split_path = tmp_path / "split.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"patch_id": "train", "split": "train"},
                {"patch_id": "test", "split": "test"},
            ]
        ),
        split_path,
    )
    members_path = tmp_path / "members.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "patch_id": "train",
                    "product_id": "s2_local",
                    "archive_path": str(archive_path),
                    "member_name": "source/train.tif",
                },
                {
                    "patch_id": "test",
                    "product_id": "s2_local",
                    "archive_path": str(archive_path),
                    "member_name": "source/test.tif",
                },
            ]
        ),
        members_path,
    )
    product = ProductSpec(
        product_id="s2_local",
        role="dense",
        bands=("B02",),
        native_gsd_m=(10.0,),
        stored_gsd_m=10.0,
        dtype="uint16",
        time_precision="month",
        already_resampled=True,
        qa_available=False,
    )

    result = compute_v2_archive_statistics(product, split_path, members_path)

    assert result["mean"] == [10_000.0]
    assert result["std"] == [0.0]
    assert result["num_observations"] == 1
    assert result["representation"] == "stored_dn"
    assert result["scaling_applied"] is False
