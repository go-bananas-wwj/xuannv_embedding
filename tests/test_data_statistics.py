from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process.statistics import compute_statistics


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
