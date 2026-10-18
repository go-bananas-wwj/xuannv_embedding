from __future__ import annotations

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window

from xuannv_embedding.data_process import preprocess as MODULE


def test_landsat_scaling_preserves_qa_pixel_bits() -> None:
    arr = np.full((7, 2, 2), 10000, dtype=np.float32)
    arr[-1] = np.array([[0, 31], [8, 16]], dtype=np.float32)
    scaled = MODULE._scale_landsat_reflectance_preserving_qa(arr, nodata=0.0)

    assert np.isclose(scaled[0, 0, 0], 0.075)
    assert np.array_equal(scaled[-1], np.array([[0, 31], [8, 16]], dtype=np.float32))
    mask = MODULE._compute_valid_mask(scaled, "landsat", 0.0)
    assert mask.tolist() == [[1, 0], [0, 0]]


def test_landsat_qa_resampling_is_nearest_neighbor() -> None:
    source = np.full((7, 2, 2), 10000, dtype=np.float32)
    source[-1] = np.array([[0, 31], [8, 16]], dtype=np.float32)
    output = MODULE._extract_patch_bounded(
        arr=source,
        src_transform=from_origin(0, 40, 20, 20),
        src_crs=CRS.from_epsg(32650),
        master_transform=from_origin(0, 40, 10, 10),
        master_crs=CRS.from_epsg(32650),
        window=Window(0, 0, 4, 4),
        nodata=-9999.0,
        source="landsat",
    )
    assert set(np.unique(output[-1]).tolist()).issubset({0.0, 8.0, 16.0, 31.0})


@pytest.mark.parametrize("sensor", ["landsat", "s2"])
def test_daily_quality_assignment_uses_every_valid_pixel(tmp_path, sensor) -> None:
    import rasterio
    import xarray as xr

    bands = (
        ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"]
        if sensor == "landsat"
        else ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"]
    )
    arr = np.full((2, len(bands), 4, 4), 10000, np.float32)
    arr[1, :-1] = 12000
    arr[:, -1] = 21824 if sensor == "landsat" else 4
    arr[0, -1, :2] = 21832 if sensor == "landsat" else 9
    if sensor == "s2":
        arr[1, -1] = 5
    frame = xr.DataArray(
        arr,
        dims=("time", "band", "y", "x"),
        coords={
            "time": np.array(["2025-01-01", "2025-01-01"], dtype="datetime64[ns]"),
            "band": bands,
            "y": [35, 25, 15, 5],
            "x": [5, 15, 25, 35],
        },
        attrs={"crs": "epsg:32650"},
    )
    path = tmp_path / "scenes.nc"
    frame.to_dataset(name="observations").assign_coords(epsg=32650).to_netcdf(path)
    config = {
        "source": sensor,
        "nodata": -9999,
        "min_valid_ratio": 0,
        "master_transform": tuple(from_origin(0, 40, 10, 10)),
        "crs": "EPSG:32650",
        "patches": [{"patch_id": "tile", "bounds": [0, 0, 40, 40]}],
        "patch_size_px": 4,
        "output_root": str(tmp_path / "out"),
        "overwrite": False,
    }
    assert MODULE.process_one_date((path, "20250101", [0, 1], config)) == 1
    output = tmp_path / "out" / "patches" / sensor / f"{sensor}_20250101_tile.tif"
    with rasterio.open(output) as source:
        result = source.read()
    with rasterio.open(output.with_stem(output.stem + "_mask")) as source:
        assert source.read(1).all()
    np.testing.assert_array_equal(result[-1, :2], 21824 if sensor == "landsat" else 5)
    np.testing.assert_array_equal(result[-1, 2:], 21824 if sensor == "landsat" else 4)
    np.testing.assert_allclose(result[0, :2], 0.13 if sensor == "landsat" else 12000, atol=1e-6)
    np.testing.assert_allclose(result[0, 2:], 0.1025 if sensor == "landsat" else 11000, atol=1e-6)
