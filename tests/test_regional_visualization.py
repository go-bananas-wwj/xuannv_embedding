from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process.regional_visualization import (
    read_rgb_preview,
    render_data_visualizations,
)


def test_rgb_preview_respects_sensor_band_order_and_missing_pixels(tmp_path: Path):
    for source, bands in [
        ("aerial", [80, 20, 10]),
        ("s2", [10, 20, 80]),
        ("landsat", [10, 20, 80]),
    ]:
        p = tmp_path / f"{source}.tif"
        values = np.array([np.full((2, 2), v) for v in bands], dtype="float32")
        values[:, 0, 1] = -9999
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=3,
            dtype="float32",
            crs="EPSG:2193",
            transform=from_origin(0, 20, 10, 10),
            nodata=-9999,
        ) as dst:
            dst.write(values)
        rgb, valid = read_rgb_preview(p, source, max_side=2)
        assert rgb[0, 0, 0] > rgb[0, 0, 2]
        assert valid.tolist() == [[True, False], [True, True]]


def test_complete_visualization_rejects_pending_downloads(tmp_path: Path):
    spec = {
        "root": str(tmp_path),
        "months": ["2024-04"],
        "bounds": [0, 0, 20, 20],
        "crs": "EPSG:2193",
        "patch_size_m": 20,
    }
    with pytest.raises(ValueError, match="incomplete"):
        render_data_visualizations(spec, require_complete=True)
    assert not (tmp_path / "visualizations/data_overview_complete.png").exists()
