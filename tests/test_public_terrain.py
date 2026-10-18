from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process import public_terrain
from xuannv_embedding.export.context import dump, sha


def ref(p):
    return {"path": str(p), "sha256": sha(p)}


def fixture_spec(tmp_path):
    x = (np.arange(12) + 0.5) * 30
    y = 360 - (np.arange(12) + 0.5) * 30
    height = (100 + 2 * x[None] + 3 * y[:, None]).astype(np.float32)
    raster = tmp_path / "height.tif"
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        count=1,
        width=12,
        height=12,
        dtype="float32",
        crs="EPSG:32650",
        transform=from_origin(0, 360, 30, 30),
        nodata=-32767,
    ) as dst:
        dst.write(height, 1)
    index = tmp_path / "index.json"
    dump(
        index,
        {"crs": "EPSG:32650", "records": [{"patch_id": "tile", "bounds": [90, 90, 270, 270]}]},
    )
    spec = {
        "schema": "public-terrain-features-v1",
        "dataset": "cop-dem-glo-30",
        "index": ref(index),
        "sources": [{"kind": "local", "raster": ref(raster)}],
        "analysis_resolution_m": 30,
        "halo_m": 60,
        "patch_size": 6,
        "output": str(tmp_path / "output"),
    }
    path = tmp_path / "spec.json"
    dump(path, spec)
    return path, spec


def test_public_terrain_plane_has_correct_metric_gradients_and_north_clockwise_aspect(tmp_path):
    path, spec = fixture_spec(tmp_path)
    public_terrain.prepare(path)
    values = np.load(Path(spec["output"]) / "features.npy")
    assert values.shape == (1, 5, 6, 6)
    np.testing.assert_allclose(values[0, 0, 0, 0], 1075, atol=1e-4)
    np.testing.assert_allclose(values[0, 1], np.sqrt(13 / 14), atol=1e-5)
    np.testing.assert_allclose(values[0, 2], 1 / np.sqrt(14), atol=1e-5)
    np.testing.assert_allclose(values[0, 3], -2 / np.sqrt(13), atol=1e-5)
    np.testing.assert_allclose(values[0, 4], -3 / np.sqrt(13), atol=1e-5)
    assert np.load(Path(spec["output"]) / "valid.npy").all()
    public_terrain.prepare(path)


def test_public_terrain_rejects_label_fields_wrong_dataset_and_no_halo(tmp_path):
    path, spec = fixture_spec(tmp_path)
    dump(path, {**spec, "labels": "must-not-be-read"})
    with pytest.raises(ValueError, match="specification"):
        public_terrain.prepare(path)
    dump(path, {**spec, "dataset": "LINZ evaluation LiDAR"})
    with pytest.raises(ValueError, match="specification"):
        public_terrain.prepare(path)
    dump(path, {**spec, "halo_m": 0})
    with pytest.raises(ValueError, match="halo"):
        public_terrain.prepare(path)
