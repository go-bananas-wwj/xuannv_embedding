import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process import resolution_control
from xuannv_embedding.export.context import dump, sha


def ref(p):
    return {"path": str(p), "sha256": sha(p)}


def fixture_spec(tmp_path):
    path = tmp_path / "rgb.tif"
    rgb = np.stack([np.arange(16, dtype=np.float32).reshape(4, 4) + k for k in [0, 20, 40]])
    valid = np.ones((4, 4), bool)
    valid[0, 0] = False
    with (
        rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),
        rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=3,
            dtype="float32",
            crs="EPSG:32650",
            transform=from_origin(0, 4, 1, 1),
            nodata=-1,
        ) as dst,
    ):
        dst.write(np.where(valid[None], rgb, -1))
        dst.write_mask(valid.astype(np.uint8) * 255)
    record = {"patch_id": "tile", "bounds": [0, 0, 4, 4], "crs": "EPSG:32650"}
    index = tmp_path / "index.json"
    dump(index, {"records": [record]})
    manifest = tmp_path / "inputs.json"
    dump(
        manifest,
        {
            "schema": "semantic-raster-inputs-v1",
            "records": [{**record, "raster": ref(path)}],
            "dependencies": [],
        },
    )
    spec = {
        "schema": "matched-resolution-control-v1",
        "index": ref(index),
        "rasters": ref(manifest),
        "resolution_m": 2,
        "output": str(tmp_path / "control"),
    }
    p = tmp_path / "spec.json"
    dump(p, spec)
    return p, spec, path, valid


def test_resolution_control_averages_only_observations_and_preserves_exact_fine_mask(tmp_path):
    path, spec, source, expected_mask = fixture_spec(tmp_path)
    digest = sha(source)
    manifest = resolution_control.prepare(path)
    result = json.loads(Path(manifest).read_text())["records"][0]["raster"]
    with rasterio.open(result["path"]) as image:
        assert image.shape == (4, 4)
        np.testing.assert_array_equal(image.dataset_mask() > 0, expected_mask)
        assert list(image.bounds) == [0, 0, 4, 4]
    with rasterio.open(Path(spec["output"]) / "coarse_000000.tif") as coarse:
        assert coarse.shape == (2, 2)
        np.testing.assert_allclose(coarse.read(1)[0, 0], 10 / 3, atol=1e-6)
    assert sha(source) == digest
    assert resolution_control.prepare(path) == manifest


def test_resolution_control_rejects_overwrite_nonintegral_grid_and_unknown_fields(tmp_path):
    path, spec, _, _ = fixture_spec(tmp_path)
    dump(path, {**spec, "evaluation": "must-not-be-read"})
    with pytest.raises(ValueError, match="specification"):
        resolution_control.prepare(path)
    dump(path, {**spec, "resolution_m": 3})
    with pytest.raises(ValueError, match="grid"):
        resolution_control.prepare(path)
    dump(path, {**spec, "output": str(tmp_path)})
    with pytest.raises(ValueError, match="overwrite"):
        resolution_control.prepare(path)
