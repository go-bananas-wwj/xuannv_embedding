import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from test_paired_multitask import fixture_spec, write_json

from xuannv_embedding.downstream.worldcover_reference import prepare, project_tile
from xuannv_embedding.export.context import sha


def raster(path, array, bounds):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        nodata=0,
        transform=from_bounds(*bounds, array.shape[1], array.shape[0]),
    ) as dst:
        dst.write(array, 1)


def test_worldcover_resampling_preserves_class_codes_and_nodata(tmp_path):
    path = tmp_path / "classes.tif"
    values = np.array([[0, 10], [50, 80]], np.uint8)
    raster(path, values, [0, 0, 20, 20])
    with rasterio.open(path) as source:
        actual, _ = project_tile(source, [0, 0, 20, 20], 4, "EPSG:3857")
    np.testing.assert_array_equal(actual, values.repeat(2, 0).repeat(2, 1))
    raster(path, np.full((2, 2), 77, np.uint8), [0, 0, 20, 20])
    with rasterio.open(path) as source, pytest.raises(ValueError, match="class legend"):
        project_tile(source, [0, 0, 20, 20], 2, "EPSG:3857")


def test_worldcover_class_selection_ignores_test_only_categories(tmp_path):
    _, base, _ = fixture_spec(tmp_path)
    path = tmp_path / "worldcover.tif"
    values = np.full((16, 48), 10, np.uint8)
    values[:, 8:16] = values[:, 24:32] = values[:, 40:48] = 50
    values[0:3, 32:35] = 80  # Water occurs only in test; must not select its task from test.
    raster(path, values, [0, 0, 480, 160])
    provenance = write_json(
        tmp_path / "provenance.json",
        {
            "dataset": "ESA WorldCover",
            "year": 2021,
            "version": "v200",
            "sha256": sha(path),
        },
    )
    domain = tmp_path / "domain.npy"
    np.save(domain, np.ones((2, 16, 16), bool))
    spec = {
        "protocol": "worldcover-reference-v1",
        "raster": {"path": str(path), "sha256": sha(path)},
        "provenance": provenance,
        "reference_cache": base["reference_cache"],
        "reference_crs": "EPSG:3857",
        "osm_labels": base["labels"],
        "training_validation_domain": {"path": str(domain), "sha256": sha(domain)},
        "min_training_tiles": 1,
        "output": str(tmp_path / "labels"),
    }
    spec_path = tmp_path / "wc_spec.json"
    write_json(spec_path, spec)
    report = prepare(spec_path)
    assert report["task_schema"]["C"]["worldcover"] == ["worldcover_tree", "worldcover_built"]
    assert report["native_pixel_crosschecks"] == 9
    with np.load(report["labels"]["test"]["path"]) as data:
        assert "esri" not in data.files
        assert "worldcover_water" not in data.files
        assert data["worldcover_tree"][0, 0, 0] == 0
    registration = json.loads((Path(spec["output"]) / "class_registration.json").read_text())
    assert registration["test_labels_opened_for_selection"] is False
