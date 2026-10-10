import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from test_paired_multitask import fixture_spec, write_json

from xuannv_embedding.downstream import native_landcover, review_readouts
from xuannv_embedding.downstream.native_landcover import aggregate_features, interior_cells


def test_native_cells_average_features_without_replicating_labels():
    source = from_origin(0, 120, 10, 10)
    target = from_origin(0, 120, 30, 30)
    values = np.arange(144, dtype=np.float32).reshape(12, 12, 1)
    valid = np.ones((12, 12), bool)
    out, mask = aggregate_features(values, valid, source, "EPSG:32650", target, "EPSG:32650", 4)
    expected = values[..., 0].reshape(4, 3, 4, 3).mean((1, 3))
    np.testing.assert_allclose(out[..., 0], expected)
    assert mask.all() and out.shape == (4, 4, 1)
    valid[0, 0] = False
    _, mask = aggregate_features(values, valid, source, "EPSG:32650", target, "EPSG:32650", 4)
    assert not mask[0, 0] and mask.sum() == 15


def test_target_cells_crossing_source_tile_boundary_are_excluded():
    transform = from_origin(-10, 130, 30, 30)
    inside = interior_cells([0, 0, 120, 120], "EPSG:32650", transform, "EPSG:32650", 5)
    assert inside.sum() == 9
    assert not inside[0].any() and not inside[:, 0].any()


def test_native_aggregation_rejects_nonfinite_valid_features():
    x = np.ones((12, 12, 2), np.float32)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        aggregate_features(
            x,
            np.ones((12, 12), bool),
            from_origin(0, 120, 10, 10),
            "EPSG:32650",
            from_origin(0, 120, 30, 30),
            "EPSG:32650",
            4,
        )


def test_native_cohort_keeps_unique_cells_and_freezes_classes_before_query(tmp_path):
    import json

    path, parent, _ = fixture_spec(tmp_path)
    shared = tmp_path / "shared"
    for phase in ["calibration", "test"]:
        review_readouts.prepare_shared(path, shared, phase)
    raster = tmp_path / "reference.tif"
    labels = (np.indices((6, 16))[1] % 2 + 1).astype(np.uint8)
    labels[:, 14:] = 3  # Query-only class must not enter the task list.
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        height=6,
        width=16,
        count=1,
        dtype="uint8",
        crs="EPSG:32650",
        transform=from_origin(0, 180, 30, 30),
        nodata=0,
    ) as ds:
        ds.write(labels, 1)
    provenance = write_json(tmp_path / "provenance.json", {"product": "synthetic"})
    spec = {
        "protocol": "native-landcover-cohort-v1",
        "parent_cohort": native_landcover._reference(path),
        "shared": str(shared),
        "raster": native_landcover._reference(raster),
        "provenance": provenance,
        "source_crs": "EPSG:32650",
        "product": "clcd",
        "year": 2024,
        "legend": {"1": "crop", "2": "forest", "3": "shrub"},
        "size": 16,
        "budgets": [1],
        "support_seeds": parent["support_seeds"][:1],
        "output": str(tmp_path / "prepared"),
    }
    job = tmp_path / "native.json"
    write_json(job, spec)
    native_landcover.prepare(job)
    root = tmp_path / "prepared"
    registration = json.loads((root / "class_registration.json").read_text())
    assert registration["tasks"] == ["clcd_crop", "clcd_forest"]
    audit = json.loads((root / "native_grid_audit.json").read_text())
    assert audit["duplicate_native_cells"] == 0 and audit["unique_native_cells"] < 100
    review_readouts.prepare_shared(root / "cohort.json", tmp_path / "native-shared", "test")
