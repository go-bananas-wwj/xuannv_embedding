import copy

import pytest

from xuannv_embedding.export.journal_temporal import subset_inputs


def observation(i):
    return {
        "raster": {"path": f"/data/r{i}.tif", "sha256": str(i) * 64},
        "mask": {"path": f"/data/m{i}.tif", "sha256": str(i) * 64},
    }


def test_temporal_removal_rebuilds_composite_from_retained_physical_observations():
    original = {
        "schema": "temporal-semantic-observations-v1",
        "records": [{"patch_id": "p", "observations": [observation(i) for i in range(3)]}],
    }
    views = [
        {
            "schema": original["schema"],
            "records": [{"patch_id": "p", "observations": [observation(i)]}],
        }
        for i in range(3)
    ]
    before = copy.deepcopy(original)
    result = subset_inputs(original, views, [0, 2])
    assert result["records"][0]["observations"] == [observation(0), observation(2)]
    assert original == before


def test_unregistered_observation_cannot_enter_a_temporal_control():
    original = {
        "schema": "temporal-semantic-observations-v1",
        "records": [{"patch_id": "p", "observations": [observation(0)]}],
    }
    view = {
        "schema": original["schema"],
        "records": [{"patch_id": "p", "observations": [observation(1)]}],
    }
    with pytest.raises(ValueError):
        subset_inputs(original, [view], [0])


def test_empty_calendar_slots_have_distinct_paths_and_preserve_zero_coverage(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    from xuannv_embedding.export.context import dump, sha
    from xuannv_embedding.export.journal_temporal import empty_slot_rasters

    raster = tmp_path / "empty.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            raster,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=3,
            dtype="uint8",
            crs="EPSG:32650",
            transform=from_bounds(0, 0, 20, 20, 2, 2),
        ) as dst:
            dst.write(np.zeros((3, 2, 2), np.uint8))
            dst.write_mask(np.zeros((2, 2), np.uint8))
    manifest = tmp_path / "rasters.json"
    dump(
        manifest,
        {
            "schema": "semantic-raster-inputs-v1",
            "records": [
                {
                    "patch_id": "p",
                    "bounds": [0, 0, 20, 20],
                    "crs": "EPSG:32650",
                    "raster": {"path": str(raster), "sha256": sha(raster)},
                }
            ],
        },
    )
    ref = {"path": str(manifest), "sha256": sha(manifest)}
    a = empty_slot_rasters(ref, tmp_path / "slot1")
    b = empty_slot_rasters(ref, tmp_path / "slot2")
    import json

    aa = json.loads(open(a["path"]).read())["records"][0]["raster"]
    bb = json.loads(open(b["path"]).read())["records"][0]["raster"]
    assert aa["path"] != bb["path"]
    assert aa["sha256"] == bb["sha256"] == sha(raster)
    with rasterio.open(aa["path"]) as src:
        assert not src.dataset_mask().any()


def test_calendar_placeholder_rejects_observed_raster_pixels(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    from xuannv_embedding.export.context import dump, sha
    from xuannv_embedding.export.journal_temporal import empty_slot_rasters

    raster = tmp_path / "observed.tif"
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=3,
        dtype="uint8",
        crs="EPSG:32650",
        transform=from_bounds(0, 0, 20, 20, 2, 2),
    ) as dst:
        dst.write(np.ones((3, 2, 2), np.uint8))
    manifest = tmp_path / "rasters.json"
    dump(
        manifest,
        {
            "schema": "semantic-raster-inputs-v1",
            "records": [
                {
                    "patch_id": "p",
                    "bounds": [0, 0, 20, 20],
                    "crs": "EPSG:32650",
                    "raster": {"path": str(raster), "sha256": sha(raster)},
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="zero coverage"):
        empty_slot_rasters({"path": str(manifest), "sha256": sha(manifest)}, tmp_path / "invalid")
