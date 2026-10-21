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
