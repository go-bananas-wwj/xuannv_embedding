import pandas as pd
import pytest

from xuannv_embedding.data_process.v5_sampling import construct_indexes


def inputs():
    registry = pd.DataFrame([{"patch_id": "p1", "split": "test"}])
    dense = pd.DataFrame(
        [
            {
                "patch_id": "p1",
                "product_id": "s2_local",
                "year": 2020,
                "month": m,
                "present": True,
                "quality_status": "passed",
            }
            for m in [1, 3, 12]
        ]
    )
    highres = pd.DataFrame(
        [
            {
                "patch_id": "p1",
                "family": "jilin1",
                "year": 2020,
                "scene_group_id": "late_year",
                "acquired_at": "2020-12-31",
                "available": True,
                "alignment_status": "passed",
                "clear_fraction": 0.8,
            }
        ]
    )
    return registry, dense, highres


def test_quarter_uses_only_its_months_but_reuses_same_year_prior():
    registry, dense, highres = inputs()
    quarters, candidates, selected = construct_indexes(registry, dense, highres)
    q1 = quarters.loc[(quarters.year == 2020) & (quarters.quarter == 1)].iloc[0]
    q4 = quarters.loc[(quarters.year == 2020) & (quarters.quarter == 4)].iloc[0]
    assert len(q1.dense_observation_ids) == 2
    assert len(q4.dense_observation_ids) == 1
    assert q1.highres_scene_ids == q4.highres_scene_ids == ["late_year"]
    assert quarters.loc[quarters.year == 2021].highres_scene_ids.map(len).sum() == 0
    assert set(quarters.split) == {"test"}
    assert len(candidates) == len(selected) == 1


def test_unknown_quality_cannot_be_silently_used_as_clear():
    registry, dense, highres = inputs()
    dense["quality_status"] = "unchecked"
    with pytest.raises(ValueError, match="quality"):
        construct_indexes(registry, dense, highres)


def test_unreliable_alignment_cannot_enter_pixel_prior():
    registry, dense, highres = inputs()
    highres["alignment_status"] = "uncertain"
    quarters, candidates, selected = construct_indexes(registry, dense, highres)
    assert candidates.empty and selected.empty
    assert quarters.highres_scene_ids.map(len).sum() == 0


def test_region_without_any_highres_still_has_dense_samples():
    registry, dense, _ = inputs()
    quarters, candidates, selected = construct_indexes(registry, dense, pd.DataFrame())
    assert len(quarters) == 8 and quarters.base_available.sum() == 2
    assert candidates.empty and selected.empty
