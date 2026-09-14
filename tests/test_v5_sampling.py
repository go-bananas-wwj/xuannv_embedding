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
                "contract_status": "verified",
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
                "candidate_qualified": True,
                "strict_pixel_fusion_candidate": False,
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


def test_unknown_alignment_keeps_qualified_annual_candidate_without_fusion_approval():
    registry, dense, highres = inputs()
    highres["alignment_status"] = "uncertain"
    quarters, candidates, selected = construct_indexes(registry, dense, highres)
    assert len(candidates) == len(selected) == 1
    assert quarters.loc[quarters.year.eq(2020)].highres_scene_ids.map(len).eq(1).all()
    assert not selected.strict_pixel_fusion_candidate.any()


def test_region_without_any_highres_still_has_dense_samples():
    registry, dense, _ = inputs()
    quarters, candidates, selected = construct_indexes(registry, dense, pd.DataFrame())
    assert len(quarters) == 8 and quarters.base_available.sum() == 2
    assert candidates.empty and selected.empty


def test_dense_contract_pending_keeps_source_reference_but_is_not_usable():
    registry, dense, highres = inputs()
    dense["contract_status"] = "pending"
    dense["quality_status"] = "pending"
    quarters, _, _ = construct_indexes(registry, dense, highres)
    q1 = quarters.loc[quarters.year.eq(2020) & quarters.quarter.eq(1)].iloc[0]
    assert len(q1.dense_inventory_ids) == 2
    assert not q1.dense_observation_ids and not q1.base_available
    assert "unknown_radiometry" in q1.base_exclusion_reasons
    assert "quality_pending" in q1.base_exclusion_reasons
    assert "s2_local:2020-02" in q1.missing_monthly_sources
    assert q1.target_year == 2020


def test_unqualified_scene_does_not_enter_annual_pool_even_if_available():
    registry, dense, highres = inputs()
    highres["candidate_qualified"] = False
    quarters, candidates, selected = construct_indexes(registry, dense, highres)
    assert candidates.empty and selected.empty
    assert not quarters.highres_scene_ids.map(len).any()


def test_calendar_and_split_inconsistencies_fail_before_index_publication():
    registry, dense, highres = inputs()
    dense["month"] = dense.month.astype(float)
    dense.loc[0, "month"] = 1.5
    with pytest.raises(ValueError, match="month"):
        construct_indexes(registry, dense, highres)
    registry, dense, highres = inputs()
    dense["split"] = "train"
    with pytest.raises(ValueError, match="split"):
        construct_indexes(registry, dense, highres)
    registry, dense, highres = inputs()
    highres["split"] = "train"
    with pytest.raises(ValueError, match="split"):
        construct_indexes(registry, dense, highres)


def test_quarter_intervals_include_leap_february_and_end_exclusively():
    registry, dense, highres = inputs()
    quarters, _, _ = construct_indexes(registry, dense, highres)
    q1 = quarters.loc[quarters.year.eq(2020) & quarters.quarter.eq(1)].iloc[0]
    q4 = quarters.loc[quarters.year.eq(2020) & quarters.quarter.eq(4)].iloc[0]
    assert q1.interval_start == "2020-01-01T00:00:00+00:00"
    assert q1.interval_end == "2020-04-01T00:00:00+00:00"
    assert q4.interval_end == "2021-01-01T00:00:00+00:00"
