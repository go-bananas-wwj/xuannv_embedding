from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from xuannv_embedding.data_process import sampling as MODULE


def test_utm_owner_zone_uses_half_open_longitude_intervals() -> None:
    assert MODULE.utm_owner_zone(72.0) == 43
    assert MODULE.utm_owner_zone(77.999999) == 43
    assert MODULE.utm_owner_zone(78.0) == 44
    assert MODULE.utm_owner_epsg(116.0, 39.0) == 32650


def test_patch_geometry_is_1280_meter_grid_cell() -> None:
    record = {"grid_epsg": 32650, "grid_col": 320, "grid_row": 330}
    geometry = MODULE.patch_geometry_utm(record)
    assert geometry.bounds == (409600.0, 422400.0, 410880.0, 423680.0)
    assert geometry.area == 1280.0**2


def test_choose_replacements_respects_owner_zone_and_unique_macro() -> None:
    inventory = [
        {
            "macro_id": "utm50n_c32_r330",
            "grid_id": "utm50n",
            "grid_epsg": 32650,
            "macro_col": 32,
            "macro_row": 330,
            "utm_bounds": [409600.0, 4224000.0, 422400.0, 4236800.0],
            "estimated_patch_count": 100.0,
            "admin1": "test",
        },
        {
            "macro_id": "utm50n_c33_r330",
            "grid_id": "utm50n",
            "grid_epsg": 32650,
            "macro_col": 33,
            "macro_row": 330,
            "utm_bounds": [422400.0, 4224000.0, 435200.0, 4236800.0],
            "estimated_patch_count": 100.0,
            "admin1": "test",
        },
    ]
    selected = MODULE.choose_replacements(
        inventory=inventory,
        country_geometry=box(115.0, 37.0, 118.0, 40.0),
        existing=[],
        count=2,
        seed=17,
        sampling_layer="semantic_audit_reserve",
        primary_reasons=["urban_and_built_up", "rare_natural_surfaces"],
        forbid_supplement_macros=set(),
    )
    assert len(selected) == 2
    assert len({item["macro_id"] for item in selected}) == 2
    assert all(
        item["grid_epsg"] == MODULE.utm_owner_epsg(item["longitude"], item["latitude"])
        for item in selected
    )
    assert [item["sampling_primary_reason"] for item in selected] == [
        "urban_and_built_up",
        "rare_natural_surfaces",
    ]


def test_normalize_record_adds_stable_identity_and_bounds() -> None:
    record = {
        "patch_id": "preview_utm50n_c32_r330_c0_r0",
        "macro_id": "utm50n_c32_r330",
        "grid_id": "utm50n",
        "grid_epsg": 32650,
        "grid_col": 320,
        "grid_row": 3300,
        "longitude": 116.0,
        "latitude": 38.0,
        "admin1": "test",
        "sampling_seed": 7,
        "sampling_tier": "base_spatial",
    }
    normalized = MODULE.normalize_record(record, registry_version="candidate-v1")
    assert normalized["sampling_layer"] == "base_expected_1pct"
    assert len(normalized["identity_hash"]) == 64
    assert len(normalized["canonical_wgs84_footprint_hash"]) == 64
    assert len(normalized["utm_bounds"]) == 4
    assert len(normalized["wgs84_bounds"]) == 4
    assert normalized["registry_status"] == "design_candidate_not_quality_eligible"


def test_cross_zone_overlap_audit_finds_metric_violation() -> None:
    records = [
        {
            "patch_id": "west",
            "grid_epsg": 32646,
            "grid_col": 607,
            "grid_row": 2911,
            "sampling_layer": "base_expected_1pct",
        },
        {
            "patch_id": "east",
            "grid_epsg": 32647,
            "grid_col": 173,
            "grid_row": 2911,
            "sampling_layer": "utm_spatial_balance_supplement",
        },
    ]
    violations = MODULE.cross_zone_overlap_violations(records)
    assert len(violations) == 1
    assert violations[0]["overlap_fraction"] > 0.01
    assert violations[0]["remove_patch_id"] == "east"


def test_enforce_macro_limits_keeps_one_base_and_one_supplement() -> None:
    records = [
        {"patch_id": "b1", "macro_id": "m1", "sampling_tier": "base_spatial"},
        {"patch_id": "b2", "macro_id": "m1", "sampling_tier": "base_spatial"},
        {
            "patch_id": "s1",
            "macro_id": "m1",
            "sampling_tier": "spatial_stratified_supplement",
        },
        {
            "patch_id": "s2",
            "macro_id": "m1",
            "sampling_tier": "coastal_ocean_adjacent_supplement",
        },
    ]
    kept, dropped = MODULE.enforce_macro_limits(records, seed=9)
    assert len(kept) == 2
    assert len(dropped) == 2
    assert sum(item["sampling_tier"] == "base_spatial" for item in kept) == 1
    assert sum(item["sampling_tier"] != "base_spatial" for item in kept) == 1


def test_sqrt_group_allocation_is_exact_and_lifts_small_groups() -> None:
    groups = {"large": [{}] * 100, "small": [{}] * 4}
    allocation = MODULE.allocate_sqrt_by_group(12, groups)
    assert sum(allocation.values()) == 12
    assert allocation == {"large": 10, "small": 2}
    assert MODULE.allocate_sqrt_by_counts(12, {"large": 100, "small": 4}) == {
        "large": 10,
        "small": 2,
    }


def test_assign_admin1_recomputes_existing_values(tmp_path: Path) -> None:
    admin_path = tmp_path / "admin.geojson"
    gpd.GeoDataFrame(
        {"shapeName": ["correct"]},
        geometry=[box(115.0, 38.0, 117.0, 40.0)],
        crs="EPSG:4326",
    ).to_file(admin_path, driver="GeoJSON")
    records = [
        {"longitude": 116.0, "latitude": 39.0, "admin1": "stale"},
        {"longitude": 120.0, "latitude": 39.0, "admin1": "stale"},
    ]
    assigned = MODULE.assign_admin1(records, admin_path)
    assert assigned == 1
    assert records[0]["admin1"] == "correct"
    assert records[1]["admin1"] == "ADM1_NOT_COVERED_BY_FROZEN_SOURCE"
