from __future__ import annotations

import json
from pathlib import Path

from xuannv_embedding.data_process import registry as MODULE


def test_one_chip_per_macrocell_and_supplement_reasons(tmp_path: Path) -> None:
    policy = {
        "sampling": {
            "macro_side_patches": 10,
            "sampling_seed": 7,
            "base_inclusion_probability": 1.0,
            "target_total": 2,
            "max_total": 10,
            "supplement_reservoir_multiplier": 3,
            "supplement_max_per_macrocell": 2,
            "supplement_quotas": {"worldcover:wetland": 2},
        }
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    records = []
    for row in range(10):
        for col in range(10):
            records.append(
                {
                    "patch_id": f"A_{row}_{col}",
                    "grid_id": "A",
                    "grid_row": row,
                    "grid_col": col,
                    "grid_epsg": 32650,
                    "wgs84_bounds": [116.0, 39.0, 116.1, 39.1],
                    "geometry_hash": f"hash-{row}-{col}",
                    "macro_candidate_count": 100,
                    "eligible": True,
                    "eligible_reasons": ["test"],
                    "admin1": "province-a",
                    "strata": ["worldcover:wetland"] if col < 2 else [],
                    "stratum_scores": {"worldcover:wetland": float(col + 1)},
                }
            )
    records.extend(
        [
            {
                "patch_id": "A_10_0",
                "grid_id": "A",
                "grid_row": 10,
                "grid_col": 0,
                "grid_epsg": 32650,
                "wgs84_bounds": [116.0, 39.0, 116.1, 39.1],
                "geometry_hash": "hash-10-0",
                "macro_candidate_count": 2,
                "eligible": True,
                "eligible_reasons": ["test"],
                "admin1": "province-a",
                "strata": [],
            },
            {
                "patch_id": "excluded",
                "grid_id": "A",
                "grid_row": 10,
                "grid_col": 1,
                "grid_epsg": 32650,
                "wgs84_bounds": [116.0, 39.0, 116.1, 39.1],
                "geometry_hash": "hash-10-1",
                "macro_candidate_count": 2,
                "eligible": False,
                "eligible_reasons": ["excluded"],
                "strata": ["worldcover:wetland"],
            },
        ]
    )
    atlas_path = tmp_path / "atlas.jsonl"
    atlas_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    registry, report = MODULE.build_registry(
        atlas_path=atlas_path,
        policy_path=policy_path,
        macro_side=None,
        seed=None,
        quota_overrides={},
    )

    selected = {record["patch_id"]: record for record in registry}
    assert report["base_selected"] == 2
    assert report["supplemental"]["worldcover:wetland"]["final_coverage"] == 2
    assert "excluded" not in selected
    assert any("supplement:worldcover:wetland" in record["sampling_reasons"] for record in registry)


def test_sparse_macrocell_has_expected_one_percent_probability(tmp_path: Path) -> None:
    policy = {
        "sampling": {
            "macro_side_patches": 10,
            "sampling_seed": 11,
            "base_inclusion_probability": 0.01,
            "target_total": 1,
            "max_total": 1,
            "supplement_reservoir_multiplier": 2,
            "supplement_quotas": {},
        }
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    record = {
        "patch_id": "only_chip",
        "grid_id": "A",
        "grid_row": 0,
        "grid_col": 0,
        "grid_epsg": 32650,
        "wgs84_bounds": [116.0, 39.0, 116.1, 39.1],
        "geometry_hash": "only",
        "macro_candidate_count": 1,
        "eligible": True,
        "eligible_reasons": ["test"],
        "strata": [],
    }
    atlas_path = tmp_path / "atlas.jsonl"
    atlas_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    registry, report = MODULE.build_registry(atlas_path, policy_path, None, None, {})
    expected = MODULE._stable_unit_hash(11, "base-macrocell:('A', 0, 0)") < 0.01
    assert bool(registry) is expected
    assert report["base_expected"] == 0.01


def test_rejects_incomplete_macrocell_atlas(tmp_path: Path) -> None:
    policy = {
        "sampling": {
            "macro_side_patches": 10,
            "sampling_seed": 1,
            "target_total": 1,
            "max_total": 1,
            "supplement_quotas": {},
        }
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    record = {
        "patch_id": "missing_neighbor",
        "grid_id": "A",
        "grid_row": 0,
        "grid_col": 0,
        "grid_epsg": 32650,
        "wgs84_bounds": [116.0, 39.0, 116.1, 39.1],
        "geometry_hash": "missing",
        "macro_candidate_count": 2,
        "eligible": True,
        "eligible_reasons": ["test"],
        "strata": [],
    }
    atlas_path = tmp_path / "atlas.jsonl"
    atlas_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="atlas records"):
        MODULE.build_registry(atlas_path, policy_path, None, None, {})


def test_supplements_respect_regional_group_cap(tmp_path: Path) -> None:
    policy = {
        "sampling": {
            "macro_side_patches": 10,
            "sampling_seed": 2,
            "base_inclusion_probability": 0.000001,
            "target_total": 3,
            "max_total": 4,
            "supplement_reservoir_multiplier": 3,
            "supplement_max_per_macrocell": 3,
            "regional_balance": {
                "group_field": "regional_group",
                "max_fraction_per_group_per_stratum": 0.5,
                "require_known_group_for_supplement": True,
            },
            "supplement_quotas": {"worldcover:wetland": 2},
        }
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    records = []
    for col, group in enumerate(("north", "north", "south")):
        records.append(
            {
                "patch_id": f"p{col}",
                "grid_id": "A",
                "grid_row": 0,
                "grid_col": col,
                "grid_epsg": 32650,
                "wgs84_bounds": [116.0, 39.0, 116.1, 39.1],
                "geometry_hash": f"p{col}",
                "macro_candidate_count": 3,
                "eligible": True,
                "eligible_reasons": ["test"],
                "strata": ["worldcover:wetland"],
                "regional_group": group,
            }
        )
    atlas_path = tmp_path / "atlas.jsonl"
    atlas_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    _, report = MODULE.build_registry(atlas_path, policy_path, None, None, {})
    counts = report["supplemental"]["worldcover:wetland"]["coverage_by_regional_group"]
    assert counts == {"north": 1, "south": 1}
