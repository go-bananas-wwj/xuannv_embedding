from __future__ import annotations

import json
from pathlib import Path

import pytest

from xuannv_embedding.cli import main
from xuannv_embedding.data_process import cli as data_cli
from xuannv_embedding.data_process import grid as grid_module
from xuannv_embedding.utils.manifest import load_manifest


@pytest.mark.parametrize(
    "command",
    ["grid", "registry", "partition", "materialize", "preprocess", "manifest", "validate"],
)
def test_data_commands_expose_their_real_help(command: str) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["data", command, "--help"])
    assert raised.value.code == 0


def test_manifest_and_validate_commands_round_trip(tmp_path: Path, capsys) -> None:
    processed = tmp_path / "processed"
    legacy_path = processed / "legacy" / "manifest.json"
    legacy_path.parent.mkdir(parents=True)
    source_path = processed / "patches" / "s2" / "sample.tif"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"test")
    legacy_path.write_text(
        json.dumps([{"patch_id": "p1", "s2": "../patches/s2/sample.tif"}]),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"

    assert (
        main(
            [
                "data",
                "manifest",
                "--legacy",
                str(legacy_path),
                "--region",
                "test-region",
                "--output",
                str(output),
                "--months",
                "202601",
            ]
        )
        == 0
    )
    document = load_manifest(output)
    assert document.records[0].sources == {"s2": "patches/s2/sample.tif"}
    capsys.readouterr()

    assert main(["data", "validate", "--manifest", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["record_count"] == 1


def test_grid_validate_applies_explicit_utm_seam_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    seam_path = tmp_path / "seam.json"
    seam_path.write_text(
        json.dumps(
            {
                "schema_version": "china_full_1280m_utm_seam_audit_v1",
                "policy_version": "adjacent-owner-zone-seam-v1",
                "overlap_pair_count": 1,
                "overlap_area_m2": 1.0,
                "total_parent_count": 1,
                "global_duplicate_area_fraction": 0.0,
                "maximum_global_duplicate_area_fraction": 0.001,
                "non_adjacent_pair_count": 0,
                "owner_order_mismatch_count": 0,
                "off_seam_pair_count": 0,
                "max_pair_overlap_fraction": 0.5,
                "passed": True,
                "seam_candidate_count": 1,
                "seam_tolerance_degrees": 0.05,
            }
        ),
        encoding="utf-8",
    )
    base = {
        "passed": False,
        "cross_zone_overlap_violation_count": 1,
        "missing": [],
        "duplicate_atlas_keys": [],
        "duplicate_sampled_keys": [],
        "hash_mismatches": {
            "identity_hash": 0,
            "footprint_hash": 0,
            "sampled_registry_footprint_hash": 0,
        },
    }
    monkeypatch.setattr(grid_module, "audit_grid_package", lambda *args, **kwargs: base)
    monkeypatch.setattr(grid_module, "read_sampled_registry_jsonl", lambda path: [])

    assert (
        data_cli.validate_main(
            [
                "--grid-root",
                str(tmp_path / "grid"),
                "--sampled-registry",
                str(tmp_path / "registry.jsonl"),
                "--utm-seam-audit",
                str(seam_path),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["legacy_cross_zone_pair_over_1pct_count"] == 1
