import numpy as np
import pytest

from xuannv_embedding.data_process.v5_negative_rules import (
    TASKS,
    expected_negatives,
    inspect_overlay,
)


def _inputs():
    shape = (2, 9, 9)
    return dict(
        states={task: np.zeros(shape, "u1") for task in TASKS},
        worldcover=np.zeros(shape, "u1"),
        worldcover_valid=np.ones(shape, bool),
        slope=np.zeros(shape, "f4"),
        slope_valid=np.ones(shape, bool),
    )


def test_negatives_require_evidence_preserve_positives_and_do_not_erode_across_patches():
    inputs = _inputs()
    inputs["worldcover"][0] = 80
    masks = expected_negatives(**inputs, erosion_pixels=2, steep_slope_degrees=20)
    assert masks["building"][0, 4, 4] and not masks["building"][1].any()
    assert not masks["building"][0, :2].any()
    inputs["states"]["building"][0, 4, 4] = 1
    assert not expected_negatives(**inputs, erosion_pixels=2, steep_slope_degrees=20)["building"][
        0, 4, 4
    ]
    inputs["slope"][1] = 25
    masks = expected_negatives(**inputs, erosion_pixels=2, steep_slope_degrees=20)
    assert masks["water_area"][1].all() and not masks["waterway"][1].any()
    inputs["slope_valid"][1] = False
    assert not expected_negatives(**inputs, erosion_pixels=2, steep_slope_degrees=20)["water_area"][
        1
    ].any()


def test_unknown_overlay_never_becomes_background_negative_and_confidence_is_exact():
    expected = np.zeros((2, 3, 3), bool)
    expected[0, 1, 1] = True
    states = np.where(expected, 3, 0).astype("u1")
    confidence = np.where(expected, 224, 0).astype("u1")
    assert inspect_overlay(expected, states, confidence, 224)["status"] == "passed"
    states[1, 1, 1] = 3
    confidence[1, 1, 1] = 224
    result = inspect_overlay(expected, states, confidence, 224)
    assert result["unexpected_negative_pixels"] == 1 and result["status"] == "failed"
    confidence[0, 1, 1] = 200
    assert inspect_overlay(expected, states, confidence, 224)["confidence_mismatch_pixels"] == 1
    with pytest.raises(ValueError):
        inspect_overlay(expected, states.astype("u2"), confidence, 224)


def test_zero_erosion_is_identity_and_unknown_state_codes_are_rejected():
    inputs = _inputs()
    inputs["worldcover"][0, 0, 0] = 80
    assert expected_negatives(**inputs, erosion_pixels=0, steep_slope_degrees=20)["building"][
        0, 0, 0
    ]
    inputs["states"]["building"][0, 0, 0] = 255
    with pytest.raises(ValueError):
        expected_negatives(**inputs, erosion_pixels=0, steep_slope_degrees=20)


def test_negative_rule_cli_keeps_the_pilot_limit_explicit(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_negative_rules

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda _: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *args: None)
    monkeypatch.setattr(
        v5_negative_rules, "audit_negative_rules", lambda *a, **k: calls.append((a, k))
    )
    argv = ["--stage", "target-negative-rules", "--max-patches", "32"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        argv += ["--" + key, str(tmp_path / key)]
    assert v5_cli.main(argv) == 0
    assert calls == [((tmp_path / "dataset-root", tmp_path / "report-root"), {"max_patches": 32})]


def test_negative_rule_full_audit_checks_years_reuses_chunks_and_rejects_changed_evidence(tmp_path):
    import hashlib

    import pandas as pd
    import zarr

    from xuannv_embedding.data_process.v5_negative_rules import RULES, audit_negative_rules
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    data, report = tmp_path / "data", tmp_path / "report"
    (data / "registry").mkdir(parents=True)
    (data / "targets").mkdir()
    report.mkdir()
    pd.DataFrame({"patch_id": ["p"], "split": ["train"]}).to_parquet(
        data / "registry/national_62000.parquet", index=False
    )
    roots, paths = {}, {}
    for family in ["osm", "static", "reliable_negative"]:
        paths[family] = tmp_path / f"{family}.zarr"
        roots[family] = zarr.open_group(str(paths[family]), mode="w")
        roots[family].attrs["patch_ids"] = ["p"]
    roots["reliable_negative"].attrs.update(
        {
            "base_osm30_sidecar": str(paths["osm"]),
            "static_sidecar": str(paths["static"]),
            "years": [2020, 2021],
            "target_names": list(TASKS),
            "rules": RULES,
            "unknown_preserved": True,
            "positive_precedence": True,
            "negative_confidence": 224,
            "erosion_pixels": 2,
            "steep_slope_degrees": 20.0,
        }
    )
    manifest, values = [], []

    def add(family, name, array):
        group, key = name.rsplit("/", 1)
        roots[family].require_group(group).array(key, array)
        manifest.append(
            {
                "family": family,
                "array": name,
                "path": str(paths[family]),
                "registry_order_verified": True,
                "source_metadata_sha256": sha256(paths[family] / ".zattrs"),
            }
        )
        values.append(
            {
                "family": family,
                "array": name,
                "path": str(paths[family]),
                "status": "values_checked_provenance_pending",
                "decoded_values_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        )

    zero = np.zeros((1, 128, 128), "u1")
    for year in [2020, 2021]:
        add("static", f"targets/worldcover_{year}", np.full_like(zero, 80 if year == 2020 else 10))
        add("static", f"valid_masks/worldcover_{year}", np.ones_like(zero, bool))
        for task in TASKS:
            add("osm", f"{year}/states/{task}", zero.copy())
            states = zero.copy()
            confidence = zero.copy()
            if year == 2020 and task == "building":
                states[:, 2:-2, 2:-2] = 3
                confidence[:, 2:-2, 2:-2] = 224
            add("reliable_negative", f"{year}/states/{task}", states)
            add("reliable_negative", f"{year}/confidence/{task}", confidence)
    add("static", "targets/dem_slope", np.zeros_like(zero, "f4"))
    add("static", "valid_masks/dem_slope", np.ones_like(zero, bool))
    pd.DataFrame(manifest).to_parquet(data / "targets/manifest.parquet", index=False)
    pd.DataFrame(values).to_parquet(report / "target_value_audit.parquet", index=False)
    write_json(
        report / "osm_temporal_progress.json",
        {
            "status": "temporal_cross_checks_finished",
            "failed_groups": 0,
            "metadata_sha256": sha256(paths["osm"] / ".zattrs"),
            "manifest_sha256": sha256(data / "targets/manifest.parquet"),
            "value_audit_sha256": sha256(report / "target_value_audit.parquet"),
        },
    )
    first = audit_negative_rules(data, report)
    assert first["processed_targets"] == 8 and first["failed_targets"] == 0
    assert first["stored_negative_pixels"] == 124 * 124
    assert audit_negative_rules(data, report)["reused_targets"] == 8
    roots["reliable_negative"]["2021/states/building"][0, 50, 50] = 3
    roots["reliable_negative"]["2021/confidence/building"][0, 50, 50] = 224
    with pytest.raises(ValueError, match="bytes changed"):
        audit_negative_rules(data, report)
    pilot = audit_negative_rules(data, report, max_patches=1)
    assert pilot["failed_targets"] == 1 and pilot["unexpected_negative_pixels"] == 1
    assert pilot["training_authorized"] is False
