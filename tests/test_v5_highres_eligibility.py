import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def inputs():
    registry = pd.DataFrame([{"patch_id": "p", "split": "train"}])
    rows = []
    for i, date in enumerate(
        ["2020-01-01", "2020-02-01", "2020-04-01", "2020-07-01", "2020-10-01", "2021-01-01"]
    ):
        rows.append(
            dict(
                observation_id=f"b{i}",
                scene_group_id=f"s{i}",
                native_observation_id=f"n{i}",
                patch_id="p",
                split="train",
                sensor="sensor",
                family="jilin1",
                product_id="jilin1_ms_5m",
                year=int(date[:4]),
                acquired_at=date,
                path=f"{i}.tif",
                file_sha256=str(i),
                band_ids=["B1", "B2"],
                valid_pixels_by_band=[10, 0],
                clear_fraction=0.1,
                quality_status="model_inferred_needs_visual_review",
                geometry_status="verified_source_grid",
                contract_status="verified_native_product",
                selected_bands_complete=False,
            )
        )
    return registry, pd.DataFrame(rows)


def test_unknown_alignment_keeps_clear_pixels_but_never_grants_strict_fusion():
    from xuannv_embedding.data_process.v5_highres_eligibility import build_views

    registry, rows = inputs()
    branches, scenes, candidates, selected = build_views(registry, rows, pd.DataFrame())
    assert branches.tolerant_reconstruction_candidate.all()
    assert not branches.strict_pixel_fusion_candidate.any()
    assert branches.alignment_status.eq("unknown").all()
    assert branches.usable_band_ids.tolist() == [["B1"]] * 6
    assert not branches.selected_bands_complete.any()
    assert len(scenes) == len(candidates) == 6
    chosen = selected.loc[selected.year.eq(2020)]
    assert chosen.scene_group_id.tolist() == ["s0", "s2", "s3", "s4"]
    assert selected.loc[selected.year.eq(2021)].scene_group_id.tolist() == ["s5"]
    assert selected.serves_quarters.tolist() == [[1, 2, 3, 4]] * 5
    again = build_views(registry, rows.sample(frac=1, random_state=7), pd.DataFrame())
    pd.testing.assert_frame_equal(selected, again[-1])


def test_native_pass_is_not_absolute_pass_and_native_failure_blocks_qa_dependents():
    from xuannv_embedding.data_process.v5_highres_eligibility import build_views

    registry, rows = inputs()
    dependent = rows.iloc[0].copy()
    dependent["observation_id"] = "b0-pan"
    dependent["product_id"] = "jilin1_b0_5m"
    rows = pd.concat([rows, pd.DataFrame([dependent])], ignore_index=True)
    audit = pd.DataFrame(
        [
            dict(
                observation_id=f"n{i}",
                patch_id="p",
                split="train",
                sensor="sensor",
                year=2020,
                status=status,
                pairs=json.dumps(
                    [
                        dict(
                            status=status,
                            residual_m=12 if i == 0 else 1,
                            translation_yx_m=[0, 12 if i == 0 else 1],
                        )
                    ]
                ),
                scope="relative native spectral bands; not alignment to the base reference",
            )
            for i, status in [(0, "over_limit"), (1, "passed")]
        ]
    )
    branches, _, _, _ = build_views(registry, rows, audit)
    rejected = branches.loc[branches.scene_group_id.eq("s0")]
    assert not rejected.tolerant_reconstruction_candidate.any()
    assert rejected.exclusion_reasons.tolist() == [["native_spectral_over_limit"]] * 2
    passed = branches.loc[branches.observation_id.eq("b1")].iloc[0]
    assert passed.native_alignment_status == "passed"
    assert passed.alignment_status == "unknown" and not passed.strict_pixel_fusion_candidate
    assert passed.tolerant_reconstruction_candidate
    audit.loc[0, "year"] = 2021
    with pytest.raises(ValueError, match="audit identity"):
        build_views(registry, rows, audit)


def test_quality_and_contract_failures_are_explicit_and_not_silently_discarded():
    from xuannv_embedding.data_process.v5_highres_eligibility import build_views

    registry, rows = inputs()
    rows.at[0, "quality_status"] = "qa_missing"
    rows.at[0, "valid_pixels_by_band"] = [0, 0]
    rows.at[1, "contract_status"] = "unknown"
    rows.at[2, "geometry_status"] = "invalid"
    rows.at[3, "valid_pixels_by_band"] = [0, 0]
    branches, scenes, candidates, _ = build_views(registry, rows, pd.DataFrame())
    assert len(branches) == len(scenes) == 6 and len(candidates) == 2
    assert "qa_missing" in branches.iloc[0].exclusion_reasons
    assert "unknown_product_contract" in branches.iloc[1].exclusion_reasons
    assert "invalid_geometry" in branches.iloc[2].exclusion_reasons
    assert "no_valid_pixels" in branches.iloc[3].exclusion_reasons
    rows.at[0, "valid_pixels_by_band"] = [1, 0]
    with pytest.raises(ValueError, match="missing QA"):
        build_views(registry, rows, pd.DataFrame())


def test_view_rejects_duplicate_split_calendar_and_nonfinite_quality():
    from xuannv_embedding.data_process.v5_highres_eligibility import build_views

    registry, rows = inputs()
    for field, value in [("split", "test"), ("year", 2021), ("clear_fraction", float("nan"))]:
        changed = rows.copy()
        changed.at[0, field] = value
        with pytest.raises(ValueError):
            build_views(registry, changed, pd.DataFrame())
    with pytest.raises(ValueError, match="duplicate"):
        build_views(registry, pd.concat([rows, rows.iloc[:1]]), pd.DataFrame())


def test_view_publication_locks_inputs_replays_without_rewriting_and_detects_tampering(
    tmp_path, monkeypatch
):
    from xuannv_embedding.data_process import v5_highres_eligibility as module
    from xuannv_embedding.data_process.v5_sources import sha256

    registry, rows = inputs()
    source = tmp_path / "qa.json"
    source.write_text("{}")
    reader = SimpleNamespace(
        registry=registry.set_index("patch_id"),
        files={str(source): sha256(source)},
        configuration={},
        verify_unchanged=lambda: None,
    )
    monkeypatch.setattr(module, "HighresQualityReader", lambda *a: reader)
    monkeypatch.setattr(module, "normalize_branches", lambda *a: rows)
    result = module.run_highres_eligibility(
        tmp_path / "data", tmp_path / "report", "jilin1", tmp_path / "qa"
    )
    root = Path(result["output"])
    assert result["scope"] == "frozen_available_QA_snapshot_candidate_views"
    assert not result["training_authorized"] and result["branches"] == 6
    files = {p: (sha256(p), p.stat().st_mtime_ns) for p in root.iterdir()}
    replay = module.run_highres_eligibility(
        tmp_path / "data", tmp_path / "report", "jilin1", tmp_path / "qa"
    )
    assert replay["reused"] and files == {p: (sha256(p), p.stat().st_mtime_ns) for p in files}
    (root / "branches.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="output seal"):
        module.run_highres_eligibility(
            tmp_path / "data", tmp_path / "report", "jilin1", tmp_path / "qa"
        )


def test_eligibility_cli_requires_family_quality_and_isolates_writers(tmp_path, monkeypatch):
    import fcntl

    from xuannv_embedding.data_process import v5_cli, v5_highres_eligibility

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(
        v5_highres_eligibility, "run_highres_eligibility", lambda *a: calls.append(a)
    )
    args = ["--stage", "highres-eligibility"]
    for k in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + k, str(tmp_path / k)]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--sensor-family", "jilin1"]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--quality-root", str(tmp_path / "qa")]
    (tmp_path / "source-root").mkdir(exist_ok=True)
    with (tmp_path / "source-root/.highres-eligibility.jilin1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another V5"):
            v5_cli.main(args)
    assert v5_cli.main(args) == 0 and calls[0][-1] is None


def test_empty_candidate_view_keeps_all_rejections_and_scene_metadata_must_agree():
    from xuannv_embedding.data_process.v5_highres_eligibility import build_views

    registry, rows = inputs()
    rows["valid_pixels_by_band"] = [[0, 0] for _ in range(len(rows))]
    branches, scenes, candidates, selected = build_views(registry, rows, pd.DataFrame())
    assert len(branches) == len(scenes) == 6 and candidates.empty and selected.empty
    rows.at[1, "scene_group_id"] = "s0"
    with pytest.raises(ValueError, match="scene branches"):
        build_views(registry, rows, pd.DataFrame())


def test_duplicate_content_prefers_eligible_alias_and_never_crosses_splits():
    from xuannv_embedding.data_process.v5_highres_eligibility import build_views

    registry, rows = inputs()
    rows.at[0, "file_sha256"] = rows.at[1, "file_sha256"]
    rows.at[0, "valid_pixels_by_band"] = [0, 0]
    branches = build_views(registry, rows, pd.DataFrame())[0]
    assert branches.iloc[1].tolerant_reconstruction_candidate
    rows.at[0, "valid_pixels_by_band"] = [10, 0]
    branches = build_views(registry, rows, pd.DataFrame())[0]
    assert branches.iloc[1].exclusion_reasons == ["duplicate_content"]
    registry = pd.concat([registry, pd.DataFrame([dict(patch_id="test-p", split="test")])])
    rows.at[1, "patch_id"] = "test-p"
    rows.at[1, "split"] = "test"
    with pytest.raises(ValueError, match="content crosses"):
        build_views(registry, rows, pd.DataFrame())


def test_native_audit_source_snapshot_and_publication_seal_are_required(tmp_path):
    from xuannv_embedding.data_process.v5_highres_eligibility import read_audit
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    rows = pd.DataFrame([dict(observation_id="n0", status="uncertain")])
    for name in ["inputs.parquet", "observations.parquet"]:
        rows.to_parquet(tmp_path / name, index=False)
    lock = {
        "inputs_sha256": sha256(tmp_path / "inputs.parquet"),
        "fingerprint": {"family": "jilin1"},
        "snapshot": {"quality_inputs_sha256": {"qa": "expected"}},
    }
    write_json(tmp_path / "inputs.lock.json", lock)
    write_json(
        tmp_path / "output.lock.json",
        {
            "inputs_lock_sha256": sha256(tmp_path / "inputs.lock.json"),
            "observations_sha256": sha256(tmp_path / "observations.parquet"),
        },
    )
    reader = SimpleNamespace(files={"qa": "different"})
    with pytest.raises(ValueError, match="snapshot differs"):
        read_audit(tmp_path, "jilin1", reader)
    reader.files = {"qa": "expected"}
    assert len(read_audit(tmp_path, "jilin1", reader)[0]) == 1
    (tmp_path / "observations.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="seal"):
        read_audit(tmp_path, "jilin1", reader)
