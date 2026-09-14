import json

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter, shift


def test_review_selection_excludes_entire_locations_and_ignores_measured_status():
    from xuannv_embedding.data_process.v5_adaptive_review import select_review_rows

    rows = pd.DataFrame(
        [
            {
                "observation_id": f"{sensor}-{year}-{i}",
                "patch_id": f"p{i}",
                "sensor": sensor,
                "year": year,
                "split": "train",
                "reference_clear_fraction": [0, 0.1, 0.4, 0.8, 0.99][i % 5],
                "status": "passed",
            }
            for sensor in ["a", "b"]
            for year in [2020, 2021]
            for i in range(150)
        ]
    )
    selected = select_review_rows(rows, {"p0", "p1"})
    assert len(selected) == 80 and selected.patch_id.is_unique
    assert not set(selected.patch_id) & {"p0", "p1"}
    changed = rows.sample(frac=1, random_state=4)
    changed["status"] = "uncertain"
    assert (
        selected.observation_id.tolist()
        == select_review_rows(changed, {"p0", "p1"}).observation_id.tolist()
    )
    rows["split"] = "test"
    with pytest.raises(ValueError, match="eligible training"):
        select_review_rows(rows, set())


def test_real_pair_consistency_checks_reverse_and_known_perturbations():
    from xuannv_embedding.data_process.v5_adaptive_review import review_pair

    reference = gaussian_filter(np.random.default_rng(71).normal(size=(256, 256)), 1.5)
    moving = shift(reference, (0, 1.5), order=1, mode="constant")
    valid = np.ones(reference.shape, bool)
    result = review_pair(reference, moving, valid, valid, gsd=5)
    assert result["candidate"]["status"] == "over_limit"
    assert result["consistency"] == "self_consistent_candidate"
    assert result["reverse_cycle_error_pixels"] <= 0.35
    assert len(result["perturbations"]) == 3
    assert all(x["error_pixels"] <= 0.35 for x in result["perturbations"])
    empty = review_pair(reference, moving, valid, np.zeros_like(valid), gsd=5)
    assert empty["consistency"] == "forward_uncertain" and empty["perturbations"] == []


def test_consistency_failure_never_becomes_fusion_approval(monkeypatch):
    from xuannv_embedding.data_process import v5_adaptive_review as module

    calls = []

    def measured(*args, **kwargs):
        calls.append(True)
        # Individually confident but contradicting opposite directions and injected shifts.
        return {"status": "passed", "translation_yx_m": [1.0, 0.0]}

    monkeypatch.setattr(module, "audit_adaptive", measured)
    result = module.review_pair(
        np.ones((160, 160)),
        np.ones((160, 160)),
        np.ones((160, 160), bool),
        np.ones((160, 160), bool),
        gsd=1,
    )
    assert result["consistency"] == "consistency_failed"
    assert not result["pixel_fusion_authorized"]
    assert len(calls) == 5


def test_review_cli_requires_frozen_roots_and_exclusion_manifest(tmp_path, monkeypatch):
    import fcntl

    from xuannv_embedding.data_process import v5_adaptive_review, v5_cli

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(v5_adaptive_review, "run_adaptive_review", lambda *a: calls.append(a))
    args = ["--stage", "adaptive-alignment-review"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + key, str(tmp_path / key)]
    for extra in [
        [],
        ["--sensor-family", "jilin1"],
        ["--quality-root", str(tmp_path / "qa")],
        ["--alignment-audit-root", str(tmp_path / "audit")],
        ["--adaptive-calibration-root", str(tmp_path / "calibration")],
    ]:
        args += extra
        with pytest.raises(SystemExit):
            v5_cli.main(args)
    args += ["--exclusion-inputs", str(tmp_path / "excluded.parquet")]
    source = tmp_path / "source-root"
    source.mkdir(exist_ok=True)
    with (source / ".adaptive-alignment-review.jilin1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another V5"):
            v5_cli.main(args)
    assert v5_cli.main(args) == 0 and calls[0][-1] == [tmp_path / "excluded.parquet"]


def test_review_freezes_before_measurement_rechecks_inputs_and_replays_readonly(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from xuannv_embedding.data_process import v5_adaptive_review as module
    from xuannv_embedding.data_process.v5_adaptive_alignment import LAYOUT
    from xuannv_embedding.data_process.v5_alignment import PARAMETERS
    from xuannv_embedding.data_process.v5_clear_audit import inspect_clear
    from xuannv_embedding.data_process.v5_intraband import _runtime_versions
    from xuannv_embedding.data_process.v5_rasters import NativeRaster
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    source = tmp_path / "source.bin"
    source.write_bytes(b"original fixture")
    texture = gaussian_filter(np.random.default_rng(14).normal(size=(160, 160)), 1.5)
    frame = NativeRaster(
        np.stack([texture] * 4),
        np.ones((4, 160, 160), bool),
        ("blue", "green", "red", "nir"),
        (8, 0, 300000, 0, -8, 4000000),
        "EPSG:32650",
    )

    class Reader:
        files = {}

        def __init__(self, *a):
            pass

        def read(self, row):
            proof = {
                "file_sha256": sha256(source),
                "quality_mask_sha256": "mask",
                "quality_status": "fixture",
                "clear_fraction_by_band": [1.0] * 4,
            }
            return frame, frame, 8, "green", proof

        def verify_unchanged(self):
            pass

    monkeypatch.setattr(module, "AuditReader", Reader)
    rows = pd.DataFrame(
        [
            {
                "observation_id": f"obs{i}",
                "patch_id": f"p{i}",
                "sensor": "GF1",
                "year": 2020,
                "split": "train",
                "path": str(source),
                "file_sha256": sha256(source),
            }
            for i in range(5)
        ]
    )
    reader = Reader()
    outputs = []
    for row in rows.itertuples():
        result = inspect_clear(reader, row)
        for key in ["pairs", "clear_fraction_by_band", "missing_bands"]:
            result[key] = json.dumps(result[key])
        outputs.append(result)
    audit, cal, base = tmp_path / "audit", tmp_path / "candidate", tmp_path / "base"
    for p in [audit, cal, base]:
        p.mkdir()
    baseline = base / "calibration.json"
    write_json(baseline, {"status": "passed"})
    implementation = {
        "v5_adaptive_alignment.py": sha256(
            Path(module.__file__).with_name("v5_adaptive_alignment.py")
        )
    }

    def seal(root, inputs, output_name, document, fingerprint, snapshot=None):
        inputs.to_parquet(root / "inputs.parquet", index=False)
        lock = {"inputs_sha256": sha256(root / "inputs.parquet"), "fingerprint": fingerprint}
        if snapshot is not None:
            lock["snapshot"] = snapshot
        write_json(root / "inputs.lock.json", lock)
        if output_name.endswith(".parquet"):
            pd.DataFrame(document).to_parquet(root / output_name, index=False)
        else:
            write_json(root / output_name, document)
        write_json(
            root / "output.lock.json",
            {
                "inputs_lock_sha256": sha256(root / "inputs.lock.json"),
                (
                    "observations_sha256"
                    if output_name.endswith(".parquet")
                    else "calibration_sha256"
                ): sha256(root / output_name),
            },
        )

    seal(
        audit,
        rows,
        "observations.parquet",
        outputs,
        {
            "family": "gaofen",
            "code_sha256": implementation,
            "calibration": {"calibration_sha256": sha256(baseline)},
        },
        {"quality_inputs_sha256": {}},
    )
    fingerprint = {
        "family": "gaofen",
        "code_sha256": implementation,
        "matcher": PARAMETERS,
        "layout": LAYOUT,
        "runtime": {**_runtime_versions(), "pandas": pd.__version__},
        "baseline_files_sha256": {str(baseline): sha256(baseline)},
    }
    seal(cal, rows.iloc[:1], "calibration.json", {"status": "passed"}, fingerprint)
    excluded = tmp_path / "excluded.parquet"
    rows.iloc[1:2].to_parquet(excluded, index=False)
    original = module.review_pair
    seen = []
    report = tmp_path / "report"

    def check(*args, **kwargs):
        frozen = list((report / "adaptive_alignment_review/gaofen").glob("*/inputs.lock.json"))
        assert len(frozen) == 1
        seen.append(sha256(frozen[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "review_pair", check)
    result = module.run_adaptive_review(
        tmp_path, report, "gaofen", tmp_path / "qa", audit, cal, [excluded]
    )
    assert (
        result["selected"] == 3 and result["pairs"] == 9 and len(seen) == 9 and len(set(seen)) == 1
    )
    root = Path(result["output"])
    files = {p: (sha256(p), p.stat().st_mtime_ns) for p in root.iterdir()}
    assert set(pd.read_parquet(root / "inputs.parquet").patch_id) == {"p2", "p3", "p4"}
    assert not json.loads((root / "review.json").read_text())["pixel_fusion_authorized"]
    monkeypatch.setattr(module, "review_pair", lambda *a, **k: pytest.fail("cache rematched"))
    repeated = module.run_adaptive_review(
        tmp_path, report, "gaofen", tmp_path / "qa", audit, cal, [excluded]
    )
    assert repeated["reused"] and files == {p: (sha256(p), p.stat().st_mtime_ns) for p in files}
    output = root / "review.json"
    saved = output.read_bytes()
    output.write_text("{}")
    with pytest.raises(ValueError, match="output seal"):
        module.run_adaptive_review(
            tmp_path, report, "gaofen", tmp_path / "qa", audit, cal, [excluded]
        )
    output.write_bytes(saved)
    new_report = tmp_path / "changed_report"

    def corrupt(*a, **k):
        result = original(*a, **k)
        path = next((new_report / "adaptive_alignment_review/gaofen").glob("*/inputs.lock.json"))
        path.write_text("{}")
        return result

    monkeypatch.setattr(module, "review_pair", corrupt)
    with pytest.raises(ValueError, match="inputs or implementation"):
        module.run_adaptive_review(
            tmp_path, new_report, "gaofen", tmp_path / "qa", audit, cal, [excluded]
        )
    assert not list((new_report / "adaptive_alignment_review/gaofen").glob("*/output.lock.json"))
    fingerprint["runtime"]["numpy"] = "changed-runtime"
    seal(cal, rows.iloc[:1], "calibration.json", {"status": "passed"}, fingerprint)
    with pytest.raises(ValueError, match="runtime"):
        module.run_adaptive_review(
            tmp_path, report, "gaofen", tmp_path / "qa", audit, cal, [excluded]
        )
