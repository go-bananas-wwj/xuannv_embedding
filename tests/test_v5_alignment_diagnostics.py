import json

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter


def test_trace_explains_mask_texture_and_correlation_without_changing_decisions():
    from xuannv_embedding.data_process.v5_alignment import audit_translation
    from xuannv_embedding.data_process.v5_alignment_diagnostics import trace_windows

    x = gaussian_filter(np.random.default_rng(10).normal(size=(256, 256)), 1)
    valid = np.ones(x.shape, bool)
    trace = trace_windows(x, x, valid)
    assert all(w["reason"] == "accepted" for w in trace)
    assert (
        sum(w["reason"] == "accepted" for w in trace)
        == audit_translation(x, x, valid, gsd=5)["valid_windows"]
    )
    empty = trace_windows(x, x, np.zeros_like(valid))
    assert all(w["reason"] == "insufficient_template_validity" for w in empty)
    flat = trace_windows(np.ones_like(x), np.ones_like(x), valid)
    assert all(w["reason"] == "insufficient_template_texture" for w in flat)
    independent = gaussian_filter(np.random.default_rng(29).normal(size=x.shape), 1)
    assert all(
        w["reason"] == "low_integer_peak_correlation" for w in trace_windows(x, independent, valid)
    )


def test_trace_marks_periodic_ambiguity_and_valid_interior_missed_by_corners():
    from xuannv_embedding.data_process.v5_alignment_diagnostics import (
        coverage_evidence,
        trace_windows,
    )

    yy, xx = np.indices((256, 256))
    x = np.sin(xx * np.pi / 4) + np.cos(yy * np.pi / 4)
    valid = np.ones(x.shape, bool)
    assert any(w["reason"] == "ambiguous_peak" for w in trace_windows(x, x, valid))
    mask = np.zeros_like(valid)
    mask[70:186, 70:186] = True
    evidence = coverage_evidence(mask, mask)
    assert evidence["native_valid_fraction"] == evidence["clear_valid_fraction"]
    assert evidence["best_64px_clear_fraction"] == 1
    assert evidence["corner_windows_meeting_validity"] == 0
    assert all(w["reason"] == "insufficient_template_validity" for w in trace_windows(x, x, mask))


def test_diagnostic_selection_is_training_only_balanced_and_reproducible():
    from xuannv_embedding.data_process.v5_alignment_diagnostics import select_diagnostics

    rows = pd.DataFrame(
        [
            dict(
                observation_id=f"{sensor}-{year}-{fraction}-{i}",
                patch_id=f"p{i}",
                sensor=sensor,
                year=year,
                split="train" if i < 8 else "test",
                status="uncertain",
                clear_fraction_by_band=json.dumps([fraction] * 6),
            )
            for sensor in ["a", "b"]
            for year in [2020, 2021]
            for fraction in [0, 0.1, 0.4, 0.8, 0.99]
            for i in range(10)
        ]
    )
    selected = select_diagnostics(rows, reference_index=3)
    assert len(selected) == 80 and selected.split.eq("train").all()
    assert selected.groupby(["sensor", "year", "coverage_bin"]).size().eq(4).all()
    assert (
        selected.observation_id.tolist()
        == select_diagnostics(
            rows.sample(frac=1, random_state=3), reference_index=3
        ).observation_id.tolist()
    )
    with pytest.raises(ValueError, match="duplicate"):
        select_diagnostics(pd.concat([rows, rows.iloc[:1]]), reference_index=3)


def test_alignment_diagnostic_cli_requires_frozen_audit_and_quality(tmp_path, monkeypatch):
    import fcntl

    from xuannv_embedding.data_process import v5_alignment_diagnostics, v5_cli

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(
        v5_alignment_diagnostics, "diagnose_alignment", lambda *a, **k: calls.append((a, k))
    )
    args = ["--stage", "alignment-diagnostics"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + key, str(tmp_path / key)]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--sensor-family", "jilin1", "--quality-root", str(tmp_path / "qa")]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--alignment-audit-root", str(tmp_path / "audit")]
    source = tmp_path / "source-root"
    source.mkdir(exist_ok=True)
    with (source / ".alignment-diagnostics.jilin1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another V5"):
            v5_cli.main(args)
    assert v5_cli.main(args) == 0 and len(calls) == 1


def test_diagnostic_pipeline_freezes_inputs_and_replays_original_audit(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    import zarr
    from test_v5_clear_intraband import gaofen_qa

    from xuannv_embedding.data_process import v5_alignment_diagnostics as module
    from xuannv_embedding.data_process.v5_clear_audit import run_clear_audit
    from xuannv_embedding.data_process.v5_clear_training import calibrate_training
    from xuannv_embedding.data_process.v5_quality import quality_masks
    from xuannv_embedding.data_process.v5_sources import sha256

    data, qa, _ = gaofen_qa(tmp_path, count=9)
    table = pd.read_parquet(qa / "observation_quality.parquet")
    table.loc[4:7, "year"] = 2021
    classes = zarr.open_group(str(qa / "classes.zarr"), mode="a")
    labels = np.zeros((9, 160, 160), "u1")
    labels[8] = 1
    labels[8, 40:120, 40:120] = 0
    classes["classes"][:] = labels
    masks = zarr.open_group(str(qa / "valid_masks.zarr"), mode="a")
    for i in range(9):
        q = quality_masks(labels[i], np.ones((160, 160), bool), gsd=8)
        for name, key in [("ms_valid_packed", "valid"), ("before_buffer_packed", "before_buffer")]:
            masks[name][i] = np.packbits(q[key], axis=-1, bitorder="little")
        table.loc[i, "clear_fraction"] = q["valid"].mean()
        table.loc[i, "ms_valid_pixels"] = q["valid"].sum()
    table.to_parquet(qa / "observation_quality.parquet", index=False)
    target = data / "quality/cloud/gaofen"
    target.mkdir(parents=True)
    shutil.copy(qa / "observation_quality.parquet", target / "observation_quality.parquet")
    report = tmp_path / "report"
    cal = calibrate_training(data, report, "gaofen", qa)
    audit = run_clear_audit(data, report, "gaofen", qa, Path(cal["output"]).parent, workers=2)
    original = module.trace_windows
    seen = []

    def traced(*a, **k):
        frozen = list((report / "alignment_diagnostics/gaofen").glob("*/inputs.parquet"))
        assert len(frozen) == 1
        seen.append(sha256(frozen[0]))
        return original(*a, **k)

    monkeypatch.setattr(module, "trace_windows", traced)
    first = module.diagnose_alignment(data, report, "gaofen", qa, Path(audit["output"]))
    assert first["selected"] == 1 and first["pairs_with_no_usable_corner_but_usable_interior"] == 3
    assert len(seen) == 3 and len(set(seen)) == 1
    output = Path(first["output"])
    files = list(output.iterdir())
    before = [(sha256(p), p.stat().st_mtime_ns) for p in files]
    second = module.diagnose_alignment(data, report, "gaofen", qa, Path(audit["output"]))
    assert second["diagnostics_sha256"] == first["diagnostics_sha256"]
    assert before == [(sha256(p), p.stat().st_mtime_ns) for p in files]
    path = output / "diagnostics.json"
    changed = json.loads(path.read_text())
    changed["selected"] = 999
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="output seal changed"):
        module.diagnose_alignment(data, report, "gaofen", qa, Path(audit["output"]))

    changed_report = tmp_path / "changed_report"
    mutated = []

    def mutate_lock(*args, **kwargs):
        if not mutated:
            lock = next(
                (changed_report / "alignment_diagnostics/gaofen").glob("*/inputs.lock.json")
            )
            content = json.loads(lock.read_text())
            content["unexpected_mutation"] = True
            lock.write_text(json.dumps(content))
            mutated.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "trace_windows", mutate_lock)
    with pytest.raises(ValueError, match="inputs or algorithm changed"):
        module.diagnose_alignment(data, changed_report, "gaofen", qa, Path(audit["output"]))
