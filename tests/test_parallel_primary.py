import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from test_paired_multitask import fixture_spec

from xuannv_embedding.downstream import paired_multitask as workflow
from xuannv_embedding.export.context import sha
from xuannv_embedding.training import experiment


def test_parallel_calibration_and_score_exactly_match_serial_without_test_access(tmp_path):
    path, spec, test_paths = fixture_spec(tmp_path)
    outputs = []
    for workers in [1, 2]:
        spec["output"] = str(tmp_path / f"workers{workers}")
        path.write_text(json.dumps(spec))
        for p in test_paths:
            p.rename(p.with_suffix(".withheld"))
        identity = workflow.calibrate(path, workers=workers)
        assert identity["workers"] == workers and identity["test_scored"] is False
        for p in test_paths:
            p.with_suffix(".withheld").rename(p)
        root = Path(spec["output"])
        result = workflow.score(path, sha(root / "calibration/identity.json"), workers=workers)
        assert result["workers"] == workers and result["parameters_refitted"] is False
        outputs.append(root)
    for stage in ["calibration", "test"]:
        rows = [json.loads((p / stage / "results.json").read_text()) for p in outputs]
        assert rows[0] == rows[1]
        for model, conditions in rows[0].items():
            for condition in conditions:
                relative = Path(stage) / "predictions" / model / (condition["key"] + ".npz")
                with np.load(outputs[0] / relative) as a, np.load(outputs[1] / relative) as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("workers", [0, 3, True, 1.5])
def test_parallel_worker_bound_rejected_before_input_access(tmp_path, workers):
    missing = tmp_path / "absent.json"
    with pytest.raises(ValueError, match="workers"):
        workflow.calibrate(missing, workers=workers)
    with pytest.raises(ValueError, match="workers"):
        workflow.score(missing, "unused", workers=workers)


def test_parallel_failure_records_scoring_that_finished_in_another_worker(tmp_path, monkeypatch):
    path, spec, _ = fixture_spec(tmp_path)
    workflow.calibrate(path)
    root = Path(spec["output"])
    original = workflow._predict
    completed = Event()

    def fail_one(stage, name, *args):
        if name == "base":
            assert completed.wait(10)
            raise RuntimeError("injected worker failure")
        result = original(stage, name, *args)
        completed.set()
        return result

    monkeypatch.setattr(workflow, "_predict", fail_one)
    with pytest.raises(RuntimeError, match="injected worker"):
        workflow.score(path, sha(root / "calibration/identity.json"), workers=2)
    status = json.loads((root / "test/status.json").read_text())
    assert status["state"] == "failed" and status["test_scored"] is True
    assert status["test_data_access_started"] is True
    assert not (root / "test/identity.json").exists()


@pytest.mark.parametrize("action", ["calibrate-primary", "score-primary"])
def test_cli_forwards_explicit_parallel_workers(tmp_path, monkeypatch, action):
    calls = []
    name = "calibrate" if action.startswith("calibrate") else "score"
    monkeypatch.setattr(workflow, name, lambda *a, **kw: calls.append((a, kw)))
    spec = tmp_path / "spec.json"
    args = [action, "--spec", str(spec), "--workers", "2"]
    if name == "score":
        args += ["--calibration-identity-sha256", "frozen"]
    experiment.main(args)
    assert calls == [((spec,) if name == "calibrate" else (spec, "frozen"), {"workers": 2})]
