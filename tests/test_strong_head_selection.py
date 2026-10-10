import json
from pathlib import Path

import numpy as np
import pytest
from test_paired_multitask import write_json
from test_strong_multitask import strong_spec

from xuannv_embedding.downstream import strong_multitask as workflow
from xuannv_embedding.downstream import strong_multitask_report as reporting
from xuannv_embedding.export.context import sha


def test_selected_heads_require_a_complete_registered_task_matrix():
    cohort = {"support_seeds": [7], "budgets": [1], "retrieval_budgets": [1]}
    conditions = reporting.conditions(cohort, heads=["mlp"])
    assert len(conditions) == 10 and {c["head"] for c in conditions} == {"mlp"}
    a = np.full((10, 1, 4), 0.2)
    b = a + 0.1
    result = reporting.compare(conditions, a, b, heads=["mlp"])
    assert set(result["heads"]) == {"mlp"}
    assert result["heads"]["mlp"]["candidate_minus_baseline"]["observed"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        reporting.compare(conditions, a, b)  # Must not infer a reduced matrix from missing rows.
    with pytest.raises(ValueError):
        reporting.compare(conditions[:-1], a[:-1], b[:-1], heads=["mlp"])


@pytest.mark.parametrize("heads", [[], ["invalid"], ["mlp", "mlp"], ["mlp", "rf"], "mlp"])
def test_invalid_head_selection_is_rejected_before_data_reads(tmp_path, heads, monkeypatch):
    path, spec, _, _ = strong_spec(tmp_path)
    spec["heads"] = heads
    spec["lock"] = write_json(
        tmp_path / "selected-lock.json",
        {"state": "locked", "contract_sha256": workflow.contract_sha256(spec)},
    )
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(
        workflow.primary, "_common", lambda *a, **k: pytest.fail("read before validation")
    )
    with pytest.raises(ValueError):
        workflow.calibrate(path)
    assert not Path(spec["output"]).exists()


def test_selected_neural_head_freezes_scores_and_reports_without_other_heads(tmp_path, monkeypatch):
    path, spec, primary, test_paths = strong_spec(tmp_path)
    spec["heads"] = ["mlp"]
    spec["lock"] = write_json(
        tmp_path / "selected-lock.json",
        {"state": "locked", "contract_sha256": workflow.contract_sha256(spec)},
    )
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(
        workflow.strong_classifiers,
        "fit_classifier",
        lambda *a, **k: pytest.fail("unselected head fitted"),
    )
    for p in test_paths:
        p.rename(p.with_suffix(".held"))
    calibrated = workflow.calibrate(path)
    assert calibrated["heads"] == ["mlp"] and len(calibrated["conditions"]) == 10
    for p in test_paths:
        p.with_suffix(".held").rename(p)
    root = Path(spec["output"])
    scored = workflow.score(path, sha(root / "calibration/identity.json"))
    assert scored["heads"] == ["mlp"] and scored["parameters_refitted"] is False

    def forbidden(*a, **k):
        raise AssertionError("report must use only archived numeric evidence")

    monkeypatch.setattr(workflow.primary, "_common", forbidden)
    monkeypatch.setattr(workflow.neural_readouts, "load_neural", forbidden)
    result = reporting.run(path, sha(root / "test/identity.json"), tmp_path / "report", repeats=4)
    assert result["heads"] == ["mlp"] and result["conditions"] == 10
    assert all(set(x["heads"]) == {"mlp"} for x in result["comparisons"])
