import json

import pytest
from test_review_readouts import cohort

from xuannv_embedding.downstream import baseline_review as baseline
from xuannv_embedding.downstream import review_readouts as review
from xuannv_embedding.downstream import strong_classifiers
from xuannv_embedding.export.context import sha


def test_rf_shared_domain_frozen_query_and_completed_resume(tmp_path, monkeypatch):
    path, cohort_spec, _ = cohort(tmp_path)
    shared = tmp_path / "shared"
    review.prepare_shared(path, shared, "calibration")
    spec = {
        "protocol": baseline.PROTOCOL,
        "cohort": {"path": str(path), "sha256": sha(path)},
        "shared": str(shared),
        "model": "candidate",
        "head": "rf",
        "device": "cpu",
        "budgets": [1],
        "support_seeds": cohort_spec["support_seeds"][:1],
        "output": str(tmp_path / "job"),
    }
    job = tmp_path / "baseline-spec.json"
    job.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="calibration"):
        baseline.run(job, "test")
    baseline.run(job, "calibration")
    before = sha(tmp_path / "job/calibration/identity.json")
    monkeypatch.setattr(
        strong_classifiers, "fit_classifier", lambda *a, **k: pytest.fail("refitting")
    )
    baseline.run(job, "calibration")
    assert sha(tmp_path / "job/calibration/identity.json") == before
    review.prepare_shared(path, shared, "test")
    baseline.run(job, "test")
    rows = json.loads((tmp_path / "job/test/results.json").read_text())
    assert len(rows) == 2
    identity = json.loads((tmp_path / "job/test/identity.json").read_text())
    assert identity["parameters_refitted"] is False
    spec["budgets"] = [99]
    job.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        baseline.run(job, "calibration")
