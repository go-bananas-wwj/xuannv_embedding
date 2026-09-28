import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_paired_multitask import fixture_spec, write_json

from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.downstream import review_readouts as review
from xuannv_embedding.downstream import strong_multitask as strong
from xuannv_embedding.export.context import sha


def cohort(tmp_path):
    path, spec, test_paths = fixture_spec(tmp_path)
    spec["task_schema"] = {
        "C": {"osm": ["osm_building"], "worldcover": ["worldcover_tree"]},
        "R": {"worldcover": ["worldcover_tree"]},
        "Q": {"osm": ["osm_building"]},
    }
    for label in spec["labels"].values():
        p = Path(label["path"])
        with np.load(p) as z:
            fields = {k: z[k] for k in ["indices", "cache_sha256", "osm_building"]}
            fields["worldcover_tree"] = (z["esri"] == 1).astype(np.int8)
        np.savez(p, **fields)
        label["sha256"] = sha(p)
    spec["lock"] = write_json(
        tmp_path / "lock.json",
        {"state": "locked", "contract_sha256": primary.contract_sha256(spec)},
    )
    path.write_text(json.dumps(spec))
    return path, spec, test_paths


def job(tmp, path, shared, model="candidate", reuse=None):
    spec = {
        "protocol": "review-neural-job-v1",
        "cohort": {"path": str(path), "sha256": sha(path)},
        "shared": str(shared),
        "model": model,
        "head": "mlp",
        "device": "cpu",
        "budgets": [1],
        "reuse": reuse or [],
        "output": str(tmp / "job"),
    }
    p = tmp / "job.json"
    p.write_text(json.dumps(spec))
    return p


def test_review_single_model_jobs_preserve_global_domain_and_freeze_before_query(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    path, spec, test_paths = cohort(tmp_path)
    for p in test_paths:
        p.rename(p.with_suffix(".withheld"))
    shared = tmp_path / "shared"
    review.prepare_shared(path, shared, "calibration")
    job_path = job(tmp_path, path, shared)
    review.calibrate(job_path)
    for p in test_paths:
        p.with_suffix(".withheld").rename(p)
    review.prepare_shared(path, shared, "test")
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("query refitting"))
    review.score(job_path)
    root = tmp_path / "job"
    identity = json.loads((root / "test/identity.json").read_text())
    rows = json.loads((root / "test/results.json").read_text())
    assert identity["parameters_refitted"] is False
    assert len(rows) == 2
    assert all(r["head"] == "mlp" and r["budget"] == 1 for r in rows)
    assert (shared / "calibration/identity.json").exists()
    # The paired candidate's invalid corner remains excluded even for a different model.
    assert not np.load(shared / "test/common_valid.npy")[:, 0, 0].any()


def test_review_imports_matching_completed_readouts_without_refitting(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    path, spec, _ = cohort(tmp_path)
    archive = {
        "protocol": strong.PROTOCOL,
        "primary_spec": {"path": str(path), "sha256": sha(path)},
        "heads": ["mlp"],
        "neural_device": "cpu",
        "output": str(tmp_path / "archived"),
    }
    archive["lock"] = write_json(
        tmp_path / "strong-lock.json",
        {"state": "locked", "contract_sha256": strong.contract_sha256(archive)},
    )
    old = tmp_path / "strong.json"
    old.write_text(json.dumps(archive))
    strong.calibrate(old)
    strong.score(old, sha(tmp_path / "archived/calibration/identity.json"))
    shared = tmp_path / "shared"
    review.prepare_shared(path, shared, "calibration")
    review.prepare_shared(path, shared, "test")
    job_path = job(
        tmp_path,
        path,
        shared,
        reuse=[{"spec": {"path": str(old), "sha256": sha(old)}, "model": "candidate"}],
    )
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("archive refitting"))
    monkeypatch.setattr(strong, "_load", lambda *a, **k: pytest.fail("archive device loading"))
    review.calibrate(job_path)
    review.score(job_path)
    for phase in ["calibration", "test"]:
        d = json.loads((tmp_path / "job" / phase / "identity.json").read_text())
        assert d["imported_conditions"] == 2 and d["computed_conditions"] == 0
    reused = json.loads((tmp_path / "job/test/results.json").read_text())
    original = json.loads((tmp_path / "archived/test/results.json").read_text())["candidate"]
    assert reused == original


def test_review_rejects_modified_shared_arrays_before_fitting(tmp_path):
    path, _, _ = cohort(tmp_path)
    shared = tmp_path / "shared"
    review.prepare_shared(path, shared, "calibration")
    p = shared / "calibration/common_valid.npy"
    a = np.load(p)
    a[0, 1, 1] = ~a[0, 1, 1]
    np.save(p, a)
    with pytest.raises(ValueError, match="shared"):
        review.calibrate(job(tmp_path, path, shared))


def test_completed_review_calibration_can_resume_without_retraining(tmp_path, monkeypatch):
    path, _, _ = cohort(tmp_path)
    shared = tmp_path / "shared"
    review.prepare_shared(path, shared, "calibration")
    p = job(tmp_path, path, shared)
    review.calibrate(p)
    monkeypatch.setattr(
        torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("completed refitting")
    )
    before = sha(tmp_path / "job/calibration/results.json")
    identity = sha(tmp_path / "job/calibration/identity.json")
    review.calibrate(p)
    assert sha(tmp_path / "job/calibration/results.json") == before
    assert sha(tmp_path / "job/calibration/identity.json") == identity


def test_review_rejects_corrupted_completed_head_payload(tmp_path):
    path, _, _ = cohort(tmp_path)
    shared = tmp_path / "shared"
    review.prepare_shared(path, shared, "calibration")
    p = job(tmp_path, path, shared)
    review.calibrate(p)
    payload = next((tmp_path / "job/calibration/readouts").rglob("parameters.npz"))
    payload.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="payload"):
        review.calibrate(p)
