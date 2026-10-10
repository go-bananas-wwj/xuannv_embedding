import json

import torch
from test_review_readouts import cohort, job

from xuannv_embedding.downstream import review_readouts as review
from xuannv_embedding.downstream import review_trajectory as convergence
from xuannv_embedding.export.context import sha


def test_convergence_worker_freezes_selection_before_query_and_resumes_completed_work(tmp_path):
    torch.set_num_threads(1)
    path, spec, test_paths = cohort(tmp_path)
    shared = tmp_path / "shared"
    review.prepare_shared(path, shared, "calibration")
    baseline = job(tmp_path, path, shared)
    d = json.loads(baseline.read_text())
    d["head"] = "conv3x3"
    d["trajectory_checkpoints"] = [100, 300, 1000]
    baseline.write_text(json.dumps(d))
    review.calibrate(baseline)
    for p in test_paths:
        p.rename(p.with_suffix(".withheld"))
    key = next(
        iter(json.loads((tmp_path / "job/calibration/identity.json").read_text())["records"])
    )
    contract = {
        "protocol": "review-trajectory-job-v1",
        "base_job": {"path": str(baseline), "sha256": sha(baseline)},
        "condition_keys": [key],
        "checkpoints": [100, 300, 1000],
        "device": "cpu",
        "output": str(tmp_path / "long"),
    }
    task = tmp_path / "long.json"
    task.write_text(json.dumps(contract))
    convergence.run(task, "calibration")
    identity = sha(tmp_path / "long/calibration/identity.json")
    convergence.run(task, "calibration")
    assert sha(tmp_path / "long/calibration/identity.json") == identity
    for p in test_paths:
        p.with_suffix(".withheld").rename(p)
    review.prepare_shared(path, shared, "test")
    convergence.run(task, "test")
    record = json.loads((tmp_path / "long/test/records" / (key + ".json")).read_text())
    assert record["selected_steps"] in [100, 300, 1000]
    chosen_ap = record["curve"][str(record["selected_steps"])]["validation_ap"]
    assert chosen_ap == max(r["validation_ap"] for r in record["curve"].values())
    assert (
        json.loads((tmp_path / "long/test/identity.json").read_text())["parameters_refitted"]
        is False
    )
