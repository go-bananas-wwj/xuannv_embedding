import json

import pytest

from xuannv_embedding.training import manuscript_queue as queue
from xuannv_embedding.training.manuscript_queue import (
    recover_dependencies,
    validate_jobs,
    worker_environment,
)


def test_retrying_producer_reopens_blocked_descendants_without_retrying_failed_jobs():
    jobs = [
        {"id": "base", "depends_on": []},
        {"id": "score", "depends_on": ["base"]},
        {"id": "summary", "depends_on": ["score"]},
        {"id": "failed", "depends_on": []},
        {"id": "still_blocked", "depends_on": ["failed"]},
    ]
    states = {
        "base": {"state": "running"},
        "score": {"state": "dependency_failed"},
        "summary": {"state": "dependency_failed"},
        "failed": {"state": "failed", "attempts": 2},
        "still_blocked": {"state": "dependency_failed"},
    }
    recover_dependencies(jobs, states)
    assert states["score"]["state"] == "pending"
    assert states["summary"]["state"] == "pending"
    assert states["failed"] == {"state": "failed", "attempts": 2}
    assert states["still_blocked"]["state"] == "dependency_failed"


def test_continuation_waits_for_controller_lock_before_resuming(monkeypatch, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"output": str(tmp_path), "timeout_seconds": 300}))
    locks, sleeps = [], []

    def flock(fd, mode):
        locks.append(mode)
        if len(locks) == 1:
            raise BlockingIOError

    monkeypatch.setattr(queue.fcntl, "flock", flock)
    monkeypatch.setattr(queue.time, "sleep", sleeps.append)
    monkeypatch.setattr(queue, "run", lambda spec: 7 if spec == path else None)
    assert queue.after_active_queue(path) == 7
    assert sleeps == [60]
    assert locks[-1] == queue.fcntl.LOCK_UN


def test_continuation_timeout_does_not_restart_active_controller(monkeypatch, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"output": str(tmp_path), "timeout_seconds": 1}))
    ticks = iter([0, 2])
    monkeypatch.setattr(queue.time, "monotonic", lambda: next(ticks))

    def busy(fd, mode):
        raise BlockingIOError

    monkeypatch.setattr(queue.fcntl, "flock", busy)
    with pytest.raises(TimeoutError):
        queue.after_active_queue(path)


def test_evaluation_continuation_waits_for_both_external_queues(monkeypatch, tmp_path):
    plan = tmp_path / "evaluation.json"
    plan.write_text(json.dumps({"output": str(tmp_path), "timeout_seconds": 300}))
    dependencies = []
    for name in ["training", "readers"]:
        root = tmp_path / name
        root.mkdir()
        path = root / "plan.json"
        path.write_text(json.dumps({"output": str(root)}))
        dependencies.append(path)
    locks = []
    monkeypatch.setattr(queue.fcntl, "flock", lambda f, m: locks.append((f.name, m)))
    monkeypatch.setattr(queue, "run", lambda path: path)
    assert queue.after_active_queue(plan, dependencies) == plan
    assert [name for name, mode in locks if mode != queue.fcntl.LOCK_UN] == [
        str(p.parent / "queue.lock") for p in dependencies
    ]


def test_worker_preserves_cann_python_paths_after_pinned_source(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/opt/cann/python/site-packages:/opt/cann/opp")
    monkeypatch.setenv("RANK", "5")
    env = worker_environment(tmp_path, "fixed-worker-commit", [3, 4])
    assert env["PYTHONPATH"].split(":") == [
        str(tmp_path / "src"),
        "/opt/cann/python/site-packages",
        "/opt/cann/opp",
    ]
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "3,4"
    assert env["XUANNV_GIT_SHA"] == "fixed-worker-commit"
    assert "RANK" not in env


def test_queue_rejects_cycles_and_unrelated_device_zero():
    jobs = [{"id": "a", "depends_on": ["b"]}, {"id": "b", "depends_on": ["a"]}]
    with pytest.raises(ValueError):
        validate_jobs(jobs, [1, 2])
    with pytest.raises(ValueError):
        validate_jobs([], [0, 1])


def test_queue_accepts_paired_forks_and_unique_producers():
    jobs = [
        {"id": "base", "depends_on": []},
        {"id": "public", "depends_on": ["base"]},
        {"id": "local", "depends_on": ["base"]},
    ]
    validate_jobs(jobs, [1, 2, 3])
