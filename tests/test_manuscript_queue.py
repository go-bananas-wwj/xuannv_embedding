import pytest

from xuannv_embedding.training.manuscript_queue import validate_jobs, worker_environment


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
