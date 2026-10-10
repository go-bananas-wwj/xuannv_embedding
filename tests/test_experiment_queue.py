import json
from pathlib import Path

import pytest

from xuannv_embedding.training import experiment_queue as queue


def test_device_process_parser_preserves_known_resident_and_blocks_unknown():
    info = """
| 0       0 | 4170061 |                    | 216 | 2687 |
| 1       0 | 1234567 | python             | 900 | 5678 |
| No running processes found in NPU 2 |
"""
    assert queue.available_devices(info, [0, 1, 2, 3], {"0": [2687]}) == [0, 2]
    assert queue.available_devices(info, [0, 1, 2], {}) == [2]


def test_assignment_releases_individual_cards_and_obeys_dependencies():
    jobs = [
        {"name": "base1", "device": 1, "state": "complete"},
        {"name": "base0", "device": 0, "state": "running"},
        {"name": "wait", "state": "pending", "depends_on": ["base0"]},
        {"name": "next", "state": "pending", "depends_on": ["base1"]},
        {"name": "last", "state": "pending", "depends_on": []},
    ]
    assert queue.assignments(jobs, [0, 1, 2]) == [("next", 1), ("last", 2)]
    jobs[0]["state"] = "finishing"
    assert queue.assignments(jobs, [0, 1, 2]) == [("last", 2)]


def test_observe_requires_completion_checkpoint_and_exit(tmp_path, monkeypatch):
    job = {"pid": 10, "output": str(tmp_path), "epochs": 800, "log": str(tmp_path / "run.log")}
    (tmp_path / "run.log").write_text("progress")
    (tmp_path / "status.json").write_text('{"state":"complete","epoch":800}')
    monkeypatch.setattr(queue, "_alive", lambda pid: False)
    assert queue.observe(job, now=0) == "failed"
    for name in ["best.pt", "latest.pt"]:
        (tmp_path / name).write_bytes(b"checkpoint")
    assert queue.observe(job, now=0) == "complete"
    monkeypatch.setattr(queue, "_alive", lambda pid: True)
    assert queue.observe(job, now=0) == "finishing"


def test_stall_is_detected_even_when_process_is_alive(tmp_path, monkeypatch):
    import os

    log = tmp_path / "run.log"
    log.write_text("stale")
    os.utime(log, (1, 1))
    monkeypatch.setattr(queue, "_alive", lambda pid: True)
    job = {"pid": 10, "output": str(tmp_path), "epochs": 800, "log": str(log)}
    assert queue.observe(job, now=1000) == "stalled"
    log.write_text("new progress")
    assert queue.observe(job, now=1000) == "running"


def test_export_completion_requires_manifest_and_all_patches(tmp_path, monkeypatch):
    job = {"action": "export", "pid": 10, "output": str(tmp_path), "epochs": 0}
    monkeypatch.setattr(queue, "_alive", lambda pid: False)
    (tmp_path / "status.json").write_text('{"state":"complete","patches":3,"total":3}')
    assert queue.observe(job, now=0) == "failed"
    (tmp_path / "manifest.json").write_text("{}")
    assert queue.observe(job, now=0) == "complete"
    (tmp_path / "status.json").write_text('{"state":"complete","patches":2,"total":3}')
    assert queue.observe(job, now=0) == "failed"


def test_plan_rejects_duplicate_destinations_and_changed_plan(tmp_path):
    plan = {
        "devices": [0, 1],
        "existing": [],
        "jobs": [
            {"name": "one", "output": "/same", "depends_on": []},
            {"name": "two", "output": "/same", "depends_on": []},
        ],
    }
    with pytest.raises(ValueError):
        queue.validate_plan(plan)
    plan["jobs"][1]["output"] = "/different"
    queue.validate_plan(plan)
    statefile = tmp_path / "state.json"
    statefile.write_text(json.dumps({"plan_sha256": "original"}))
    with pytest.raises(ValueError):
        queue.load_state(statefile, plan, "changed")


def test_uncertain_launch_is_not_requeued_after_controller_restart(tmp_path):
    statefile = tmp_path / "state.json"
    state = {"plan_sha256": "fixed", "jobs": [{"name": "one", "state": "launching"}]}
    statefile.write_text(json.dumps(state))
    assert queue.load_state(statefile, {}, "fixed") == state
    assert queue.assignments(state["jobs"], [1]) == []


def test_deferred_checkpoint_must_belong_to_its_declared_producer():
    plan = {
        "devices": [0],
        "existing": [],
        "jobs": [
            {"name": "train", "output": "/train"},
            {
                "name": "export",
                "output": "/export",
                "depends_on": ["train"],
                "checkpoint_from": "train",
                "checkpoint": "/different/best.pt",
            },
        ],
    }
    with pytest.raises(ValueError, match="producer"):
        queue.validate_plan(plan)


def test_deferred_adaptation_parent_must_match_dependency():
    plan = {
        "devices": [0],
        "existing": [],
        "jobs": [
            {"name": "first", "output": "/first"},
            {
                "name": "second",
                "output": "/second",
                "depends_on": ["first"],
                "initialize_from": "first",
                "adaptation": {"initialize": "/wrong/best.pt"},
            },
        ],
    }
    with pytest.raises(ValueError, match="producer"):
        queue.validate_plan(plan)
    plan["jobs"][1]["adaptation"]["initialize"] = "/first/best.pt"
    queue.validate_plan(plan)


def test_launch_uses_spawn_and_isolated_runtime(tmp_path, monkeypatch):
    from types import SimpleNamespace

    code = tmp_path / "code"
    code.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cache.json").write_text("{}")
    job = {
        "name": "one",
        "code": str(code),
        "git_sha": "fixed",
        "config": str(config),
        "config_sha256": queue._sha(config),
        "cache": str(cache),
        "cache_sha256": queue._sha(cache / "cache.json"),
        "runtime": str(tmp_path / "runtime"),
        "log": str(tmp_path / "run.log"),
        "output": str(tmp_path / "output"),
        "epochs": 800,
    }
    monkeypatch.setattr(
        queue.subprocess, "check_output", lambda cmd, **kw: "fixed" if "rev-parse" in cmd else ""
    )
    seen = []

    def popen(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return SimpleNamespace(pid=999)

    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    queue.launch(job, 1)
    assert job["pid"] == 999 and job["state"] == "running"
    assert "set_start_method('spawn')" in seen[0][0][3]
    assert seen[0][1]["cwd"] == Path(job["runtime"])
    assert seen[0][1]["env"]["PYTHONPATH"].startswith(str(code / "src"))


def test_controller_persists_completed_jobs_and_never_relaunches_them(tmp_path, monkeypatch):
    jobs = [
        {
            "name": f"job{i}",
            "output": str(tmp_path / f"job{i}"),
            "epochs": 800,
            "log": str(tmp_path / f"job{i}.log"),
            "depends_on": [],
        }
        for i in range(2)
    ]
    plan = {"devices": [1], "existing": [], "jobs": jobs}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    launches = []

    def launch(job, device):
        # Launch intent must be on disk before a process can be started.
        disk = json.loads((tmp_path / "queue_state.json").read_text())
        assert next(j for j in disk["jobs"] if j["name"] == job["name"])["state"] == "launching"
        launches.append(job["name"])
        job.update(pid=1, state="running", device=device)

    monkeypatch.setattr(queue, "launch", launch)
    monkeypatch.setattr(queue, "observe", lambda job, now: "complete")
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        queue.subprocess, "check_output", lambda *a, **kw: "No running processes found in NPU 1 "
    )
    queue.main(plan_path)
    assert launches == ["job0", "job1"]
    assert json.loads((tmp_path / "queue_status.json").read_text())["state"] == "finished"
    queue.main(plan_path)
    assert launches == ["job0", "job1"]
