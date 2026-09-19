"""Persistent CPU evaluation queue driven by immutable artifact dependencies."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from xuannv_embedding.training.experiment import _json, _sha


def ready(job):
    for dependency in job.get("requires", []):
        path = Path(dependency["path"])
        if not path.is_file():
            return False
        if dependency.get("state"):
            if json.loads(path.read_text()).get("state") != dependency["state"]:
                return False
    return True


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    root = plan_path.parent
    if (root / "state.json").exists():
        raise FileExistsError("evaluation queue exists; inspect before restarting")
    digest = _sha(plan_path)
    jobs = [{**j, "state": "pending"} for j in plan["jobs"]]
    if len({j["output"] for j in jobs}) != len(jobs):
        raise ValueError("evaluation outputs must be unique")
    for job in jobs:
        if job["arguments"][0] not in {"comparison", "probe", "test-readout"}:
            raise ValueError("unsupported CPU evaluation action")
        if Path(job["output"]).exists():
            raise FileExistsError("evaluation queue must not overwrite outputs")
    active = {}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with (root / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if _sha(plan_path) != digest:
                raise ValueError("CPU queue plan changed")
            for index, process in list(active.items()):
                code = process.poll()
                if code is not None:
                    job = jobs[index]
                    path = Path(job["output"]) / "status.json"
                    status = json.loads(path.read_text()) if path.exists() else {}
                    job.update(
                        state=(
                            "complete"
                            if code == 0 and status.get("state") == "complete"
                            else "failed"
                        ),
                        exit_code=code,
                    )
                    del active[index]
            for index, job in enumerate(jobs):
                if len(active) >= 3:
                    break
                if job["state"] != "pending" or not ready(job):
                    continue
                command = [
                    sys.executable,
                    "-u",
                    "-c",
                    "from xuannv_embedding.cli import main; raise SystemExit(main())",
                    "experiment",
                    *job["arguments"],
                ]
                log_path = root / (job["name"] + ".log")
                with log_path.open("x") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                active[index] = process
                job.update(state="running", pid=process.pid, log=str(log_path))
            counts = {s: sum(j["state"] == s for j in jobs) for s in {j["state"] for j in jobs}}
            _json(root / "state.json", {"plan_sha256": digest, "jobs": jobs})
            _json(
                root / "status.json",
                {
                    "counts": counts,
                    "updated_utc": time.time(),
                    "state": (
                        "complete" if all(j["state"] == "complete" for j in jobs) else "monitoring"
                    ),
                },
            )
            if all(j["state"] in {"complete", "failed"} for j in jobs):
                return
            time.sleep(15)
