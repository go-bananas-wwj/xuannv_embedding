"""Durable per-device queue for registered, independent baseline experiments."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from xuannv_embedding.training.experiment import _json, _sha
from xuannv_embedding.training.experiment_schedule import _alive


def available_devices(info: str, devices: list[int], residents: dict) -> list[int]:
    processes: dict[int, set[int]] = {}
    for line in info.splitlines():
        empty = re.search(r"No running processes found in NPU (\d+)\b", line)
        if empty:
            processes.setdefault(int(empty[1]), set())
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) == 5 and re.fullmatch(r"\d+\s+\d+", parts[0]):
            if parts[1].isdigit() and parts[4].isdigit():
                processes.setdefault(int(parts[0].split()[0]), set()).add(int(parts[4]))
    return [d for d in devices if d in processes and processes[d] <= set(residents.get(str(d), []))]


def assignments(jobs: list[dict], devices: list[int]) -> list[tuple[str, int]]:
    occupied = {
        j.get("device")
        for j in jobs
        if j["state"] in {"running", "finishing", "stalled", "launching"}
    }
    completed = {j["name"] for j in jobs if j["state"] == "complete"}
    ready = [
        j for j in jobs if j["state"] == "pending" and set(j.get("depends_on", [])) <= completed
    ]
    free = [d for d in devices if d not in occupied]
    return [(j["name"], d) for j, d in zip(ready, free)]


def observe(job: dict, *, now: float) -> str:
    destination = Path(job["output"])
    path = destination / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    alive = _alive(job["pid"])
    if job.get("action") == "export" and status.get("state") == "complete":
        manifest = destination / "manifest.json"
        if not manifest.is_file() or status.get("patches", 0) != status.get("total", -1):
            return "failed"
        return "finishing" if alive else "complete"
    if status.get("state") == "complete" and status.get("epoch") == job["epochs"]:
        if not all(
            (destination / n).is_file() and (destination / n).stat().st_size > 0
            for n in ["best.pt", "latest.pt"]
        ):
            return "failed"
        return "finishing" if alive else "complete"
    if not alive:
        return "failed"
    paths = [Path(job["log"]), path, destination / "metrics.jsonl"]
    updated = max((p.stat().st_mtime for p in paths if p.exists()), default=0)
    return "stalled" if now - updated > 600 else "running"


def validate_plan(plan: dict) -> None:
    devices = plan["devices"]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("queue devices must be unique and nonempty")
    if any(type(d) is not int or d not in range(8) for d in devices):
        raise ValueError("invalid queue device")
    jobs = plan["existing"] + plan["jobs"]
    names, outputs = [j["name"] for j in jobs], [j["output"] for j in jobs]
    if len(set(names)) != len(names) or len(set(outputs)) != len(outputs):
        raise ValueError("queue names and output directories must be unique")
    resolved = {j["name"] for j in plan["existing"]}
    unresolved = list(plan["jobs"])
    while unresolved:
        ready = [j for j in unresolved if set(j.get("depends_on", [])) <= resolved]
        if not ready:
            raise ValueError("queue dependencies are missing or cyclic")
        resolved.update(j["name"] for j in ready)
        unresolved = [j for j in unresolved if j not in ready]


def load_state(path: Path, plan: dict, digest: str) -> dict:
    if path.exists():
        state = json.loads(path.read_text())
        if state["plan_sha256"] != digest:
            raise ValueError("queue plan changed; refusing to reuse incompatible state")
        return state
    return {
        "plan_sha256": digest,
        "jobs": [
            {**copy.deepcopy(j), "state": "running", "external": True} for j in plan["existing"]
        ]
        + [{**copy.deepcopy(j), "state": "pending"} for j in plan["jobs"]],
    }


def launch(job: dict, device: int) -> None:
    code = Path(job["code"])
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
        != job["git_sha"]
    ):
        raise ValueError("training snapshot changed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=code, text=True).strip():
        raise ValueError("training snapshot is dirty")
    if _sha(Path(job["config"])) != job["config_sha256"]:
        raise ValueError("registered configuration changed")
    if _sha(Path(job["cache"]) / "cache.json") != job["cache_sha256"]:
        raise ValueError("registered cache changed")
    if Path(job["output"]).exists() or Path(job["log"]).exists():
        raise FileExistsError("queue never overwrites an existing run")
    runtime = Path(job["runtime"])
    runtime.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        env.pop(key, None)
    command = [
        sys.executable,
        "-u",
        "-c",
        "import multiprocessing as mp; mp.set_start_method('spawn'); "
        "from xuannv_embedding.cli import main; raise SystemExit(main())",
        "experiment",
        job.get("action", "run"),
        "--config",
        job["config"],
        "--cache",
        job["cache"],
        "--output",
        job["output"],
        "--device",
        f"npu:{device}",
    ]
    if job.get("action") == "export":
        if _sha(Path(job["checkpoint"])) != job["checkpoint_sha256"]:
            raise ValueError("registered export checkpoint changed")
        command += ["--checkpoint", job["checkpoint"]]
    else:
        command += ["--epochs", str(job["epochs"])]
        if job.get("pilot"):
            command += ["--pilot"]
        if job.get("adaptation"):
            a = job["adaptation"]
            for name in ("initialize", "base_config"):
                if _sha(Path(a[name])) != a[name + "_sha256"]:
                    raise ValueError(f"registered adaptation {name} changed")
            command += [
                "--initialize",
                a["initialize"],
                "--base-config",
                a["base_config"],
                "--highres-encoding",
                a["highres_encoding"],
            ]
            if a["freeze_base"]:
                command += ["--freeze-base"]
    with Path(job["log"]).open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=runtime,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    job.update(
        pid=process.pid,
        device=device,
        command=command,
        state="running",
        started_utc=time.time(),
        worker_start_method="spawn",
    )


def main(plan_path: Path) -> None:
    root = plan_path.parent
    state_path, status_path = root / "queue_state.json", root / "queue_status.json"
    with (root / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            digest = _sha(plan_path)
            plan = json.loads(plan_path.read_text())
            validate_plan(plan)
            state = load_state(state_path, plan, digest)
            while True:
                if _sha(plan_path) != digest:
                    raise ValueError("queue plan changed during execution")
                for job in state["jobs"]:
                    if job["state"] in {"running", "finishing", "stalled"}:
                        before = job["state"]
                        job["state"] = observe(job, now=time.time())
                        if before != job["state"]:
                            print(
                                json.dumps({"job": job["name"], "state": job["state"]}), flush=True
                            )
                _json(state_path, state)
                info = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
                free = available_devices(info, plan["devices"], plan.get("residents", {}))
                for name, device in assignments(state["jobs"], free):
                    job = next(j for j in state["jobs"] if j["name"] == name)
                    job.update(state="launching", device=device)
                    # Persist intent before spawning. An uncertain launch on restart is held
                    # for inspection rather than risking a second copy of the experiment.
                    _json(state_path, state)
                    try:
                        launch(job, device)
                    except Exception as exc:
                        job.update(state="failed", error=str(exc))
                    _json(state_path, state)
                    print(
                        json.dumps({"job": name, "state": job["state"], "device": device}),
                        flush=True,
                    )
                counts = {
                    s: sum(j["state"] == s for j in state["jobs"])
                    for s in {j["state"] for j in state["jobs"]}
                }
                issues = [
                    {"name": j["name"], "state": j["state"], "error": j.get("error")}
                    for j in state["jobs"]
                    if j["state"] in {"failed", "stalled", "launching"}
                ]
                done = all(j["state"] in {"complete", "failed"} for j in state["jobs"])
                _json(
                    status_path,
                    {
                        "state": "finished" if done else "monitoring",
                        "updated_utc": time.time(),
                        "counts": counts,
                        "issues": issues,
                    },
                )
                if done:
                    return
                time.sleep(15)
        except BaseException as exc:
            _json(status_path, {"state": "controller_failed", "error": str(exc)})
            raise
