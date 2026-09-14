"""Advance a complete calibration matrix to independent full baseline runs."""

from __future__ import annotations

import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from xuannv_embedding.training.experiment import _json, _sha


def select_learning_rate(records: list[dict]) -> tuple[float, dict[float, float]]:
    expected = {(lr, seed) for lr in (1e-4, 3e-4) for seed in (41, 42, 43)}
    if len(records) != len(expected) or {(r["lr"], r["seed"]) for r in records} != expected:
        raise ValueError("selection requires both learning rates and all three paired seeds")
    if not all(math.isfinite(r["score"]) for r in records):
        raise ValueError("all calibration scores must be finite")
    means = {lr: sum(r["score"] for r in records if r["lr"] == lr) / 3 for lr in (1e-4, 3e-4)}
    return min(means, key=lambda lr: (means[lr], lr)), means


def _alive(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
        return status != "Z"
    except FileNotFoundError:
        return False


def follow(root: Path) -> None:
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "formal_registry.json").exists():
            raise FileExistsError("formal runs already registered; refusing duplicate launch")
        registry = json.loads((root / "experiment_registry.json").read_text())
        _json(root / "controller_status.json", {"state": "waiting_for_calibration"})
        while True:
            scores = []
            for job in registry["runs"]:
                path = Path(job["output"]) / "status.json"
                status = json.loads(path.read_text()) if path.exists() else {}
                alive = _alive(job["pid"])
                if status.get("state") == "complete" and status["epoch"] == job["epochs"]:
                    if not alive:
                        scores.append(
                            {
                                "lr": job["lr"],
                                "seed": job["seed"],
                                "score": status["best_validation"],
                            }
                        )
                elif not alive:
                    raise RuntimeError(f"calibration failed: {job['name']}; inspect its log")
            if len(scores) == 6:
                break
            time.sleep(15)
        selected, means = select_learning_rate(scores)
        _json(
            root / "selection.json",
            {
                "selected_lr": selected,
                "mean_best_validation": means,
                "records": scores,
                "rule": "paired_three_seed_mean",
            },
        )
        code = Path(registry["code_snapshot"])
        actual_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=code, text=True
        ).strip()
        if actual_sha != registry["git_sha"]:
            raise ValueError("training code snapshot changed")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=code, text=True).strip():
            raise ValueError("training code snapshot is dirty")
        if _sha(root / "cache/cache.json") != registry["cache_sha256"]:
            raise ValueError("cache registry changed")
        devices = [1, 2, 3]
        _json(root / "controller_status.json", {"state": "waiting_for_devices", "devices": devices})
        while True:
            output = subprocess.check_output(["npu-smi", "info"], text=True)
            if all(f"No running processes found in NPU {i} " in output for i in devices):
                break
            time.sleep(15)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(code / "src") + os.pathsep + env.get("PYTHONPATH", "")
        for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
            env.pop(key, None)
        formal = {"state": "launching", "selected_lr": selected, "git_sha": actual_sha, "runs": []}
        _json(root / "formal_registry.json", formal)
        for device, seed in zip(devices, (41, 42, 43)):
            pilot = next(j for j in registry["runs"] if j["seed"] == seed and j["lr"] == selected)
            config = Path(pilot["config"])
            if _sha(config) != pilot["config_sha256"]:
                raise ValueError("calibration configuration changed")
            name = f"B0_full_seed{seed}"
            destination = root / "runs" / name
            logpath = root / f"{name}.log"
            if destination.exists() or logpath.exists():
                raise FileExistsError(name)
            # CANN writes compiler reports to cwd; keep them out of the code snapshot
            # and prevent independent runs from overwriting one another's reports.
            runtime = root / "runtime" / name
            runtime.mkdir(parents=True, exist_ok=False)
            command = [
                sys.executable,
                "-u",
                "-c",
                "from xuannv_embedding.cli import main; raise SystemExit(main())",
                "experiment",
                "run",
                "--config",
                str(config),
                "--cache",
                str(root / "cache"),
                "--output",
                str(destination),
                "--device",
                f"npu:{device}",
                "--epochs",
                "800",
            ]
            with logpath.open("x") as log:
                process = subprocess.Popen(
                    command,
                    cwd=runtime,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            formal["runs"].append(
                {
                    "name": name,
                    "pid": process.pid,
                    "device": device,
                    "seed": seed,
                    "lr": selected,
                    "output": str(destination),
                    "log": str(logpath),
                    "runtime": str(runtime),
                    "command": command,
                    "initialization": "scratch",
                    "epochs": 800,
                }
            )
            _json(root / "formal_registry.json", formal)
        formal["state"] = "running"
        _json(root / "formal_registry.json", formal)
        _json(root / "controller_status.json", {"state": "formal_baselines_running"})
        print(json.dumps({"formal_baselines_started": formal["runs"]}), flush=True)
        while True:
            complete = []
            for job in formal["runs"]:
                path = Path(job["output"]) / "status.json"
                status = json.loads(path.read_text()) if path.exists() else {}
                if status.get("state") == "complete" and status.get("epoch") == job["epochs"]:
                    complete.append(job["name"])
                elif not _alive(job["pid"]):
                    raise RuntimeError(f"formal baseline failed: {job['name']}")
            if len(complete) == 3:
                _json(
                    root / "controller_status.json",
                    {
                        "state": "baselines_complete",
                        "completed": complete,
                        "next": "highres adapter implementation and evaluation",
                    },
                )
                return
            time.sleep(30)


def main(root: Path) -> None:
    try:
        follow(root)
    except BaseException as exc:
        # A duplicate controller must not overwrite the active controller's status.
        if not isinstance(exc, (BlockingIOError, FileExistsError)):
            _json(root / "controller_status.json", {"state": "failed", "error": str(exc)})
        raise
