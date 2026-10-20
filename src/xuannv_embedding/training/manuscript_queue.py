"""Pinned-source DAG queue with bounded recovery and a quiet local watchdog."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.experiment_queue import available_devices
from xuannv_embedding.training.frozen_fusion import verified


def validate_jobs(jobs, devices):
    if (
        not devices
        or len(set(devices)) != len(devices)
        or any(d not in range(1, 8) for d in devices)
    ):
        raise ValueError("Only unique explicitly idle-authorized devices1..7 may be leased")
    names = [j["id"] for j in jobs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate queue producers")
    done, pending = set(), list(jobs)
    while pending:
        ready = [j for j in pending if set(j["depends_on"]) <= done]
        if not ready:
            raise ValueError("Queue dependency missing or cyclic")
        done.update(j["id"] for j in ready)
        pending = [j for j in pending if j not in ready]


def _terminate(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def worker_environment(source, git_sha, cards):
    env = os.environ.copy()
    paths = [str(Path(source) / "src"), *env.get("PYTHONPATH", "").split(os.pathsep)]
    env.update(
        PYTHONPATH=os.pathsep.join(dict.fromkeys(p for p in paths if p)),
        XUANNV_GIT_SHA=git_sha,
        ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, cards)),
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        env.pop(key, None)
    return env


def run(path):
    plan = json.loads(path.read_text())
    if (
        set(plan)
        != {
            "schema",
            "source",
            "git_sha",
            "source_files",
            "output",
            "devices",
            "jobs",
            "poll_seconds",
            "timeout_seconds",
            "min_free_gib",
            "publication_repository",
        }
        or plan["schema"] != "manuscript-queue-v1"
    ):
        raise ValueError("Invalid pinned manuscript queue")
    validate_jobs(plan["jobs"], plan["devices"])
    if plan["poll_seconds"] != 60 or plan["min_free_gib"] < 100:
        raise ValueError("Watchdog cadence or disk floor changed")
    source = Path(plan["source"])
    files = json.loads(verified(plan["source_files"]).read_text())
    for name, digest in files.items():
        if sha(source / name) != digest:
            raise ValueError("Pinned source content changed")
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "queue-state.json"
        identity = sha(path)
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        if previous and previous.get("plan_sha256") != identity:
            raise ValueError("Queue plan changed")
        state = previous or {
            "plan_sha256": identity,
            "jobs": {j["id"]: {"state": "pending", "attempts": 0} for j in plan["jobs"]},
        }
        for row in state["jobs"].values():
            if row["state"] == "running":
                pid = row.get("pid")
                if pid and Path(f"/proc/{pid}").exists():
                    raise RuntimeError("Recorded child still exists; inspect before resuming queue")
                row["state"] = "pending"
        active = {}
        started = time.monotonic()

        def save(status="running"):
            state.update(state=status, updated_at=time.time(), scientific_goal_complete=False)
            dump(state_path, state)

        def publish():
            repo = Path(plan["publication_repository"])
            doc = repo / "docs/operations/manuscript-comparison-status.md"
            lines = [
                "# 三区原稿重训与融合对照状态",
                "",
                "自动阶段记录，不代表精度验收或论文结论。",
                "",
                "| 实验 | 状态 |",
                "| --- | --- |",
            ]
            lines += [f"| {k} | {v['state']} |" for k, v in state["jobs"].items()]
            doc.write_text("\n".join(lines) + "\n")
            subprocess.run(
                ["git", "add", "docs/operations/manuscript-comparison-status.md"],
                cwd=repo,
                check=True,
            )
            changed = subprocess.run(
                [
                    "git",
                    "diff",
                    "--cached",
                    "--quiet",
                    "--",
                    "docs/operations/manuscript-comparison-status.md",
                ],
                cwd=repo,
            ).returncode
            if changed:
                subprocess.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        "docs: record manuscript experiment milestone",
                        "--",
                        "docs/operations/manuscript-comparison-status.md",
                    ],
                    cwd=repo,
                    check=True,
                )
                subprocess.run(["git", "push", "origin", "HEAD"], cwd=repo, check=True, timeout=120)

        def safe_publish():
            try:
                publish()
            except (OSError, subprocess.SubprocessError) as error:
                dump(
                    root / "publication-error.json",
                    {"time": time.time(), "error": str(error), "training_interrupted": False},
                )

        try:
            while True:
                if time.monotonic() - started > plan["timeout_seconds"]:
                    raise RuntimeError("Bounded queue timeout")
                if (root / "STOP").exists() or shutil.disk_usage(root).free < plan[
                    "min_free_gib"
                ] * 2**30:
                    raise RuntimeError("STOP requested or disk reserve reached")
                for name, item in list(active.items()):
                    process, job, card, launched, stream = item
                    code = process.poll()
                    if code is None and time.monotonic() - launched <= job["timeout_seconds"]:
                        continue
                    if code is None:
                        _terminate(process)
                        code = process.returncode
                    stream.close()
                    del active[name]
                    row = state["jobs"][name]
                    if code == 0:
                        receipt = Path(job["receipt"])
                        result = json.loads(receipt.read_text())
                        if result.get("state") not in {"complete", "validation_complete"}:
                            raise ValueError("Successful process lacks completion receipt")
                        row.update(
                            state="complete",
                            receipt={"path": str(receipt), "sha256": sha(receipt)},
                            completed_at=time.time(),
                        )
                        if job.get("publish", False):
                            save()
                            safe_publish()
                    else:
                        row.update(
                            state="pending" if row["attempts"] < 2 else "failed", returncode=code
                        )
                for job in plan["jobs"]:
                    row = state["jobs"][job["id"]]
                    if row["state"] == "pending" and any(
                        state["jobs"][d]["state"] in {"failed", "dependency_failed"}
                        for d in job["depends_on"]
                    ):
                        row["state"] = "dependency_failed"
                pending = [j for j in plan["jobs"] if state["jobs"][j["id"]]["state"] == "pending"]
                if not pending and not active:
                    failed = any(v["state"] != "complete" for v in state["jobs"].values())
                    save("failed" if failed else "complete")
                    safe_publish()
                    return 1 if failed else 0
                info = subprocess.check_output(["npu-smi", "info"], text=True, timeout=30)
                free = available_devices(info, plan["devices"], {})
                leased = [d for x in active.values() for d in x[2]]
                free = [d for d in free if d not in leased]
                for job in pending:
                    count = job.get("device_count", 1)
                    if len(free) < count:
                        continue
                    if any(state["jobs"][d]["state"] != "complete" for d in job["depends_on"]):
                        continue
                    cards = free[:count]
                    del free[:count]
                    env = worker_environment(source, plan["git_sha"], cards)
                    work = root / "runtime" / job["id"]
                    work.mkdir(parents=True, exist_ok=True)
                    log = root / "logs" / (job["id"] + ".log")
                    log.parent.mkdir(exist_ok=True)
                    stream = log.open("a")
                    args = [sys.executable, "-u", "-m"]
                    if count > 1:
                        args += [
                            "torch.distributed.run",
                            "--standalone",
                            "--nnodes=1",
                            f"--nproc-per-node={count}",
                            "--module",
                        ]
                    args += ["xuannv_embedding.cli", "experiment", *job["arguments"]]
                    p = subprocess.Popen(
                        args,
                        cwd=work,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    active[job["id"]] = (p, job, cards, time.monotonic(), stream)
                    row = state["jobs"][job["id"]]
                    row.update(
                        state="running",
                        attempts=row["attempts"] + 1,
                        pid=p.pid,
                        devices=cards,
                        log=str(log),
                        started_at=time.time(),
                    )
                save()
                time.sleep(plan["poll_seconds"])
        except BaseException as error:
            for process, _, _, _, stream in active.values():
                _terminate(process)
                stream.close()
            state["error"] = str(error)
            save("failed")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args.plan)
