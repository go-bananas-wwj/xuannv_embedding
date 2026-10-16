"""Supervise owned regional runs without restarting healthy or completed training."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.regional_base import read_spec, require_training_approval
from xuannv_embedding.training.regional_resume import validate_resume_checkpoint


@dataclass(frozen=True)
class Process:
    pid: int
    group: int
    started: int
    arguments: tuple[str, ...]


def processes() -> dict[int, Process]:
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split(") ", 1)[1].split()
            if fields[0] == "Z":
                continue
            arguments = tuple(
                v.decode(errors="replace")
                for v in (entry / "cmdline").read_bytes().split(b"\0")
                if v
            )
            result[int(entry.name)] = Process(
                int(entry.name), int(fields[2]), int(fields[19]), arguments
            )
        except (OSError, IndexError, ValueError):
            continue
    return result


def matching_processes(spec: Path, *, evaluation=False) -> dict[int, Process]:
    expected = str(spec.resolve())
    result = {}
    for pid, info in processes().items():
        args = info.arguments
        try:
            if args[args.index("--spec") + 1] != expected:
                continue
            phase = args[args.index("--phase") + 1]
            if evaluation:
                matches = "regional-benchmark" in args and phase in {
                    "score",
                    "regression",
                    "export",
                }
            else:
                matches = "regional-base" in args and phase == "train"
            if matches:
                result[pid] = info
        except (ValueError, IndexError):
            continue
    return result


def stop_owned(spec: Path, *, timeout=30) -> None:
    matched = matching_processes(spec)
    groups = {info.group for info in matched.values()}
    if os.getpgrp() in groups:
        raise RuntimeError("Refusing to terminate a shared caller process group")
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while matching_processes(spec) and time.monotonic() < deadline:
        time.sleep(0.2)
    # Revalidate ownership before escalating. Never kill an unrelated reused PID.
    for group in {info.group for info in matching_processes(spec).values()}:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while matching_processes(spec) and time.monotonic() < deadline:
        time.sleep(0.2)
    if matching_processes(spec):
        raise RuntimeError("Owned training workers have not stopped")


def decision(*, alive, complete, stale, attempts, max_restarts, disk_ok, checkpoint_ok):
    if complete:
        return "complete"
    if alive and not stale:
        return "running"
    if not disk_ok:
        return "blocked_disk"
    if attempts >= max_restarts:
        return "blocked_retry_limit"
    if not checkpoint_ok:
        return "blocked_no_checkpoint"
    return "restart"


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def select_checkpoint(output: Path, spec: dict, config) -> tuple[Path | None, int, list]:
    candidates, rejected = [], []
    paths = set(output.glob("step*.pt")) | {output / "latest.pt", output / "final.pt"}
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            step = validate_resume_checkpoint(
                path,
                config_sha256=spec["config"]["sha256"],
                sources={k: asdict(v) for k, v in config.model.input_sources.items()},
                regions=[d.region for d in config.data.datasets],
                max_steps=spec["training"]["steps"],
            )
            candidates.append((step, path.name == "latest.pt", path))
        except (
            OSError,
            ValueError,
            RuntimeError,
            EOFError,
            KeyError,
            pickle.UnpicklingError,
        ) as error:
            rejected.append({"file": path.name, "error": type(error).__name__})
    if not candidates:
        return None, 0, rejected
    step, _, path = max(candidates)
    return path, step, rejected


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.trial = args.trial.resolve()
        self.repository = args.repository.resolve()
        self.root = self.trial / "watchdog"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = _json(self.root / "state.json") or {
            "schema": "regional-watchdog-v1",
            "models": {},
        }
        if self.state.get("schema") != "regional-watchdog-v1":
            raise ValueError("Unknown watchdog state schema")
        self.state.update(pid=os.getpid(), state="monitoring")
        self.state["policy"] = {
            "poll_seconds": args.poll_seconds,
            "stale_seconds": args.stale_seconds,
            "startup_grace_seconds": args.startup_grace,
            "max_restarts_per_model": args.max_restarts,
        }
        self.children = []

    def save(self):
        self.state["updated_utc"] = time.time()
        dump(self.root / "state.json", self.state)

    def event(self, model, action, **details):
        with (self.root / "events.jsonl").open("a") as handle:
            handle.write(
                json.dumps({"utc": time.time(), "model": model, "action": action, **details}) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def environment(self, devices=None):
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH=str(self.repository / "src") + ":" + environment.get("PYTHONPATH", ""),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MALLOC_ARENA_MAX="2",
        )
        if devices is None:
            environment.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        else:
            environment["ASCEND_RT_VISIBLE_DEVICES"] = devices
        return environment

    def check_model(self, name: str):
        now = time.time()
        path = self.trial / f"{name}-spec.json"
        spec, config, _ = read_spec(path)
        require_training_approval(path, self.trial / f"{name}-approval.json")
        output = Path(spec["output"])
        status_path = output / "status.json"
        status = _json(status_path)
        record_path = self.trial / f"{name}-train-process.json"
        record = _json(record_path)
        model = self.state["models"].setdefault(name, {"restarts": 0})
        step = status.get("optimizer_steps", 0)
        if model.get("observed_step") != step:
            model.update(
                observed_step=step,
                progress_at=(
                    min(now, status_path.stat().st_mtime) if status_path.exists() else now
                ),
            )
        matched = matching_processes(path)
        complete = (
            status.get("state") == "complete"
            and step == spec["training"]["steps"]
            and (output / "final.pt").is_file()
        )
        stale = (
            now - model.get("progress_at", now) >= self.args.stale_seconds
            and now - record.get("launched_utc", now) >= self.args.startup_grace
        )
        disk_ok = shutil.disk_usage(output).free >= spec["training"]["min_free_gib"] * 2**30
        action = decision(
            alive=bool(matched),
            complete=complete,
            stale=stale,
            attempts=model["restarts"],
            max_restarts=self.args.max_restarts,
            disk_ok=disk_ok,
            checkpoint_ok=True,
        )
        model.update(
            state=action,
            alive_pids=sorted(matched),
            heartbeat_age_seconds=round(now - model.get("progress_at", now), 1),
        )
        if action != "restart":
            # A live orphaned rank can still make progress after its launcher dies.
            if matched and record.get("pid") not in matched and not complete:
                record.update(pid=min(matched), adopted_worker=True)
                dump(record_path, record)
            return
        reason = "no_progress_timeout" if matched else "process_exit"
        delay = min(600, 60 * 2 ** max(0, model["restarts"] - 1)) if model["restarts"] else 0
        if now - model.get("last_restart_utc", 0) < delay:
            model["state"] = "backoff"
            return
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src"], cwd=self.repository, text=True
        ).strip()
        if dirty:
            model["state"] = "blocked_uncommitted_source"
            return
        checkpoint, saved_step, rejected = select_checkpoint(output, spec, config)
        if checkpoint is None:
            model.update(state="blocked_no_checkpoint", rejected_checkpoints=rejected)
            return
        devices = str(record["devices"])
        ids = [int(value) for value in devices.split(",")]
        micro = record["micro_batch"]
        if (
            not ids
            or min(ids) < 0
            or len(set(ids)) != len(ids)
            or type(micro) is not int
            or micro < 1
            or 48 % (len(ids) * micro)
        ):
            raise ValueError("Invalid registered device or micro batch settings")
        model["state"] = "recovering"
        self.save()
        stop_owned(path)
        checkpoint, saved_step, rejected = select_checkpoint(output, spec, config)
        if checkpoint is None:
            model["state"] = "blocked_no_checkpoint"
            return
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repository, text=True
        ).strip()
        environment = self.environment(devices)
        environment.update(
            XUANNV_GIT_SHA=revision, HCCL_WHITELIST_DISABLE="1", HCCL_CONNECT_TIMEOUT="300"
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={len(ids)}",
            "-m",
            "xuannv_embedding.cli",
            "experiment",
            "regional-base",
            "--spec",
            str(path),
            "--approval",
            str(self.trial / f"{name}-approval.json"),
            "--phase",
            "train",
            "--device",
            "npu:0",
            "--micro-batch",
            str(micro),
            "--resume",
            str(checkpoint),
        ]
        attempt = model["restarts"] + 1
        with (self.root / f"{name}-restart-{attempt}.log").open("ab") as log:
            child = subprocess.Popen(
                command,
                cwd=self.trial,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        self.children.append(child)
        record.update(
            pid=child.pid,
            command=command,
            git_sha=revision,
            launched_utc=time.time(),
            resume_checkpoint=str(checkpoint),
            resume_step=saved_step,
        )
        dump(record_path, record)
        model.update(
            restarts=attempt,
            last_restart_utc=time.time(),
            observed_step=saved_step,
            progress_at=time.time(),
            checkpoint_step=saved_step,
            recovery_pid=child.pid,
            last_restart_reason=reason,
        )
        self.event(
            name,
            "restarted",
            checkpoint=checkpoint.name,
            checkpoint_step=saved_step,
            checkpoint_sha256=sha(checkpoint),
            attempt=attempt,
            git_sha=revision,
            rejected_checkpoints=rejected,
            reason=reason,
        )

    def check_evaluation(self):
        spec = self.trial / "benchmark-spec.json"
        if not spec.exists():
            return
        status = _json(self.trial / "benchmark/follow-status.json")
        if status.get("stage") == "complete":
            self.state["evaluation"] = "complete"
            return
        record_path = self.trial / "benchmark-follow-process.json"
        record = _json(record_path)
        info = processes().get(record.get("pid"))
        if (
            info is not None
            and "regional-benchmark" in info.arguments
            and "follow" in info.arguments
        ):
            self.state["evaluation"] = "running"
            return
        # Let existing readouts finish before replacing their failed scheduler.
        if matching_processes(spec, evaluation=True):
            self.state["evaluation"] = "waiting_for_existing_readouts"
            return
        if any(
            v["state"] != "complete" and v["state"] != "running"
            for v in self.state["models"].values()
        ):
            self.state["evaluation"] = "waiting_for_training_recovery"
            return
        attempts = self.state.get("evaluation_restarts", 0)
        if attempts >= self.args.max_restarts:
            self.state["evaluation"] = "blocked_retry_limit"
            return
        command = [
            sys.executable,
            "-m",
            "xuannv_embedding.cli",
            "experiment",
            "regional-benchmark",
            "--spec",
            str(spec),
            "--phase",
            "follow",
        ]
        with (self.trial / "benchmark-follow.log").open("ab") as log:
            child = subprocess.Popen(
                command,
                cwd=self.trial,
                env=self.environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        self.children.append(child)
        dump(record_path, {"pid": child.pid, "command": command})
        self.state.update(evaluation="restarted", evaluation_restarts=attempts + 1)
        self.event("evaluation", "scheduler_restarted", attempt=attempts + 1)

    def tick(self):
        for child in self.children:
            child.poll()
        for name in self.args.models:
            try:
                self.check_model(name)
            except Exception as error:
                self.state["models"].setdefault(name, {"restarts": 0}).update(
                    state="blocked_validation", error=f"{type(error).__name__}: {error}"
                )
        self.check_evaluation()
        self.save()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment watchdog")
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["public", "highres"])
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument("--startup-grace", type=int, default=600)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--disable", action="store_true")
    args = parser.parse_args(argv)
    if min(args.poll_seconds, args.stale_seconds, args.startup_grace, args.max_restarts) < 1 or any(
        not n or not all(c.isalnum() or c in "_-" for c in n) for n in args.models
    ):
        raise ValueError("Positive watchdog settings and safe registered model names are required")
    torch.set_num_threads(1)
    supervisor = Supervisor(args)
    stop = supervisor.root / "STOP"
    if args.disable:
        stop.touch()
        return 0
    with (supervisor.root / "watchdog.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor.event(
            "watchdog",
            "audit_started" if args.once else "started",
            pid=os.getpid(),
            policy=supervisor.state["policy"],
        )
        while not stop.exists():
            supervisor.tick()
            if args.once:
                return 0
            if (
                all(v["state"] == "complete" for v in supervisor.state["models"].values())
                and supervisor.state.get("evaluation", "complete") == "complete"
            ):
                supervisor.state["state"] = "complete"
                supervisor.save()
                return 0
            time.sleep(args.poll_seconds)
        supervisor.state["state"] = "disabled"
        supervisor.save()
    return 0
