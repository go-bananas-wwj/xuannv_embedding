"""One-shot, provenance-checked export and validation after a registered training PID.

Never restarts training, starts a new sweep, scores test labels, or publishes unchecked
results. A successful controller stops at ready_to_publish for report review and sync.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

import torch

from xuannv_embedding.downstream.multitask import normalized_score
from xuannv_embedding.training.experiment import _json, _sha
from xuannv_embedding.training.experiment_queue import available_devices


def process_identity(pid: int) -> dict | None:
    root = Path("/proc") / str(pid)
    try:
        fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_ticks": int(fields[19]),
            "command_sha256": hashlib.sha256((root / "cmdline").read_bytes()).hexdigest(),
        }
    except FileNotFoundError:
        return None


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def audit_checkpoint(state, base, registration, *, epochs: int, steps: int) -> dict:
    if state["epoch"] + 1 != epochs or state["scheduler"]["last_epoch"] != epochs:
        raise ValueError("checkpoint epoch does not match the registered stop")
    for field in ("git_sha", "config_sha256"):
        if state[field] != registration[field]:
            raise ValueError(f"checkpoint {field} differs from registration")
    metrics = state["metrics"]
    if len(metrics["rank_random_states"]) != registration["world_size"]:
        raise ValueError("checkpoint rank random states are incomplete")
    actual = sorted({float(v["step"]) for v in state["optimizer"]["state"].values() if "step" in v})
    if actual != [steps] or metrics["optimizer_steps"] != steps:
        raise ValueError("actual optimizer updates differ from the requested budget")
    for key, value in base["model"].items():
        if "base." + key not in state["model"] or not torch.equal(
            value, state["model"]["base." + key]
        ):
            raise ValueError(f"frozen model changed: {key}")
    for key, value in base["criterion"].items():
        if key not in state["criterion"] or not torch.equal(value, state["criterion"][key]):
            raise ValueError(f"frozen criterion changed: {key}")
    for group in (state["model"], state["criterion"], *state["optimizer"]["state"].values()):
        if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all() for v in group.values()):
            raise ValueError("nonfinite checkpoint tensor")
    return {
        "actual_optimizer_steps": actual,
        "epoch": epochs,
        "frozen_model_tensors": len(base["model"]),
        "frozen_criterion_tensors": len(base["criterion"]),
        "world_size": registration["world_size"],
    }


def paired_report(baseline: Path, candidate: Path) -> dict:
    left = json.loads((baseline / "results.json").read_text())
    right = json.loads((candidate / "results.json").read_text())
    identities = [json.loads((p / "identity.json").read_text()) for p in (baseline, candidate)]
    for field in ("protocol", "cache_sha256", "active_indices", "tasks", "test_scored"):
        if identities[0][field] != identities[1][field]:
            raise ValueError(f"paired evaluation identity differs: {field}")
    if identities[1]["test_scored"]:
        raise ValueError("screening must not score test labels")
    by_key = {row["key"]: row for row in left}
    for rows in (left, right):
        for row in rows:
            expected_error = (
                row["metrics"]["rmse"] if row["family"] == "R" else 1 - row["metrics"]["ap"]
            )
            if abs(row["error"] - expected_error) > 1e-12:
                raise ValueError("selection error is inconsistent with the measured metric")
    score = normalized_score(left, right)
    for row in right:
        other = by_key[row["key"]]
        for key in ("family", "source", "task", "seed", "budget"):
            if row[key] != other[key]:
                raise ValueError("paired evaluation condition mismatch")
        keys = {
            "C": ("support_tiles", "support_positions_sha256", "validation_pixels"),
            "R": ("support_tiles", "support_blocks", "validation_blocks"),
            "Q": ("queries",),
        }
        for key in keys[row["family"]]:
            if row["metrics"][key] != other["metrics"][key]:
                raise ValueError(f"paired support mismatch: {key}")
    summaries = []
    for rows in (left, right):
        summary = {}
        for family, source, metric in (
            ("C", "osm", "ap"),
            ("C", "esri", "ap"),
            ("R", "esri", "rmse"),
            ("Q", "osm", "ap"),
        ):
            values = [
                r["metrics"][metric]
                for r in rows
                if r["family"] == family and r["source"] == source
            ]
            summary[f"{family}_{source}_{metric}"] = mean(values)
        summaries.append(summary)
    return {
        "paired_conditions": len(right),
        "normalized_score": score,
        "baseline": summaries[0],
        "candidate": summaries[1],
        "test_scored": False,
    }


def run(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    fields = {
        "run_id",
        "training_process",
        "training_directory",
        "config",
        "config_sha256",
        "cache",
        "cache_sha256",
        "code",
        "code_sha",
        "code_tree_sha256",
        "epochs",
        "steps",
        "output",
        "device",
        "evaluation_template",
        "evaluation_template_sha256",
        "baseline",
        "baseline_results_sha256",
        "verifier",
        "verifier_sha256",
    }
    if set(plan) != fields:
        raise ValueError("followup plan has missing or unknown fields")
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=False)
    _json(
        root / "controller.json",
        {"process": process_identity(os.getpid()), "plan_sha256": _sha(plan_path)},
    )
    started = time.time()

    def status(phase, **extra):
        _json(
            root / "status.json",
            {"state": phase, "started_at": started, "updated_at": time.time(), **extra},
        )

    def unchanged():
        for key, digest_key in (
            ("config", "config_sha256"),
            ("evaluation_template", "evaluation_template_sha256"),
            ("verifier", "verifier_sha256"),
        ):
            if _sha(Path(plan[key])) != plan[digest_key]:
                raise ValueError(f"registered {key} changed")
        if _sha(Path(plan["cache"]) / "cache.json") != plan["cache_sha256"]:
            raise ValueError("training cache changed")
        if source_digest(Path(plan["code"])) != plan["code_tree_sha256"]:
            raise ValueError("pinned execution source changed")
        if _sha(Path(plan["baseline"]) / "results.json") != plan["baseline_results_sha256"]:
            raise ValueError("baseline results changed")

    def stage(name, command, env):
        status(name, command_sha256=hashlib.sha256(json.dumps(command).encode()).hexdigest())
        runtime = root / ("runtime_" + name)
        runtime.mkdir()
        with (root / (name + ".log")).open("x") as stream:
            process = subprocess.Popen(
                command,
                cwd=runtime,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            status(name, child_process=process_identity(process.pid))
            if process.wait() != 0:
                raise RuntimeError(f"{name} subprocess failed; inspect its dedicated log")

    try:
        unchanged()
        training = Path(plan["training_directory"])
        expected = plan["training_process"]
        while True:
            try:
                identity = process_identity(expected["pid"])
            except OSError:
                status("training_observation_unavailable")
                time.sleep(15)
                continue
            if identity is None:
                break
            if identity != expected:
                raise RuntimeError("training PID identity changed; never attach to a reused PID")
            path = training / "status.json"
            try:
                current = json.loads(path.read_text()) if path.exists() else {}
            except (OSError, json.JSONDecodeError):
                status("training_status_unavailable")
                time.sleep(15)
                continue
            status("watching_training", training_epoch=current.get("epoch"))
            time.sleep(15)
        current = json.loads((training / "status.json").read_text())
        if current.get("state") != "complete" or current.get("epoch") != plan["epochs"]:
            raise RuntimeError("training handle ended without the registered terminal checkpoint")
        status("checking_checkpoint")
        unchanged()
        registration = json.loads((training / "run.json").read_text())
        if (
            registration["git_sha"] != plan["code_sha"]
            or registration["config_sha256"] != plan["config_sha256"]
        ):
            raise ValueError("training registration identity changed")
        checkpoint = training / f"epoch_{plan['epochs']:04d}.pt"
        parent = Path(registration["adaptation"]["base_checkpoint"])
        if _sha(parent) != registration["adaptation"]["base_checkpoint_sha256"]:
            raise ValueError("parent checkpoint changed")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        base = torch.load(parent, map_location="cpu", weights_only=True)
        audit = audit_checkpoint(
            state, base, registration, epochs=plan["epochs"], steps=plan["steps"]
        )
        audit["checkpoint_sha256"] = _sha(checkpoint)
        del state, base
        _json(root / "checkpoint_verification.json", audit)
        while True:
            try:
                info = subprocess.check_output(["npu-smi", "info"], text=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                status("device_observation_unavailable")
                time.sleep(15)
                continue
            if plan["device"] in available_devices(info, [plan["device"]], {}):
                break
            status("waiting_export_device")
            time.sleep(15)
        env = os.environ.copy()
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            env.pop(key, None)
        env.update(
            PYTHONPATH=str(Path(plan["code"]) / "src") + os.pathsep + env.get("PYTHONPATH", ""),
            XUANNV_GIT_SHA=plan["code_sha"],
            ASCEND_RT_VISIBLE_DEVICES=str(plan["device"]),
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        )
        cli = [sys.executable, "-u", "-m", "xuannv_embedding.cli"]
        stage(
            "export",
            [
                *cli,
                "experiment",
                "export",
                "--config",
                plan["config"],
                "--cache",
                plan["cache"],
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(root / "export"),
                "--device",
                "npu:0",
                "--batch-size",
                "4",
            ],
            env,
        )
        spec = json.loads(Path(plan["evaluation_template"]).read_text())
        spec.update(
            output=str(root / "validation"),
            models={plan["run_id"]: {"manifest": str(root / "export/manifest.json")}},
        )
        _json(root / "evaluation.json", spec)
        stage(
            "evaluation",
            [
                *cli,
                "audit",
                "multitask",
                "--spec",
                str(root / "evaluation.json"),
                "--model",
                plan["run_id"],
            ],
            env,
        )
        stage(
            "verification",
            [
                sys.executable,
                plan["verifier"],
                str(root / "validation"),
                str(root / "verification.json"),
                plan["run_id"],
            ],
            env,
        )
        candidate = root / "validation" / plan["run_id"]
        report = paired_report(Path(plan["baseline"]), candidate)
        _json(root / "paired_summary.json", report)
        status(
            "ready_to_publish",
            report=str(root / "paired_summary.json"),
            test_scored=False,
            next_action="review, compile and sync to Overleaf before next training group",
        )
    except BaseException as exc:
        status("failed", error_type=type(exc).__name__, error=str(exc))
        raise
