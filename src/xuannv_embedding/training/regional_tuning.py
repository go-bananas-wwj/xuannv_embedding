"""Bounded regional tuning with validation selection and persistent process supervision."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from xuannv_embedding.downstream import regional_benchmark as benchmark
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.feature_preservation import audit_frozen_checkpoint
from xuannv_embedding.training.regional_base import read_spec
from xuannv_embedding.training.watchdog import select_checkpoint


def select_validation_winner(candidates: dict) -> str:
    if not candidates:
        raise ValueError("Validation candidates must not be empty")
    for candidate in candidates.values():
        value = candidate["validation_macro_f1"]
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid validation score")
    return max(candidates, key=lambda name: candidates[name]["validation_macro_f1"])


def stage_schedule(candidate: dict, budget: int) -> list[int]:
    stages = candidate.get("stages", [budget])
    if (
        not isinstance(stages, list)
        or not stages
        or any(type(v) is not int or not 1 < v <= budget for v in stages)
        or stages != sorted(set(stages))
        or stages[-1] != budget
    ):
        raise ValueError("Stages must be increasing update counts ending at the registered budget")
    return stages


def stop_after_validation(scores: list[float], *, patience: int) -> bool:
    if type(patience) is not int or patience < 1:
        raise ValueError("Validation patience must be positive")
    if not scores:
        return False
    select_validation_winner(
        {str(i): {"validation_macro_f1": score} for i, score in enumerate(scores)}
    )
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return len(scores) - 1 - best_index >= patience


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    expected = {"protocol", "output", "baseline", "candidates", "selection", "github"}
    if (
        not expected.issubset(plan)
        or set(plan) - expected - {"public_fallback"}
        or plan["protocol"] != "regional-validation-tuning-v1"
    ):
        raise ValueError("Unknown tuning plan")
    if type(plan.get("public_fallback", False)) is not bool:
        raise ValueError("public_fallback must be a boolean")
    if (
        set(plan["baseline"]) != {"path", "sha256"}
        or sha(plan["baseline"]["path"]) != plan["baseline"]["sha256"]
    ):
        raise ValueError("Baseline registration changed")
    if plan["selection"] != {"fold": 0, "budget": 200, "heads": ["mlp", "conv3x3"]}:
        raise ValueError("Unregistered validation selection protocol")
    devices, names = [], []
    root = Path(plan["output"]).resolve()
    for candidate in plan["candidates"]:
        required_candidate = {"name", "spec", "approval", "devices", "micro_batch"}
        if not required_candidate.issubset(candidate) or set(candidate) - required_candidate - {
            "stages"
        }:
            raise ValueError("Unknown tuning candidate fields")
        name = candidate["name"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise ValueError("Candidate names must be unique and path-safe")
        names.append(name)
        spec, _, _ = read_spec(Path(candidate["spec"]))
        stage_schedule(candidate, spec["training"]["steps"])
        if Path(spec["output"]).resolve() != root / name / "run":
            raise ValueError("Candidate output must be isolated beneath the tuning directory")
        cards = candidate["devices"]
        if not cards or any(type(d) is not int or d < 1 for d in cards):
            raise ValueError("Tuning devices must be explicit positive indices")
        devices.extend(cards)
        if (
            type(candidate["micro_batch"]) is not int
            or candidate["micro_batch"] < 1
            or 48 % (len(cards) * candidate["micro_batch"])
        ):
            raise ValueError("Candidate micro batch must preserve effective batch 48")
    if not names or len(devices) != len(set(devices)):
        raise ValueError("Tuning candidates require disjoint devices")
    github = plan["github"]
    if set(github) != {"repository", "report"}:
        raise ValueError("Unknown GitHub publication fields")
    repo = Path(github["repository"]).resolve()
    target = (repo / github["report"]).resolve()
    if not target.is_relative_to(repo / "docs") or target.suffix != ".md":
        raise ValueError("Tuning summaries must be inside repository docs")
    return plan


def baseline_spec(plan):
    return json.loads(Path(plan["baseline"]["path"]).read_text())


def candidate_benchmark(plan, candidate, *, step=None):
    baseline = baseline_spec(plan)
    source = Path(baseline["output"])
    directory = Path(plan["output"]) / candidate["name"]
    root = (
        directory / "benchmark"
        if step is None
        else (directory / "stages" / f"step{step:06d}" / "benchmark")
    )
    root.mkdir(parents=True, exist_ok=True)

    def link(path, target):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            if path.resolve() != target.resolve():
                raise ValueError("Frozen reference link changed")
        else:
            path.symlink_to(target, target_is_directory=target.is_dir())

    for name in [
        "prepared.json",
        "targets.npz",
        "support.json",
        "aef.npy",
        "raw_public.npy",
        "raw_highres.npy",
    ]:
        link(root / name, source / name)
    link(root / "representations/public", source / "representations/public")
    for sub in ["scores", "regression"]:
        (root / sub).mkdir(exist_ok=True)
        for path in (source / sub).iterdir():
            if path.name.startswith(("aef", "public", "raw_")):
                link(root / sub / path.name, path)
    return {**baseline, "output": str(root)}


def validate_candidate(plan, candidate, *, device_name, step=None):
    root = Path(candidate_benchmark(plan, candidate, step=step)["output"])
    report_path = root / "validation-selection.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    metadata = json.loads((root / "prepared.json").read_text())
    with np.load(root / "targets.npz") as z:
        labels, valid = z["labels"], z["valid"]
    classes = metadata["classes"]
    target = np.full(labels.shape, -1, np.int64)
    for i, label in enumerate(classes):
        target[(labels == label) & valid] = i
    fold = metadata["folds"][plan["selection"]["fold"]]
    budget = plan["selection"]["budget"]
    support = json.loads((root / "support.json").read_text())[f"{fold['fold']}:{budget}"]
    features = np.load(benchmark.feature_path(root, "highres"), mmap_mode="r")
    result = {
        "selection": "mean_validation_macro_f1_only",
        "heads": {},
        "fold": fold["fold"],
        "test_scored": False,
    }
    for kind in plan["selection"]["heads"]:
        head, scaler, device, details = benchmark._fit_neural(
            kind, features, target, valid, support, fold, classes, device_name
        )
        matrix = benchmark._confusion(
            head, features, target, valid, fold["validation"], classes, scaler, device
        )
        result["heads"][kind] = {**details, **benchmark.classification_metrics(matrix, classes)}
        torch.save(
            {
                "head": head.cpu().state_dict(),
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
            root / f"validation-{kind}-head.pt",
        )
        del head
    result["validation_macro_f1"] = float(
        np.mean([value["macro_f1"] for value in result["heads"].values()])
    )
    dump(report_path, result)
    return result


def run_candidate(plan, candidate):
    from xuannv_embedding.export.regional_trial import export

    torch.set_num_threads(2)
    output = Path(plan["output"]) / candidate["name"]
    spec, config, _ = read_spec(Path(candidate["spec"]))
    status = output / "status.json"
    if status.exists() and json.loads(status.read_text()).get("stage") == "complete":
        return
    training = Path(spec["output"])
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(candidate["devices"])),
        "--module",
        "xuannv_embedding.cli",
        "experiment",
        "regional-base",
        "--spec",
        candidate["spec"],
        "--approval",
        candidate["approval"],
        "--phase",
        "train",
        "--device",
        "npu:0",
        "--micro-batch",
        str(candidate["micro_batch"]),
    ]
    completed = (
        json.loads((training / "status.json").read_text())
        if (training / "status.json").exists()
        else {}
    )
    stages = stage_schedule(candidate, spec["training"]["steps"])
    staged = "stages" in candidate
    if completed.get("state") != "complete":
        checkpoint, step, rejected = select_checkpoint(training, spec, config)
        if completed and checkpoint is None:
            raise RuntimeError("Stopped candidate has no valid recovery checkpoint")
        if checkpoint is None:
            dump(status, {"stage": "one_update_gate", "utc": time.time()})
            subprocess.run([*command, "--stop-after-updates", "1"], check=True)
            checkpoint, step, rejected = select_checkpoint(training, spec, config)
            if checkpoint is None or step != 1:
                raise RuntimeError("Two-card real-input gate did not save a valid first update")
            dump(
                output / "gate.json",
                {
                    "successful_updates": step,
                    "rejected": rejected,
                    "world_size": len(candidate["devices"]),
                    "effective_batch": 48,
                },
            )
            if spec["training"].get("freeze_public", False):
                dump(
                    output / "frozen-gate.json",
                    audit_frozen_checkpoint(checkpoint, spec["checkpoint"]["path"]),
                )
    history = []
    for target_step in stages:
        eval_step = target_step if staged else None
        root = Path(candidate_benchmark(plan, candidate, step=eval_step)["output"])
        existing_validation = root / "validation-selection.json"
        if not existing_validation.exists():
            checkpoint, current_step, _ = select_checkpoint(training, spec, config)
            if current_step < target_step:
                dump(
                    status,
                    {
                        "stage": "training",
                        "utc": time.time(),
                        "target_step": target_step,
                        "resumed_step": current_step,
                    },
                )
                subprocess.run(
                    [
                        *command,
                        "--resume",
                        str(checkpoint),
                        "--stop-after-updates",
                        str(target_step),
                    ],
                    check=True,
                )
            selected = (
                training / "final.pt"
                if target_step == spec["training"]["steps"]
                else training / f"step{target_step:06d}.pt"
            )
            if spec["training"].get("freeze_public", False):
                dump(
                    root / "frozen-audit.json",
                    audit_frozen_checkpoint(selected, spec["checkpoint"]["path"]),
                )
            dump(status, {"stage": "export", "utc": time.time(), "optimizer_steps": target_step})
            export(
                Path(candidate["spec"]),
                selected,
                root / "representations/highres",
                device_name="npu:0",
            )
            dump(
                status, {"stage": "validation", "utc": time.time(), "optimizer_steps": target_step}
            )
        validation = validate_candidate(plan, candidate, device_name="npu:0", step=eval_step)
        history.append(
            {"step": target_step, "validation_macro_f1": validation["validation_macro_f1"]}
        )
        dump(output / "stage-history.json", history)
        if stop_after_validation([entry["validation_macro_f1"] for entry in history], patience=2):
            break
    chosen = max(history, key=lambda value: value["validation_macro_f1"])
    chosen_checkpoint = (
        training / "final.pt"
        if chosen["step"] == spec["training"]["steps"]
        else training / f"step{chosen['step']:06d}.pt"
    )
    selected_alias = training / "selected.pt"
    if selected_alias.exists() or selected_alias.is_symlink():
        if selected_alias.resolve() != chosen_checkpoint.resolve():
            raise ValueError("Selected checkpoint alias changed")
    else:
        selected_alias.symlink_to(chosen_checkpoint)
    if staged:
        alias = output / "benchmark"
        target = output / "stages" / f"step{chosen['step']:06d}" / "benchmark"
        if alias.is_symlink():
            if alias.resolve() != target.resolve():
                raise ValueError("Candidate selected-stage alias changed")
        elif alias.exists():
            raise ValueError("Refusing to replace an existing benchmark directory")
        else:
            alias.symlink_to(target, target_is_directory=True)
    dump(
        status,
        {
            "stage": "complete",
            "utc": time.time(),
            "validation_macro_f1": chosen["validation_macro_f1"],
            "selected_step": chosen["step"],
            "completed_steps": history[-1]["step"],
            "registered_budget": spec["training"]["steps"],
        },
    )


def parent_validation(plan):
    source = Path(baseline_spec(plan)["output"])
    path = source / "validation-selection.json"
    if path.exists():
        report = json.loads(path.read_text())
        if report.get("test_scored") is not False or report["fold"] != plan["selection"]["fold"]:
            raise ValueError("Parent validation provenance differs")
        return {
            "validation_macro_f1": report["validation_macro_f1"],
            "selection": "validation_only",
        }
    fold, budget = plan["selection"]["fold"], plan["selection"]["budget"]
    values = []
    for head in plan["selection"]["heads"]:
        item = json.loads(
            (source / "scores" / f"highres_{head}_fold{fold}_n{budget}/result.json").read_text()
        )
        values.append(max(point["validation_macro_f1"] for point in item["curve"]))
    return {"validation_macro_f1": float(np.mean(values)), "selection": "validation_only"}


def public_validation(plan):
    source = Path(baseline_spec(plan)["output"])
    values = []
    for head in plan["selection"]["heads"]:
        path = source / "scores" / f"public_{head}_fold0_n200/result.json"
        item = json.loads(path.read_text())
        values.append(max(point["validation_macro_f1"] for point in item["curve"]))
    return {"validation_macro_f1": float(np.mean(values)), "selection": "validation_only"}


def evaluate_selected(plan, name):
    if name in {"parent", "public"}:
        return {"winner": name, "state": f"{name}_retained_no_validation_improvement"}
    candidate = next(c for c in plan["candidates"] if c["name"] == name)
    spec = candidate_benchmark(plan, candidate)
    root = Path(spec["output"])
    # Fold zero's validation-trained heads can be tested directly, without refitting.
    metadata = json.loads((root / "prepared.json").read_text())
    classes, fold = metadata["classes"], metadata["folds"][0]
    with np.load(root / "targets.npz") as z:
        labels, valid = z["labels"], z["valid"]
    target = np.full(labels.shape, -1, np.int64)
    for index, label in enumerate(classes):
        target[(labels == label) & valid] = index
    from xuannv_embedding.downstream.heads import build_head
    from xuannv_embedding.training.cli import _setup_device

    features = np.load(benchmark.feature_path(root, "highres"), mmap_mode="r")
    device, _, _ = _setup_device("npu:7")
    for kind in plan["selection"]["heads"]:
        job = {"feature": "highres", "head": kind, "fold": 0, "budget": 200}
        output = root / "scores" / f"highres_{kind}_fold0_n200"
        output.mkdir(exist_ok=True)
        if not (output / "result.json").exists():
            saved = torch.load(root / f"validation-{kind}-head.pt", weights_only=True)
            head = build_head(kind, embed_dim=64, num_classes=len(classes)).to(device)
            head.load_state_dict(saved["head"], strict=True)
            scaler = StandardScaler()
            scaler.mean_, scaler.scale_ = np.array(saved["mean"]), np.array(saved["scale"])
            maps = np.zeros_like(labels)
            matrix = benchmark._confusion(
                head,
                features,
                target,
                valid,
                fold["test"],
                classes,
                scaler,
                device,
                prediction_maps=maps,
            )
            dump(
                output / "result.json",
                {
                    "state": "complete",
                    **job,
                    "metrics": benchmark.classification_metrics(matrix, classes),
                    "selection": "validation_only_reused_selected_head",
                },
            )
            np.savez_compressed(
                output / "predictions.npz",
                labels=maps,
                test_tiles=np.asarray(fold["test"], np.int32),
            )
        for number in range(1, 5):
            benchmark.score_job(spec, {**job, "fold": number}, device_name="npu:7")
    benchmark.score_regression(spec, "highres")
    from xuannv_embedding.downstream.regional_figures import render
    from xuannv_embedding.training.regional_followup import summarize

    summary = summarize(spec)
    render(spec)
    return {
        "winner": name,
        "selected_step": json.loads((Path(plan["output"]) / name / "status.json").read_text()).get(
            "selected_step"
        ),
        "state": "complete",
        "comparisons": summary["comparisons"],
        "scope": "adaptive exploration of an already inspected regional benchmark; "
        "five-fold pooled score is not a new blind test; fold0 test is scored "
        "only after current-round validation selection",
    }


def plot_curves(plan):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(plan["output"])
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for path in (root / "diagnostics").glob("*-snapshots.json"):
        data = json.loads(path.read_text())["snapshots"]
        x = [row["step"] for row in data]
        for ax, key in zip(axes, ["total", "recon_esa_worldcover"]):
            ax.plot(
                x,
                [row["losses"].get(key, np.nan) for row in data],
                "o-",
                label=path.stem.replace("-snapshots", ""),
            )
            ax.set_title(key)
            ax.set_xlabel("Original scheduled optimizer update")
    for ax in axes:
        ax.legend()
    fig.suptitle("Fixed-sample unmasked checkpoint losses (not the original training curve)")
    fig.savefig(figures / "checkpoint_diagnostic_losses.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for candidate in plan["candidates"]:
        path = root / candidate["name"] / "run/training-curve.jsonl"
        if not path.exists():
            continue
        data = [json.loads(row) for row in path.read_text().splitlines()]
        axes[0].plot(
            [r["optimizer_steps"] for r in data],
            [r["losses"]["total"] for r in data],
            label=candidate["name"],
        )
        axes[1].plot(
            [r["optimizer_steps"] for r in data],
            [r["learning_rates"].get("public", 0) for r in data],
            label=candidate["name"],
        )
    for ax, title in zip(axes, ["Mean training objective", "Public-backbone learning rate"]):
        ax.set_title(title)
        ax.set_xlabel("Additional successful optimizer updates")
        ax.legend()
    fig.savefig(figures / "tuning_training_curves.png", dpi=160)
    plt.close(fig)


def publish(plan, status, stage):
    repo = Path(plan["github"]["repository"])
    report = plan["github"]["report"]
    content = "# Regional high-resolution tuning progress\n\n```json\n"
    content += json.dumps(status, indent=2) + "\n```\n"
    if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=repo).strip():
        raise RuntimeError("Unrelated staged changes prevent progress publication")
    (repo / report).write_text(content)
    for arguments in [
        ["add", "--", report],
        ["commit", "-m", f"docs: record tuning {stage}"],
        ["push"],
    ]:
        subprocess.run(["git", *arguments], cwd=repo, check=True)


def supervise(plan_path):
    plan = read_plan(plan_path)
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=True)
    active, done, attempts, published_stages = {}, {}, {}, {}
    identity = sha(plan_path)
    with (root / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while len(done) < len(plan["candidates"]):
                if sha(plan_path) != identity or (root / "STOP").exists():
                    raise RuntimeError("Tuning plan changed or manual STOP requested")
                if shutil.disk_usage(root).free < 100 * 2**30:
                    raise RuntimeError("Storage reserve reached")
                for candidate in plan["candidates"]:
                    name = candidate["name"]
                    if name in done:
                        continue
                    directory = root / name
                    directory.mkdir(exist_ok=True)
                    if name in active:
                        process, started = active[name]
                        stamps = [
                            p.stat().st_mtime
                            for p in [
                                directory / "status.json",
                                directory / "run/status.json",
                                directory / "benchmark/representations/highres/status.json",
                            ]
                            if p.exists()
                        ]
                        stalled = time.time() - max([started, *stamps]) > 1200
                        if process.poll() is None and not stalled:
                            continue
                        if stalled and process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        del active[name]
                        path = directory / "status.json"
                        state = json.loads(path.read_text()) if path.exists() else {}
                        if process.returncode == 0 and state.get("stage") == "complete":
                            done[name] = state
                            publish(plan, {"stage": "candidate_complete", "completed": done}, name)
                            continue
                        attempts[name] = attempts.get(name, 0) + 1
                        if attempts[name] > 2:
                            raise RuntimeError(f"Candidate repeatedly failed: {name}")
                    environment = os.environ.copy()
                    environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
                        map(str, candidate["devices"])
                    )
                    command = [
                        sys.executable,
                        "-m",
                        "xuannv_embedding.cli",
                        "experiment",
                        "regional-tuning",
                        "--plan",
                        str(plan_path),
                        "--phase",
                        "candidate",
                        "--candidate",
                        name,
                    ]
                    with (directory / "process.log").open("ab") as log:
                        process = subprocess.Popen(
                            command,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                    active[name] = process, time.time()
                state = {
                    "stage": "training_and_validation",
                    "utc": time.time(),
                    "supervisor_pid": os.getpid(),
                    "completed": done,
                    "restarts": attempts,
                }
                state["active"] = {name: process.pid for name, (process, _) in active.items()}
                state["progress"] = {}
                for candidate in plan["candidates"]:
                    directory = root / candidate["name"]
                    history_path = directory / "stage-history.json"
                    if history_path.exists():
                        history = json.loads(history_path.read_text())
                        state["progress"][candidate["name"]] = history
                dump(root / "status.json", state)
                if not done and not attempts and not (root / "started-published.json").exists():
                    publish(plan, state, "started")
                    dump(root / "started-published.json", {"utc": time.time()})
                for name, history in state["progress"].items():
                    step = history[-1]["step"]
                    if published_stages.get(name) != step:
                        publish(plan, state, f"{name}_validation_{step}")
                        published_stages[name] = step
                if len(done) < len(plan["candidates"]):
                    time.sleep(30)
            scores = {"parent": parent_validation(plan)}
            if plan.get("public_fallback", False):
                scores["public"] = public_validation(plan)
            for name, value in done.items():
                scores[name] = {"validation_macro_f1": value["validation_macro_f1"]}
            winner = select_validation_winner(scores)
            selection = {"winner": winner, "candidates": scores, "test_used_for_selection": False}
            dump(root / "selection.json", selection)
            dump(root / "status.json", {"stage": "final_evaluation", "selection": selection})
            plot_curves(plan)
            result = evaluate_selected(plan, winner)
            dump(root / "result.json", result)
            final = {
                "stage": "complete",
                "winner": winner,
                "validation": scores,
                "overleaf_modified": False,
                "restarts": attempts,
            }
            dump(root / "status.json", final)
            publish(plan, final, "complete")
        except BaseException as error:
            for process, _ in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            dump(root / "status.json", {"stage": "stopped", "reason": str(error)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment regional-tuning")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", choices=["run", "candidate", "curves"], required=True)
    parser.add_argument("--candidate")
    args = parser.parse_args(argv)
    plan = read_plan(args.plan)
    if args.phase == "run":
        supervise(args.plan)
    elif args.phase == "curves":
        plot_curves(plan)
    else:
        candidate = next(c for c in plan["candidates"] if c["name"] == args.candidate)
        run_candidate(plan, candidate)
    return 0
