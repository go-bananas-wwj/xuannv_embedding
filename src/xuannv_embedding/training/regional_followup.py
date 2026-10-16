"""Run registered downstream jobs as encoder artifacts become available."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from xuannv_embedding.downstream import regional_benchmark as benchmark
from xuannv_embedding.export.context import dump, sha


def read_spec(path: Path) -> dict:
    spec = json.loads(path.read_text())
    expected = {
        "protocol",
        "trial",
        "data",
        "reference",
        "output",
        "github_repository",
        "github_report",
        "public_devices",
        "highres_devices",
        "reserved_device",
    }
    if set(spec) != expected or spec["protocol"] != "paired-regional-benchmark-v1":
        raise ValueError("Unknown or missing benchmark specification fields")
    devices = spec["public_devices"] + spec["highres_devices"] + [spec["reserved_device"]]
    if any(type(v) is not int or v < 0 for v in devices) or len(devices) != len(set(devices)):
        raise ValueError("Evaluation devices must be nonnegative and disjoint")
    repository = Path(spec["github_repository"]).resolve()
    report = (repository / spec["github_report"]).resolve()
    if not report.is_relative_to(repository / "docs") or report.suffix != ".md":
        raise ValueError("Public progress must be a Markdown file inside repository docs")
    return spec


def jobs() -> list[dict]:
    result = []
    combinations = [
        (feature, head)
        for feature in ["aef", "public", "highres"]
        for head in ["mlp", "linear", "conv3x3"]
    ]
    combinations += [
        (feature, head) for feature in ["raw_public", "raw_highres"] for head in ["rf", "unet"]
    ]
    for feature, head in combinations:
        for budget in [200, 50, 1000]:
            for fold in range(5):
                result.append({"feature": feature, "head": head, "budget": budget, "fold": fold})
    for model in ["public", "highres"]:
        for representation in ["april", "mar_may"]:
            for fold in range(5):
                result.append(
                    {
                        "feature": f"{model}_{representation}",
                        "head": "mlp",
                        "budget": 200,
                        "fold": fold,
                    }
                )
    return result


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def snapshot(spec: dict, stage: str) -> dict:
    trial, output = Path(spec["trial"]), Path(spec["output"])
    models = {}
    for model in ["public", "highres"]:
        status = _json(trial / "runs" / model / "status.json")
        models[model] = {
            k: status[k]
            for k in ["state", "optimizer_steps", "elapsed_seconds", "loss"]
            if k in status
        }
    prepared = _json(output / "prepared.json")
    return {
        "stage": stage,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "reference": {
            k: v
            for k, v in _json(Path(spec["reference"]).parent / "status.json").items()
            if k in {"state", "valid_fraction"}
        },
        "evaluation_classes": prepared.get("classes", []),
        "classification_jobs_complete": len(list((output / "scores").glob("*/result.json"))),
        "classification_jobs_total": len(jobs()),
        "overleaf_modified": False,
    }


def publish(spec: dict, stage: str) -> None:
    repository = Path(spec["github_repository"])
    path = repository / spec["github_report"]
    content = (
        "# Regional experiment progress\n\n"
        "Generated from registered local artifacts; no runtime payloads are published.\n\n"
        "```json\n" + json.dumps(snapshot(spec, stage), indent=2) + "\n```\n"
    )
    if path.exists() and path.read_text() == content:
        return
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], cwd=repository, text=True
    ).strip()
    if staged:
        raise RuntimeError("Refusing to commit unrelated staged work")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    subprocess.run(["git", "add", "--", spec["github_report"]], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"docs: record regional experiment {stage}"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "push"], cwd=repository, check=True)


def summarize(spec: dict) -> dict:
    root = Path(spec["output"])
    classes = _json(root / "prepared.json")["classes"]
    groups = {}
    for path in sorted((root / "scores").glob("*/result.json")):
        item = _json(path)
        key = f"{item['feature']}:{item['head']}:{item['budget']}"
        group = groups.setdefault(
            key, {"folds": [], "confusion": np.zeros((len(classes), len(classes)), np.int64)}
        )
        group["folds"].append(item["fold"])
        group["confusion"] += np.asarray(item["metrics"]["confusion"], np.int64)
    result = {}
    for key, value in groups.items():
        if sorted(value["folds"]) == list(range(5)):
            result[key] = benchmark.classification_metrics(value["confusion"], classes)
    report = {
        "state": "complete" if len(result) == 43 else "partial",
        "classes": classes,
        "class_names": _json(root / "prepared.json").get("class_names", {}),
        "comparisons": result,
        "scope": "pooled held-out predictions across all five spatial folds",
    }
    dump(root / "comparison.json", report)
    return report


def follow(spec_path: Path) -> None:
    spec = read_spec(spec_path)
    root, trial = Path(spec["output"]), Path(spec["trial"])
    root.mkdir(parents=True, exist_ok=True)
    identity = sha(spec_path)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MALLOC_ARENA_MAX="2")
    environment.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    active, completed, exporting = {}, set(), set()
    planned = jobs()
    with (root / "follow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not Path(spec["reference"]).is_file():
            dump(root / "follow-status.json", snapshot(spec, "waiting_for_reference"))
            time.sleep(30)
        benchmark.prepare(spec)
        publish(spec, "data_prepared")
        while True:
            if sha(spec_path) != identity:
                raise ValueError("The registered benchmark specification changed")
            for key, running in list(active.items()):
                code = running["process"].poll()
                if code is None:
                    continue
                if code:
                    dump(
                        root / "follow-status.json",
                        {**snapshot(spec, "failed"), "failed_job": key, "exit_code": code},
                    )
                    publish(spec, "job_failed")
                    raise RuntimeError(f"Registered downstream job failed: {key}")
                del active[key]
                completed.add(key)
                if key.startswith("export:"):
                    publish(spec, key.replace(":", "_"))
            devices = [spec["reserved_device"]]
            for model in ["public", "highres"]:
                model_status = _json(trial / "runs" / model / "status.json")
                if model_status.get("state") == "complete":
                    devices += spec[model + "_devices"]
                else:
                    registration = _json(trial / f"{model}-train-process.json")
                    pid = registration.get("pid")
                    process_path = Path(f"/proc/{pid}/stat")
                    alive = process_path.exists()
                    if alive:
                        alive = process_path.read_text().split(") ", 1)[1].split()[0] != "Z"
                    if pid is not None and not alive:
                        dump(root / "follow-status.json", snapshot(spec, "training_failed"))
                        publish(spec, "training_failed")
                        raise RuntimeError(f"Registered encoder process stopped: {model}")
            busy = {v["device"] for v in active.values() if v["device"] is not None}
            available = [d for d in devices if d not in busy]

            def launch(key, arguments, device=None):
                command = [
                    sys.executable,
                    "-m",
                    "xuannv_embedding.cli",
                    "experiment",
                    "regional-benchmark",
                    "--spec",
                    str(spec_path),
                    *arguments,
                ]
                (root / "logs").mkdir(exist_ok=True)
                with (root / "logs" / (key.replace(":", "_") + ".log")).open("ab") as log:
                    process = subprocess.Popen(
                        command,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                active[key] = {"process": process, "device": device}

            for model in ["public", "highres"]:
                status = _json(trial / "runs" / model / "status.json")
                export_status = _json(root / "representations" / model / "status.json")
                if status.get("state") == "complete" and export_status.get("state") != "complete":
                    key = "export:" + model
                    if key not in active and available:
                        device = available.pop(0)
                        launch(
                            key,
                            ["--phase", "export", "--model", model, "--device", f"npu:{device}"],
                            device,
                        )
                        exporting.add(model)
            for i, job in enumerate(planned):
                key = f"job:{i:03d}"
                if key in active or key in completed:
                    continue
                path = benchmark.feature_path(root, job["feature"])
                if not path.is_file():
                    continue
                result_path = (
                    root
                    / "scores"
                    / (f"{job['feature']}_{job['head']}_" f"fold{job['fold']}_n{job['budget']}")
                    / "result.json"
                )
                if result_path.exists():
                    completed.add(key)
                    continue
                cpu = job["head"] == "rf"
                if cpu and sum(v["device"] is None for v in active.values()) >= 2:
                    continue
                if not cpu and not available:
                    continue
                device = None if cpu else available.pop(0)
                launch(
                    key,
                    [
                        "--phase",
                        "score",
                        "--job-index",
                        str(i),
                        "--device",
                        "cpu" if cpu else f"npu:{device}",
                    ],
                    device,
                )
            for feature in ["aef", "raw_public", "raw_highres", "public", "highres"]:
                key = "regression:" + feature
                if (
                    key not in active
                    and key not in completed
                    and benchmark.feature_path(root, feature).is_file()
                    and sum(v["device"] is None for v in active.values()) < 2
                ):
                    if (root / "regression" / f"{feature}.json").exists():
                        completed.add(key)
                    else:
                        launch(key, ["--phase", "regression", "--model", feature])
            state = snapshot(spec, "evaluating")
            state["active_jobs"] = {k: v["device"] for k, v in active.items()}
            dump(root / "follow-status.json", state)
            if len([k for k in completed if k.startswith("job:")]) == len(planned) and not active:
                summarize(spec)
                publish(spec, "evaluation_complete")
                dump(root / "follow-status.json", snapshot(spec, "complete"))
                return
            time.sleep(30)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv experiment regional-benchmark")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["prepare", "export", "score", "regression", "follow", "report"],
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args(argv)
    spec = read_spec(args.spec)
    if args.phase == "follow":
        follow(args.spec)
    elif args.phase == "prepare":
        benchmark.prepare(spec)
    elif args.phase == "export":
        from xuannv_embedding.export.regional_trial import export

        if args.model not in {"public", "highres"}:
            raise ValueError("Export requires one registered regional model")
        trial, root = Path(spec["trial"]), Path(spec["output"])
        export(
            trial / f"{args.model}-spec.json",
            trial / "runs" / args.model / "final.pt",
            root / "representations" / args.model,
            device_name=args.device,
        )
    elif args.phase == "score":
        if args.job_index is None or not 0 <= args.job_index < len(jobs()):
            raise ValueError("A registered nonnegative downstream job index is required")
        benchmark.score_job(spec, jobs()[args.job_index], device_name=args.device)
    elif args.phase == "regression":
        if args.model not in {"public", "highres", "aef", "raw_public", "raw_highres"}:
            raise ValueError("Regression requires a registered representation")
        benchmark.score_regression(spec, args.model)
    else:
        summarize(spec)
    return 0
