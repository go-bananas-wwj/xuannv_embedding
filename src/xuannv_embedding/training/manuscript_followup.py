"""Identical downstream contracts for new paired models and frozen controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from xuannv_embedding.data_process.retention import apply_manifest, file_identity
from xuannv_embedding.downstream import full_region_readouts as full
from xuannv_embedding.downstream.manuscript_evidence import paired_statistics
from xuannv_embedding.export import manuscript_controls, manuscript_model
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.frozen_fusion import verified
from xuannv_embedding.training.shared_fusion_search import reader_jobs


def score(path):
    spec = json.loads(path.read_text())
    keys = {"schema", "region", "case", "producer", "output", "retire_arrays"}
    if set(spec) != keys or spec["schema"] != "manuscript-followup-v1":
        raise ValueError("Invalid self-contained downstream follow-up")
    region, case = spec["region"], spec["case"]
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "result.json").exists():
        old_result = json.loads((root / "result.json").read_text())
        if old_result.get("spec_sha256") != sha(path):
            raise ValueError("Completed comparison specification changed")
        for job in old_result["jobs"]:
            directory = Path(job["output"]) / "calibration"
            identity = json.loads((directory / "identity.json").read_text())
            if identity["implementation"] != full.code() or identity["results_sha256"] != sha(
                directory / "results.json"
            ):
                raise ValueError("Completed comparison results changed")
            for row in json.loads((directory / "results.json").read_text()):
                if (
                    sha(directory / "predictions/candidate" / (row["key"] + ".npz"))
                    != row["prediction_sha256"]
                ):
                    raise ValueError("Completed prediction changed")
        return old_result
    producer = spec["producer"]
    if producer["kind"] == "trained":
        status = manuscript_model.export(
            verified(producer["spec"]), producer["phase"], "npu:0", fraction=producer["coverage"]
        )
        feature = verified(status["features"])
    elif producer["kind"] == "control":
        status = manuscript_controls.export(verified(producer["spec"]), "npu:0")
        feature = verified(status["features"])
    elif producer["kind"] == "frozen":
        feature = verified(producer["features"])
    else:
        raise ValueError("Unregistered feature producer")
    jobs = reader_jobs(
        {"output": str(root / "readers")},
        region,
        {"name": case, "kind": "registered_manuscript_comparison", "producer": producer},
        feature,
    )
    results = []
    for job in jobs:
        full.main(["--spec", job["spec"], "--phase", "calibration"])
        sp = json.loads(Path(job["spec"]).read_text())
        data = json.loads(verified(sp["data"]).read_text())
        reference = json.loads(verified(region["evaluation"]).read_text())
        for key in [
            "valid",
            "labels",
            "indices",
            "records",
            "calibration_positions",
            "train_positions",
            "calibration_cohort",
        ]:
            if data[key] != reference[key]:
                raise ValueError("Downstream geography, targets or cohort changed")
        directory = Path(job["output"]) / "calibration"
        identity = json.loads((directory / "identity.json").read_text())
        if identity["implementation"] != full.code() or identity["results_sha256"] != sha(
            directory / "results.json"
        ):
            raise ValueError("Readout implementation or results changed")
        rows = json.loads((directory / "results.json").read_text())
        for row in rows:
            record = json.loads((directory / "records" / (row["key"] + ".json")).read_text())
            anchor = json.loads(verified(region["aef_jobs"][job["head"]]).read_text())
            original = Path(anchor["output"]) / "calibration/records" / (row["key"] + ".json")
            old = json.loads(original.read_text())
            if (
                record["support"] != old["support"]
                or row["metrics"]["observations"] != old["row"]["metrics"]["observations"]
            ):
                raise ValueError("Support labels, pixels, positions or AEF scoring scope differ")
            candidate_prediction = directory / "predictions" / sp["model"] / (row["key"] + ".npz")
            baseline_prediction = (
                Path(anchor["output"])
                / "calibration/predictions"
                / anchor["model"]
                / (row["key"] + ".npz")
            )
            if (
                sha(candidate_prediction) != row["prediction_sha256"]
                or sha(baseline_prediction) != old["row"]["prediction_sha256"]
            ):
                raise ValueError("Saved paired prediction identity changed")
            index = json.loads(verified(region["index"]).read_text())
            with np.load(candidate_prediction) as first, np.load(baseline_prediction) as second:
                evidence = paired_statistics(
                    first,
                    second,
                    row["metrics"]["threshold"],
                    old["row"]["metrics"]["threshold"],
                    data["calibration_positions"],
                    index["records"],
                    data["size"],
                )
            expected_difference = row["metrics"]["f1"] - old["row"]["metrics"]["f1"]
            if abs(evidence["f1_difference"] - expected_difference) > 1e-10:
                raise ValueError("Frozen thresholds do not reproduce saved F1 counts")
            dump(root / (row["key"] + "-evidence.json"), evidence)
            results.append({"case": case, "region": region["name"], "head": job["head"], **row})
    result = {
        "state": "complete",
        "spec_sha256": sha(path),
        "rows": results,
        "jobs": jobs,
        "all_support_and_scored_positions_match_AEF": True,
        "calibration_only": True,
        "scientific_goal_complete": False,
    }
    dump(root / "result.json", result)
    if spec["retire_arrays"]:
        files = [feature] if producer["kind"] in {"control", "trained"} else []
        files += [Path(job["spec"]).parent / "evaluation-features.npy" for job in jobs]
        files = sorted(set(files))
        manifest = {
            "protocol": "artifact-retention-v1",
            "allowed_roots": (
                [str(root), str(feature.parent)]
                if producer["kind"] in {"control", "trained"}
                else [str(root)]
            ),
            "protected": [str(root / "result.json")],
            "requirements": [file_identity(root / "result.json")],
            "files": [
                {
                    **file_identity(f),
                    "role": "derived_features",
                    "reason": "completed control; retained source and recipe permit exact rebuild",
                }
                for f in files
                if f.exists()
            ],
        }
        ledger = root / "derived-array-retention.json"
        if not (root / "deletions.jsonl").exists():
            dump(ledger, manifest)
            dump(root / "retention-result.json", apply_manifest(ledger, root / "deletions.jsonl"))
    return result


def summarize(path):
    spec = json.loads(path.read_text())
    if (
        set(spec) != {"schema", "prior", "followups", "output"}
        or spec["schema"] != "manuscript-summary-v1"
    ):
        raise ValueError("Invalid summary registration")
    prior = json.loads(verified(spec["prior"]).read_text())
    records, scores = [], {}
    for entry in spec["followups"]:
        f = json.loads(verified(entry).read_text())
        result_path = Path(f["output"]) / "result.json"
        result = json.loads(result_path.read_text())
        if result["state"] != "complete" or result["spec_sha256"] != sha(Path(entry["path"])):
            raise ValueError("Incomplete registered comparison")
        records.extend(result["rows"])
        for head in ["mlp", "conv3x3"]:
            rows = [r for r in result["rows"] if r["head"] == head]
            scores.setdefault(f["case"], {}).setdefault(f["region"]["name"], {})[head] = float(
                np.mean([r["metrics"]["f1"] for r in rows])
            )
    aef = prior["scores"]["aef"]["reference"]
    reference = prior["reference"]
    public = scores["original_public"]
    candidates = {
        "original_highres": scores["original_highres"],
        "viewwise075": prior["scores"]["viewwise075"]["highres"],
    }
    gates = []
    for name, values in candidates.items():
        gaps, means = [], {}
        for region in public:
            for head in ["mlp", "conv3x3"]:
                floor = max(
                    reference[region][head]["public"], public[region][head], aef[region][head]
                )
                gaps.append(values[region][head] - floor)
            means[region] = float(
                np.mean([values[region][h] - aef[region][h] for h in ["mlp", "conv3x3"]])
            )
        gates.append(
            {
                "case": name,
                "minimum_gain": min(gaps),
                "mean_gain": float(np.mean(gaps)),
                "regional_AEF_gains": means,
                "eligible": min(gaps) > 0 and min(means.values()) >= 0.005,
            }
        )
    eligible = [g for g in gates if g["eligible"]]
    selected = (
        max(eligible, key=lambda g: (g["minimum_gain"], g["mean_gain"]))["case"]
        if eligible
        else None
    )
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "state": "validation_complete",
        "scores": scores,
        "gates": gates,
        "selected_for_query": selected,
        "primary_references_unchanged": True,
        "all_task_rows": records,
        "scientific_goal_complete": False,
        "next_step": (
            "conditional ablations and frozen queries"
            if selected
            else "stop expansion and report all positive/negative findings"
        ),
    }
    dump(out / "result.json", result)
    with (out / "all-task-results.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["case", "region", "head", "task", "f1", "iou", "ap"]
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    **{k: row[k] for k in ["case", "region", "head", "task"]},
                    **{k: row["metrics"][k] for k in ["f1", "iou", "ap"]},
                }
            )
    lines = [
        "# 原稿架构与逐期融合：统一校准比较",
        "",
        "本轮采用固定的区域制图校准，不属于盲测。",
        "",
        "| 方法 | 区域 | MLP F1（%） | CNN F1（%） |",
        "| --- | --- | ---: | ---: |",
    ]
    for case, regions in {**scores, "viewwise075": candidates["viewwise075"], "AEF": aef}.items():
        for region, heads in regions.items():
            lines.append(
                f"| {case} | {region} | {100*heads['mlp']:.3f} | {100*heads['conv3x3']:.3f} |"
            )
    lines += [
        "",
        f"通过共同门槛的方法：{selected or '暂无'}。",
        "未通过则停止本轮扩展；不能把软件检查或平均提升写成全面胜出。",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["score", "summary"], required=True)
    args = parser.parse_args(argv)
    score(args.spec) if args.phase == "score" else summarize(args.spec)
    return 0
