"""Render verified validation results into a reviewable, path-free TeX bundle.

This does not publish files, select a winner, or start the next training group.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import yaml

from xuannv_embedding.training.experiment import _json, _sha
from xuannv_embedding.training.multitask_followup import paired_report


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def verified_results(root: Path, proof: dict) -> list[dict]:
    status, identity = load(root / "status.json"), load(root / "identity.json")
    if status.get("state") != "complete" or status.get("conditions_complete") != 220:
        raise ValueError("evaluation is not complete with 220 conditions")
    if any(x.get("test_scored") is not False for x in (status, identity, proof)):
        raise ValueError("validation report cannot contain test scoring")
    if proof["results_sha256"] != _sha(root / "results.json"):
        raise ValueError("verified results changed")
    if proof["conditions_verified"] != 220 or not 0 <= proof["max_abs_difference"] < 1e-10:
        raise ValueError("independent verification did not pass")
    rows = load(root / "results.json")
    expected = {("C", "osm"): 40, ("C", "esri"): 60, ("R", "esri"): 60, ("Q", "osm"): 60}
    if Counter((r["family"], r["source"]) for r in rows) != expected:
        raise ValueError("evaluation condition families are incomplete")
    digests = {r["key"]: r["sha256"] for r in proof["prediction_arrays"]}
    keys = [r["key"] for r in rows]
    if len(set(keys)) != 220 or len(digests) != 220 or set(keys) != set(digests):
        raise ValueError("prediction verification coverage differs")
    for row in rows:
        key = row["key"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise ValueError("invalid prediction key")
        if _sha(root / (key + ".npz")) != digests[key]:
            raise ValueError("verified prediction array changed")
        for value in (row["error"], *row["metrics"].values()):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("nonfinite reported metric")
    return rows


def summarize(rows: list[dict]) -> dict:
    result = {}
    for family, source in (("C", "osm"), ("C", "esri"), ("R", "esri"), ("Q", "osm")):
        selected = [r for r in rows if (r["family"], r["source"]) == (family, source)]
        metrics = {
            "C": ("ap", "f1", "iou", "ba"),
            "R": ("rmse", "mae", "r2", "bias"),
            "Q": ("ap", "precision_top0.01", "recall_top0.01"),
        }[family]
        summary = {}
        for metric in metrics:
            values = [r["metrics"][metric] for r in selected if r["metrics"][metric] is not None]
            summary[metric] = mean(values) if values else None
            summary[metric + "_defined_conditions"] = len(values)
        result[f"{family}_{source}"] = summary
    return result


def tex(value: str) -> str:
    escapes = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(escapes.get(c, c) for c in value)


def render_pair(
    baseline: Path,
    candidate: Path,
    baseline_proof: dict,
    candidate_proof: dict,
    *,
    name: str,
    output: Path,
    training: dict | None = None,
    reference: Path | None = None,
    reference_proof: dict | None = None,
) -> None:
    if output.exists():
        raise FileExistsError("report staging directory already exists")
    if name == "B0":
        raise ValueError("candidate must have a distinct name")
    rows = {
        "B0": verified_results(baseline, baseline_proof),
        name: verified_results(candidate, candidate_proof),
    }
    paired = paired_report(baseline, candidate)
    roots = [("B0", baseline, baseline_proof), (name, candidate, candidate_proof)]
    reference_pair = None
    if reference is not None:
        if name == "T0" or reference_proof is None:
            raise ValueError("T0 reference requires a distinct candidate and verification")
        rows["T0"] = verified_results(reference, reference_proof)
        reference_pair = paired_report(baseline, reference)
        roots.append(("T0", reference, reference_proof))
    summaries = {key: summarize(value) for key, value in rows.items()}
    provenance = {}
    for label, root, proof in roots:
        identity = load(root / "identity.json")
        provenance[label] = {
            key: identity[key]
            for key in ("protocol", "code_commit", "spec_sha256", "cache_sha256", "test_scored")
        }
        provenance[label].update(
            results_sha256=_sha(root / "results.json"),
            identity_sha256=_sha(root / "identity.json"),
            elapsed_seconds=load(root / "status.json")["elapsed_seconds"],
            verification={k: v for k, v in proof.items() if k != "prediction_arrays"},
        )
    summary = {
        "state": "complete",
        "split": "validation",
        "test_scored": False,
        "model": name,
        "paired": paired,
        "summaries": summaries,
        "provenance": provenance,
        "training": training,
        "publication_state": "staged_for_review",
    }
    if reference_pair:
        summary["reference_T0"] = reference_pair
        summary["score_difference_vs_T0"] = (
            paired["normalized_score"]["score"] - reference_pair["normalized_score"]["score"]
        )
    lines = [
        r"\section{" + tex(name) + "：多任务验证结果}",
        r"\paragraph{状态。}验证评价完成，测试集未评分。本记录不构成多种子稳定性或跨地区泛化结论。",
        r"\paragraph{固定协议。}multitask-v3；每个表示220个条件，包含100项分类、60项地图覆盖比例回归和60项相似地物检索。"
        "B0与候选使用相同支持位置及验证参考域；指标从保存预测独立复算，并复核预测文件哈希。",
        r"\begin{center}\begin{tabular}{lrrr}\toprule 指标 & B0 & "
        + tex(name)
        + r" & 候选减B0 \\\midrule",
    ]
    display = [
        ("C_osm", "ap", r"OSM分类AP／\%", 100),
        ("C_esri", "ap", r"ESRI分类AP／\%", 100),
        ("R_esri", "rmse", "覆盖比例RMSE", 1),
        ("R_esri", "mae", "覆盖比例MAE", 1),
        ("R_esri", "r2", r"覆盖比例$R^2$", 1),
        ("Q_osm", "ap", r"地物检索AP／\%", 100),
    ]
    for family, metric, label, scale in display:
        a, b = (summaries[key][family][metric] for key in ("B0", name))
        values = ["--" if v is None else f"{v * scale:.4f}" for v in (a, b)]
        difference = "--" if a is None or b is None else f"{(b-a)*scale:+.4f}"
        lines.append(label + " & " + " & ".join([*values, difference]) + r" \\")
    score = paired["normalized_score"]
    lines += [
        r"\bottomrule\end{tabular}\end{center}",
        r"\paragraph{筛选分数。}$S="
        + f"{score['score']:+.6f}"
        + "$；"
        + "，".join(f"{k}族 {v:+.6f}" for k, v in score["families"].items())
        + "。"
        "正值为相对B0的平均归一化误差收益，负值为退步；不能解释为精度百分比。"
        "AP差值单位为百分点；RMSE、MAE及其差值为类别覆盖比例单位，越低越好。",
        r"\paragraph{来源与限制。}OSM与训练监督同源；ESRI为地图参考，覆盖比例不是独立实测的生物物理变量。"
        "未定义的指标记为--，定义条件数保存在summary.json。少数类及逐条件结果完整保留在conditions.csv。",
        r"\paragraph{待完成。}严格遮挡重建、逐月缺源诊断、共同解码器、多种子复验和AEF新协议比较另行记录；"
        "不能从本表推断这些项目已经完成。",
    ]
    if reference_pair:
        lines += [
            r"\paragraph{与同预算原设置T0配对。}$\Delta S="
            + f"{summary['score_difference_vs_T0']:+.6f}"
            + "$；保持以B0误差归一化，再作候选减T0，不更换分母。",
            r"\begin{center}\begin{tabular}{lrrr}\toprule 指标 & T0 & "
            + tex(name)
            + r" & 候选减T0 \\\midrule",
        ]
        for family, metric, label, scale in display:
            a, b = (summaries[key][family][metric] for key in ("T0", name))
            values = ["--" if v is None else f"{v * scale:.4f}" for v in (a, b)]
            difference = "--" if a is None or b is None else f"{(b-a)*scale:+.4f}"
            lines.append(label + " & " + " & ".join([*values, difference]) + r" \\")
        lines.append(r"\bottomrule\end{tabular}\end{center}")
    if training:
        lines += [
            r"\paragraph{训练与追溯。}种子"
            + str(training["seed"])
            + "，"
            + str(training["epoch"])
            + " Epoch，实际"
            + str(training["optimizer_updates"])
            + "次更新；训练代码\\texttt{"
            + training["code_commit"][:12]
            + "}。"
            "配置、父权重、checkpoint、核验记录哈希及资源用量见summary.json。"
        ]
        if "parameters" in training:
            parameters = training["parameters"]["training"]
            lines.append(
                r"\paragraph{本组参数。}学习率"
                + str(parameters["lr"])
                + "，weight decay "
                + str(parameters["weight_decay"])
                + "，语义权重"
                + str(parameters["semantic_probe_weight"])
                + "；完整遮挡、调度、损失权重配置见summary.json的parameters字段。"
            )
    output.mkdir(parents=True)
    _json(output / "summary.json", summary)
    with (output / "conditions.csv").open("w") as stream:
        keys = [
            "model",
            "family",
            "source",
            "task",
            "seed",
            "budget",
            "error",
            "ap",
            "f1",
            "iou",
            "ba",
            "rmse",
            "mae",
            "r2",
            "bias",
            "precision_top0.01",
            "recall_top0.01",
            "precision_top0.05",
            "recall_top0.05",
        ]
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for model, records in rows.items():
            for row in records:
                all_values = {"model": model, **row, **row["metrics"]}
                writer.writerow({k: all_values.get(k, "") for k in keys})
    (output / "result.tex").write_text("\n".join(lines) + "\n")


def run(args) -> None:
    plan = load(args.plan)
    root = Path(plan["output"])
    if load(root / "status.json").get("state") != "ready_to_publish":
        raise ValueError("followup is not ready_to_publish")
    if load(root / "controller.json")["plan_sha256"] != _sha(args.plan):
        raise ValueError("controller plan changed")
    audit = load(root / "checkpoint_verification.json")
    directory = Path(plan["training_directory"])
    registration, status = load(directory / "run.json"), load(directory / "status.json")
    checkpoint = directory / f"epoch_{plan['epochs']:04d}.pt"
    if status["state"] != "complete" or status["epoch"] != plan["epochs"]:
        raise ValueError("training terminal status differs")
    if audit["epoch"] != plan["epochs"] or audit["actual_optimizer_steps"] != [plan["steps"]]:
        raise ValueError("checkpoint verification budget differs")
    if audit["checkpoint_sha256"] != _sha(checkpoint):
        raise ValueError("verified checkpoint changed")
    if (
        registration["git_sha"] != plan["code_sha"]
        or registration["config_sha256"] != plan["config_sha256"]
    ):
        raise ValueError("training registration changed")
    if _sha(Path(plan["config"])) != plan["config_sha256"]:
        raise ValueError("configuration changed")
    candidate = root / "validation" / plan["run_id"]
    baseline = Path(plan["baseline"])
    identity = load(candidate / "identity.json")
    manifest = load(root / "export/manifest.json")
    if identity["feature_identity"]["manifest_sha256"] != _sha(root / "export/manifest.json"):
        raise ValueError("evaluated export identity differs")
    if (
        manifest["checkpoint_sha256"] != audit["checkpoint_sha256"]
        or manifest["checkpoint_epoch"] != plan["epochs"]
    ):
        raise ValueError("export checkpoint differs")
    paired = paired_report(baseline, candidate)
    if paired != load(root / "paired_summary.json"):
        raise ValueError("paired summary changed")
    if _sha(baseline / "results.json") != plan["baseline_results_sha256"]:
        raise ValueError("registered baseline results changed")
    training = {
        "epoch": plan["epochs"],
        "optimizer_updates": plan["steps"],
        "seed": registration["seed"],
        "code_commit": registration["git_sha"],
        "config_sha256": registration["config_sha256"],
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "parent_checkpoint_sha256": registration["adaptation"]["base_checkpoint_sha256"],
        "elapsed_seconds": status["elapsed_seconds"],
        "peak_memory_bytes": status["peak_memory_bytes"],
        "world_size": registration["world_size"],
        "nominal_global_batch_size": registration["global_batch_size"],
        "plan_sha256": _sha(args.plan),
        "checkpoint_verification_sha256": _sha(root / "checkpoint_verification.json"),
    }
    configuration = yaml.safe_load(Path(plan["config"]).read_text())
    training["parameters"] = {
        "training": configuration["training"],
        "target_heads": configuration["model"]["target_heads"],
    }
    reference, reference_proof = None, None
    if plan["run_id"] != "T0":
        if args.reference_followup is None:
            raise ValueError("training candidate report requires same-budget T0 reference")
        reference_root = args.reference_followup
        if load(reference_root / "status.json")["state"] != "ready_to_publish":
            raise ValueError("T0 reference is not ready_to_publish")
        reference_audit = load(reference_root / "checkpoint_verification.json")
        if any(reference_audit[k] != audit[k] for k in ("epoch", "actual_optimizer_steps")):
            raise ValueError("T0 reference has a different training budget")
        reference = reference_root / "validation/T0"
        reference_proof = load(reference_root / "verification.json")["T0"]
    render_pair(
        baseline,
        candidate,
        load(args.baseline_verification)["B0"],
        load(root / "verification.json")[plan["run_id"]],
        name=plan["run_id"],
        output=args.output,
        training=training,
        reference=reference,
        reference_proof=reference_proof,
    )
