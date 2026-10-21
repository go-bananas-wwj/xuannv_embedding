"""Complete per-target calibration and fixed-reader intervention tables."""

from __future__ import annotations

import json
from pathlib import Path

from xuannv_embedding.export.context import sha

CALIBRATION = {
    "simple_concat": ("直接组合", "Direct combination"),
    "dino_context": ("视觉上下文", "Visual context"),
    "without_geometry": ("无结构补充", "No structural term"),
    "binary_coverage": ("二值覆盖校准", "Binary coverage (cal.)"),
    "aef_highres_guide": ("AEF+地方引导", "AEF+local guide"),
    "aef_public_guide": ("AEF+公共引导", "AEF+public guide"),
    "aef_degraded_guide": ("AEF+降质引导", "AEF+degraded guide"),
}
FROZEN = {
    "no_guide": ("关闭末端", "No terminal guide"),
    "mean_first": ("先合成再引导", "Image-first guide"),
    "coverage50": ("末端覆盖50", "Terminal coverage50"),
    "coverage25": ("末端覆盖25", "Terminal coverage25"),
    "three_spread": ("末端分散三期", "Terminal3 spread"),
    "three_consecutive": ("末端连续三期", "Terminal3 consecutive"),
    "one_observation": ("末端单期", "Terminal1 date"),
    "binary_coverage_matched": ("同批次二值覆盖", "Binary presence"),
    "without_geometry_fixed": ("关闭结构补充", "Structural term off"),
    "aef_guidance_nchw": ("AEF同引导", "AEF same guide"),
}
TARGETS = {
    "clcd_crop": ("耕地", "Cropland"),
    "clcd_forest": ("林地", "Forest"),
    "clcd_water": ("水体", "Water"),
    "clcd_impervious": ("不透水面", "Impervious"),
}


def _longtable(path, caption, rows):
    lines = [
        r"\small\begin{longtable}{lllrrrr}",
        r"\caption{" + caption + r"}\\\toprule",
        r"Condition & Reader & Target & F1 & IoU & AP & BF1\\\midrule\endhead",
        *[" & ".join(row) + r"\\" for row in rows],
        r"\bottomrule\end{longtable}\normalsize",
        "",
    ]
    path.write_text("\n".join(lines))


def build(study, paper):
    study, paper = Path(study), Path(paper)
    root, goal = study / "jstars-revision", study.parent.parent
    source_case = json.loads((root / "guidance-controls/no_guide/case.json").read_text())
    region = source_case["region"]["name"]
    sources, calibration, frozen = [], [], []

    def read(path):
        sources.append({"path": str(path), "sha256": sha(path)})
        return json.loads(path.read_text())

    def append(target, labels, rows, key="metrics"):
        for head in ["mlp", "conv3x3"]:
            selected = [x for x in rows if x["head"] == head]
            if {x["task"] for x in selected} != set(TARGETS) or len(selected) != 4:
                raise ValueError("Control class or reader inventory differs")
            for row in selected:
                target.append(dict(labels=labels, head=head, task=row["task"], metrics=row[key]))

    jobs = read(goal / "waves/viewwise_guided_refinement_v12/jobs.json")
    for role, labels in [
        ("highres", ("原生完整", "Native complete")),
        ("public", ("公共角色", "Public role")),
        ("degraded", ("降质角色", "Degraded role")),
    ]:
        rows = []
        for head in ["mlp", "conv3x3"]:
            job = next(
                x for x in jobs if x["region"] == region and x["role"] == role and x["head"] == head
            )
            path = Path(job["output"]) / "calibration/results.json"
            identity = read(path.with_name("identity.json"))
            if identity["state"] != "complete" or identity["results_sha256"] != sha(path):
                raise ValueError("Calibration control receipt changed")
            rows += read(path)
        append(calibration, labels, rows)
    aef_rows = []
    for head in ["mlp", "conv3x3"]:
        job = read(Path(source_case["region"]["aef_jobs"][head]["path"]))
        aef_rows += read(Path(job["output"]) / "calibration/results.json")
    append(calibration, ("AEF", "AEF"), aef_rows)
    for condition, labels in CALIBRATION.items():
        result = read(study / "comparisons" / condition / region / "result.json")
        if result["state"] != "complete" or not result["calibration_only"]:
            raise ValueError("Expected completed calibration-only control")
        append(calibration, labels, result["rows"])
    matched_path = root / "matched-feature-fusion/comparison/result.json"
    if matched_path.exists():
        matched = read(matched_path)
        if matched["state"] == "complete":
            append(calibration, ("同预训练特征拼接", "Shared-feature concat"), matched["rows"])
    for name, labels in [
        ("selected_method", ("原生完整", "Native complete")),
        ("AEF", ("AEF", "AEF")),
    ]:
        result = read(study / "full-region" / name / "result.json")
        append(frozen, labels, result["rows"], "all_region")
    for condition, labels in FROZEN.items():
        path = root / "guidance-controls" / condition / "evaluation/result.json"
        if not path.exists():
            continue
        result = read(path)
        if result["state"] == "complete":
            append(frozen, labels, result["rows"])
    for condition, labels in [
        ("coverage50", ("完整来源覆盖50", "Whole-source50")),
        ("coverage25", ("完整来源覆盖25", "Whole-source25")),
    ]:
        result = read(root / "raw-missingness" / condition / "evaluation/result.json")
        append(frozen, labels, result["rows"])
    result = read(root / "whole-source-zero/evaluation/result.json")
    append(frozen, ("地方来源全缺失", "Local sources absent"), result["rows"])
    for condition, labels in [
        ("three_spread", ("完整来源分散三期", "Whole-source3 spread")),
        ("three_consecutive", ("完整来源连续三期", "Whole-source3 consecutive")),
        ("one_observation", ("完整来源单期", "Whole-source1 date")),
    ]:
        path = root / "temporal-source" / condition / "evaluation/result.json"
        if path.exists():
            append(frozen, labels, read(path)["rows"])
    for english in [False, True]:
        out = paper / "tables" / ("en" if english else "")
        for name, values, caption in [
            (
                "calibration_controls",
                calibration,
                (
                    "Complete calibration comparisons: all four targets and both readers. "
                    "BF1 uses one reference-cell tolerance; CLCD reference cells are 30m."
                    if english
                    else "完整校准对照：保留四类及两种读取器。"
                    "BF1为单个参考像元容差的边界F1，CLCD参考像元为30米。"
                ),
            ),
            (
                "frozen_controls",
                frozen,
                (
                    "Complete fixed-reader interventions on all-region reference cells. "
                    "Thresholds remain fixed. BF1 uses one reference-cell tolerance (30m for CLCD)."
                    if english
                    else "完整固定读取器干预：全区参考像元，阈值固定；"
                    "BF1采用一个参考像元容差（CLCD为30米）。"
                ),
            ),
        ]:
            rows = []
            for value in values:
                row = [
                    value["labels"][int(english)],
                    "MLP" if value["head"] == "mlp" else "CNN",
                    TARGETS[value["task"]][int(english)],
                ]
                for key in ["f1", "iou", "ap", "boundary_f1_1px"]:
                    score = value["metrics"].get(key)
                    row.append("--" if score is None else f"{100*score:.3f}")
                rows.append(row)
            _longtable(out / ("supplement_" + name + ".tex"), caption, rows)
    ledger = dict(
        sources=sources,
        calibration_rows=len(calibration),
        frozen_rows=len(frozen),
        code_sha256=sha(__file__),
    )
    (paper / "notes/jstars-control-class-ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n"
    )
    return ledger
