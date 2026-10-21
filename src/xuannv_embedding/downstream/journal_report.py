"""Build journal tables and their source ledger from completed experiment receipts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xuannv_embedding.export.context import sha


def _load(path):
    return json.loads(Path(path).read_text())


def _table(path, caption, label, columns, header, rows):
    text = [
        r"\begin{table*}[t]\centering\small",
        r"\caption{" + caption + "}\\label{" + label + "}",
        r"\begin{tabular}{" + columns + r"}\toprule",
        " & ".join(header) + r"\\\midrule",
    ]
    text += [" & ".join(row) + r"\\" for row in rows]
    text += [r"\bottomrule\end{tabular}\end{table*}", ""]
    path.write_text("\n".join(text))


def build(root, paper):
    root, paper = Path(root), Path(paper)
    out = paper / "tables"
    out.mkdir(exist_ok=True)
    sources = []

    def read(path):
        path = Path(path)
        sources.append({"path": str(path), "sha256": sha(path)})
        return _load(path)

    full = {
        "AEF": read(root / "full-region/AEF/result.json"),
        "区域三源模型": read(root / "comparisons/original_public/haidian/frozen-query/result.json"),
        "同方法公共条件": read(root / "jstars-revision/public-full-region/result.json"),
        "完整逐期融合": read(root / "full-region/selected_method/result.json"),
    }
    rows = []
    for name, result in full.items():
        values = [name]
        for scope in ["all_region", "outside_support_and_calibration"]:
            for head in ["mlp", "conv3x3"]:
                a = [x[scope]["f1"] for x in result["rows"] if x["head"] == head]
                assert len(a) == 4
                values.append(f"{100*np.mean(a):.3f}")
        rows.append(values)
    _table(
        out / "jstars_region.tex",
        "海淀CLCD2025四类平均F1（百分数）。各条件复用自己的校准分类头与阈值。全区覆盖320图块，另列排除支持及校准图块后的结果；编码器采用区域内训练。",
        "tab:jstars-region",
        "lrrrr",
        ["表示", "全区MLP", "全区CNN", "查询范围MLP", "查询范围CNN"],
        rows,
    )
    tasks = ["clcd_crop", "clcd_forest", "clcd_water", "clcd_impervious"]
    names = ["耕地", "林地", "水体", "不透水面"]
    rows = []
    for task, name in zip(tasks, names):
        values = [name]
        for head in ["mlp", "conv3x3"]:
            a = next(
                x["all_region"]["f1"]
                for x in full["AEF"]["rows"]
                if x["head"] == head and x["task"] == task
            )
            b = next(
                x["all_region"]["f1"]
                for x in full["完整逐期融合"]["rows"]
                if x["head"] == head and x["task"] == task
            )
            values.extend([f"{100*a:.2f}", f"{100*b:.2f}", f"{100*(b-a):+.2f}"])
        rows.append(values)
    _table(
        out / "jstars_classes.tex",
        "全区逐类F1及完整逐期融合相对AEF的差值。F1为百分数，差值为百分点。",
        "tab:jstars-classes",
        "lrrrrrr",
        ["类别", "AEF MLP", "融合MLP", "差值", "AEF CNN", "融合CNN", "差值"],
        rows,
    )
    prior = read(root / "prior-haidian.json")
    controls = read(root / "report/completed-control-means.json")
    rows = []
    for role, name in [
        ("highres", "原生地方观测"),
        ("public", "同方法公共观测"),
        ("degraded", "同景降质地方观测"),
    ]:
        x = prior["scores"]["viewwise075"][role]["haidian"]
        rows.append([name, f'{100*x["mlp"]:.3f}', f'{100*x["conv3x3"]:.3f}'])
    labels = {
        "simple_concat": "直接拼接与PCA",
        "dino_context": "视觉上下文对照",
        "without_geometry": "去除额外几何补充",
        "binary_coverage": "二值覆盖",
        "aef_highres_guide": "AEF与相同地方引导",
        "aef_public_guide": "AEF与公共RGB引导",
        "aef_degraded_guide": "AEF与降质地方引导",
    }
    for key, name in labels.items():
        x = controls[key]
        rows.append([name, f'{100*x["mlp"]:.3f}', f'{100*x["conv3x3"]:.3f}'])
    _table(
        out / "jstars_mechanisms.tex",
        "固定校准位置下的融合方案比较，四类平均F1（百分数）。本表与全区冻结查询分别报告。同方法原生、公共及降质条件用于分析观测贡献，其他条件用于检验方案组成和通用引导。",
        "tab:jstars-mechanisms",
        "lrr",
        ["条件", "MLP", "CNN"],
        rows,
    )
    completed = []
    rows = []
    cases = {
        "no_guide": "关闭末端引导",
        "mean_first": "先汇总再引导",
        "coverage50": "保留50\\%引导覆盖",
        "coverage25": "保留25\\%引导覆盖",
        "three_spread": "三份分散观测引导",
        "three_consecutive": "三份连续观测引导",
        "one_observation": "一份观测引导",
    }
    for case, name in cases.items():
        path = root / "jstars-revision/guidance-controls" / case / "evaluation/result.json"
        if not path.exists():
            continue
        result = read(path)
        if result["state"] != "complete":
            continue
        completed.append(case)
        values = [name]
        for head in ["mlp", "conv3x3"]:
            x = [r["metrics"]["f1"] for r in result["rows"] if r["head"] == head]
            assert len(x) == 4
            values.append(f"{100*np.mean(x):.3f}")
        rows.append(values)
    if rows:
        _table(
            out / "jstars_guidance.tex",
            "固定完整方案分类头与阈值的末端引导干预，报告全区四类平均F1（百分数）。上游区域表示固定，覆盖与观测期变化作用于末端引导。",
            "tab:jstars-guidance",
            "lrr",
            ["引导条件", "MLP", "CNN"],
            rows,
        )
    source_coverage = []
    for fraction, relative in [
        (0.5, "raw-missingness/coverage50/evaluation/result.json"),
        (0.25, "raw-missingness/coverage25/evaluation/result.json"),
        (0.0, "whole-source-zero/evaluation/result.json"),
    ]:
        path = root / "jstars-revision" / relative
        if path.exists():
            result = read(path)
            if result["state"] == "complete":
                source_coverage.append((fraction, result))
    if source_coverage:

        def means(rows, key):
            return [
                f"{100*np.mean([x[key]['f1'] for x in rows if x['head']==h]):.3f}"
                for h in ["mlp", "conv3x3"]
            ]

        rows = [["100" + chr(92) + "%"] + means(full["完整逐期融合"]["rows"], "all_region")]
        for fraction, result in source_coverage:
            rows.append([f"{100*fraction:.0f}" + chr(92) + "%"] + means(result["rows"], "metrics"))
        _table(
            out / "jstars_source_coverage.tex",
            "完整地方来源缺测的固定模型与固定分类头评价。覆盖比例相对原有有效观测计算；原始影像与派生特征同步处理，模型、PCA及阈值固定。",
            "tab:jstars-source-coverage",
            "lrr",
            ["地方覆盖保留", "MLP", "CNN"],
            rows,
        )
    ledger = {
        "method": "viewwise075",
        "sources": sources,
        "full_region_methods": list(full),
        "complete_guidance_cases": completed,
        "complete_raw_source_fractions": [x[0] for x in source_coverage],
        "pending_guidance_cases": [x for x in cases if x not in completed],
        "historical_manuscript_metrics_reassigned": False,
    }
    (paper / "notes/jstars-evidence-ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n"
    )
    return ledger
