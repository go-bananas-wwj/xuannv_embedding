"""Figures and paired block uncertainty for the fixed product comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from xuannv_embedding.downstream.product_bootstrap import run_all
from xuannv_embedding.export.context import dump, sha

MODELS = ("xuannv", "AlphaEarth", "raw")
COLORS = ("#267e82", "#7253a0", "#bc7533")


def paired_intervals(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("paired task/seed/block shapes differ")
    rng = np.random.default_rng(20260921)
    indices = rng.integers(a.shape[1], size=(2000, a.shape[1]))
    result = {}
    for name in ("f1", "iou"):
        means = []
        for x in (a, b):
            counts = x[:, indices].sum(2)
            tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
            values = (
                (2 * tp / np.maximum(1, 2 * tp + fp + fn))
                if name == "f1"
                else (tp / np.maximum(1, tp + fp + fn))
            )
            means.append(values.mean(0))
        result[name] = np.percentile(means[1] - means[0], [2.5, 97.5]).tolist()
    return result


def run(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    spec = json.loads(args.spec.read_text())
    root = Path(spec["output"])
    if not (root / "complete.json").exists():
        raise ValueError("comparison has unfinished jobs")
    run_all(root, spec["workers"], spec["threads"])
    identity = json.loads((root / "identity.json").read_text())
    out = root / "report"
    out.mkdir(exist_ok=True)
    rows = []
    for p in sorted((root / "runs").glob("*/*/*.json")):
        r = json.loads(p.read_text())
        if "metrics" in r:
            rows.append(r)
    groups = {}
    for r in rows:
        key = (r["family"], r["task"], r["head"], r["budget"], r["model"])
        groups.setdefault(key, []).append(r)
    for values in groups.values():
        values.sort(key=lambda r: r["seed"])
        if [r["seed"] for r in values] != spec["seeds"]:
            raise ValueError("missing or duplicated support draws")
    summary = []
    families = ("OSM", "ESRI")
    for family in families:
        tasks = [t for t, d in identity["tasks"].items() if d["family"] == family]
        for head, budgets in (("ridge", spec["budgets"]), ("rf", [5]), ("svm", [5])):
            for budget in budgets:
                eligible = [
                    t for t in tasks if all((family, t, head, budget, m) in groups for m in MODELS)
                ]
                for task in eligible + ["macro"]:
                    chosen = eligible if task == "macro" else [task]
                    item = {
                        "family": family,
                        "task": task,
                        "head": head,
                        "budget": budget,
                        "tasks": chosen,
                        "models": {},
                        "differences": {},
                    }
                    gathered = {}
                    for model in MODELS:
                        subset = [
                            r for t in chosen for r in groups[(family, t, head, budget, model)]
                        ]
                        gathered[model] = subset
                        item["models"][model] = {
                            k: float(np.mean([r["metrics"][k] for r in subset]))
                            for k in ("f1", "ap", "iou")
                        }
                    for reference in ("AlphaEarth", "raw"):
                        a, b = gathered[reference], gathered["xuannv"]
                        for ra, rb in zip(a, b, strict=True):
                            if any(
                                ra[k] != rb[k]
                                for k in (
                                    "seed",
                                    "task",
                                    "sample_positions_sha256",
                                    "support_label_sha256",
                                )
                            ):
                                raise ValueError("unpaired support schedule")
                        item["differences"][reference] = {
                            "point": {
                                k: item["models"]["xuannv"][k] - item["models"][reference][k]
                                for k in ("f1", "ap", "iou")
                            },
                            "ci95": paired_intervals(
                                [r["metrics"]["block_counts"] for r in a],
                                [r["metrics"]["block_counts"] for r in b],
                            ),
                        }
                        ap_draws = {}
                        for m, rr in ((reference, a), ("xuannv", b)):
                            ap_draws[m] = np.mean(
                                [
                                    np.load(
                                        root
                                        / "ap_bootstrap"
                                        / r["task"]
                                        / r["head"]
                                        / f"{r['model']}_{r['seed']}_{r['budget']}.npy"
                                    )
                                    for r in rr
                                ],
                                axis=0,
                            )
                        item["differences"][reference]["ci95"]["ap"] = np.percentile(
                            ap_draws["xuannv"] - ap_draws[reference], [2.5, 97.5]
                        ).tolist()
                    summary.append(item)
    dump(out / "summary.json", summary)
    with (out / "all_metrics.csv").open("w") as f:
        fields = [
            "family",
            "task",
            "head",
            "budget",
            "model",
            "seed",
            "f1",
            "ap",
            "iou",
            "selected_parameter",
            "selected_fit_seconds",
            "validation_seconds",
            "test_seconds",
            "labeled_pixels",
            "fitted_pixels",
            "kernel_backend",
            "implementation_sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            r = {
                **r,
                "kernel_backend": r.get(
                    "kernel_backend", "numpy_float64" if r["head"] == "svm" else "sklearn"
                ),
            }
            writer.writerow(
                {k: r["metrics"][k] if k in ("f1", "ap", "iou") else r[k] for k in fields}
            )
    lookup = {(r["family"], r["task"], r["head"], r["budget"]): r for r in summary}
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )

    def save(fig, name):
        for suffix in ("pdf", "png"):
            fig.savefig(out / f"{name}.{suffix}", dpi=190, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), layout="constrained")
    for j, family in enumerate(families):
        tasks = [t for t, d in identity["tasks"].items() if d["family"] == family]
        x = np.arange(len(tasks))
        for i, model in enumerate(MODELS):
            axes[j, 0].bar(
                x + (i - 1) * 0.24,
                [100 * lookup[(family, t, "ridge", 5)]["models"][model]["f1"] for t in tasks],
                0.24,
                label=model,
                color=COLORS[i],
            )
        axes[j, 0].set_xticks(
            x, [identity["tasks"][t]["name"] for t in tasks], rotation=30, ha="right"
        )
        axes[j, 0].set_title(f"{family}: five-tile Ridge F1")
        axes[j, 0].set_ylabel("F1 (%)")
        axes[j, 0].legend(frameon=False)
        for i, ref in enumerate(("AlphaEarth", "raw")):
            ds = [lookup[(family, t, "ridge", 5)]["differences"][ref] for t in tasks]
            vals = np.array([d["point"]["f1"] for d in ds]) * 100
            bounds = np.array([d["ci95"]["f1"] for d in ds]) * 100
            yy = x + (i - 0.5) * 0.2
            axes[j, 1].hlines(yy, bounds[:, 0], bounds[:, 1], color=COLORS[i + 1])
            axes[j, 1].scatter(vals, yy, label=f"xuannv - {ref}", color=COLORS[i + 1], s=18)
        axes[j, 1].axvline(0, color="gray", lw=0.8)
        axes[j, 1].set_yticks(x, [identity["tasks"][t]["name"] for t in tasks])
        axes[j, 1].set_xlabel("F1 difference (percentage points); paired 95% CI")
        axes[j, 1].legend(frameon=False, fontsize=9)
    save(fig, "product_tasks")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for j, family in enumerate(families):
        for k, metric in enumerate(("f1", "ap")):
            for model, color in zip(MODELS, COLORS, strict=True):
                values = [
                    100 * lookup[(family, "macro", "ridge", b)]["models"][model][metric]
                    for b in spec["budgets"]
                ]
                axes[j, k].plot(spec["budgets"], values, "o-", label=model, color=color)
            axes[j, k].set_xticks(spec["budgets"])
            axes[j, k].set_xlabel("Fully labeled support tiles")
            axes[j, k].set_ylabel(metric.upper() + " (%)")
            axes[j, k].set_title(f"{family}: macro {metric.upper()}")
            axes[j, k].legend(frameon=False)
    save(fig, "product_budget")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    heads = ("ridge", "rf", "svm")
    for j, family in enumerate(families):
        for k, metric in enumerate(("f1", "ap")):
            for i, model in enumerate(MODELS):
                axes[j, k].bar(
                    np.arange(3) + (i - 1) * 0.23,
                    [100 * lookup[(family, "macro", h, 5)]["models"][model][metric] for h in heads],
                    0.23,
                    color=COLORS[i],
                    label=model,
                )
            axes[j, k].set_xticks(range(3), ["Ridge", "Random forest", "RBF-SVM"])
            axes[j, k].set_title(f"{family}: {metric.upper()}, five support tiles")
            axes[j, k].set_ylabel(metric.upper() + " (%)")
            axes[j, k].legend(frameon=False)
    save(fig, "product_heads")
    tasks = ("osm_building", "osm_road", "osm_water", "osm_green")
    fig, axes = plt.subplots(4, 4, figsize=(9, 9), layout="constrained")
    cmap = ListedColormap(["#f3f3f3", "#208b69", "#c94878", "#347bbb"])
    for j, task in enumerate(tasks):
        if not identity["rois"][task]:
            for ax in axes[j]:
                ax.set_axis_off()
            continue
        index = identity["rois"][task][0]
        y = np.load(root / "prepared" / f"label_{task}.npy")[index]
        axes[j, 0].imshow(y == 1, cmap=ListedColormap(["#f3f3f3", "#208b69"]), vmin=0, vmax=1)
        for k, model in enumerate(("raw", "AlphaEarth", "xuannv"), 1):
            p = root / "runs" / task / "ridge" / f"{model}_{spec['seeds'][0]}_5_predictions.npz"
            with np.load(p) as z:
                pos = list(z["test_indices"]).index(index)
                pred = z["scores"][pos] >= float(z["threshold"])
            image = np.zeros_like(y)
            image[(y == 1) & pred] = 1
            image[(y == 0) & pred] = 2
            image[(y == 1) & ~pred] = 3
            axes[j, k].imshow(image, cmap=cmap, vmin=0, vmax=3)
        for k, ax in enumerate(axes[j]):
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_title(
                    ("Reference", "Raw + Ridge", "AlphaEarth + Ridge", "xuannv + Ridge")[k]
                )
        axes[j, 0].set_ylabel(
            f"{task.removeprefix('osm_')}\n{identity['records'][index]['patch_id']}"
        )
    fig.legend(
        handles=[
            Patch(color=c, label=label)
            for c, label in zip(
                cmap.colors,
                ["Background", "True positive", "False positive", "False negative"],
                strict=True,
            )
        ],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    save(fig, "product_examples")
    # Tables and text use exact same summary objects as plots.
    macro = [r for r in summary if r["task"] == "macro"]
    lines = [
        "# 官方嵌入与传统机器学习对比",
        "",
        "固定编码器；所有方法采用支持样本标准化和验证集选参；两类标签分别汇总。",
        "",
        "| 标签 | 头 | 图块数 | xuannv F1 | AlphaEarth F1 | 原始特征 F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in macro:
        lines.append(
            f"| {r['family']} | {r['head']} | {r['budget']} | "
            + " | ".join(f"{100*r['models'][m]['f1']:.2f}" for m in MODELS)
            + " |"
        )
    lines.extend(
        [
            "",
            "AP、F1、IoU逐项指标与成对区间见summary.json；630组配置的完整指标见all_metrics.csv。",
            "",
            "B0保留在上一轮结果中；上一轮未执行本轮的支持集标准化和验证集正则选择，不将旧数字直接放入本轮同协议排名。",
            "",
            "AlphaEarth使用2025年年度官方COG，已修正旧缓存先插值后反量化的处理顺序；本轮先反量化再插值并单位化。",
            "xuannv使用2026年5月输出、2025年12月至2026年5月上下文；原始特征使用相同六个月和静态汇聚高分输入。",
            "OSM与上游监督关联；ESRI来自2023年，只提供跨来源、跨年份参考一致性；旧权重见过全部320图块。",
            "置信区间以当前57个空间图块为条件，不代表跨区域泛化；AP、F1及IoU均报告2000次配对图块重采样区间；AP按重采样后的全体像元排序精确重算。",
        ]
    )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    table = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{相同五图块预算下的产品比较。数值为三次支持抽样均值（\%）；OSM与ESRI分别汇总，不能合并解释为独立区域泛化精度。}\label{tab:product-main}",
        r"\begin{tabular}{llrrrrrr}\toprule",
        (
            r"参考标签 & 分类器 & \multicolumn{2}{c}{xuannv} & "
            r"\multicolumn{2}{c}{AlphaEarth Foundations} & \multicolumn{2}{c}{原始多源特征}\\"
        ),
        r" & & F1 & AP & F1 & AP & F1 & AP\\\midrule",
    ]
    for r in macro:
        if r["budget"] != 5:
            continue
        table.append(
            f"{r['family']} & {dict(ridge='Ridge',rf='随机森林',svm='RBF-SVM')[r['head']]} & "
            + " & ".join(
                f"{100*r['models'][m][metric]:.2f}" for m in MODELS for metric in ("f1", "ap")
            )
            + r"\\"
        )
    table.extend([r"\bottomrule\end{tabular}", r"\end{table*}"])
    (out / "product_main.tex").write_text("\n".join(table) + "\n")
    paired_svm_cost_cases = set.intersection(
        *[
            {
                (r["task"], r["seed"])
                for r in rows
                if r["model"] == model
                and r["head"] == "svm"
                and r.get("kernel_backend") == "numexpr_float64"
            }
            for model in MODELS
        ]
    )
    costs = {}
    for family in families:
        for h in ("ridge", "rf", "svm"):
            for model in MODELS:
                rr = [
                    r
                    for r in rows
                    if r["family"] == family
                    and r["head"] == h
                    and r["model"] == model
                    and r["budget"] == 5
                    and (h != "svm" or (r["task"], r["seed"]) in paired_svm_cost_cases)
                ]
                costs[f"{family}/{h}/{model}"] = {
                    k: float(np.median([r[k] for r in rr]))
                    for k in (
                        "selected_fit_seconds",
                        "validation_seconds",
                        "test_seconds",
                        "labeled_pixels",
                        "fitted_pixels",
                    )
                }
                costs[f"{family}/{h}/{model}"]["measured_configurations"] = len(rr)
    storage = {
        model: {
            "dimensions": identity["feature_dimensions"][model],
            "array_bytes": int(np.load(root / "prepared" / f"{model}.npy", mmap_mode="r").nbytes),
            "file_bytes": (root / "prepared" / f"{model}.npy").stat().st_size,
        }
        for model in MODELS
    }
    dump(out / "storage.json", storage)
    budgets = {}
    for family in families:
        for budget in spec["budgets"]:
            selected = [
                r
                for r in rows
                if r["family"] == family
                and r["budget"] == budget
                and r["head"] == "ridge"
                and r["model"] == "xuannv"
            ]
            budgets[f"{family}/{budget}"] = {
                key: {
                    "min": min(r[key] for r in selected),
                    "median": float(np.median([r[key] for r in selected])),
                    "max": max(r[key] for r in selected),
                }
                for key in ("labeled_pixels", "fitted_pixels")
            }
    dump(out / "annotation_budget.json", budgets)
    dump(out / "costs.json", costs)
    cost_lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{五图块设置的CPU读出成本中位数（秒）。候选拟合包括全部正则候选；"
        r"验证包含候选预测、AP计算与阈值选择；测试预测覆盖57图块。"
        r"测量在最多四任务并行、每任务四线程的运行条件下进行，"
        r"SVM仅汇总三种表征共有任务与抽样的融合CPU实现计时，N为计时样本数；不包括影像准备、嵌入生成和人工标注。}\label{tab:product-cost}",
        r"\begin{tabular}{llrrrr}\toprule",
        r"特征 & 分类器 & N & 候选拟合 & 验证与选择 & 测试预测\\\midrule",
    ]
    for model in MODELS:
        for head in ("ridge", "rf", "svm"):
            selected = [
                r
                for r in rows
                if r["model"] == model
                and r["head"] == head
                and r["budget"] == 5
                and (head != "svm" or (r["task"], r["seed"]) in paired_svm_cost_cases)
            ]
            fit = np.median([sum(r["fit_seconds_candidates"]) for r in selected])
            val = np.median([r["validation_seconds"] for r in selected])
            test = np.median([r["test_seconds"] for r in selected])
            title = {"ridge": "Ridge", "rf": "随机森林", "svm": "RBF-SVM"}[head]
            cost_lines.append(
                f"{model} & {title} & {len(selected)} & {fit:.2f} & {val:.2f} & {test:.2f}" + r"\\"
            )
    cost_lines.extend([r"\bottomrule\end{tabular}", r"\end{table*}"])
    (out / "product_cost.tex").write_text("\n".join(cost_lines) + "\n")
    dump(
        out / "artifacts.json",
        {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "artifacts.json"},
    )
    print("report complete", len(rows), "selected configurations", flush=True)
