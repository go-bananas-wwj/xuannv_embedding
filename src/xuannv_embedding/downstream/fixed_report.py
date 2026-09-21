"""Generate publication figures and numerical tables from paired audit results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from xuannv_embedding.downstream.fixed_audit import paired_bootstrap
from xuannv_embedding.export.context import dump, sha

NAMES = {
    "building": "建筑",
    "road": "道路",
    "water": "水体",
    "green": "绿地",
    "forest": "林地",
    "agriculture": "农田",
    "bare": "裸地",
    "education": "教育用地",
}
COLORS = {"B0": "#527caa", "xuannv": "#c56838"}


def configure():
    from matplotlib import font_manager

    font = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font)
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=font).get_name(),
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", dpi=220)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def run(args):
    configure()
    spec = json.loads(args.spec.read_text())
    root = Path(spec["output"])
    out = root / "report"
    out.mkdir(exist_ok=True)
    tasks = list(spec["tasks"])
    models = list(spec["models"])
    rows = []
    retr = []
    for task in tasks:
        rows.extend(json.loads((root / "probes" / task / "results.json").read_text()))
        retr.extend(json.loads((root / "probes" / task / "retrieval.json").read_text()))
    cache = json.loads(Path(spec["cache"]).read_text())
    identity = json.loads((root / "prepared" / "identity.json").read_text())
    description = json.loads((root / "description" / "results.json").read_text())

    def selected(model, task=None, budget=5, head="ridge"):
        return [
            r
            for r in rows
            if r["model"] == model
            and (task is None or r["task"] == task)
            and r["budget"] == budget
            and r["head"] == head
        ]

    def value(model, task=None, budget=5, head="ridge", metric="f1"):
        values = [
            r["metrics"][metric]
            for r in selected(model, task, budget, head)
            if r["metrics"][metric] is not None
        ]
        return float(np.mean(values)) if values else None

    table = []
    for task in tasks:
        a, b = [np.array([r["metrics"]["block_counts"] for r in selected(m, task)]) for m in models]
        ci = paired_bootstrap(a, b, repeats=spec["bootstrap_repeats"])
        record = {"task": task, "f1_difference_ci": ci}
        for m in models:
            record[m] = {
                k: value(m, task, metric=k)
                for k in (
                    "f1",
                    "ap",
                    "iou",
                    "boundary_f1_10m",
                    "boundary_f1_20m",
                    "small_object_recall",
                )
            }
        table.append(record)
    summary = {
        "protocol_sha256": sha(args.spec),
        "models": spec["models"],
        "table_5tile_ridge": table,
        "mean_by_budget": {
            m: {
                str(b): {k: value(m, budget=b, metric=k) for k in ("f1", "ap", "iou")}
                for b in spec["budgets"]
            }
            for m in models
        },
        "mean_knn_5": {
            m: {k: value(m, head="knn", metric=k) for k in ("f1", "ap", "iou")} for m in models
        },
        "description": description,
        "sample_seed_count": len(spec["seeds"]),
        "scope": (
            "product comparison; old xuannv trained across evaluation area; "
            "OSM reference overlap"
        ),
        "confidence": (
            "paired tile bootstrap; mean of three support-seed F1 differences; "
            "conditional on fixed models, region and labels"
        ),
    }
    costs = {}
    for m in models:
        r = selected(m)
        costs[m] = {
            "ridge_fit_median_seconds": float(np.median([x["fit_seconds"] for x in r])),
            "ridge_predict_median_seconds": float(np.median([x["predict_seconds"] for x in r])),
            "knn_predict_median_seconds": float(
                np.median([x["predict_seconds"] for x in selected(m, head="knn")])
            ),
            "retrieval_median_seconds": float(
                np.median([x["seconds"] for x in retr if x.get("model") == m and x["budget"] == 3])
            ),
        }
    summary["readout_cost"] = costs
    summary["macro_f1_difference_ci"] = paired_bootstrap(
        np.array([r["metrics"]["block_counts"] for r in selected("B0")]),
        np.array([r["metrics"]["block_counts"] for r in selected("xuannv")]),
        repeats=spec["bootstrap_repeats"],
    )
    dump(out / "summary.json", summary)
    flat = []
    for r in rows:
        item = {
            k: r[k]
            for k in (
                "model",
                "task",
                "seed",
                "budget",
                "head",
                "threshold",
                "labeled_pixels",
                "fitted_pixels",
                "fit_seconds",
                "predict_seconds",
            )
        }
        item.update({k: v for k, v in r["metrics"].items() if not k.startswith("block_")})
        flat.append(item)
    with (out / "all_metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    with (out / "retrieval_metrics.csv").open("w") as f:
        rr = [
            {**{k: r[k] for k in ("model", "task", "seed", "budget", "seconds")}, **r["metrics"]}
            for r in retr
            if "metrics" in r
        ]
        w = csv.DictWriter(f, fieldnames=list(rr[0]))
        w.writeheader()
        w.writerows(rr)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    xx = np.arange(len(tasks))
    for j, m in enumerate(models):
        axes[0, 0].bar(
            xx + (j - 0.5) * 0.36,
            [100 * value(m, t) for t in tasks],
            width=0.36,
            label=m,
            color=COLORS[m],
        )
        axes[0, 1].plot(
            spec["budgets"],
            [100 * value(m, budget=b) for b in spec["budgets"]],
            "-o",
            label=m,
            color=COLORS[m],
        )
        axes[1, 1].bar(
            xx + (j - 0.5) * 0.36,
            [100 * value(m, t, head="knn") for t in tasks],
            width=0.36,
            label=m,
            color=COLORS[m],
        )
    for ax in (axes[0, 0], axes[1, 1]):
        ax.set_xticks(xx, [NAMES[t] for t in tasks], rotation=35, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("F1 / %")
        ax.legend(frameon=False)
    axes[0, 0].set_title("(a) 跨任务复用：5 个支持图块，线性读出")
    axes[0, 1].set_title("(b) 标注预算与八任务宏平均")
    axes[0, 1].set_xlabel("完整标注图块数")
    axes[0, 1].set_ylabel("F1 / %")
    axes[0, 1].set_xticks(spec["budgets"])
    axes[0, 1].legend(frameon=False)
    delta = np.array([r["xuannv"]["f1"] - r["B0"]["f1"] for r in table]) * 100
    # Center interval on bootstrap functional; observed seed-mean difference shown separately.
    ci = np.array([r["f1_difference_ci"] for r in table]) * 100
    axes[1, 0].hlines(xx, ci[:, 0], ci[:, 1], color="#444444", lw=2)
    axes[1, 0].scatter(delta, xx, c=np.where(delta >= 0, COLORS["xuannv"], COLORS["B0"]), zorder=3)
    axes[1, 0].axvline(0, color="grey", ls="--")
    axes[1, 0].set_yticks(xx, [NAMES[t] for t in tasks])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("xuannv − B0 / 百分点")
    axes[1, 0].set_title("(c) 配对空间块差异与 95% 区间")
    axes[1, 1].set_title("(d) 固定 kNN 读出核查：5 个支持图块")
    save(fig, out, "audit_tasks")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for j, m in enumerate(models):
        rgb = np.load(root / "description" / f"{m}_halo_rgb.npy")
        axes[0, j].imshow(rgb, interpolation="nearest")
        axes[0, j].axis("off")
        axes[0, j].set_title(f"({chr(97+j)}) {m}：共享 PCA 色阶")
        mat = np.array(description["semantic"][m]["centroid_cosine_distances"])
        image = axes[1, j].imshow(mat, vmin=0, vmax=2, cmap="Blues")
        cn = ["水体", "树木", "低矮植被", "农田", "建成区", "裸地"]
        axes[1, j].set_xticks(range(6), cn, rotation=45, ha="right")
        axes[1, j].set_yticks(range(6), cn)
        axes[1, j].set_title(f"({chr(100+j)}) {m}：类中心余弦距离")
    axes[0, 2].bar(
        models,
        [description["semantic"][m]["silhouette_cosine"] for m in models],
        color=[COLORS[m] for m in models],
    )
    axes[0, 2].axhline(0, color="grey", lw=0.8)
    axes[0, 2].set_title("(c) 原始 64 维 Silhouette（↑）")
    axes[1, 2].bar(
        models,
        [description["semantic"][m]["davies_bouldin_euclidean"] for m in models],
        color=[COLORS[m] for m in models],
    )
    axes[1, 2].set_title("(f) Davies–Bouldin（↓）")
    fig.colorbar(image, ax=axes[1, :2], shrink=0.55, label="余弦距离")
    save(fig, out, "audit_semantics")

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    for m in models:
        ax.plot(
            [1, 3, 5],
            [
                100
                * np.mean(
                    [r["metrics"]["ap"] for r in retr if r.get("model") == m and r["budget"] == b]
                )
                for b in (1, 3, 5)
            ],
            "-o",
            color=COLORS[m],
            label=m,
        )
        yy = [
            100
            * np.mean(
                [
                    r["metrics"]["ap"]
                    for r in retr
                    if r.get("model") == m and r["budget"] == 3 and r["task"] == t
                ]
            )
            for t in tasks
        ]
        ax2.plot(yy, np.arange(len(tasks)), "o-", label=m, color=COLORS[m])
    ax.set_title("(a) 正类样例检索：八任务宏平均 AP")
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("参考标签连通地物样例数")
    ax.set_ylabel("AP / %")
    ax.legend(frameon=False)
    ax2.set_yticks(range(len(tasks)), [NAMES[t] for t in tasks])
    ax2.invert_yaxis()
    ax2.set_xlabel("AP / %")
    ax2.set_title("(b) 3 样例：全部任务")
    task = "water"
    i = identity["representative_rois"][task][0]
    ti = cache["split"]["test"].index(i)
    y = np.load(root / "prepared" / f"label_{task}.npy")[i]
    for j, title in enumerate(["参考水体", "B0 相似度", "xuannv 相似度"]):
        a = fig.add_subplot(gs[1, j])
        a.axis("off")
        a.set_title(f"({chr(99+j)}) {title}")
        if j == 0:
            a.imshow(y, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        else:
            s = np.load(root / "probes" / task / f"retrieval_{models[j-1]}_20260916_3.npz")[
                "scores"
            ][ti]
            a.imshow(s, cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
            a.contour(y, levels=[0.5], colors="white", linewidths=0.6)
    save(fig, out, "audit_retrieval")

    fig = plt.figure(figsize=(13, 10), layout="constrained")
    gs = fig.add_gridspec(3, 4)
    ax = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    for j, m in enumerate(models):
        ax.bar(
            np.arange(2) + (j - 0.5) * 0.3,
            [100 * value(m, metric=k) for k in ("boundary_f1_10m", "boundary_f1_20m")],
            width=0.3,
            label=m,
            color=COLORS[m],
        )
        ax2.bar(
            np.arange(2) + (j - 0.5) * 0.3,
            [description["seams"][m][s]["ratio"] for s in ("without_context", "with_context")],
            width=0.3,
            label=m,
            color=COLORS[m],
        )
    ax.set_xticks([0, 1], ["10 m 容差", "20 m 容差"])
    ax.set_ylabel("八任务平均边界 F1 / %")
    ax.set_title("(a) 边界匹配")
    ax.legend(frameon=False)
    ax2.set_xticks([0, 1], ["无扩边", "16 像素扩边"])
    ax2.set_ylabel("拼接边缘 / 相邻内部的特征距离")
    ax2.set_title("(b) 接缝诊断（低不等于分类准确）")
    ax2.legend(frameon=False)
    for row, task in enumerate(["building", "road"], 1):
        i = identity["representative_rois"][task][0]
        ti = cache["split"]["test"].index(i)
        y = np.load(root / "prepared" / f"label_{task}.npy")[i]
        for j in range(4):
            ax = fig.add_subplot(gs[row, j])
            ax.axis("off")
            if j == 0:
                r = cache["records"][i]
                b = np.array([r0["bounds"] for r0 in cache["records"]])
                x0 = round((r["bounds"][0] - b[:, 0].min()) / 10)
                y0 = round((b[:, 3].max() - r["bounds"][3]) / 10)
                canvas = np.load(root / "description" / "xuannv_halo_rgb.npy", mmap_mode="r")
                ax.imshow(canvas[y0 : y0 + 128, x0 : x0 + 128])
                title = f"{NAMES[task]}：预选位置 PCA"
            elif j == 1:
                ax.imshow(y, cmap="Greys", vmin=0, vmax=1)
                title = "参考标签"
            else:
                m = models[j - 2]
                r = [q for q in selected(m, task) if q["seed"] == 20260916][0]
                data = np.load(root / "probes" / task / f"{m}_20260916_5_ridge_predictions.npz")
                pred = data["scores"][len(cache["split"]["validation"]) + ti] >= r["threshold"]
                image = np.ones((128, 128, 3)) * 0.95
                image[pred & (y == 1)] = [0.15, 0.6, 0.3]
                image[pred & (y == 0)] = [0.8, 0.25, 0.18]
                image[~pred & (y == 1)] = [0.2, 0.4, 0.85]
                ax.imshow(image, interpolation="nearest")
                title = f"{m}：误差分布"
            ax.set_title(title, fontsize=10.5)
    save(fig, out, "audit_spatial")

    fig = plt.figure(figsize=(13, 8), layout="constrained")
    gs = fig.add_gridspec(3, 6, height_ratios=[2, 1, 1])
    for j, m in enumerate(models):
        ax = fig.add_subplot(gs[0, j * 3 : (j + 1) * 3])
        im = ax.imshow(
            description["temporal"][m]["mean_cosine_similarity"], vmin=0, vmax=1, cmap="viridis"
        )
        ax.set_xticks(range(6), ["12月", "1月", "2月", "3月", "4月", "5月"])
        ax.set_yticks(range(6), ["12月", "1月", "2月", "3月", "4月", "5月"])
        ax.set_title(f"({chr(97+j)}) {m}：同位置跨月余弦相似度")
        fig.colorbar(im, ax=ax, shrink=0.75)
        i = identity["representative_rois"]["green"][0]
        rgb = np.load(root / "description" / f"{m}_monthly_roi_{i}.npy")
        for k in range(6):
            a = fig.add_subplot(gs[j + 1, k])
            a.imshow(rgb[k])
            a.axis("off")
            a.set_title(f"{m} · {k+1 if k else 12}月" if k == 0 else f"{m} · {k}月", fontsize=10.5)
    save(fig, out, "audit_monthly")
    architecture(out, root, table)
    conceptual_figures(out)
    latex_tables(out, summary, rows, retr)
    inventory = {
        p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "artifact_hashes.json"
    }
    dump(out / "artifact_hashes.json", inventory)
    print(
        json.dumps(
            {"mean_by_budget": summary["mean_by_budget"], "readout_cost": costs},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def architecture(out, root, table):
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    def box(ax, x, y, w, h, text, color="#e5eef6"):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.015", fc=color, ec="#678099", lw=0.8)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12)

    def arrow(ax, x, y, u, v):
        ax.add_patch(
            FancyArrowPatch((x, y), (u, v), arrowstyle="-|>", mutation_scale=14, color="#52616e")
        )

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(a) 同一地物的多源观测与训练目标", loc="left")
    box(ax, 0.03, 0.7, 0.4, 0.2, "公开时序观测\nS2 · S1 · Landsat")
    box(ax, 0.56, 0.7, 0.4, 0.2, "地方高分观测\n光学 · SAR", "#f4e5d9")
    box(ax, 0.03, 0.38, 0.4, 0.18, "质量控制与月度组织")
    box(ax, 0.56, 0.38, 0.4, 0.18, "有效观测汇聚\n映射到公共网格", "#f4e5d9")
    arrow(ax, 0.23, 0.7, 0.23, 0.57)
    arrow(ax, 0.76, 0.7, 0.76, 0.57)
    box(
        ax,
        0.2,
        0.04,
        0.6,
        0.2,
        "重建观测 + OSM 弱语义 + 均匀性\n训练遮挡；地图不参与导出",
        "#e5eee7",
    )
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) 时空编码与高分后融合", loc="left")
    box(ax, 0.02, 0.68, 0.27, 0.2, "分源浅层编码\n可用性门控")
    box(ax, 0.37, 0.68, 0.59, 0.2, "共享空间 / 时间 / 精细路径\n月度特征")
    arrow(ax, 0.29, 0.78, 0.37, 0.78)
    box(ax, 0.02, 0.31, 0.27, 0.2, "高分卷积编码\n静态特征", "#f4e5d9")
    box(ax, 0.39, 0.3, 0.57, 0.23, "可用性嵌入 + 拼接融合\n逐像元球面归一化", "#e5eee7")
    arrow(ax, 0.29, 0.41, 0.39, 0.41)
    arrow(ax, 0.66, 0.68, 0.66, 0.54)
    ax.text(
        0.5,
        0.09,
        "输出：10 m 网格 × 64 维 × 时间索引\n公开时序随月份变化；高分特征跨月共享",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(np.load(root / "description" / "xuannv_halo_rgb.npy"), interpolation="nearest")
    ax.axis("off")
    ax.set_title("(c) 海淀区域嵌入：实际 PCA 投影", loc="left")
    ax = fig.add_subplot(gs[1, 1])
    xx = np.arange(len(table))
    for j, m in enumerate(("B0", "xuannv")):
        ax.bar(
            xx + (j - 0.5) * 0.35,
            [100 * r[m]["f1"] for r in table],
            width=0.35,
            label=m,
            color=COLORS[m],
        )
    ax.set_xticks(xx, [NAMES[r["task"]] for r in table], rotation=35, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("F1 / %")
    ax.legend(frameon=False)
    ax.set_title("(d) 冻结嵌入跨任务复用：5 图块线性读出", loc="left")
    save(fig, out, "audit_architecture")


def latex_tables(out, summary, rows, retr):
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{五个完整支持图块下的线性读出结果，三次支持抽样的平均值。AP、F1和IoU以百分数表示；区间为配对图块bootstrap的F1差值区间。两模型上游训练范围不同，结果用于产品能力比较。}\label{tab:audit-main}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"任务 & \multicolumn{3}{c}{B0} & \multicolumn{3}{c}{xuannv} & F1差值95\%区间 \\",
        r" & AP & F1 & IoU & AP & F1 & IoU & 百分点 \\",
        r"\midrule",
    ]
    for r in summary["table_5tile_ridge"]:
        vals = [100 * r[m][k] for m in ("B0", "xuannv") for k in ("ap", "f1", "iou")]
        ci = np.array(r["f1_difference_ci"]) * 100
        lines.append(
            NAMES[r["task"]]
            + " & "
            + " & ".join(f"{x:.2f}" for x in vals)
            + f" & [{ci[0]:.2f}, {ci[1]:.2f}] "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (out / "audit_main.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{固定嵌入上的实际运行成本中位数。线性及kNN预测覆盖67个验证和57个测试图块；检索覆盖57个测试图块。编码累计包含有、无扩边两次前向及结果取回，不含原始数据准备和写盘；下游计时假定特征已载入内存。均不包含人工标注和交互界面耗时。}\label{tab:audit-cost}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"操作（未注明者为s） & B0 & xuannv \\",
        r"\midrule",
    ]
    for k, label in [
        ("ridge_fit_median_seconds", "线性头拟合"),
        ("ridge_predict_median_seconds", "线性头预测"),
        ("knn_predict_median_seconds", "kNN预测"),
        ("retrieval_median_seconds", "3样例检索"),
    ]:
        lines.append(
            label
            + " & "
            + " & ".join(f"{summary['readout_cost'][m][k]:.3f}" for m in ("B0", "xuannv"))
            + r" \\"
        )
    for key, label, divisor in [
        ("forward_export_seconds", "双设置编码累计 / s", 1),
        ("canonical_float32_6months_bytes", "六个月嵌入 / GiB", 1024**3),
    ]:
        values = [summary["description"]["cost"][m][key] / divisor for m in ("B0", "xuannv")]
        lines.append(label + " & " + " & ".join(f"{v:.2f}" for v in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (out / "audit_cost.tex").write_text("\n".join(lines) + "\n")
    # Full task/head/budget means for reproducibility, kept out of the main narrative.
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{全部支持预算与读出方式的三次抽样均值（\%）。}\label{tab:audit-full}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"任务 & 读出/图块数 & \multicolumn{3}{c}{B0} & \multicolumn{3}{c}{xuannv} \\",
        r" & & AP & F1 & IoU & AP & F1 & IoU \\",
        r"\midrule",
    ]
    for task in NAMES:
        for budget, head in [(1, "ridge"), (5, "ridge"), (10, "ridge"), (5, "knn")]:
            vals = [
                np.mean(
                    [
                        r["metrics"][k]
                        for r in rows
                        if r["task"] == task
                        and r["budget"] == budget
                        and r["head"] == head
                        and r["model"] == m
                    ]
                )
                * 100
                for m in ("B0", "xuannv")
                for k in ("ap", "f1", "iou")
            ]
            lines.append(
                NAMES[task]
                + f" & {head}/{budget} & "
                + " & ".join(f"{x:.2f}" for x in vals)
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (out / "audit_full.tex").write_text("\n".join(lines) + "\n")


def conceptual_figures(out):
    """Conceptual figures are schematics, without invented measurements."""

    def box(ax, x, y, w, h, text, color="#e5eef6"):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.008", fc=color, ec="#708497", lw=0.9)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12)

    def arrow(ax, x, y, u, v):
        ax.add_patch(
            FancyArrowPatch((x, y), (u, v), arrowstyle="-|>", mutation_scale=14, color="#52616e")
        )

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.4), layout="constrained")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    ax = axes[0]
    ax.set_title("(a) 按任务组织的建模流程", loc="left")
    box(ax, 0.02, 0.23, 0.2, 0.54, "同一区域\n多源遥感观测")
    for y, name in [(0.72, "水体"), (0.39, "建筑"), (0.06, "其他目标")]:
        box(ax, 0.32, y, 0.36, 0.22, f"{name}样本 → 特征学习与模型训练", "#f3e7de")
        box(ax, 0.78, y, 0.2, 0.22, f"{name}专题结果", "#e5eee7")
        arrow(ax, 0.22, 0.5, 0.31, y + 0.11)
        arrow(ax, 0.69, y + 0.11, 0.77, y + 0.11)
    ax = axes[1]
    ax.set_title("(b) 区域共享表征与任务读出", loc="left")
    box(ax, 0.02, 0.23, 0.2, 0.54, "同一区域\n多源遥感观测")
    box(ax, 0.3, 0.23, 0.22, 0.54, "区域联合编码\n预计算嵌入", "#e5eee7")
    arrow(ax, 0.22, 0.5, 0.29, 0.5)
    for y, name in [(0.72, "水体"), (0.39, "建筑"), (0.06, "其他目标")]:
        box(ax, 0.61, y, 0.37, 0.22, f"{name}少量标签 / 样例 → 轻量读出", "#f3e7de")
        arrow(ax, 0.52, 0.5, 0.60, y + 0.11)
    save(fig, out, "audit_introduction")
    fig, ax = plt.subplots(figsize=(12, 5.2), layout="constrained")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5, 0.97, "遥感共享表征：学习目标、访问方式与应用组织", ha="center", va="top", fontsize=14
    )
    cards = [
        (0.01, "观测重建", "SatMAE · Prithvi", "遮挡输入 / 重建观测"),
        (0.34, "潜空间学习", "Galileo · OlmoEarth", "跨视图关系 / 特征目标"),
        (0.67, "跨模态对齐", "RemoteCLIP · SatCLIP", "影像—文本 / 影像—位置"),
    ]
    for x, title, examples, detail in cards:
        box(ax, x, 0.55, 0.31, 0.28, f"{title}\n{examples}\n{detail}")
        arrow(ax, x + 0.155, 0.54, 0.50, 0.44)
    box(
        ax,
        0.08,
        0.19,
        0.52,
        0.25,
        (
            "共享特征与地理嵌入\nMOSAIKS · AlphaEarth Foundations · TESSERA\n"
            "预计算 / 位置查询 / 下游复用"
        ),
        "#e5eee7",
    )
    box(
        ax,
        0.70,
        0.19,
        0.28,
        0.25,
        "交互与工具编排\nEarth-Agent 等\n任务理解 / 专业工具调用",
        "#f3e7de",
    )
    arrow(ax, 0.61, 0.315, 0.69, 0.315)
    ax.text(
        0.5,
        0.045,
        "不同路径可以组合；图示并非互斥分类，也不表示严格的时间先后关系。",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    save(fig, out, "audit_related")
