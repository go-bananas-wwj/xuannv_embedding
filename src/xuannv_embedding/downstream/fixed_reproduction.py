"""Audit an archived embedding product without fitting any encoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier

from xuannv_embedding.downstream.fixed_audit import paired_bootstrap, score_tiles, threshold
from xuannv_embedding.export.context import dump, sha


def checked_positions(labels, positions, reference):
    if hashlib.sha256(positions.tobytes()).hexdigest() != reference["sample_positions_sha256"]:
        raise ValueError("support positions differ")
    if hashlib.sha256(labels.tobytes()).hexdigest() != reference["support_label_sha256"]:
        raise ValueError("support labels differ")
    if (labels[positions] < 0).any():
        raise ValueError("invalid support labels")


def run(args):
    definition = json.loads(args.spec.read_text())
    base = Path(definition["source_audit"])
    root = Path(definition["output"])
    archive = Path(definition["archive"])
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads((base / "spec.json").read_text())
    cache = json.loads(Path(spec["cache"]).read_text())
    meta = json.loads((archive / "meta.json").read_text())
    model = definition["model"]
    checkpoint = spec["models"][model]
    actual = sha(checkpoint["checkpoint"])
    if actual != checkpoint["sha256"] or not actual.startswith(meta["checkpoint_sha256"]):
        raise ValueError("checkpoint provenance mismatch")
    if len(meta["checkpoint_sha256"]) < 16:
        raise ValueError("archived checksum prefix too short")
    current = np.load(base / "prepared" / f"{model}.npy", mmap_mode="r")
    old = np.lib.format.open_memmap(
        root / "archived.npy", mode="w+", dtype="float32", shape=current.shape
    )
    records = []
    month = spec["month"].replace("-", "")
    # Select every existing evaluation tile, never select on old/new performance.
    for i, r in enumerate(cache["records"]):
        p = archive / definition["region_directory"] / r["patch_id"]
        p = p / f"{month}_embedding_map.pt"
        z = torch.load(p, weights_only=True, map_location="cpu").numpy()
        if z.shape != (64, 128, 128) or not np.isfinite(z).all():
            raise ValueError("invalid archived tensor")
        a = z.transpose(1, 2, 0).astype("float32")
        norm = np.linalg.norm(a, axis=-1)
        if norm.min() < 0.99 or norm.max() > 1.01:
            raise ValueError("archived embeddings are not unit vectors")
        old[i] = a
        b = np.asarray(current[i])
        cos = (a * b).sum(-1) / (norm * np.linalg.norm(b, axis=-1))
        border = np.ones((128, 128), bool)
        border[16:-16, 16:-16] = False
        records.append(
            {
                "patch_id": r["patch_id"],
                "sha256": sha(p),
                "mean_cosine": float(cos.mean()),
                "interior_cosine": float(cos[~border].mean()),
                "border_cosine": float(cos[border].mean()),
                "max_abs_difference": float(np.abs(a - b).max()),
            }
        )
    old.flush()
    dump(root / "inputs.json", {"archive_metadata": meta, "records": records})
    print("Archived embeddings verified:", len(records), flush=True)
    val, test = cache["split"]["validation"], cache["split"]["test"]
    query_ids = val + test
    query = np.asarray(old[query_ids]).reshape(-1, 64)
    rows = []
    for task in spec["tasks"]:
        y = np.load(base / "prepared" / f"label_{task}.npy")
        for seed in spec["seeds"]:
            key = f"{model}_{seed}_5_ridge"
            folder = base / "probes" / task
            reference = json.loads((folder / f"{key}.json").read_text())
            with np.load(folder / f"{key}_head.npz") as h:
                selected, picks = h["support_indices"], h["pixel_indices"]
                sy = y[selected].reshape(-1)
                checked_positions(sy, picks, reference)
                frozen = (query @ h["coef"].T + h["intercept"]).ravel()
            fit = RidgeClassifier(alpha=1, class_weight="balanced", solver="cholesky")
            fit.fit(np.asarray(old[selected]).reshape(-1, 64)[picks], sy[picks])
            refitted = fit.decision_function(query)
            np.savez(
                root / f"{task}_{seed}_head.npz",
                coef=fit.coef_,
                intercept=fit.intercept_,
                support_indices=selected,
                pixel_indices=picks,
            )
            for variant, scores in (("archive_refit", refitted), ("archive_frozen_head", frozen)):
                scores = scores.reshape(len(query_ids), 128, 128).astype("float32")
                cut = (
                    threshold(y[val].ravel(), scores[: len(val)].ravel())
                    if variant == "archive_refit"
                    else reference["threshold"]
                )
                result = score_tiles(y[test], scores[len(val) :], cut)
                row = {
                    "task": task,
                    "seed": seed,
                    "variant": variant,
                    "threshold": cut,
                    "support_patch_ids": reference["support_patch_ids"],
                    "support_label_sha256": reference["support_label_sha256"],
                    "sample_positions_sha256": reference["sample_positions_sha256"],
                    "metrics": result,
                }
                np.savez_compressed(
                    root / f"{task}_{seed}_{variant}_predictions.npz",
                    scores=scores,
                    patch_indices=query_ids,
                    threshold=cut,
                )
                rows.append(row)
            for variant, m in (("current", model), ("B0", "B0")):
                ref = json.loads((folder / f"{m}_{seed}_5_ridge.json").read_text())
                for field in ("support_label_sha256", "sample_positions_sha256"):
                    if ref[field] != reference[field]:
                        raise ValueError("unpaired reference evaluation")
                rows.append({**ref, "variant": variant})
        print(task, "complete", flush=True)
        dump(root / "results.json", rows)
    summary = {}
    for task in spec["tasks"]:
        t = {}
        counts = {}
        for variant in ("B0", "current", "archive_refit", "archive_frozen_head"):
            selected = [r for r in rows if r["task"] == task and r["variant"] == variant]
            t[variant] = {
                k: float(np.mean([r["metrics"][k] for r in selected]))
                for k in ("f1", "ap", "iou", "boundary_f1_10m", "boundary_f1_20m")
            }
            counts[variant] = [r["metrics"]["block_counts"] for r in selected]
        t["archive_minus_current_f1_ci"] = paired_bootstrap(
            counts["current"], counts["archive_refit"]
        )
        t["archive_minus_B0_f1_ci"] = paired_bootstrap(counts["B0"], counts["archive_refit"])
        summary[task] = t
    dump(root / "summary.json", summary)
    dump(
        root / "provenance.json",
        {
            "spec_sha256": sha(args.spec),
            "checkpoint_sha256": actual,
            "source_spec_sha256": sha(base / "spec.json"),
            "archived_metadata_sha256": sha(archive / "meta.json"),
            "label_identity_sha256": sha(base / "prepared" / "identity.json"),
            "new_encoder_training": False,
            "protocol": (
                "eight tasks, three seeds, five support tiles, "
                "identical Ridge; validation thresholds"
            ),
            "frozen_head_protocol": (
                "current head and threshold applied unchanged to archived embeddings"
            ),
            "georeferencing_limit": "archive tensors carry tile IDs, but no embedded CRS or bounds",
        },
    )
    report(root)


def report(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    rows = json.loads((root / "results.json").read_text())
    inputs = json.loads((root / "inputs.json").read_text())
    variants = ("B0", "current", "archive_refit", "archive_frozen_head")
    metrics = ("f1", "ap", "iou", "boundary_f1_10m", "boundary_f1_20m")
    macro = {
        v: {k: float(np.mean([t[v][k] for t in summary.values()])) for k in metrics}
        for v in variants
    }
    counts = {
        v: [r["metrics"]["block_counts"] for r in rows if r["variant"] == v] for v in variants
    }
    macro["archive_minus_current_f1_ci"] = paired_bootstrap(
        counts["current"], counts["archive_refit"]
    )
    macro["archive_minus_B0_f1_ci"] = paired_bootstrap(counts["B0"], counts["archive_refit"])
    dump(root / "macro.json", macro)
    labels = {
        "building": "建筑",
        "road": "道路",
        "water": "水体",
        "green": "绿地",
        "forest": "林地",
        "agriculture": "农田",
        "bare": "裸地",
        "education": "教育用地",
    }
    lines = [
        "# 旧版生产嵌入复现核查",
        "",
        "固定同一编码器权重，2026年5月，八任务×三种子，五支持图块；只重拟合轻量Ridge头。",
        "沿用原评价全部支持像素、67个验证图块与57个测试图块；阈值仅由验证集选择。",
        "",
        "| 类别 | B0 F1 | 本次导出 F1 | 旧版存档 F1 | 旧版存档 AP |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, t in summary.items():
        lines.append(
            f"| {labels.get(task, task)} | {100*t['B0']['f1']:.2f} | "
            f"{100*t['current']['f1']:.2f} | {100*t['archive_refit']['f1']:.2f} | "
            f"{100*t['archive_refit']['ap']:.2f} |"
        )
    lines.append(
        f"| 平均 | {100*macro['B0']['f1']:.2f} | {100*macro['current']['f1']:.2f} | "
        f"{100*macro['archive_refit']['f1']:.2f} | {100*macro['archive_refit']['ap']:.2f} |"
    )
    for key, label in (
        ("archive_minus_current_f1_ci", "旧版存档−本次导出"),
        ("archive_minus_B0_f1_ci", "旧版存档−B0"),
    ):
        lo, hi = macro[key]
        lines.extend(
            ["", f"{label}，配对图块bootstrap 95%区间：[{100*lo:.2f}, {100*hi:.2f}]个百分点。"]
        )
    cosine = np.array([r["mean_cosine"] for r in inputs["records"]])
    lines.extend(
        [
            "",
            f"320个图块平均旧/新嵌入余弦相似度：{cosine.mean():.4f}。",
            f"沿用本次读出头及阈值直接评旧版存档时，平均F1为{100*macro['archive_frozen_head']['f1']:.2f}%。",
            "该交叉应用仅诊断导出变化，不替代各自拟合的公平能力比较。",
            "",
            "限制：旧存档只保留权重摘要前16位和代码提交，没有逐源输入哈希或张量内嵌地理坐标；",
            "按相同图块ID和月份对应，不能仅凭本次结果把所有导出差异归因于扩边。",
            "旧模型训练区域覆盖全部320图块，OSM上游监督与本次标签关联；仍非独立空间泛化评估。",
            "本次没有增加编码器训练，没有据结果替换类别或重新选择支持样本。",
            "",
            "results.json保存逐种子指标；预测、读出头、输入文件哈希及配置均保留在本目录。",
        ]
    )
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    x = np.arange(len(summary))
    colors = ["#455a64", "#c97936", "#267e82"]
    for j, (v, title, color) in enumerate(
        zip(variants[:3], ["B0", "Re-exported xuannv", "Archived xuannv"], colors, strict=True)
    ):
        axes[0].bar(
            x + (j - 1) * 0.24,
            [100 * t[v]["f1"] for t in summary.values()],
            0.24,
            label=title,
            color=color,
        )
    axes[0].set_xticks(x, list(summary), rotation=35, ha="right")
    axes[0].set_ylabel("F1 (%)")
    axes[0].set_title("Same 5 support tiles; mean of 3 draws")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(cosine, bins=24, color=colors[2], edgecolor="white")
    axes[1].set_xlabel("Archived / re-exported mean cosine per tile")
    axes[1].set_ylabel("Tiles")
    axes[1].set_title("Same weights and month; different export products")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"reproduction_comparison.{suffix}", dpi=180)
    plt.close(fig)
