"""Standalone image/mask review gallery; never substitutes for completed data gates."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import zarr
from matplotlib.colors import ListedColormap

from xuannv_embedding.data_process.v5_sources import now, write_json


def review_gaofen(quality_root: Path, report_root: Path, *, sample_size: int = 32) -> dict:
    frame = pd.read_parquet(quality_root / "observation_quality.parquet")
    order = pd.read_parquet(quality_root / "observation_order.parquet")
    position = {pair_id: i for i, pair_id in enumerate(order.pair_id)}
    candidate = frame.drop_duplicates("patch_id")
    third = max(1, sample_size // 3)
    selected = pd.concat(
        [
            candidate.nsmallest(third, "clear_fraction"),
            candidate.assign(distance=(candidate.clear_fraction - 0.6).abs()).nsmallest(
                third, "distance"
            ),
            candidate.nlargest(sample_size, "clear_fraction"),
        ]
    )
    selected = selected.drop_duplicates("pair_id").head(sample_size)
    source_classes = zarr.open_group(str(quality_root / "classes.zarr"), mode="r")
    source_masks = zarr.open_group(str(quality_root / "valid_masks.zarr"), mode="r")
    directory = report_root / "visual_review"
    directory.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, row in enumerate(selected.itertuples()):
        index = position[row.pair_id]
        if not bool(source_masks["completed"][index]):
            raise ValueError("review cannot use an incomplete QA row")
        with rasterio.open(row.ms_path) as source:
            values = source.read([3, 2, 1]).astype("f4")
            valid = np.all(source.read_masks([3, 2, 1]) > 0, axis=0)
        rgb = np.zeros(values.shape, dtype="f4")
        if valid.any():
            for band in range(3):
                low, high = np.percentile(values[band, valid], [2, 98])
                rgb[band] = np.clip((values[band] - low) / max(float(high - low), 1), 0, 1)
        rgb[:, ~valid] = 0
        rgb = rgb.transpose(1, 2, 0)
        labels = np.asarray(source_classes["classes"][index])
        clear = np.unpackbits(
            source_masks["ms_valid_packed"][index], axis=-1, count=160, bitorder="little"
        ).astype(bool)
        before = np.unpackbits(
            source_masks["before_buffer_packed"][index], axis=-1, count=160, bitorder="little"
        ).astype(bool)
        fig, axes = plt.subplots(1, 5, figsize=(17, 4))
        axes[0].imshow(rgb, extent=(0, 1280, 0, 1280))
        axes[0].set_title("Native RGB (display stretch)")
        axes[1].imshow(
            labels,
            cmap=ListedColormap(["#28965a", "#ffffff", "#68cce7", "#555555"]),
            vmin=0,
            vmax=3,
            interpolation="nearest",
            extent=(0, 1280, 0, 1280),
        )
        axes[1].set_title("Clear / cloud / thin / shadow")
        axes[2].imshow(before, cmap="gray", vmin=0, vmax=1, extent=(0, 1280, 0, 1280))
        axes[2].set_title("Clear pixels before buffer")
        axes[3].imshow(clear, cmap="gray", vmin=0, vmax=1, extent=(0, 1280, 0, 1280))
        axes[3].set_title(f"After 30m buffer: {row.clear_fraction:.1%}")
        axes[4].imshow(rgb, extent=(0, 1280, 0, 1280))
        overlay = np.zeros((*clear.shape, 4), dtype="f4")
        overlay[~clear] = [1, 0, 0, 0.45]
        axes[4].imshow(overlay, extent=(0, 1280, 0, 1280))
        axes[4].set_title("Invalid overlay / 320m grid")
        for axis in axes:
            axis.set_xticks([0, 640, 1280])
            axis.set_yticks([0, 640, 1280])
        axes[4].set_xticks(np.arange(0, 1281, 320))
        axes[4].set_yticks(np.arange(0, 1281, 320))
        axes[4].grid(alpha=0.5)
        fig.suptitle(
            f"{row.sensor} | {row.acquired_at} | {row.patch_id} | registration pending", fontsize=11
        )
        fig.tight_layout()
        filename = f"gaofen_{i:02d}.png"
        fig.savefig(directory / filename, dpi=120)
        plt.close(fig)
        cards.append(
            {
                "family": "gaofen",
                "pair_id": row.pair_id,
                "patch_id": row.patch_id,
                "image": filename,
                "clear_fraction": float(row.clear_fraction),
                "strict_scene_qualified": bool(row.strict_scene_qualified),
                "alignment_status": "not_audited",
            }
        )
    pd.DataFrame(cards).to_csv(directory / "samples.csv", index=False)
    body = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>V5 数据质量样例</title>',
        "<style>body{font-family:system-ui;max-width:1500px;margin:30px auto;"
        "padding:20px;background:#f5f6f8}img{width:100%;background:white}"
        "article{margin:25px 0}h1{font-size:26px}</style>",
        "<h1>V5 数据质量样例：当前高分批次</h1>",
        "<p>这是实际处理样例，不是最终验收报告。吉林一号和基础影像样例尚未齐备，配准未验收。</p>",
        f"<p>实际展示{len(cards)}组独立位置；绿色=晴空，白色=厚云，蓝色=薄云，灰色=云影。RGB拉伸仅供显示，不改变训练数值。</p>",
    ]
    for card in cards:
        body.append(
            f'<article><p>{html.escape(card["pair_id"])}；'
            f'有效比例{card["clear_fraction"]:.1%}；'
            f'整景60%筛选：{card["strict_scene_qualified"]}</p>'
            f'<img src="{card["image"]}" loading="lazy"></article>'
        )
    body.append("</html>")
    (directory / "index.html").write_text("\n".join(body))
    result = {
        "rendered_gaofen": len(cards),
        "rendered_jilin": 0,
        "rendered_dense": 0,
        "required_total": 96,
        "status": "partial_review_material",
        "generated_at": now(),
        "source_quality_root": str(quality_root.resolve()),
    }
    write_json(report_root / "visual_review_summary.json", result)
    return result
