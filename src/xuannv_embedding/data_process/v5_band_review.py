"""Frozen visual diagnostics for measured native-band offsets; never rewrites sources."""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.ndimage import shift

from xuannv_embedding.data_process.v5_intraband import (
    _code_fingerprint,
    _inventory,
    _read_row,
    _root,
)
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def select_examples(rows: list[dict], *, sample_size: int = 12) -> list[dict]:
    if sample_size <= 0:
        raise ValueError("sample size must be positive")
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["observation_id"]):
        if row["status"] in {"passed", "uncertain", "over_limit"}:
            groups[(row["status"], row["sensor"])].append(row)
    selected, used = [], set()
    while len(selected) < sample_size:
        added = False
        for key in sorted(groups):
            while groups[key] and groups[key][0]["patch_id"] in used:
                groups[key].pop(0)
            if groups[key] and len(selected) < sample_size:
                row = groups[key].pop(0)
                selected.append(row)
                used.add(row["patch_id"])
                added = True
        if not added:
            break
    return selected


def stretch_band(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    result = np.zeros(values.shape, dtype="f4")
    if valid.any():
        low, high = np.percentile(values[valid], [2, 98])
        result[valid] = np.clip((values[valid] - low) / max(float(high - low), 1e-12), 0, 1)
    return result


def _overlay(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.stack([a, b, np.zeros_like(a)], axis=-1)
    result[~valid] = 0
    return result


def render_example(row, family: str, record: dict, output: Path) -> None:
    frame, gsd, ref_name = _read_row(row, family)
    bands = frame.band_ids
    ref = bands.index(ref_name)
    pairs = record["result"]["pairs"]
    rgb_names = ("B5", "B4", "B3") if family == "jilin1" else ("red", "green", "blue")
    rgb = np.stack(
        [
            stretch_band(frame.values[bands.index(b)], frame.valid[bands.index(b)])
            for b in rgb_names
        ],
        axis=-1,
    )
    fig, axes = plt.subplots(len(pairs) + 1, 4, figsize=(16, 3 * (len(pairs) + 1)), squeeze=False)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original RGB / display stretch only")
    axes[0, 1].imshow(frame.valid.all(axis=0), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Joint data mask (not cloud QA)")
    for ax in axes[0, 2:]:
        ax.axis("off")
    axes[0, 2].text(
        0,
        0.9,
        "Red = reference; green = moving band\nYellow = similar intensity\n"
        "Spectral differences can also create colors.\nOffsets are algorithm estimates.\n"
        "Preview translation changes no source file.\nNo base-image alignment is certified.",
        va="top",
        fontsize=11,
    )
    for index, pair in enumerate(pairs, 1):
        moving = bands.index(pair["moving_band"])
        joint = frame.valid[ref] & frame.valid[moving]
        a = stretch_band(frame.values[ref], frame.valid[ref])
        b = stretch_band(frame.values[moving], frame.valid[moving])
        axes[index, 0].imshow(a, cmap="gray", vmin=0, vmax=1)
        axes[index, 0].set_title(f"Reference {ref_name}")
        axes[index, 1].imshow(b, cmap="gray", vmin=0, vmax=1)
        axes[index, 1].set_title(f"Moving {pair['moving_band']}")
        axes[index, 2].imshow(_overlay(a, b, joint))
        axes[index, 2].set_title(f"Original overlay: {pair['status']}")
        for window in pair["windows"]:
            y, x = window["origin_yx"]
            size = pair["parameters"]["window_pixels"]
            axes[index, 2].add_patch(
                Rectangle((x, y), size, size, fill=False, edgecolor="cyan", lw=0.7)
            )
        delta = pair.get("translation_yx_m")
        if pair["status"] != "uncertain" and delta is not None:
            pixels = np.asarray(delta) / gsd
            moved = shift(b, pixels, order=1, mode="constant", cval=0)
            moved_valid = (
                shift(frame.valid[moving].astype("f4"), pixels, order=1, mode="constant", cval=0)
                >= 1 - 1e-6
            )
            axes[index, 3].imshow(_overlay(a, moved, frame.valid[ref] & moved_valid))
            axes[index, 3].set_title(f"Preview only: y={delta[0]:.2f}, x={delta[1]:.2f} m")
        else:
            axes[index, 3].axis("off")
            axes[index, 3].text(
                0,
                0.6,
                "No reliable preview translation\n" + pair.get("reason", ""),
                wrap=True,
                fontsize=10,
            )
        for ax in axes[index]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"{row.sensor} / {row.year} / {row.patch_id}\n"
        "Native band audit; full source and receipt hashes in samples.csv",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    temporary = output.with_suffix(".partial.png")
    fig.savefig(temporary, dpi=100)
    plt.close(fig)
    temporary.replace(output)


def review_native_bands(
    dataset_root: Path,
    report_root: Path,
    family: str,
    *,
    version: str = "v1",
    sample_size: int = 12,
) -> dict:
    audit = _root(dataset_root, family, version)
    calibration_path = audit / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    if (
        calibration["status"] != "passed"
        or calibration["fingerprint"]["code_sha256"] != _code_fingerprint()
    ):
        raise ValueError("current calibrated algorithm is required for band review")
    output = report_root / "diagnostics/native_band" / family / version
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "samples.json"
    inventory = _inventory(dataset_root, family)
    owners = {r.observation_id: r for r in inventory.itertuples()}
    fingerprint = {
        "family": family,
        "version": version,
        "calibration_sha256": sha256(calibration_path),
        "sample_size": sample_size,
        "review_code_sha256": sha256(Path(__file__)),
    }
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["fingerprint"] != fingerprint:
            raise ValueError("frozen review changed; use a new review artifact directory")
    else:
        candidates = []
        for receipt in sorted((audit / "receipts").glob("*/*.json")):
            record = json.loads(receipt.read_text())
            result = record["result"]
            if record["fingerprint"].get("calibration_sha256") != fingerprint["calibration_sha256"]:
                continue
            candidates.append(
                {
                    **{k: result[k] for k in ["observation_id", "patch_id", "sensor", "status"]},
                    "receipt_path": str(receipt),
                    "receipt_sha256": sha256(receipt),
                }
            )
        selected = select_examples(candidates, sample_size=sample_size)
        if not selected:
            raise ValueError("no completed band measurements available for review")
        selection = {
            "fingerprint": fingerprint,
            "scope": "fixed diagnostic selection from completed receipts; not national rates",
            "available_receipts": len(candidates),
            "samples": selected,
            "selected_at": now(),
        }
        write_json(selection_path, selection)
    cards = []
    for index, sample in enumerate(selection["samples"]):
        receipt = Path(sample["receipt_path"])
        if sha256(receipt) != sample["receipt_sha256"]:
            raise ValueError("selected measurement receipt changed")
        record = json.loads(receipt.read_text())
        row = owners[sample["observation_id"]]
        expected = {
            k: getattr(row, k)
            for k in ["observation_id", "patch_id", "sensor", "split", "file_sha256"]
        }
        expected["year"] = int(row.year)
        if any(record["fingerprint"].get(k) != v for k, v in expected.items()):
            raise ValueError("selected measurement no longer matches catalog")
        path = output / f"sample_{index:02d}.png"
        render_example(row, family, record, path)
        cards.append({**sample, **expected, "image": path.name, "image_sha256": sha256(path)})
    pd.DataFrame(cards).to_csv(output / "samples.csv", index=False)
    body = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>原生波段配准诊断</title>',
        "<style>body{font:16px system-ui;margin:30px auto;max-width:1500px}"
        "img{width:100%}article{margin:35px 0}</style>",
        "<h1>原生波段配准诊断：固定实际样例</h1>",
        "<p>这些样例按状态和卫星分组抽取，不代表全国比例。红绿叠加中的颜色也可能来自光谱差异；超限是算法估计，仍需人工判断。右列只预览可靠估计的平移，未修改原图。该审核不证明与基础影像的绝对配准，不授权融合或训练。</p>",
        f"<p>实际 {len(cards)} 组，目标 {sample_size} 组；样例不足不重复补齐。</p>",
    ]
    for card in cards:
        body.append(
            f'<article><p>{html.escape(card["observation_id"])} / {card["status"]}</p>'
            f'<img loading="lazy" src="{card["image"]}"></article>'
        )
    (output / "index.html").write_text("\n".join(body) + "</html>")
    result = {
        "status": "diagnostic_review_material",
        "scope": selection["scope"],
        "rendered": len(cards),
        "required": sample_size,
        "counts": dict(Counter(c["status"] for c in cards)),
        "selection_sha256": sha256(selection_path),
        "output": str(output),
        "training_authorized": False,
        "pixel_fusion_authorized": False,
        "finished_at": now(),
    }
    write_json(output / "summary.json", result)
    return result
