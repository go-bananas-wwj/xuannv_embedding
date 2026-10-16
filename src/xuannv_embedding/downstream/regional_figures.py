"""Render fixed scenes chosen by reference land cover, independently of model scores."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def choose_scenes(
    labels: np.ndarray, valid: np.ndarray, classes: list[int]
) -> list[tuple[int, int]]:
    selected, used = [], set()
    for label in [1, 2, 30, 33, 40, 69, 71, 70, 22]:
        if label not in classes:
            continue
        counts = ((labels == label) & valid).sum(axis=(1, 2))
        order = np.argsort(-counts, kind="stable")
        tile = next((int(i) for i in order if int(i) not in used and counts[i] > 0), None)
        if tile is not None:
            selected.append((label, tile))
            used.add(tile)
    return selected


def render(spec: dict) -> None:
    root, trial = Path(spec["output"]), Path(spec["trial"])
    metadata = json.loads((root / "prepared.json").read_text())
    comparison = json.loads((root / "comparison.json").read_text())
    with np.load(root / "targets.npz") as z:
        labels, valid = z["labels"], z["valid"]
        heights, height_valid = z["heights"], z["height_valid"]
    classes, names = metadata["classes"], metadata["class_names"]
    scenes = choose_scenes(labels, valid, classes)
    output = root / "figures"
    output.mkdir(exist_ok=True)
    predictions = {}
    for model in ["aef", "public", "highres"]:
        maps, owned = np.zeros_like(labels), set()
        for fold in range(5):
            path = root / "scores" / f"{model}_mlp_fold{fold}_n200" / "predictions.npz"
            with np.load(path) as z:
                indices = z["test_tiles"].tolist()
                if owned.intersection(indices):
                    raise ValueError("Duplicated held-out prediction tiles")
                maps[indices] = z["labels"][indices]
                owned.update(indices)
        if len(owned) != len(labels):
            raise ValueError("Held-out predictions do not cover every tile")
        predictions[model] = maps
    colors = plt.get_cmap("tab20")(np.arange(len(classes)))
    cmap = ListedColormap(np.vstack([[0.85, 0.85, 0.85, 1], colors]))
    norm = BoundaryNorm(np.arange(len(classes) + 2) - 0.5, cmap.N)

    def indices(values, mask):
        result = np.zeros_like(values)
        for i, label in enumerate(classes, 1):
            result[(values == label) & mask] = i
        return result

    config = json.loads((trial / "cache/highres/cache.json").read_text())
    data_root = Path(spec["data"])
    stats = json.loads((data_root / "statistics/highres_optical_stats.json").read_text())
    fig, axes = plt.subplots(len(scenes), 5, figsize=(16, len(scenes) * 2.9), squeeze=False)
    for row, (label, tile) in enumerate(scenes):
        sample = torch.load(config["records"][tile]["path"], weights_only=True, mmap=True)
        image = sample["highres_frames"]["highres_optical"].numpy()
        mask = sample["highres_masks"]["highres_optical"][0].numpy() > 0
        rgb = (
            image * np.asarray(stats["std"])[:, None, None]
            + np.asarray(stats["mean"])[:, None, None]
        )
        low, high = np.percentile(rgb[:, mask], [2, 98]) if mask.any() else (0, 1)
        rgb = np.clip((rgb - low) / max(high - low, 1e-6), 0, 1).transpose(1, 2, 0)
        rgb[~mask] = 1
        axes[row, 0].imshow(rgb)
        arrays = [labels[tile], *[predictions[m][tile] for m in ["aef", "public", "highres"]]]
        for column, values in enumerate(arrays, 1):
            axes[row, column].imshow(
                indices(values, valid[tile]), cmap=cmap, norm=norm, interpolation="nearest"
            )
        axes[row, 0].set_ylabel("\n".join(textwrap.wrap(names[str(label)], 23)) + f"\nTile {tile}")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    for axis, title in zip(
        axes[0],
        ["LINZ RGB (3 m)", "LCDB reference (10 m)", "AEF + MLP", "Public + MLP", "LINZ + MLP"],
    ):
        axis.set_title(title)
    fig.legend(
        handles=[Patch(color=colors[i], label=names[str(c)]) for i, c in enumerate(classes)],
        loc="lower center",
        ncol=3,
        fontsize=9,
    )
    fig.suptitle(
        "Fixed reference-selected scenes; held-out predictions; 200 labels per class\n"
        "Each tile spans 1.28 km. Grey pixels are excluded from evaluation."
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    fig.savefig(output / "landcover_comparison.png", dpi=160)
    plt.close(fig)
    scores = {
        model: comparison["comparisons"][f"{model}:mlp:200"]["per_class"]
        for model in ["aef", "public", "highres"]
    }
    fig, axis = plt.subplots(figsize=(14, 5))
    x = np.arange(len(classes))
    for offset, model in [(-0.15, "public"), (0.15, "highres")]:
        difference = [
            100 * (scores[model][str(c)]["f1"] - scores["aef"][str(c)]["f1"]) for c in classes
        ]
        axis.bar(x + offset, difference, 0.3, label=model)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, ["\n".join(textwrap.wrap(names[str(c)], 17)) for c in classes], fontsize=8)
    axis.set_ylabel("F1 difference from AEF (percentage points)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "per_class_f1_difference.png", dpi=160)
    plt.close(fig)
    height_maps = {}
    for model in ["aef", "public", "highres"]:
        with np.load(root / "regression" / f"{model}_predictions.npz") as z:
            height_maps[model] = z["values"]
    subset = [item for label in [69, 71, 70, 1, 33] for item in scenes if item[0] == label]
    fig, axes = plt.subplots(len(subset), 4, figsize=(13, len(subset) * 2.8), squeeze=False)
    for row, (_, tile) in enumerate(subset):
        arrays = [
            heights[tile, 3],
            *[height_maps[m][tile, :, :, 3] for m in ["aef", "public", "highres"]],
        ]
        for column, values in enumerate(arrays):
            image = np.ma.array(values, mask=~height_valid[tile])
            artist = axes[row, column].imshow(image, vmin=0, vmax=50, cmap="viridis")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    for axis, title in zip(
        axes[0], ["LiDAR reference", "AEF + Ridge", "Public + Ridge", "LINZ + Ridge"]
    ):
        axis.set_title(title)
    fig.suptitle(
        "Surface-height p90 (trees and buildings); shared 0-50 m scale, higher values saturated"
    )
    fig.colorbar(artist, ax=axes.ravel().tolist(), label="metres", fraction=0.02)
    fig.savefig(output / "surface_height_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    (output / "scenes.json").write_text(
        json.dumps(
            {"selection": "reference class pixel counts, no predictions used", "scenes": scenes},
            indent=2,
        )
    )
