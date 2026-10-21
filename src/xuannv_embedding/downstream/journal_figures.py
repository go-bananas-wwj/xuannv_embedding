"""Reference-selected geographic cases on a common physical footprint."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

from xuannv_embedding.export import viewwise_guided_refinement as viewwise
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.frozen_fusion import verified


def geographic_cases(path):
    spec = json.loads(Path(path).read_text())
    if set(spec) != {"guide", "evaluation", "results", "output", "slot"}:
        raise ValueError("Invalid geographic case specification")
    guide = viewwise.specification(verified(spec["guide"]))
    source = viewwise.SourceData(guide)
    data = json.loads(verified(spec["evaluation"]).read_text())
    native = json.loads(verified(guide["index"]).read_text())
    with np.load(verified(data["labels"])) as z:
        labels = {k: z[k] for k in z.files}
    valid = np.load(verified(data["valid"]))
    results = {
        name: json.loads(verified(ref).read_text())["rows"] for name, ref in spec["results"].items()
    }
    n, size = len(data["records"]), data["size"]
    tasks = data["tasks"]["C"]
    tasks = next(iter(tasks.values()))
    fig, axes = plt.subplots(len(tasks), 2 + len(results), figsize=(13.5, 3.1 * len(tasks)))
    cmap = ListedColormap(["#eeeeee", "#314859", "#25ac9c"])
    selected = []
    names = {
        "clcd_crop": "Cropland",
        "clcd_forest": "Forest",
        "clcd_water": "Water",
        "clcd_impervious": "Impervious",
    }
    for row, task in enumerate(tasks):
        entries = {
            name: next(x for x in rows if x["task"] == task and x["head"] == "conv3x3")
            for name, rows in results.items()
        }
        excluded = set(next(iter(entries.values()))["excluded_tile_ids"])
        fraction = ((labels[task] == 1) & valid).sum((1, 2)) / np.maximum(valid.sum((1, 2)), 1)
        candidates = [
            i
            for i in range(n)
            if data["indices"][i] not in excluded and 0.05 <= fraction[i] <= 0.95
        ]
        if not candidates:
            raise ValueError("No query tile with mixed reference coverage")
        median = float(np.median(fraction[candidates]))
        i = min(candidates, key=lambda x: (abs(fraction[x] - median), data["indices"][x]))
        image, coverage = list(source.raw.frames(i))[spec["slot"]]
        h, w = image.shape[-2:]
        bounds = native["records"][i]["bounds"]
        native_crs = native.get("crs", native["records"][i].get("crs"))
        dst_transform = from_bounds(*bounds, w, h)
        src_transform = from_bounds(*data["records"][i]["bounds"], size, size)

        def aligned(values):
            out = np.full((h, w), -1, np.float32)
            reproject(
                values.astype(np.float32),
                out,
                src_transform=src_transform,
                src_crs=data["crs"],
                dst_transform=dst_transform,
                dst_crs=native_crs,
                src_nodata=-1,
                dst_nodata=-1,
                resampling=Resampling.nearest,
            )
            return out

        rgb = image.transpose(1, 2, 0).copy()
        observed = coverage[0] > 0
        if observed.any():
            low, high = np.percentile(rgb[observed], [2, 98], axis=0)
            rgb = np.clip((rgb - low) / np.maximum(high - low, 1), 0, 1)
        rgb[~observed] = 1
        axes[row, 0].imshow(rgb)
        reference = aligned(np.where(valid[i], labels[task][i], -1))
        axes[row, 1].imshow(reference + 1, vmin=0, vmax=2, cmap=cmap, interpolation="nearest")
        for col, (name, entry) in enumerate(entries.items(), 2):
            p = verified(entry["prediction"])
            with np.load(p) as z:
                dense = np.full((n, size, size), -1, np.float32)
                dense.ravel()[z["valid_indices"]] = (
                    z["scores"] >= entry["all_region"]["threshold"]
                ).astype(np.float32)
            axes[row, col].imshow(
                aligned(dense[i]) + 1, vmin=0, vmax=2, cmap=cmap, interpolation="nearest"
            )
            if row == 0:
                axes[row, col].set_title(name)
        axes[row, 0].set_ylabel(names.get(task, task), fontsize=11)
        axes[row, 0].plot(
            [w * 0.08, w * 0.08 + 200 / (1280 / w)], [h * 0.92] * 2, color="white", linewidth=3
        )
        axes[row, 0].text(w * 0.08, h * 0.88, "200 m", color="white", fontsize=9)
        selected.append(
            {
                "task": task,
                "position": i,
                "patch_id": native["records"][i]["patch_id"],
                "reference_fraction": float(fraction[i]),
                "rule": "median mixed-reference query tile; no score selection",
                "native_bounds": bounds,
                "native_crs": native_crs,
                "reference_crs": data["crs"],
            }
        )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("Local optical")
    axes[0, 1].set_title("CLCD reference")
    fig.tight_layout()
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "geographic-cases.pdf", bbox_inches="tight")
    fig.savefig(out / "geographic-cases.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    dump(
        out / "manifest.json",
        {
            "state": "complete",
            "spec_sha256": sha(path),
            "code_sha256": sha(__file__),
            "slot": spec["slot"],
            "cases": selected,
            "coordinate_alignment": "nearest-neighbour reprojection to original optical footprint",
            "pdf_sha256": sha(out / "geographic-cases.pdf"),
        },
    )
    return out
