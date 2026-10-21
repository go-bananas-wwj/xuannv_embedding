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
    if len(native["records"]) != n or any(
        a["patch_id"] != b["patch_id"] for a, b in zip(native["records"], data["records"])
    ):
        raise ValueError("Geographic case record order differs from optical sources")
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
                if not np.array_equal(
                    np.asarray(data["indices"])[z["valid_indices"] // (size * size)], z["tiles"]
                ):
                    raise ValueError("Geographic prediction global IDs do not match coordinates")
                dense = np.full((n, size, size), -1, np.float32)
                dense.ravel()[z["valid_indices"]] = (
                    z["scores"] >= entry["all_region"]["threshold"]
                ).astype(np.float32)
            axes[row, col].imshow(
                aligned(dense[i]) + 1, vmin=0, vmax=2, cmap=cmap, interpolation="nearest"
            )
            if row == 0:
                axes[row, col].set_title(name, fontsize=14)
        axes[row, 0].set_ylabel(names.get(task, task), fontsize=14)
        axes[row, 0].plot(
            [w * 0.08, w * 0.08 + 200 / (1280 / w)], [h * 0.92] * 2, color="white", linewidth=3
        )
        axes[row, 0].text(w * 0.08, h * 0.88, "200 m", color="white", fontsize=14)
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
    axes[0, 0].set_title("Local optical", fontsize=14)
    axes[0, 1].set_title("CLCD reference", fontsize=14)
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


def architecture(output):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_xlim(-1, 16)
    ax.set_ylim(-1.2, 3.5)
    ax.axis("off")
    nodes = {
        "public": (0.2, 2.1, "Public observations\nS1 / S2 / Landsat\n+ terrain"),
        "anchor": (
            3.0,
            2.1,
            "Public feature blocks\n16T + 16V + 32S\n64D anchor + stats95",
        ),
        "local": (
            0.2,
            -0.1,
            "Local optical images\nAcquisition intervals\nCoverage masks",
        ),
        "vision": (
            3.0,
            -0.1,
            "Frozen DINOv3\nFine: 8 m, PCA64\nContext: 32 m, 1024D",
        ),
        "fusion": (6.1, 1.0, "Regional fusion\nSemantic branch\n+ structural branch"),
        "base": (8.8, 1.0, "Shared feature base\nB = .85Es + .15Eg"),
        "guide": (11.6, 1.0, "Per-view AnyUp\nFrozen weights\nCoverage increments"),
        "output": (14.3, 1.0, "Reusable embedding\n64D / 10 m grid"),
    }
    for name, (x, y, label) in nodes.items():
        fill = "#e8f5f2" if name in ["fusion", "guide", "output"] else "#edf1f6"
        ax.add_patch(
            FancyBboxPatch(
                (x - 1.15, y - 0.48),
                2.3,
                0.96,
                boxstyle="round,pad=0.06",
                facecolor=fill,
                edgecolor="#365366",
                linewidth=1.1,
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=14)

    def arrow(a, b, curve=0):
        x, y, _ = nodes[a]
        xx, yy, _ = nodes[b]
        ax.add_patch(
            FancyArrowPatch(
                (x + 1.2, y),
                (xx - 1.2, yy),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#526775",
                connectionstyle=f"arc3,rad={curve}",
            )
        )

    for a, b in [
        ("public", "anchor"),
        ("local", "vision"),
        ("anchor", "fusion"),
        ("vision", "fusion"),
        ("fusion", "base"),
        ("base", "guide"),
        ("guide", "output"),
    ]:
        arrow(a, b)
    ax.add_patch(
        FancyArrowPatch(
            (1.4, -0.18),
            (11.6, 0.48),
            arrowstyle="-|>",
            mutation_scale=12,
            color="#2b9787",
            connectionstyle="arc3,rad=.22",
        )
    )
    ax.text(
        6.0,
        2.7,
        "Regional weak supervision: WorldCover + OSM",
        fontsize=13,
        ha="center",
        color="#365366",
    )
    ax.annotate(
        "",
        xy=(6.1, 1.54),
        xytext=(6.1, 2.43),
        arrowprops={"arrowstyle": "-|>", "color": "#526775", "linestyle": ":"},
    )
    ax.text(
        14.3,
        -0.1,
        "Classification / retrieval\nCoverage estimation",
        ha="center",
        va="center",
        fontsize=13,
        color="#365366",
    )
    ax.annotate(
        "",
        xy=(14.3, 0.15),
        xytext=(14.3, 0.47),
        arrowprops={"arrowstyle": "-|>", "color": "#526775"},
    )
    fig.tight_layout()
    fig.savefig(output / "architecture.pdf", bbox_inches="tight")
    fig.savefig(output / "architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output / "architecture.pdf"
