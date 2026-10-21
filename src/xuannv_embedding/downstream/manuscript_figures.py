"""Plot all registered classes from completed frozen regional queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.frozen_fusion import verified


def score_matrix(document, tasks, scope):
    rows = document["rows"]
    expected = {(head, task) for head in ["mlp", "conv3x3"] for task in tasks}
    observed = [(row["head"], row["task"]) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Every registered task must have exactly one result for both heads")
    lookup = {(row["head"], row["task"]): row for row in rows}
    try:
        result = np.array(
            [[lookup[head, task][scope]["f1"] for task in tasks] for head in ["mlp", "conv3x3"]],
            dtype=float,
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Missing registered query metric") from error
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError("Query F1 must be finite and between zero and one")
    return result


def render(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = json.loads(Path(path).read_text())
    if (
        set(spec) != {"schema", "sources", "tasks", "scope", "output", "title", "baseline"}
        or spec["schema"] != "manuscript-score-figures-v1"
        or spec["scope"] not in {"all_region", "outside_support_and_calibration"}
        or not spec["tasks"]
        or len(spec["sources"]) < 2
        or spec["baseline"] not in spec["sources"]
    ):
        raise ValueError("Invalid self-contained score figure specification")
    matrices = {}
    observations = None
    for label, entry in spec["sources"].items():
        data = json.loads(verified(entry).read_text())
        if (
            data.get("state") != "complete"
            or data.get("heads_refitted") is not False
            or data.get("thresholds_refitted") is not False
        ):
            raise ValueError("Figures require completed frozen queries")
        matrices[label] = score_matrix(data, list(spec["tasks"]), spec["scope"])
        counts = {
            (row["head"], row["task"]): row[spec["scope"]]["observations"] for row in data["rows"]
        }
        if observations is not None and observations != counts:
            raise ValueError("Compared methods must score the same registered observations")
        observations = counts
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    ticks = np.arange(len(spec["tasks"]))
    palette = ["#71869d", "#da923a", "#147e79", "#a3629b"]
    baseline = matrices[spec["baseline"]]
    outputs = []
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        for difference in [False, True]:
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
            displayed = [name for name in matrices if not difference or name != spec["baseline"]]
            width = 0.74 / len(displayed)
            for head, ax in enumerate(axes):
                for j, name in enumerate(displayed):
                    scores = matrices[name][head]
                    values = 100 * (scores - baseline[head] if difference else scores)
                    x = ticks + (j - (len(displayed) - 1) / 2) * width
                    color = palette[list(matrices).index(name) % len(palette)]
                    artist = ax.bar(x, values, width, label=name, color=color)
                    ax.bar_label(
                        artist, fmt="%.2f" if difference else "%.1f", fontsize=8, padding=3
                    )
                ax.set_xticks(ticks, list(spec["tasks"].values()))
                ax.set_title(["MLP", "3 x 3 CNN"][head])
                ax.axhline(0, color="#555555", linewidth=0.7)
                ax.grid(axis="y", alpha=0.18)
                ax.set_axisbelow(True)
                ax.set_ylabel("F1 gain (percentage points)" if difference else "F1 (%)")
                if not difference:
                    ax.set_ylim(0, 105)
                else:
                    ax.margins(y=0.2)
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle(spec["title"])
            stem = "per_class_gain" if difference else "per_class_f1"
            for extension in ["png", "pdf"]:
                destination = root / f"{stem}.{extension}"
                fig.savefig(destination, dpi=180)
                outputs.append({"path": str(destination), "sha256": sha(destination)})
            plt.close(fig)
    result = {
        "state": "complete",
        "spec_sha256": sha(path),
        "implementation_sha256": sha(__file__),
        "sources": spec["sources"],
        "tasks": spec["tasks"],
        "scope": spec["scope"],
        "all_registered_tasks_displayed": True,
        "predictions_or_scores_modified": False,
        "outputs": outputs,
    }
    dump(root / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    render(args.spec)
    return 0
