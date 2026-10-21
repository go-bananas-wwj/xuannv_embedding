"""Paired tile resampling for building-composition errors of fixed readers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xuannv_embedding.export.context import sha


def paired_mae_interval(ours, baseline, *, repeats=1000, seed=41):
    ours, baseline = np.asarray(ours, float), np.asarray(baseline, float)
    if (
        ours.ndim != 2
        or ours.shape[1] != 2
        or baseline.shape != ours.shape
        or not np.isfinite(ours).all()
        or not np.isfinite(baseline).all()
        or (ours < 0).any()
        or (baseline < 0).any()
        or not np.array_equal(ours[:, 0], baseline[:, 0])
        or type(repeats) is not int
        or repeats < 100
    ):
        raise ValueError("Paired tile errors require identical reference-cell counts")
    keep = ours[:, 0] > 0
    ours, baseline = ours[keep], baseline[keep]
    if not len(ours):
        raise ValueError("No eligible query tiles")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        ids = rng.integers(len(ours), size=len(ours))
        differences.append((baseline[ids, 1].sum() - ours[ids, 1].sum()) / ours[ids, 0].sum())
    return dict(
        baseline_minus_fusion_mae=float(
            (baseline[:, 1].sum() - ours[:, 1].sum()) / ours[:, 0].sum()
        ),
        percentile95=np.quantile(differences, [0.025, 0.975]).tolist(),
        eligible_tiles=len(ours),
        observations=int(ours[:, 0].sum()),
        repeats=repeats,
        seed=seed,
        unit="Paired query tiles; MAE weights pixels within each sampled set",
        scope="Conditional on the fixed representations and fitted readers",
    )


def query_statistics(rows, indices, excluded, stratum):
    if len(rows) != len(indices) or len(set(indices)) != len(indices):
        raise ValueError("Invalid tile error inventory")
    return [[x[stratum][0], x[stratum][2]] for x, i in zip(rows, indices) if i not in excluded]


def build(root, paper):
    root, paper = Path(root), Path(paper)
    mixed = root / "mixed-composition-v2"
    if not (mixed / "spec.json").exists():
        mixed = root / "mixed-composition"
    native_path = mixed / "native/result.json"
    native = json.loads(native_path.read_text())
    spec = json.loads((mixed / "spec.json").read_text())
    data = json.loads(Path(spec["reference"]["path"]).read_text())
    results, sources = [], [{"path": str(native_path), "sha256": sha(native_path)}]
    for model in ["AEF", "public", "Raw180", "degraded"]:
        path = mixed / model / "result.json"
        if not path.exists():
            continue
        baseline = json.loads(path.read_text())
        sources.append({"path": str(path), "sha256": sha(path)})
        for a, b in zip(native["rows"], baseline["rows"]):
            for key in ["budget_tiles", "support_tile_ids", "calibration_tile_ids"]:
                if a[key] != b[key]:
                    raise ValueError("Composition comparison partitions differ")
            excluded = set(a["support_tile_ids"]) | set(a["calibration_tile_ids"])
            if len(a["tile_error_statistics"]) != len(b["tile_error_statistics"]):
                raise ValueError("Composition tile inventories differ")
            for stratum in ["all_cells", "mixed_cells"]:
                aa = query_statistics(
                    a["tile_error_statistics"], data["indices"], excluded, stratum
                )
                bb = query_statistics(
                    b["tile_error_statistics"], data["indices"], excluded, stratum
                )
                result = paired_mae_interval(aa, bb)
                scope = "outside_support_and_calibration"
                expected = b["metrics"][scope][stratum]["mae"] - a["metrics"][scope][stratum]["mae"]
                if not np.isclose(result["baseline_minus_fusion_mae"], expected, atol=1e-12):
                    raise ValueError("Bootstrap point estimate differs from frozen query")
                results.append(
                    dict(baseline=model, budget=a["budget_tiles"], stratum=stratum, **result)
                )
    ledger = dict(sources=sources, results=results, code_sha256=sha(__file__))
    (paper / "notes/jstars-composition-intervals.json").write_text(
        json.dumps(ledger, indent=2) + "\n"
    )
    return ledger
