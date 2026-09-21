"""Validation-only paired classification, coverage regression and cosine retrieval.

This protocol evaluates fixed embeddings, never selects against held-out test labels,
and treats map-derived coverage regression as a proxy, not biophysical ground truth.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from xuannv_embedding.downstream.fixed_audit import balanced_positions, nested_support, threshold
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.cli import _git_sha

PROTOCOL = "multitask-v3"


def check_partition(split, count):
    groups = [split[k] for k in ("train", "validation", "test", "buffer")]
    flat = [i for group in groups for i in group]
    if len(flat) != len(set(flat)):
        raise ValueError("partition overlap or duplicate")
    if sorted(flat) != list(range(count)) or any(not g for g in groups[:3]):
        raise ValueError(
            "partition must cover all records exactly, with nonempty evaluation splits"
        )


def block_regression_data(x, y, indices, *, block=16, minimum=0.8):
    """Average features and binary reference coverage on matching valid pixels."""
    h, w = y.shape[1:]
    if block < 1 or h % block or w % block or not 0 < minimum <= 1:
        raise ValueError("invalid block aggregation geometry or minimum coverage")
    features, fractions, tile_ids = [], [], []
    for i in indices:
        for a in range(0, h, block):
            for b in range(0, w, block):
                yy = y[i, a : a + block, b : b + block]
                valid = yy >= 0
                if valid.mean() >= minimum:
                    features.append(x[i, a : a + block, b : b + block][valid].mean(0))
                    fractions.append((yy[valid] == 1).mean())
                    tile_ids.append(i)
    if not features:
        raise ValueError("no eligible regression blocks")
    return np.asarray(features), np.asarray(fractions), np.asarray(tile_ids)


def regression_metrics(y, predictions):
    residual = np.asarray(predictions) - y
    variance = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": float(1 - np.sum(residual**2) / variance) if variance > 1e-15 else None,
    }


def retrieval_prototypes(x, y, train, ids, budget, seed):
    candidates = []
    for i in train:
        cc, n = ndi.label(y[i] == 1, structure=np.ones((3, 3)))
        for component in range(1, n + 1):
            pixels = np.flatnonzero(cc.ravel() == component)
            if len(pixels) >= 4:
                key = hashlib.sha256(f"{seed}:{ids[i]}:{component}".encode()).digest()
                candidates.append((key, i, component, pixels))
    candidates.sort(key=lambda r: r[0])
    if len(candidates) < budget:
        raise ValueError("insufficient training components")
    selected = candidates[:budget]
    prototypes = np.stack([x[i].reshape(-1, x.shape[-1])[p].mean(0) for _, i, _, p in selected])
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError("zero query embedding")
    prototypes /= norms
    samples = [{"tile": i, "component": c, "pixels": len(p)} for _, i, c, p in selected]
    return prototypes, samples


def normalized_score(baseline, candidate):
    a, b = ({r["key"]: r for r in rows} for rows in (baseline, candidate))
    if a.keys() != b.keys() or len(a) != len(baseline) or len(b) != len(candidate):
        raise ValueError("results must be uniquely paired")
    groups = {}
    for key, row in a.items():
        if row["error"] < 1e-6:
            raise ValueError("degenerate baseline requires preregistered exclusion")
        other = b[key]
        if row["family"] != other["family"] or row.get("source") != other.get("source"):
            raise ValueError("paired result schema differs")
        value = (row["error"] - other["error"]) / row["error"]
        if not np.isfinite(value):
            raise ValueError("nonfinite score")
        groups.setdefault((row["family"], row.get("source", "all")), []).append(value)
    families = {}
    for (family, _), values in groups.items():
        families.setdefault(family, []).append(float(np.mean(values)))
    if set(families) != {"C", "R", "Q"}:
        raise ValueError("all three task families are required")
    scores = {k: float(np.mean(v)) for k, v in families.items()}
    return {"score": float(np.mean(list(scores.values()))), "families": scores}


def _task_labels(prepared, active):
    result = {}
    for task in ("building", "road", "water", "green"):
        array = np.load(prepared / f"label_{task}.npy", mmap_mode="r")
        result[f"osm_{task}"] = np.asarray(array[active])
    semantic = np.asarray(np.load(prepared / "semantic.npy", mmap_mode="r")[active])
    for i, task in enumerate(("water", "trees", "range", "crops", "built", "bare")):
        result[f"esri_{task}"] = np.where(semantic < 0, -1, semantic == i).astype(np.int8)
    return result


def _features(model, records, active, out):
    """Validate identity before using a prepared array or a per-tile export."""
    manifest_path = Path(model["manifest"])
    manifest = json.loads(manifest_path.read_text())
    exported = manifest["records"]
    if len(exported) != len(records):
        raise ValueError("export record count mismatch")
    for i in active:
        if any(exported[i][k] != records[i][k] for k in ("patch_id", "bounds")):
            raise ValueError("export grid/order mismatch")
    metadata = {"manifest_sha256": sha(manifest_path)}
    if "array" in model:
        path = Path(model["array"])
        metadata["array_sha256"] = sha(path)
        array = np.load(path, mmap_mode="r")
        if array.shape[:3] != (len(records), 128, 128):
            raise ValueError("prepared embedding geometry mismatch")
        x = np.asarray(array[active])
    else:
        x = np.lib.format.open_memmap(
            out / "active_embeddings.npy",
            mode="w+",
            dtype="float32",
            shape=(len(active), 128, 128, 64),
        )
        digests = {}
        for j, i in enumerate(active):
            path = Path(exported[i]["path"])
            digest = sha(path)
            if "sha256" in exported[i] and exported[i]["sha256"] != digest:
                raise ValueError("embedding digest mismatch")
            digests[records[i]["patch_id"]] = digest
            with np.load(path) as f:
                x[j] = f["embedding"][-1].transpose(1, 2, 0)
        x.flush()
        metadata["tile_sha256"] = digests
    if not np.isfinite(x).all():
        raise ValueError("nonfinite embedding")
    return x, metadata


def _classification(x, y, train, val, ids, budget, seed, out):
    support = nested_support(y, ids, train, budget, seed)
    labels = y[support].ravel()
    positions = balanced_positions(labels, 4096, seed)
    scaler = StandardScaler()
    features = scaler.fit_transform(x[support].reshape(-1, x.shape[-1])[positions])
    target = labels[positions]
    valid = y[val].ravel() >= 0
    truth = y[val].ravel()[valid]
    query = x[val].reshape(-1, x.shape[-1])[valid]
    query = scaler.transform(query)
    trials = []
    best = None
    for alpha in (10.0, 1.0, 0.1):
        model = RidgeClassifier(alpha=alpha).fit(features, target)
        scores = model.decision_function(query)
        ap = float(average_precision_score(truth, scores))
        trials.append({"alpha": alpha, "ap": ap})
        if best is None or ap > best[0] + 1e-12:
            best = (ap, alpha, scores)
    ap, alpha, scores = best
    cut = threshold(truth, scores)
    predictions = scores >= cut
    tp = int((predictions & (truth == 1)).sum())
    fp = int((predictions & (truth == 0)).sum())
    fn = int((~predictions & (truth == 1)).sum())
    np.savez_compressed(out, scores=scores, truth=truth, valid_indices=np.flatnonzero(valid))
    return {
        "ap": ap,
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "iou": tp / max(1, tp + fp + fn),
        "ba": float(balanced_accuracy_score(truth, predictions)),
        "alpha": alpha,
        "threshold": cut,
        "support_tiles": support,
        "support_positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
        "support_pixels": len(positions),
        "validation_pixels": len(truth),
        "candidates": trials,
    }


def _regression(x, y, train, val, ids, budget, seed, out):
    support = sorted(train, key=lambda i: hashlib.sha256(f"{seed}:{ids[i]}".encode()).digest())[
        :budget
    ]
    features, target, _ = block_regression_data(x, y, support)
    query, truth, tiles = block_regression_data(x, y, val)
    scaler = StandardScaler().fit(features)
    features, query = scaler.transform(features), scaler.transform(query)
    best, trials = None, []
    for alpha in (10.0, 1.0, 0.1):
        predictions = np.clip(Ridge(alpha=alpha).fit(features, target).predict(query), 0, 1)
        metrics = regression_metrics(truth, predictions)
        trials.append({"alpha": alpha, **metrics})
        if best is None or metrics["rmse"] < best[0]["rmse"] - 1e-12:
            best = (metrics, alpha, predictions)
    metrics, alpha, predictions = best
    np.savez_compressed(out, predictions=predictions, truth=truth, tiles=tiles)
    return {
        **metrics,
        "alpha": alpha,
        "support_tiles": support,
        "support_blocks": len(target),
        "validation_blocks": len(truth),
        "candidates": trials,
        "training_mean_baseline": regression_metrics(truth, np.full_like(truth, target.mean())),
    }


def _retrieval(x, y, train, val, ids, budget, seed, out):
    prototypes, samples = retrieval_prototypes(x, y, train, ids, budget, seed)
    query = x[val].reshape(-1, x.shape[-1])
    valid = y[val].ravel() >= 0
    query = query[valid]
    query /= np.maximum(1e-12, np.linalg.norm(query, axis=1, keepdims=True))
    scores = (query @ prototypes.T).max(1)
    truth = y[val].ravel()[valid]
    metrics = {"ap": float(average_precision_score(truth, scores)), "queries": samples}
    order = np.argsort(-scores, kind="stable")
    for fraction in (0.01, 0.05):
        k = max(1, int(np.ceil(len(scores) * fraction)))
        hits = int(truth[order[:k]].sum())
        metrics[f"precision_top{fraction}"] = hits / k
        metrics[f"recall_top{fraction}"] = hits / max(1, int(truth.sum()))
    np.savez_compressed(out, scores=scores, truth=truth, valid_indices=np.flatnonzero(valid))
    return metrics


def run(args):
    spec = json.loads(args.spec.read_text())
    if spec["protocol"] != PROTOCOL:
        raise ValueError("unsupported multitask protocol")
    out = Path(spec["output"]) / args.model
    if out.exists():
        raise FileExistsError("never overwrite an evaluation; use a new run id")
    cache_path = Path(spec["cache"])
    cache = json.loads(cache_path.read_text())
    check_partition(cache["split"], len(cache["records"]))
    prepared_identity = json.loads((Path(spec["prepared"]) / "identity.json").read_text())
    if prepared_identity["cache_sha256"] != sha(cache_path):
        raise ValueError("label cache identity mismatch")
    export_manifest = json.loads(Path(spec["models"][args.model]["manifest"]).read_text())
    if export_manifest["months"][-1] != spec["month"]:
        raise ValueError("evaluation month differs from export")
    for part in ("train", "validation", "test", "buffer"):
        if export_manifest["split"][part] != cache["split"][part]:
            raise ValueError("export partition mismatch")
    active = cache["split"]["train"] + cache["split"]["validation"]
    train = list(range(len(cache["split"]["train"])))
    val = list(range(len(train), len(active)))
    ids = [cache["records"][i]["patch_id"] for i in active]
    labels = _task_labels(Path(spec["prepared"]), active)
    # Label-only audit runs before any scores; no task may be silently dropped.
    audit = {}
    for task, y in labels.items():
        if np.unique(y[val][y[val] >= 0]).size != 2:
            raise ValueError(f"validation lacks positive/negative references: {task}")
        for seed in spec["seeds"]:
            nested_support(y, ids, train, max(spec["budgets"]), seed)
        audit[task] = {
            "validation_positive": int((y[val] == 1).sum()),
            "validation_negative": int((y[val] == 0).sum()),
            "active_label_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        }
    out.mkdir(parents=True)
    started = time.monotonic()
    dump(out / "status.json", {"state": "running", "phase": "prepare"})
    try:
        x, feature_identity = _features(spec["models"][args.model], cache["records"], active, out)
        dump(
            out / "identity.json",
            {
                "protocol": PROTOCOL,
                "code_commit": _git_sha(),
                "spec_sha256": sha(args.spec),
                "cache_sha256": sha(cache_path),
                "active_indices": active,
                "tasks": audit,
                "feature_identity": feature_identity,
                "test_scored": False,
                "selection_split": "validation",
                "label_provenance": "archived OSM/ESRI maps",
            },
        )
        rows = []
        for family in ("C", "R", "Q"):
            tasks = [
                t
                for t in labels
                if family == "C" or t.startswith("esri" if family == "R" else "osm")
            ]
            budgets = (1, 3, 5) if family == "Q" else spec["budgets"]
            runner = {"C": _classification, "R": _regression, "Q": _retrieval}[family]
            for task in tasks:
                for seed in spec["seeds"]:
                    for budget in budgets:
                        key = f"{family}_{task}_{seed}_{budget}"
                        tic = time.monotonic()
                        metrics = runner(
                            x, labels[task], train, val, ids, budget, seed, out / f"{key}.npz"
                        )
                        row = {
                            "key": key,
                            "family": family,
                            "source": task.split("_")[0],
                            "task": task,
                            "seed": seed,
                            "budget": budget,
                            "metrics": metrics,
                            "error": metrics["rmse"] if family == "R" else 1 - metrics["ap"],
                            "seconds": time.monotonic() - tic,
                        }
                        rows.append(row)
                        dump(out / f"{key}.json", row)
                dump(
                    out / "status.json",
                    {
                        "state": "running",
                        "family": family,
                        "task": task,
                        "conditions_complete": len(rows),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                print(
                    json.dumps(
                        {"model": args.model, "family": family, "task": task, "rows": len(rows)}
                    ),
                    flush=True,
                )
        dump(out / "results.json", rows)
        dump(
            out / "status.json",
            {
                "state": "complete",
                "conditions_complete": len(rows),
                "elapsed_seconds": time.monotonic() - started,
                "test_scored": False,
            },
        )
    except BaseException as exc:
        dump(out / "status.json", {"state": "failed", "error": repr(exc)})
        raise
