"""Exact pooled average precision under paired spatial-block resampling."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads

from xuannv_embedding.export.context import dump, sha


def block_weights(blocks: int) -> np.ndarray:
    indices = np.random.default_rng(20260921).integers(blocks, size=(2000, blocks))
    return np.array([np.bincount(row, minlength=blocks) for row in indices], dtype=np.int64)


@njit(parallel=True, cache=True)
def _weighted_ap(y, blocks, ends, weights):
    result = np.zeros(len(weights), dtype=np.float64)
    for r in prange(len(weights)):
        tp = 0.0
        total = 0.0
        previous_tp = 0.0
        area = 0.0
        for i in range(len(y)):
            w = weights[r, blocks[i]]
            total += w
            tp += w * y[i]
            if ends[i]:
                if tp > previous_tp:
                    area += (tp - previous_tp) * tp / total
                previous_tp = tp
        if tp > 0:
            result[r] = area / tp
    return result


def bootstrap_ap(labels, scores, weights):
    labels, scores, weights = np.asarray(labels), np.asarray(scores), np.asarray(weights)
    if labels.shape != scores.shape or labels.ndim < 2:
        raise ValueError("labels and scores must share block/pixel dimensions")
    if weights.ndim != 2 or weights.shape[1] != labels.shape[0]:
        raise ValueError("bootstrap weights must match the number of blocks")
    if np.any(weights < 0) or not np.equal(weights, np.floor(weights)).all():
        raise ValueError("block multiplicities must be nonnegative integers")
    valid = labels >= 0
    y, s = labels[valid], scores[valid]
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all():
        raise ValueError("valid labels must be binary with finite scores")
    blocks = np.broadcast_to(
        np.arange(labels.shape[0]).reshape((-1,) + (1,) * (labels.ndim - 1)), labels.shape
    )[valid]
    order = np.argsort(-s, kind="stable")
    ordered = s[order]
    ends = np.r_[ordered[:-1] != ordered[1:], True] if len(y) else np.empty(0, bool)
    return _weighted_ap(y[order], blocks[order], ends, weights.astype(np.int64))


def _one(arguments):
    path, root, threads = arguments
    set_num_threads(threads)
    row = json.loads(path.read_text())
    pred = path.with_name(path.stem + "_predictions.npz")
    label = root / "prepared" / f"label_{row['task']}.npy"
    output = root / "ap_bootstrap" / row["task"] / row["head"] / path.stem
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = output.with_suffix(".json")
    array = output.with_suffix(".npy")
    identity = {
        "prediction_sha256": sha(pred),
        "label_sha256": sha(label),
        "implementation_sha256": sha(Path(__file__)),
        "seed": 20260921,
        "replicates": 2000,
    }
    if meta.exists() and array.exists():
        old = json.loads(meta.read_text())
        if old.get("inputs") == identity and old.get("array_sha256") == sha(array):
            return str(path.relative_to(root))
    with np.load(pred) as z:
        y = np.load(label, mmap_mode="r")[z["test_indices"]]
        values = bootstrap_ap(y, z["scores"], block_weights(len(y)))
    temporary = array.with_suffix(".tmp")
    with temporary.open("wb") as f:
        np.save(f, values)
    temporary.replace(array)
    dump(meta, {"inputs": identity, "array_sha256": sha(array)})
    return str(path.relative_to(root))


def run_all(root: Path, workers: int, threads: int):
    if not (1 <= workers <= 4 and 1 <= threads <= 4):
        raise ValueError("at most four workers with four threads each")
    paths = [
        p
        for p in sorted((root / "runs").glob("*/*/*.json"))
        if "metrics" in json.loads(p.read_text())
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, name in enumerate(pool.map(_one, [(p, root, threads) for p in paths]), 1):
            print("AP bootstrap", i, "/", len(paths), name, flush=True)
