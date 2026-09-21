"""Paired spatial uncertainty for fixed C/Q AP and R RMSE predictions.

This module does not fit readouts, select a model, load test labels, or aggregate task families.
Callers must supply the same fixed evaluation domain and draw schedule for every condition.
Metrics are pooled over the resampled tiles, then averaged over training/support seeds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Sequence

import numpy as np

from xuannv_embedding.downstream.product_bootstrap import _weighted_ap

Metric = Literal["ap", "rmse"]


def _tile_ids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(v, str) or not v for v in result):
        raise ValueError("tile IDs must be nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("tile IDs must be unique")
    return result


def _seeds(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(isinstance(v, bool) or not isinstance(v, Integral) for v in result):
        raise ValueError("seed identities must be nonempty integer sequences")
    if len(set(result)) != len(result):
        raise ValueError("seed identities must be unique")
    return result


def tile_weights(
    tile_ids: Sequence[str], *, repeats: int = 2000, seed: int = 20260921
) -> np.ndarray:
    """Draw equally likely tiles with replacement, retaining within-tile dependence."""
    count = len(_tile_ids(tile_ids))
    if isinstance(repeats, bool) or not isinstance(repeats, Integral) or repeats < 1:
        raise ValueError("bootstrap repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    indices = np.random.default_rng(seed).integers(count, size=(repeats, count))
    return np.array([np.bincount(row, minlength=count) for row in indices], dtype=np.int64)


def _weights(values: np.ndarray, count: int) -> np.ndarray:
    weights = np.asarray(values)
    if weights.ndim != 2 or weights.shape[1] != count or not len(weights):
        raise ValueError("weights must have shape [draws, tiles]")
    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or (weights > count).any()
        or not np.equal(weights, np.floor(weights)).all()
        or not np.equal(weights.sum(1), count).all()
    ):
        raise ValueError("each draw must contain exactly tile-count integer multiplicities")
    return weights.astype(np.int64)


def _digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SeedMetricDraws:
    metric: Metric
    tile_ids: tuple[str, ...]
    training_seeds: tuple[int, ...]
    support_seeds: tuple[int, ...]
    domain_sha256: str
    weights_sha256: str
    observed: np.ndarray  # [training seed, support seed]
    draws: np.ndarray  # [training seed, support seed, bootstrap draw]


def seed_metric_draws(
    truth: np.ndarray,
    predictions: np.ndarray,
    tiles: np.ndarray,
    weights: np.ndarray,
    *,
    tile_ids: Sequence[str],
    training_seeds: Sequence[int],
    support_seeds: Sequence[int],
    metric: Metric,
) -> SeedMetricDraws:
    """Score already fixed predictions; excluded/invalid observations must be removed upstream.

    ``tiles`` maps each retained observation to its position in ``tile_ids``. The tile list
    includes tiles with no eligible observations. AP without a sampled positive and RMSE
    without sampled observations are undefined (NaN), never silently replaced by zero.
    """
    ids, train, support = _tile_ids(tile_ids), _seeds(training_seeds), _seeds(support_seeds)
    truth = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    tiles = np.asarray(tiles)
    if truth.ndim != 1 or tiles.shape != truth.shape:
        raise ValueError("truth and tile positions must be matching one-dimensional arrays")
    if predictions.shape != (len(train), len(support), len(truth)):
        raise ValueError("predictions must match training seeds, support seeds and observations")
    if not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError("truth and predictions must be finite on the fixed domain")
    if (
        not np.isfinite(tiles).all()
        or not np.equal(tiles, np.floor(tiles)).all()
        or (tiles < 0).any()
        or (tiles >= len(ids)).any()
    ):
        raise ValueError("tile positions must be integers within the registered tile list")
    tiles = tiles.astype(np.int64)
    weights = _weights(weights, len(ids))
    if metric not in ("ap", "rmse"):
        raise ValueError("only AP and RMSE primary metrics are supported")
    if metric == "ap" and not np.isin(truth, (0, 1)).all():
        raise ValueError("AP requires binary truth")
    all_weights = np.concatenate([np.ones((1, len(ids)), dtype=np.int64), weights])
    counts = all_weights @ np.bincount(tiles, minlength=len(ids))
    positives = all_weights @ np.bincount(tiles, weights=truth, minlength=len(ids))
    output = np.empty((len(train), len(support), len(all_weights)), dtype=np.float64)
    for i in range(len(train)):
        for j in range(len(support)):
            prediction = predictions[i, j]
            if metric == "ap":
                order = np.argsort(-prediction, kind="stable")
                ordered = prediction[order]
                ends = np.r_[ordered[:-1] != ordered[1:], True] if len(truth) else np.empty(0, bool)
                values = _weighted_ap(truth[order], tiles[order], ends, all_weights)
                values[positives == 0] = np.nan
            else:
                squares = np.bincount(tiles, weights=(prediction - truth) ** 2, minlength=len(ids))
                values = np.full(len(all_weights), np.nan)
                valid = counts > 0
                values[valid] = np.sqrt((all_weights[valid] @ squares) / counts[valid])
            output[i, j] = values
    output.setflags(write=False)
    return SeedMetricDraws(
        metric,
        ids,
        train,
        support,
        _digest(truth, tiles),
        _digest(weights),
        output[:, :, 0],
        output[:, :, 1:],
    )


def paired_seed_summary(baseline: SeedMetricDraws, candidate: SeedMetricDraws) -> dict:
    """Average seeds within each common spatial draw, then subtract candidate minus baseline.

    Methods may have different counts of training seeds (e.g. one published embedding), but
    must have the same support-seed identities and spatial draws. The interval is withheld
    if any draw is undefined; no condition or seed is silently dropped with ``nanmean``.
    """
    for key in ("metric", "tile_ids", "support_seeds", "domain_sha256", "weights_sha256"):
        if getattr(baseline, key) != getattr(candidate, key):
            raise ValueError(f"paired inputs differ: {key}")
    if baseline.draws.shape[-1] != candidate.draws.shape[-1]:
        raise ValueError("paired draw counts differ")
    differences = candidate.draws.mean((0, 1)) - baseline.draws.mean((0, 1))
    observed = float(candidate.observed.mean() - baseline.observed.mean())
    defined = np.isfinite(differences)
    return {
        "metric": baseline.metric,
        "difference_direction": "candidate_minus_baseline",
        "observed_difference": observed if np.isfinite(observed) else None,
        "differences": differences,
        "defined_draws": int(defined.sum()),
        "total_draws": len(differences),
        "interval95": (np.percentile(differences, [2.5, 97.5]).tolist() if defined.all() else None),
        "aggregation": "mean of metrics over training and support seeds within each paired draw",
    }
