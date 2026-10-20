"""Fixed-threshold spatial uncertainty and boundary/interior attribution."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def _f1(counts):
    tp, fp, fn = np.moveaxis(np.asarray(counts), -1, 0)
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=denominator > 0,
    )


def _counts(truth, prediction, groups, n):
    return np.stack(
        [
            np.bincount(groups, weights=mask, minlength=n)
            for mask in [truth & prediction, ~truth & prediction, truth & ~prediction]
        ],
        -1,
    )


def paired_statistics(
    candidate, baseline, threshold, baseline_threshold, query_positions, records, size
):
    for key in ["truth", "tiles", "valid_indices"]:
        if not np.array_equal(candidate[key], baseline[key]):
            raise ValueError("Paired target values or positions differ")
    truth = np.asarray(candidate["truth"]).astype(bool)
    prediction = np.asarray(candidate["scores"]) >= threshold
    other = np.asarray(baseline["scores"]) >= baseline_threshold
    positions = np.asarray(candidate["valid_indices"])
    if (
        prediction.shape != truth.shape
        or other.shape != truth.shape
        or positions.shape != truth.shape
    ):
        raise ValueError("Prediction vectors or positions differ")
    bounds = np.array([r["bounds"] for r in records])
    center = (bounds[:, :2] + bounds[:, 2:]) / 2
    widths = np.median(bounds[:, 2:] - bounds[:, :2], axis=0)
    cells = np.floor((center - center.min(0)) / (2 * widths) + 1e-6).astype(int)
    _, block = np.unique(cells, axis=0, return_inverse=True)
    local_tile = positions // size**2
    group = block[np.asarray(query_positions)[local_tile]]
    n = int(block.max() + 1)
    a, b = _counts(truth, prediction, group, n), _counts(truth, other, group, n)
    rng = np.random.default_rng(41)
    weights = np.stack([np.bincount(rng.integers(n, size=n), minlength=n) for _ in range(1000)])
    draws = _f1(weights @ a) - _f1(weights @ b)
    finite = draws[np.isfinite(draws)]
    interval = np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) and n >= 5 else None
    dense = np.zeros((len(query_positions), size, size), bool)
    known = np.zeros_like(dense)
    dense.ravel()[positions] = truth
    known.ravel()[positions] = True
    boundary = np.zeros_like(dense)
    for i in range(len(dense)):
        boundary[i] = (
            binary_dilation(dense[i] & known[i], iterations=2)
            & binary_dilation(~dense[i] & known[i], iterations=2)
            & known[i]
        )
    near = boundary.ravel()[positions]
    strata = {}
    for name, mask in [("near_boundary", near), ("interior", ~near)]:
        counts = _counts(truth[mask], prediction[mask], np.zeros(mask.sum(), int), 1).sum(0)
        value = float(_f1(counts))
        strata[name] = {
            "observations": int(mask.sum()),
            "f1": value if np.isfinite(value) else None,
        }
    observed = float(_f1(a.sum(0)) - _f1(b.sum(0)))
    return {
        "f1_difference": observed,
        "interval95": interval,
        "spatial_blocks": n,
        "bootstrap_repeats": 1000,
        "finite_draws": len(finite),
        "strata": strata,
        "thresholds_refitted": False,
        "scope": "conditional uncertainty on an adaptively inspected calibration cohort",
        "boundary_band_pixels": 2,
        "boundary_pixel_size_must_follow_reference": True,
    }
