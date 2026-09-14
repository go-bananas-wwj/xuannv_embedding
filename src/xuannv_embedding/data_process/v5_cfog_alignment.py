"""Experimental masked CFOG matching; independent calibration is required before use.

Unsigned derivative channels and smoothing follow Ye et al., TGRS 2019,
https://arxiv.org/abs/1808.06194v8. Mask support, channel-centered NCC,
window rejection and subpixel refinement are explicit project adaptations.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, correlate1d, gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.signal import correlate

from xuannv_embedding.data_process.v5_adaptive_alignment import select_windows

PARAMETERS = {
    "method": "masked_CFOG_channel_centered_NCC_candidate_v1",
    "orientation_channels": 9,
    "gaussian_sigma_pixels": 0.8,
    "gaussian_radius_pixels": 3,
    "derivative_kernel": [-1, 0, 1],
    "orientation_smoothing": [0.25, 0.5, 0.25],
    "minimum_valid_fraction": 0.70,
    "minimum_correlation": 0.80,
    "minimum_peak_margin": 0.01,
    "peak_exclusion_radius_pixels": 3,
    "window_pixels": 64,
    "maximum_shift_pixels": 16,
    "minimum_windows": 3,
    "maximum_window_deviation_pixels": 0.5,
    "maximum_residual_m": 5.0,
    "upsample_factor": 20,
    "thresholds_calibrated": False,
}


def cfog_features(values, valid):
    values, valid = np.asarray(values, dtype="f8"), np.asarray(valid)
    if values.ndim != 2 or valid.dtype != bool or values.shape != valid.shape:
        raise ValueError("CFOG requires a two-dimensional image and boolean valid mask")
    if not np.isfinite(values[valid]).all():
        raise ValueError("valid source values must be finite")
    radius = PARAMETERS["gaussian_radius_pixels"]
    support = binary_erosion(valid, structure=np.ones((2 * radius + 3,) * 2, bool))
    output = np.zeros((PARAMETERS["orientation_channels"], *values.shape), "f8")
    if not valid.any() or not support.any() or values[valid].std() <= 1e-12:
        return output, support
    normalized = np.zeros_like(values)
    normalized[valid] = (values[valid] - values[valid].mean()) / values[valid].std()
    dx = correlate1d(normalized, [-1, 0, 1], axis=1, mode="constant")
    dy = correlate1d(normalized, [-1, 0, 1], axis=0, mode="constant")
    for i in range(len(output)):
        theta = i * np.pi / len(output)
        output[i] = gaussian_filter(
            np.abs(np.cos(theta) * dx + np.sin(theta) * dy),
            PARAMETERS["gaussian_sigma_pixels"],
            radius=radius,
            mode="constant",
        )
    output = (np.roll(output, 1, axis=0) + 2 * output + np.roll(output, -1, axis=0)) / 4
    output[:, ~support] = 0
    return output, support


def feature_ncc(template, search, tvalid, svalid):
    """FFT overlap sums with a separate spatial mean for each orientation channel."""
    tm, sm = tvalid.astype("f8"), svalid.astype("f8")

    def corr(a, b):
        return correlate(a, b, mode="valid", method="fft")

    count = np.maximum(corr(sm, tm), 0)
    safe = np.maximum(count, 1)
    cov, vart, vars_ = [np.zeros_like(count) for _ in range(3)]
    for t, s in zip(template, search, strict=True):
        t, s = np.where(tvalid, t, 0), np.where(svalid, s, 0)
        st, ss = corr(sm, t), corr(s, tm)
        cov += corr(s, t) - st * ss / safe
        vart += np.maximum(corr(sm, t * t) - st * st / safe, 0)
        vars_ += np.maximum(corr(s * s, tm) - ss * ss / safe, 0)
    denominator = np.sqrt(vart * vars_)
    allowed = (count >= tvalid.size * PARAMETERS["minimum_valid_fraction"]) & (denominator > 1e-10)
    scores = np.full(count.shape, -np.inf)
    np.divide(cov, denominator, out=scores, where=allowed)
    return np.clip(scores, -1, 1, where=np.isfinite(scores), out=scores)


def _match(template, search, tvalid, svalid):
    score = feature_ncc(template, search, tvalid, svalid)
    if not np.isfinite(score).any():
        return None
    peak = np.array(np.unravel_index(np.argmax(score), score.shape), dtype="f8")
    best = float(score[tuple(peak.astype(int))])
    yy, xx = np.indices(score.shape)
    far = np.hypot(yy - peak[0], xx - peak[1]) > PARAMETERS["peak_exclusion_radius_pixels"]
    others = score[far & np.isfinite(score)]
    margin = best - float(others.max()) if others.size else 0.0
    if best < PARAMETERS["minimum_correlation"] or margin < PARAMETERS["minimum_peak_margin"]:
        return None
    grid = np.indices(tvalid.shape, dtype="f8")

    def evaluate(position):
        coords = grid + np.asarray(position)[:, None, None]
        valid = tvalid & (
            map_coordinates(svalid.astype(float), coords, order=1, mode="constant", cval=0)
            >= 1 - 1e-6
        )
        count = int(valid.sum())
        if count < tvalid.size * PARAMETERS["minimum_valid_fraction"]:
            return -1.0, count
        a = template[:, valid]
        b = np.stack(
            [map_coordinates(c, coords, order=1, mode="constant", cval=0)[valid] for c in search]
        )
        a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        return (float(np.sum(a * b) / denom) if denom > 1e-10 else -1.0), count

    if best < 1 - 1e-8:
        bounds = [(max(0, p - 0.75), min(score.shape[i] - 1, p + 0.75)) for i, p in enumerate(peak)]
        refined = minimize(
            lambda p: 1 - evaluate(p)[0],
            peak,
            method="Powell",
            bounds=bounds,
            options={"xtol": 0.005, "ftol": 1e-8, "maxiter": 20},
        )
        if refined.success and evaluate(refined.x)[0] >= evaluate(peak)[0]:
            peak = refined.x
    peak = np.round(peak * PARAMETERS["upsample_factor"]) / PARAMETERS["upsample_factor"]
    correlation, count = evaluate(peak)
    if correlation < PARAMETERS["minimum_correlation"]:
        return None
    return {
        "translation_yx_pixels": (PARAMETERS["maximum_shift_pixels"] - peak).tolist(),
        "correlation": correlation,
        "peak_margin": margin,
        "valid_pixels": count,
    }


def audit_cfog(reference, moving, reference_valid, moving_valid, *, gsd):
    reference, moving = np.asarray(reference, dtype="f8"), np.asarray(moving, dtype="f8")
    if reference.shape != moving.shape or not np.isfinite(gsd) or gsd <= 0:
        raise ValueError("corresponding grids and finite positive GSD required")
    left, lmask = cfog_features(reference, reference_valid)
    right, rmask = cfog_features(moving, moving_valid)
    # Freeze the same reference-only layout as the baseline; descriptor support is
    # enforced in matching, without double-eroding the layout qualification mask.
    selection = select_windows(reference, reference_valid)
    size, pad = PARAMETERS["window_pixels"], PARAMETERS["maximum_shift_pixels"]
    windows = []
    for y, x in selection["origins_yx"]:
        mask = lmask[y : y + size, x : x + size]
        if mask.mean() < PARAMETERS["minimum_valid_fraction"]:
            continue
        match = _match(
            left[:, y : y + size, x : x + size],
            right[:, y - pad : y + size + pad, x - pad : x + size + pad],
            mask,
            rmask[y - pad : y + size + pad, x - pad : x + size + pad],
        )
        if match:
            windows.append({"origin_yx": [y, x], **match})
    result = {
        "status": "uncertain",
        "valid_windows": len(windows),
        "windows": windows,
        "selection": selection,
        "parameters": PARAMETERS,
        "gsd_m": float(gsd),
        "modifies_source_pixels": False,
        "pixel_fusion_authorized": False,
    }
    if len(windows) < PARAMETERS["minimum_windows"]:
        result["reason"] = "insufficient reliable structural windows"
        return result
    shifts = np.array([w["translation_yx_pixels"] for w in windows])
    median = np.median(shifts, axis=0)
    deviation = float(np.max(np.linalg.norm(shifts - median, axis=1)))
    result.update(
        translation_yx_m=(median * gsd).tolist(),
        residual_m=float(np.linalg.norm(median) * gsd),
        maximum_window_deviation_pixels=deviation,
    )
    if deviation > PARAMETERS["maximum_window_deviation_pixels"]:
        result["reason"] = "inconsistent local translations"
    else:
        result["status"] = (
            "passed" if result["residual_m"] <= PARAMETERS["maximum_residual_m"] else "over_limit"
        )
    return result
