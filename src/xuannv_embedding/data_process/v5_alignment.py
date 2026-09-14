"""Masked edge matching without circular-window or shared-NoData alignment bias."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, map_coordinates, sobel
from scipy.optimize import minimize
from scipy.signal import correlate

PARAMETERS = {
    "method": "masked_edge_normalized_cross_correlation",
    "window_pixels": 64,
    "minimum_valid_fraction": 0.70,
    "minimum_correlation": 0.65,
    "minimum_windows": 3,
    "maximum_window_deviation_pixels": 0.5,
    "maximum_shift_pixels": 16,
    "upsample_factor": 20,
    "minimum_peak_margin": 1e-4,
    "peak_exclusion_radius_pixels": 3,
    "maximum_residual_m": 5.0,
}


def _edges(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = array[valid]
    if not selected.size or float(selected.std()) <= 1e-12:
        return np.zeros_like(array, dtype="f8")
    values = np.where(valid, (array - float(selected.mean())) / float(selected.std()), 0)
    return np.hypot(sobel(values, axis=0), sobel(values, axis=1)) / 8


def _masked_ncc(
    template: np.ndarray, search: np.ndarray, template_valid: np.ndarray, search_valid: np.ndarray
) -> np.ndarray:
    tmask = template_valid.astype("f8")
    smask = search_valid.astype("f8")
    t = np.where(template_valid, template, 0)
    s = np.where(search_valid, search, 0)

    def corr(a, b):
        return correlate(a, b, mode="valid", method="fft")

    count = np.maximum(corr(smask, tmask), 0)
    safe = np.maximum(count, 1)
    sum_t = corr(smask, t)
    sum_s = corr(s, tmask)
    var_t = np.maximum(corr(smask, t * t) - sum_t * sum_t / safe, 0)
    var_s = np.maximum(corr(s * s, tmask) - sum_s * sum_s / safe, 0)
    denominator = np.sqrt(var_t * var_s)
    allowed = (count >= template.size * PARAMETERS["minimum_valid_fraction"]) & (
        denominator > 1e-10
    )
    score = np.full(count.shape, -np.inf)
    np.divide(corr(s, t) - sum_t * sum_s / safe, denominator, out=score, where=allowed)
    return np.clip(score, -1, 1, where=np.isfinite(score), out=score)


def _match_window(
    template: np.ndarray, search: np.ndarray, tvalid: np.ndarray, svalid: np.ndarray
) -> dict | None:
    score = _masked_ncc(template, search, tvalid, svalid)
    if not np.isfinite(score).any():
        return None
    peak = np.array(np.unravel_index(np.argmax(score), score.shape), dtype="f8")
    best = float(score[tuple(peak.astype(int))])
    yy, xx = np.indices(score.shape)
    far = np.hypot(yy - peak[0], xx - peak[1]) > PARAMETERS["peak_exclusion_radius_pixels"]
    competitors = score[far & np.isfinite(score)]
    margin = best - float(competitors.max()) if competitors.size else 0.0
    if best < PARAMETERS["minimum_correlation"] or margin < PARAMETERS["minimum_peak_margin"]:
        return None
    grid = np.indices(template.shape, dtype="f8")
    minimum = int(np.ceil(template.size * PARAMETERS["minimum_valid_fraction"]))

    def evaluate(position):
        coords = grid + np.asarray(position)[:, None, None]
        mask = tvalid & (
            map_coordinates(svalid.astype("f8"), coords, order=1, mode="constant", cval=0)
            >= 1 - 1e-6
        )
        count = int(mask.sum())
        if count < minimum:
            return -1.0, count
        moving = map_coordinates(search, coords, order=1, mode="constant", cval=0)
        a = template[mask]
        b = moving[mask]
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
        return (float(np.dot(a, b) / denom) if denom > 1e-10 else -1.0), count

    if best < 1 - 1e-8:
        bounds = [(max(0, v - 0.75), min(score.shape[i] - 1, v + 0.75)) for i, v in enumerate(peak)]
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
    delta = PARAMETERS["maximum_shift_pixels"] - peak
    return {
        "translation_yx_pixels": delta.tolist(),
        "correlation": correlation,
        "peak_margin": margin,
        "valid_pixels": count,
    }


def audit_translation(
    reference: np.ndarray, moving: np.ndarray, valid: np.ndarray, *, gsd: float
) -> dict:
    reference = np.asarray(reference, dtype="f8")
    moving = np.asarray(moving, dtype="f8")
    valid = np.asarray(valid, dtype=bool)
    if (
        reference.shape != moving.shape
        or valid.shape != moving.shape
        or moving.ndim != 2
        or min(moving.shape) < 128
        or gsd <= 0
    ):
        raise ValueError("alignment requires corresponding 2D grids of at least 128 pixels")
    valid = valid & np.isfinite(reference) & np.isfinite(moving)
    left, right = _edges(reference, valid), _edges(moving, valid)
    # Sobel needs a complete 3x3 neighborhood; missing pixels must not create matched edges.
    valid = binary_erosion(valid, structure=np.ones((3, 3), bool), border_value=0)
    size = PARAMETERS["window_pixels"]
    pad = PARAMETERS["maximum_shift_pixels"]
    windows = []
    for y in (pad, reference.shape[0] - size - pad):
        for x in (pad, reference.shape[1] - size - pad):
            mask = valid[y : y + size, x : x + size]
            if mask.mean() < PARAMETERS["minimum_valid_fraction"]:
                continue
            template = left[y : y + size, x : x + size]
            search = right[y - pad : y + size + pad, x - pad : x + size + pad]
            search_valid = valid[y - pad : y + size + pad, x - pad : x + size + pad]
            if float(template[mask].std()) < 1e-4:
                continue
            match = _match_window(template, search, mask, search_valid)
            if match:
                windows.append({"origin_yx": [y, x], **match})
    result = {
        "status": "uncertain",
        "valid_windows": len(windows),
        "windows": windows,
        "parameters": PARAMETERS,
        "gsd_m": gsd,
        "modifies_source_pixels": False,
    }
    if len(windows) < PARAMETERS["minimum_windows"]:
        result["reason"] = "insufficient reliable textured and unambiguous windows"
        return result
    displacements = np.asarray([w["translation_yx_pixels"] for w in windows])
    median = np.median(displacements, axis=0)
    deviation = float(np.max(np.linalg.norm(displacements - median, axis=1)))
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
