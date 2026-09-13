"""Conservative translation audit; uncertain matching never becomes an alignment pass."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

PARAMETERS = {
    "window_pixels": 64,
    "minimum_valid_fraction": 0.70,
    "minimum_correlation": 0.65,
    "minimum_windows": 3,
    "maximum_window_deviation_pixels": 0.5,
    "maximum_shift_pixels": 16,
    "upsample_factor": 20,
    "maximum_residual_m": 5.0,
}


def audit_translation(
    reference: np.ndarray, moving: np.ndarray, valid: np.ndarray, *, gsd: float
) -> dict:
    reference = np.asarray(reference, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
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
    windows = []
    for y in (0, reference.shape[0] - 64):
        for x in (0, reference.shape[1] - 64):
            mask = valid[y : y + 64, x : x + 64].copy()
            left, right = reference[y : y + 64, x : x + 64], moving[y : y + 64, x : x + 64]
            if mask.mean() < PARAMETERS["minimum_valid_fraction"]:
                continue
            if min(float(left[mask].std()), float(right[mask].std())) < 1e-4:
                continue
            left = np.where(mask, left - left[mask].mean(), 0)
            right = np.where(mask, right - right[mask].mean(), 0)
            delta, _, _ = phase_cross_correlation(
                left, right, upsample_factor=PARAMETERS["upsample_factor"], normalization=None
            )
            if np.max(np.abs(delta)) > PARAMETERS["maximum_shift_pixels"]:
                continue
            corrected = shift(right, delta, order=1, mode="constant", cval=0)
            overlap = shift(mask.astype(np.uint8), delta, order=0, mode="constant", cval=0) > 0
            overlap &= mask
            overlap[:8] = overlap[-8:] = False
            overlap[:, :8] = overlap[:, -8:] = False
            if overlap.sum() < 512:
                continue
            a, b = left[overlap], corrected[overlap]
            if min(float(a.std()), float(b.std())) < 1e-4:
                continue
            correlation = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(correlation) or correlation < PARAMETERS["minimum_correlation"]:
                continue
            windows.append(
                {
                    "origin_yx": [y, x],
                    "translation_yx_pixels": delta.tolist(),
                    "correlation": correlation,
                    "valid_pixels": int(overlap.sum()),
                }
            )
    result = {
        "status": "uncertain",
        "valid_windows": len(windows),
        "windows": windows,
        "parameters": PARAMETERS,
        "gsd_m": gsd,
        "modifies_source_pixels": False,
    }
    if len(windows) < PARAMETERS["minimum_windows"]:
        result["reason"] = "insufficient reliable textured windows"
        return result
    displacements = np.asarray([w["translation_yx_pixels"] for w in windows])
    median = np.median(displacements, axis=0)
    maximum_deviation = float(np.max(np.linalg.norm(displacements - median, axis=1)))
    result.update(
        translation_yx_m=(median * gsd).tolist(),
        residual_m=float(np.linalg.norm(median) * gsd),
        maximum_window_deviation_pixels=maximum_deviation,
    )
    if maximum_deviation > PARAMETERS["maximum_window_deviation_pixels"]:
        result["reason"] = "inconsistent local translations"
    else:
        result["status"] = (
            "passed" if result["residual_m"] <= PARAMETERS["maximum_residual_m"] else "over_limit"
        )
    return result
