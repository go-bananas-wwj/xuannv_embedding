"""Reference-only non-overlapping window layouts; unchanged native matching thresholds."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion

from xuannv_embedding.data_process.v5_alignment import PARAMETERS, _edges, _match_window

LAYOUT = {
    "method": "reference_only_four_tile_layout_v1",
    "preserve_qualified_corners": True,
    "minimum_template_edge_std": 1e-4,
    "maximum_templates": 4,
    "overlapping_templates": False,
    "selection_uses_moving_pixels_or_matches": False,
    "ranking": "qualified_count, minimum_qualified_coverage, total_qualified_coverage, y, x",
}


def _sums(array, size):
    integral = np.pad(array.astype("f8"), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def select_windows(reference: np.ndarray, valid: np.ndarray) -> dict:
    """Freeze a layout before reading moving pixels or estimating any displacement."""
    reference = np.asarray(reference, dtype="f8")
    valid = np.asarray(valid, dtype=bool)
    size, pad = PARAMETERS["window_pixels"], PARAMETERS["maximum_shift_pixels"]
    if (
        reference.ndim != 2
        or valid.shape != reference.shape
        or min(reference.shape) < 2 * (size + pad)
    ):
        raise ValueError("independent windows require corresponding grids of at least 160 pixels")
    valid = valid & np.isfinite(reference)
    edges = _edges(reference, valid)
    mask = binary_erosion(valid, structure=np.ones((3, 3), bool), border_value=0)
    counts = _sums(mask, size)
    safe = np.maximum(counts, 1)
    mean = _sums(np.where(mask, edges, 0), size) / safe
    variance = np.maximum(_sums(np.where(mask, edges * edges, 0), size) / safe - mean * mean, 0)
    qualified = (counts >= size * size * PARAMETERS["minimum_valid_fraction"]) & (
        variance >= LAYOUT["minimum_template_edge_std"] ** 2
    )
    h, w = reference.shape
    corners = [(y, x) for y in (pad, h - size - pad) for x in (pad, w - size - pad)]
    if sum(bool(qualified[y, x]) for y, x in corners) >= PARAMETERS["minimum_windows"]:
        origins, layout = corners, "original_corners"
    else:
        # Enumerate translations of four disjoint 64px tiles. At most four matches are
        # attempted; a failed match never causes a different layout to be tried.
        nh, nw = h - 2 * (size + pad) + 1, w - 2 * (size + pad) + 1
        tile_counts = np.stack(
            [
                counts[pad + y : pad + y + nh, pad + x : pad + x + nw]
                for y, x in ((0, 0), (0, size), (size, 0), (size, size))
            ]
        )
        tile_ok = np.stack(
            [
                qualified[pad + y : pad + y + nh, pad + x : pad + x + nw]
                for y, x in ((0, 0), (0, size), (size, 0), (size, size))
            ]
        )
        number = tile_ok.sum(0)
        minimum = np.where(tile_ok, tile_counts, np.inf).min(0)
        minimum[number == 0] = 0
        total = np.where(tile_ok, tile_counts, 0).sum(0)
        yy, xx = np.indices(number.shape)
        order = np.lexsort(
            (xx.ravel(), yy.ravel(), -total.ravel(), -minimum.ravel(), -number.ravel())
        )
        y, x = int(yy.ravel()[order[0]] + pad), int(xx.ravel()[order[0]] + pad)
        origins = [(y, x), (y, x + size), (y + size, x), (y + size, x + size)]
        layout = "translated_four_tiles"
    return {
        "layout": layout,
        "origins_yx": [list(origin) for origin in origins],
        "reference_qualified": [bool(qualified[y, x]) for y, x in origins],
        "reference_valid_fractions": [float(counts[y, x] / size**2) for y, x in origins],
        "parameters": LAYOUT,
    }


def audit_adaptive(reference, moving, reference_valid, moving_valid, *, gsd: float) -> dict:
    reference, moving = np.asarray(reference, dtype="f8"), np.asarray(moving, dtype="f8")
    reference_valid, moving_valid = np.asarray(reference_valid, bool), np.asarray(
        moving_valid, bool
    )
    if (
        reference.ndim != 2
        or reference.shape != moving.shape
        or reference_valid.shape != reference.shape
        or moving_valid.shape != reference.shape
        or not np.isfinite(gsd)
        or gsd <= 0
    ):
        raise ValueError("finite positive GSD and corresponding native grids required")
    selection = select_windows(reference, reference_valid)
    joint = reference_valid & moving_valid & np.isfinite(reference) & np.isfinite(moving)
    left, right = _edges(reference, joint), _edges(moving, joint)
    valid = binary_erosion(joint, structure=np.ones((3, 3), bool), border_value=0)
    size, pad = PARAMETERS["window_pixels"], PARAMETERS["maximum_shift_pixels"]
    windows = []
    for y, x in selection["origins_yx"]:
        mask = valid[y : y + size, x : x + size]
        if mask.mean() < PARAMETERS["minimum_valid_fraction"]:
            continue
        template = left[y : y + size, x : x + size]
        if float(template[mask].std()) < LAYOUT["minimum_template_edge_std"]:
            continue
        match = _match_window(
            template,
            right[y - pad : y + size + pad, x - pad : x + size + pad],
            mask,
            valid[y - pad : y + size + pad, x - pad : x + size + pad],
        )
        if match:
            windows.append({"origin_yx": [y, x], **match})
    result = {
        "status": "uncertain",
        "valid_windows": len(windows),
        "windows": windows,
        "selection": selection,
        "parameters": PARAMETERS,
        "gsd_m": gsd,
        "modifies_source_pixels": False,
        "pixel_fusion_authorized": False,
    }
    if len(windows) < PARAMETERS["minimum_windows"]:
        result["reason"] = "insufficient reliable textured and unambiguous windows"
        return result
    displacements = np.array([w["translation_yx_pixels"] for w in windows])
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
