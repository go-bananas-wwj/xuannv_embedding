"""Supervision-only aggregation; unknown support never becomes background."""

from __future__ import annotations

import numpy as np


def block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    h, w = values.shape[-2:]
    if h % factor or w % factor:
        raise ValueError("Supervision grids must divide exactly")
    return values.reshape(*values.shape[:-2], h // factor, factor, w // factor, factor).mean(
        axis=(-3, -1), dtype=np.float32
    )


def aggregate_continuous(values, valid, factor, *, minimum_coverage=0.5):
    values, valid = np.asarray(values, dtype=np.float32), np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or not np.isfinite(values[valid]).all():
        raise ValueError("Invalid continuous target values or mask")
    support = block_mean(valid, factor)
    numerator = block_mean(np.where(valid, values, 0), factor)
    mask = (support >= minimum_coverage) & (support > 0)
    return {
        "values": np.where(mask, numerator / np.maximum(support, 1e-12), 0).astype(np.float32),
        "valid": mask,
        "weight": np.where(mask, support, 0).astype(np.float32),
    }


def clcd_fractions(values, valid):
    values, valid = np.asarray(values), np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or (valid & ~np.isin(values, range(1, 10))).any():
        raise ValueError("Illegal valid CLCD code or mask shape")
    support = block_mean(valid, 4)
    # Channel axis follows any leading position dimensions.
    fractions = np.stack([block_mean(valid & (values == code), 4) for code in range(1, 10)], -3)
    mask = support >= 0.5
    fractions = fractions / np.maximum(support[..., None, :, :], 1e-12)
    return {
        "values": np.where(mask[..., None, :, :], fractions, 0).astype(np.float32),
        "valid": mask,
        "weight": np.where(mask, support, 0).astype(np.float32),
    }


def dsm_slope(values, valid):
    """Central differences on the locked40m DSM; omit patch edges and missing neighbors."""
    values, valid = np.asarray(values, dtype=np.float32), np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or not np.isfinite(values[valid]).all():
        raise ValueError("Invalid DSM for slope derivation")
    mask = np.zeros_like(valid)
    mask[..., 1:-1, 1:-1] = (
        valid[..., 1:-1, 1:-1]
        & valid[..., :-2, 1:-1]
        & valid[..., 2:, 1:-1]
        & valid[..., 1:-1, :-2]
        & valid[..., 1:-1, 2:]
    )
    dx = (values[..., 1:-1, 2:] - values[..., 1:-1, :-2]) / 80
    dy = (values[..., 2:, 1:-1] - values[..., :-2, 1:-1]) / 80
    slope = np.zeros_like(values)
    slope[..., 1:-1, 1:-1] = np.degrees(np.arctan(np.hypot(dx, dy)))
    return {"values": np.where(mask, slope, 0), "valid": mask, "weight": mask.astype(np.float32)}


def osm_evidence(values, states, confidence, source_bits, negative_states, negative_confidence):
    """Historical positives take precedence over coarse, rule-derived negatives."""
    fields = [
        np.asarray(a)
        for a in (values, states, confidence, source_bits, negative_states, negative_confidence)
    ]
    if len({a.shape for a in fields}) != 1:
        raise ValueError("OSM evidence shapes differ")
    values, states, confidence, source_bits, negative_states, negative_confidence = fields
    if not np.isin(states, [0, 1, 2]).all() or not np.isin(negative_states, [0, 3]).all():
        raise ValueError("Illegal OSM evidence state")
    if any(((a < 0) | (a > 255)).any() for a in (values, confidence, negative_confidence)):
        raise ValueError("OSM target/confidence outside uint8 encoding")
    positive = np.isin(states, [1, 2])
    if (positive & ((source_bits & 1) == 0)).any():
        raise ValueError("Historical OSM positive lacks historical source bit")
    negative = (negative_states == 3) & ~positive
    weight = np.where(positive, confidence, np.where(negative, negative_confidence, 0)) / 255.0
    valid = (positive | negative) & (weight > 0)
    return {
        "values": np.where(positive & valid, values.astype(np.float32) / 255, 0),
        "valid": valid,
        "weight": np.where(valid, weight, 0).astype(np.float32),
    }


def aggregate_structure(evidence):
    weight = evidence["weight"] * evidence["valid"]
    support = block_mean(weight, 2)
    numerator = block_mean(evidence["values"] * weight, 2)
    return {
        "values": (numerator / np.maximum(support, 1e-12)).astype(np.float32),
        "valid": support > 0,
        "weight": support,
    }
