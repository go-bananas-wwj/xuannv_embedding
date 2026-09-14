"""Rotate registered spatial groups without changing or duplicating cached samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xuannv_embedding.training.experiment import _json, _sha


def derive_fold_cache(cache: Path, output: Path, test_group: int) -> None:
    if test_group not in range(5):
        raise ValueError("test group must be between zero and four")
    source = cache / "cache.json"
    document = json.loads(source.read_text())
    records = document["records"]
    count = len(records)
    groups = [document["split"][f"group{i}"] for i in range(5)]
    members = [index for group in groups for index in group]
    if any(not group for group in groups) or sorted(members) != list(range(count)):
        raise ValueError("five groups must partition all cached samples exactly once")
    if [record["index"] for record in records] != list(range(count)):
        raise ValueError("cache record indices must match their positions")
    bounds = np.asarray([record["bounds"] for record in records], dtype=np.float64)
    if bounds.shape != (count, 4) or not np.isfinite(bounds).all():
        raise ValueError("cache bounds must be finite projected rectangles")
    sizes = bounds[:, 2:] - bounds[:, :2]
    tile_size = sizes[0, 0]
    if tile_size <= 0 or not np.allclose(sizes, tile_size, rtol=0, atol=0.01):
        raise ValueError("fold rotation requires uniform square projected tiles")
    centers = (bounds[:, 2:] + bounds[:, :2]) / 2
    validation_group = (test_group - 1) % 5
    test, validation = groups[test_group], groups[validation_group]
    held = centers[test + validation]
    distance = np.max(np.abs(centers[:, None] - held[None]), axis=-1).min(axis=1)
    candidates = set(range(count)) - set(test + validation)
    train = sorted(index for index in candidates if distance[index] > tile_size + 0.01)
    if not train:
        raise ValueError("spatial buffer leaves no training patches")
    document["split"] = {
        "train": train,
        "validation": validation,
        "test": test,
        "buffer": sorted(candidates - set(train)),
        **{f"group{i}": group for i, group in enumerate(groups)},
    }
    document["fold_provenance"] = {
        "parent_cache": str(source.resolve()),
        "parent_sha256": _sha(source),
        "test_group": test_group,
        "validation_group": validation_group,
        "rule": "fixed groups; previous cyclic group validates; one-tile training buffer",
        "tile_size": float(tile_size),
    }
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "cache.json", document)
