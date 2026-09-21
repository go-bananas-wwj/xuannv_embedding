"""Explicit annual/monthly feature contracts for the final multi-task evaluator.

Reads fixed exported features only. It does not select a model, open reference labels,
resample geography, or turn an annual product into monthly observations.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from xuannv_embedding.downstream.multitask import check_partition
from xuannv_embedding.export.context import sha


def _month(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[1-9][0-9]{3}-(0[1-9]|1[0-2])", value))


@dataclass(frozen=True)
class FeatureSelection:
    kind: Literal["monthly", "annual", "raw"]
    period: str
    evaluation_month: str
    channels: int

    def __post_init__(self):
        if self.kind not in ("monthly", "annual", "raw") or not _month(self.evaluation_month):
            raise ValueError("invalid feature kind or evaluation month")
        if type(self.channels) is not int or self.channels < 1:
            raise ValueError("feature channels must be a positive integer")
        if self.kind == "annual":
            if not isinstance(self.period, str) or not re.fullmatch(r"[1-9][0-9]{3}", self.period):
                raise ValueError("annual feature period must be a year")
        elif not _month(self.period):
            raise ValueError("monthly/raw feature period must be YYYY-MM")


@dataclass(frozen=True)
class FeatureBatch:
    values: np.ndarray  # [selected tile, height, width, original feature dimension]
    valid: np.ndarray  # separate feature validity; never inferred from zero vector magnitude
    indices: tuple[int, ...]
    identity: dict


def _registered_json(path: Path, expected: str, name: str) -> dict:
    if sha(path) != expected:
        raise ValueError(f"registered {name} digest changed")
    return json.loads(path.read_text())


def _period_index(selection: FeatureSelection, manifest: dict) -> int:
    months = manifest["months"]
    if selection.kind == "annual":
        if months != ["annual_" + selection.period]:
            raise ValueError("annual product period differs from the registered manifest")
        return 0
    if not months or not all(_month(m) for m in months) or months != sorted(set(months)):
        raise ValueError("monthly periods must be unique ordered calendar months")
    if selection.kind == "raw" and manifest.get("kind") != "raw":
        raise ValueError("raw observation features require an explicit raw export")
    if selection.period not in months:
        raise ValueError("feature period is absent from the export")
    index = months.index(selection.period)
    cutoff = manifest.get("input_ablation", {}).get("last_visible_month_index")
    if cutoff is not None:
        if type(cutoff) is not int or not 0 <= cutoff < len(months) or index > cutoff:
            raise ValueError("requested feature month exceeds the registered input cutoff")
    return index


def _grid(cache: dict, manifest: dict) -> tuple[int, dict]:
    records, exported = cache["records"], manifest["records"]
    if any(
        type(i) is not int
        for group in ("train", "validation", "test", "buffer")
        for i in cache["split"][group]
    ):
        raise ValueError("partition indices must be integers")
    check_partition(cache["split"], len(records))
    if manifest["split"] != cache["split"] or len(exported) != len(records):
        raise ValueError("export split or record count differs from its registered cache")
    size = cache["data"]["patch_size"]
    if type(size) is not int or size < 1:
        raise ValueError("invalid registered patch size")
    ids, layout = set(), []
    for source, other in zip(records, exported):
        patch = source["patch_id"]
        bounds = np.asarray(source["bounds"], dtype=float)
        if not isinstance(patch, str) or not patch or patch in ids:
            raise ValueError("cache tile IDs must be unique nonempty strings")
        ids.add(patch)
        if (
            bounds.shape != (4,)
            or not np.isfinite(bounds).all()
            or not (bounds[0] < bounds[2] and bounds[1] < bounds[3])
        ):
            raise ValueError("invalid registered tile bounds")
        if other["patch_id"] != patch or other["bounds"] != source["bounds"]:
            raise ValueError("export grid or tile order differs from its registered cache")
        layout.append({"patch_id": patch, "bounds": source["bounds"]})
    return size, {"patch_size": size, "records": layout, "split": cache["split"]}


def _tile(path: Path, selection: FeatureSelection, manifest: dict, index: int, size: int):
    with np.load(path, allow_pickle=False) as archive:
        array = archive["embedding"]
        if array.shape != (len(manifest["months"]), selection.channels, size, size):
            raise ValueError("embedding period/channel/grid shape differs from its contract")
        if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
            raise ValueError("embedding must have a real numeric dtype")
        if selection.kind == "annual":
            if "timestamps" in archive:
                raise ValueError("annual export must not claim monthly timestamps")
        elif "timestamps" in archive:
            expected = np.array([int(m.replace("-", "")) for m in manifest["months"]])
            if not np.array_equal(archive["timestamps"], expected):
                raise ValueError("embedding timestamps differ from registered periods")
        elif selection.kind != "raw":
            raise ValueError("monthly embedding requires explicit timestamps")
        valid = np.ones((size, size), dtype=bool)
        if "valid_mask" in archive:
            mask = archive["valid_mask"]
            if mask.dtype != np.bool_:
                raise ValueError("feature validity mask must be boolean")
            if mask.shape == (len(manifest["months"]), size, size):
                valid = mask[index]
            elif mask.shape == (size, size):
                valid = mask
            else:
                raise ValueError("feature validity mask shape differs from the grid")
        values = array[index].transpose(1, 2, 0).astype(np.float32)
        if not np.isfinite(values[valid]).all():
            raise ValueError("nonfinite feature on the declared valid domain")
        values[~valid] = 0
        return values, valid.copy(), "valid_mask" in archive


def read_features(
    manifest_path: Path,
    cache_path: Path,
    *,
    manifest_sha256: str,
    cache_sha256: str,
    tile_sha256: Mapping[str, str],
    selection: FeatureSelection,
    splits: Sequence[str] = ("train", "validation"),
    output: Path | None = None,
) -> FeatureBatch:
    """Read only explicit splits with manifest/cache/tile identities fixed by the caller.

    Original dimensions and a separate validity mask are retained. Full-grid validity when
    no mask is stored refers to exported representations, not observation availability.
    Geographic provenance is inherited from the cache-bound export; CRS/reprojection audits
    for external products remain a required upstream check, not an inference from bounds.
    """
    manifest_path, cache_path = Path(manifest_path), Path(cache_path)
    manifest = _registered_json(manifest_path, manifest_sha256, "manifest")
    cache = _registered_json(cache_path, cache_sha256, "cache")
    if manifest["cache_sha256"] != cache_sha256:
        raise ValueError("export is not bound to the supplied source cache")
    if selection.evaluation_month not in cache["data"]["months"]:
        raise ValueError("evaluation month is outside the registered reference period")
    size, layout = _grid(cache, manifest)
    index = _period_index(selection, manifest)
    splits = tuple(splits)
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(s not in ("train", "validation", "test", "buffer") for s in splits)
    ):
        raise ValueError("requested splits must be distinct registered names")
    indices = tuple(i for split in splits for i in cache["split"][split])
    if not indices:
        raise ValueError("requested splits have no records")
    for i in indices:
        if manifest["records"][i]["patch_id"] not in tile_sha256:
            raise ValueError("registered tile digest inventory is incomplete")
    shape = (len(indices), size, size, selection.channels)
    if output is None:
        values, valid = np.empty(shape, np.float32), np.empty(shape[:-1], bool)
    else:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        values = np.lib.format.open_memmap(
            output / "features.npy", mode="w+", dtype="float32", shape=shape
        )
        valid = np.lib.format.open_memmap(
            output / "valid.npy", mode="w+", dtype="bool", shape=shape[:-1]
        )
    digests, mask_kinds = {}, []
    for j, i in enumerate(indices):
        record = manifest["records"][i]
        path = Path(record["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        digest = sha(path)
        if digest != tile_sha256[record["patch_id"]] or record.get("sha256", digest) != digest:
            raise ValueError("registered embedding tile digest changed")
        values[j], valid[j], explicit = _tile(path, selection, manifest, index, size)
        digests[record["patch_id"]] = digest
        mask_kinds.append("explicit" if explicit else "full exported grid")
    identity = {
        "manifest_sha256": manifest_sha256,
        "cache_sha256": cache_sha256,
        "spatial_layout_sha256": hashlib.sha256(
            json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "kind": selection.kind,
        "feature_period": selection.period,
        "evaluation_month": selection.evaluation_month,
        "temporal_resolution": "annual" if selection.kind == "annual" else "monthly",
        "channels": selection.channels,
        "splits": list(splits),
        "indices": list(indices),
        "tile_sha256": digests,
        "valid_pixels": [int(v.sum()) for v in valid],
        "validity": mask_kinds,
        "input_ablation": manifest.get("input_ablation"),
        "test_records_read": bool(set(indices) & set(cache["split"]["test"])),
        "labels_read": False,
        "training_time_causality_claim": False,
        "spatial_provenance": "cache-bound export; external CRS audit is an upstream requirement",
    }
    if output is not None:
        values.flush()
        valid.flush()
        (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return FeatureBatch(values, valid, indices, identity)
