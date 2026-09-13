"""Read every label value without granting source provenance or training approval."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def inspect_label_block(
    family: str,
    name: str,
    values: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    base_states: np.ndarray | None = None,
) -> dict:
    if valid is None:
        valid = np.ones(values.shape, dtype=bool)
    if valid.shape != values.shape:
        raise ValueError("label and validity shapes disagree")
    finite = np.isfinite(values)
    if np.any(valid & ~finite):
        raise ValueError("nonfinite valid label")
    selected = values[valid & finite]
    if name.startswith("valid_masks/") and not np.isin(values, [0, 1]).all():
        raise ValueError("invalid binary label mask")
    if family == "static" and name.startswith("targets/"):
        if "worldcover_" in name:
            allowed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
        elif "clcd_" in name:
            allowed = list(range(1, 10))
        else:
            allowed = None
        if allowed is not None and not np.isin(selected, allowed).all():
            raise ValueError("invalid categorical class or valid NoData")
        if "dem_slope" in name and ((selected < 0).any() or (selected > 90).any()):
            raise ValueError("slope outside degrees contract")
    if "/states/" in name:
        allowed = [0, 3] if family == "reliable_negative" else [0, 1, 2, 3]
        if not np.isin(values, allowed).all():
            raise ValueError("unknown label state code")
        if family == "reliable_negative":
            if base_states is None or base_states.shape != values.shape:
                raise ValueError("overlay requires corresponding base states")
            if np.any((values == 3) & (base_states != 0)):
                raise ValueError("negative overlay conflicts with positive base evidence")
    histogram = {}
    if values.dtype in (np.dtype("uint8"), np.dtype("bool")):
        counts = np.bincount(selected.astype(np.uint8, copy=False), minlength=256)
        histogram = {str(int(key)): int(counts[key]) for key in np.flatnonzero(counts)}
    elif np.issubdtype(values.dtype, np.integer):
        ids, counts = np.unique(selected, return_counts=True)
        histogram = {str(int(key)): int(count) for key, count in zip(ids, counts, strict=True)}
    return {
        "pixels": int(values.size),
        "valid_pixels": int(selected.size),
        "nonfinite_pixels": int((~finite).sum()),
        "minimum": float(selected.min()) if selected.size else None,
        "maximum": float(selected.max()) if selected.size else None,
        "value_counts": histogram,
    }


def audit_target_values(dataset_root: Path, report_root: Path, *, families=None) -> dict:
    manifest_path = dataset_root / "targets/manifest.parquet"
    frame = pd.read_parquet(manifest_path)
    if families is not None:
        frame = frame.loc[frame.family.isin(families)]
    if frame.empty or not frame.registry_order_verified.all():
        raise ValueError("label metadata audit and matching registry are required")
    count = len(pd.read_parquet(dataset_root / "registry/national_62000.parquet"))
    roots = {path: zarr.open_group(path, mode="r") for path in frame.path.unique()}
    source_hash = sha256(manifest_path)
    code_hash = sha256(Path(__file__))
    rows = []
    for item in frame.itertuples():
        root = roots[item.path]
        array = root[item.array]
        key = hashlib.sha256(f"{item.family}/{item.array}".encode()).hexdigest()
        output = report_root / "target_value_shards" / (key + ".json")
        # Read and hash actual values on every run: metadata alone cannot detect pixel changes.
        digest = hashlib.sha256()
        histogram = Counter()
        record = {
            "family": item.family,
            "array": item.array,
            "path": item.path,
            "pixels": 0,
            "valid_pixels": 0,
            "nonfinite_pixels": 0,
            "minimum": None,
            "maximum": None,
            "issues": [],
            "source_manifest_sha256": source_hash,
            "code_sha256": code_hash,
            "provenance_status": "not_granted_by_value_audit",
        }
        if array.shape[0] != count:
            raise ValueError("label first dimension does not match registry")
        valid_array = None
        if item.family == "static" and item.array.startswith("targets/"):
            valid_array = root[item.array.replace("targets/", "valid_masks/", 1)]
        base_array = None
        if item.family == "reliable_negative" and "/states/" in item.array:
            base = zarr.open_group(root.attrs["base_osm30_sidecar"], mode="r")
            if list(base.attrs["patch_ids"]) != list(root.attrs["patch_ids"]):
                raise ValueError("negative overlay and base patch ordering disagree")
            base_array = base[item.array]
        for start in range(0, count, array.chunks[0]):
            stop = min(count, start + array.chunks[0])
            try:
                values = np.asarray(array[start:stop])
                digest.update(values.tobytes(order="C"))
                valid = (
                    np.asarray(valid_array[start:stop], dtype=bool)
                    if valid_array is not None
                    else None
                )
                states = np.asarray(base_array[start:stop]) if base_array is not None else None
                block = inspect_label_block(
                    item.family, item.array, values, valid, base_states=states
                )
                for field in ("pixels", "valid_pixels", "nonfinite_pixels"):
                    record[field] += block[field]
                for field, fn in (("minimum", min), ("maximum", max)):
                    if block[field] is not None:
                        record[field] = (
                            block[field]
                            if record[field] is None
                            else fn(record[field], block[field])
                        )
                histogram.update(block["value_counts"])
            except Exception as exc:
                record["issues"].append(
                    {"start": start, "stop": stop, "type": type(exc).__name__, "reason": str(exc)}
                )
        record.update(
            value_counts=dict(histogram),
            decoded_values_sha256=digest.hexdigest(),
            status="failed" if record["issues"] else "values_checked_provenance_pending",
            finished_at=now(),
        )
        write_json(output, record)
        rows.append(record)
        write_json(
            report_root / "target_value_progress.json",
            {
                "processed_arrays": len(rows),
                "selected_arrays": len(frame),
                "failed_arrays": sum(bool(row["issues"]) for row in rows),
                "status": "running",
                "updated_at": now(),
            },
        )
    flat = [
        {**r, "issues": json.dumps(r["issues"]), "value_counts": json.dumps(r["value_counts"])}
        for r in rows
    ]
    atomic_parquet(pd.DataFrame(flat), report_root / "target_value_audit.parquet")
    result = {
        "processed_arrays": len(rows),
        "selected_arrays": len(frame),
        "failed_arrays": sum(bool(row["issues"]) for row in rows),
        "status": "value_audit_finished_provenance_pending",
        "source_manifest_sha256": source_hash,
        "finished_at": now(),
    }
    write_json(report_root / "target_value_progress.json", result)
    return result
