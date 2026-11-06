"""Trace negative overlays to their recorded annual and cross-class evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import zarr
from scipy.ndimage import binary_erosion

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

TASKS = ("building", "road_all", "water_area", "waterway")
RULES = {
    "building": "eroded WorldCover/OSM water interior",
    "road_all": "eroded known-building interior",
    "water_area": "eroded known-building interior or valid steep DEM",
    "waterway": "eroded known-building interior",
}


def expected_negatives(
    *,
    states: dict[str, np.ndarray],
    worldcover: np.ndarray,
    worldcover_valid: np.ndarray,
    slope: np.ndarray,
    slope_valid: np.ndarray,
    erosion_pixels: int,
    steep_slope_degrees: float,
) -> dict[str, np.ndarray]:
    shape = worldcover.shape
    if set(states) != set(TASKS) or len(shape) not in {2, 3}:
        raise ValueError("four corresponding negative evidence tasks are required")
    arrays = [*states.values(), worldcover_valid, slope, slope_valid]
    if any(a.shape != shape for a in arrays):
        raise ValueError("negative evidence arrays disagree")
    if any(a.dtype != np.dtype("u1") or not np.isin(a, [0, 1]).all() for a in states.values()):
        raise ValueError("negative rules require the audited historical positive/unknown base")
    if (
        type(erosion_pixels) is not int
        or erosion_pixels < 0
        or not np.isfinite(steep_slope_degrees)
        or steep_slope_degrees <= 0
    ):
        raise ValueError("invalid negative evidence parameters")
    if worldcover_valid.dtype != bool or slope_valid.dtype != bool:
        raise ValueError("evidence masks must be boolean")
    if np.any(slope_valid & ~np.isfinite(slope)):
        raise ValueError("valid nonfinite slope")
    positives = {task: a == 1 for task, a in states.items()}

    def erode(mask):
        if erosion_pixels == 0:
            return mask.copy()
        # A unit batch dimension prevents neighborhoods crossing between positions.
        structure = np.ones((1, 3, 3) if mask.ndim == 3 else (3, 3), bool)
        return binary_erosion(mask, structure=structure, iterations=erosion_pixels, border_value=0)

    water_core = erode((worldcover_valid & (worldcover == 80)) | positives["water_area"])
    building_core = erode(positives["building"])
    evidence = {
        "building": water_core,
        "road_all": building_core,
        "water_area": building_core | (slope_valid & (slope >= steep_slope_degrees)),
        "waterway": building_core,
    }
    return {task: mask & (states[task] == 0) for task, mask in evidence.items()}


def inspect_overlay(
    expected: np.ndarray, states: np.ndarray, confidence: np.ndarray, negative_confidence: int
) -> dict:
    if (
        states.shape != expected.shape
        or confidence.shape != expected.shape
        or states.dtype != np.dtype("u1")
        or confidence.dtype != np.dtype("u1")
    ):
        raise ValueError("overlay arrays must have matching shapes and uint8 dtype")
    if type(negative_confidence) is not int or not 1 <= negative_confidence <= 255:
        raise ValueError("invalid negative confidence")
    if not np.isin(states, [0, 3]).all():
        raise ValueError("unexpected overlay state")
    actual = states == 3
    unexpected = int((actual & ~expected).sum())
    missing = int((expected & ~actual).sum())
    confidence_mismatch = int((confidence != np.where(actual, negative_confidence, 0)).sum())
    return {
        "status": "failed" if unexpected or missing or confidence_mismatch else "passed",
        "expected_negative_pixels": int(expected.sum()),
        "stored_negative_pixels": int(actual.sum()),
        "unexpected_negative_pixels": unexpected,
        "missing_negative_pixels": missing,
        "confidence_mismatch_pixels": confidence_mismatch,
    }


def audit_negative_rules(
    dataset_root: Path, report_root: Path, *, max_patches: int | None = None
) -> dict:
    if max_patches is not None and max_patches <= 0:
        raise ValueError("invalid negative rule patch limit")
    registry_path = dataset_root / "registry/national_62000.parquet"
    manifest_path = dataset_root / "targets/manifest.parquet"
    value_path = report_root / "target_value_audit.parquet"
    temporal_path = report_root / "osm_temporal_progress.json"
    registry, manifest = pd.read_parquet(registry_path), pd.read_parquet(manifest_path)
    temporal = json.loads(temporal_path.read_text())
    if (
        temporal.get("status") != "temporal_cross_checks_finished"
        or temporal.get("failed_groups") != 0
    ):
        raise ValueError("completed historical OSM temporal checks are required")
    roots, paths, metadata = {}, {}, {}
    for family in ["osm", "static", "reliable_negative"]:
        entries = manifest.loc[manifest.family == family]
        if (
            entries.empty
            or entries.path.nunique() != 1
            or not entries.registry_order_verified.all()
        ):
            raise ValueError("one verified label registry per evidence family is required")
        path = Path(entries.path.iloc[0])
        meta = sha256(path / ".zattrs")
        if set(entries.source_metadata_sha256) != {meta}:
            raise ValueError("label metadata changed after inventory")
        root = zarr.open_group(str(path), mode="r")
        if list(root.attrs.get("patch_ids", [])) != registry.patch_id.tolist():
            raise ValueError("negative evidence grid ordering changed")
        roots[family], paths[family], metadata[family] = root, str(path), meta
    if registry.empty or registry.patch_id.duplicated().any():
        raise ValueError("invalid negative evidence registry")
    attrs = dict(roots["reliable_negative"].attrs)
    if (
        attrs.get("base_osm30_sidecar") != paths["osm"]
        or attrs.get("static_sidecar") != paths["static"]
        or attrs.get("years") != [2020, 2021]
        or attrs.get("target_names") != list(TASKS)
        or attrs.get("rules") != RULES
        or attrs.get("unknown_preserved") is not True
        or attrs.get("positive_precedence") is not True
    ):
        raise ValueError("unverified overlay evidence contract")
    parameters = {
        k: attrs[k] for k in ["erosion_pixels", "steep_slope_degrees", "negative_confidence"]
    }
    if (
        temporal.get("metadata_sha256") != metadata["osm"]
        or temporal.get("manifest_sha256") != sha256(manifest_path)
        or temporal.get("value_audit_sha256") != sha256(value_path)
    ):
        raise ValueError("temporal audit does not match current evidence version")
    values = pd.read_parquet(value_path)
    fingerprint = {
        "registry_sha256": sha256(registry_path),
        "manifest_sha256": sha256(manifest_path),
        "value_audit_sha256": sha256(value_path),
        "temporal_audit_sha256": sha256(temporal_path),
        "metadata_sha256": metadata,
        "parameters": parameters,
        "rules": RULES,
        "code_sha256": sha256(Path(__file__)),
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__},
    }
    scope = "full" if max_patches is None else f"pilot_{max_patches}"
    directory = dataset_root / "quality/targets/negative_rules" / scope
    directory.mkdir(parents=True, exist_ok=True)
    progress = report_root / f"negative_rule_audit_{scope}.json"
    selected = registry if max_patches is None else registry.iloc[:max_patches]
    results, reused = [], 0
    totals = Counter()
    for year in [2020, 2021]:
        required = [("osm", f"{year}/states/{task}") for task in TASKS]
        required += [
            ("static", f"{group}/{target}")
            for group in ["targets", "valid_masks"]
            for target in [f"worldcover_{year}", "dem_slope"]
        ]
        required += [
            ("reliable_negative", f"{year}/{group}/{task}")
            for group in ["states", "confidence"]
            for task in TASKS
        ]
        arrays, receipts = {}, {}
        for family, name in required:
            key = (family, name)
            array = roots[family][name]
            record = values.loc[
                (values.family == family) & (values.array == name) & (values.path == paths[family])
            ]
            if (
                array.shape != (len(registry), 128, 128)
                or len(record) != 1
                or record.iloc[0].status != "values_checked_provenance_pending"
            ):
                raise ValueError("missing or failed evidence value receipt")
            arrays[key], receipts[key] = array, record.iloc[0].decoded_values_sha256
        digests = {key: hashlib.sha256() for key in required}
        for start in range(0, len(selected), 32):
            stop = min(len(selected), start + 32)
            block = {key: np.asarray(a[start:stop]) for key, a in arrays.items()}
            digest = hashlib.sha256()
            for key in required:
                payload = block[key].tobytes()
                digests[key].update(payload)
                digest.update(payload)
            expected = {
                **fingerprint,
                "year": year,
                "start": start,
                "stop": stop,
                "decoded_chunk_sha256": digest.hexdigest(),
            }
            receipt = directory / "chunks" / f"{year}_{start:06d}.json"
            cached = json.loads(receipt.read_text()) if receipt.exists() else {}
            if cached.get("fingerprint") == expected:
                chunk = cached["rows"]
                reused += len(chunk)
            else:
                masks = expected_negatives(
                    states={t: block[("osm", f"{year}/states/{t}")] for t in TASKS},
                    worldcover=block[("static", f"targets/worldcover_{year}")],
                    worldcover_valid=block[("static", f"valid_masks/worldcover_{year}")],
                    slope=block[("static", "targets/dem_slope")],
                    slope_valid=block[("static", "valid_masks/dem_slope")],
                    erosion_pixels=parameters["erosion_pixels"],
                    steep_slope_degrees=parameters["steep_slope_degrees"],
                )
                chunk = []
                for local, row in enumerate(selected.iloc[start:stop].itertuples()):
                    for task in TASKS:
                        checked = inspect_overlay(
                            masks[task][local],
                            block[("reliable_negative", f"{year}/states/{task}")][local],
                            block[("reliable_negative", f"{year}/confidence/{task}")][local],
                            parameters["negative_confidence"],
                        )
                        chunk.append(
                            {
                                "patch_id": row.patch_id,
                                "year": year,
                                "task": task,
                                "split": row.split,
                                **checked,
                            }
                        )
                write_json(receipt, {"fingerprint": expected, "rows": chunk})
            results.extend(chunk)
            for row in chunk:
                totals["failed_targets"] += int(row["status"] == "failed")
                for field in [
                    "expected_negative_pixels",
                    "stored_negative_pixels",
                    "unexpected_negative_pixels",
                    "missing_negative_pixels",
                    "confidence_mismatch_pixels",
                ]:
                    totals[field] += row[field]
            write_json(
                progress,
                {
                    "status": "running",
                    "scope": scope,
                    "processed_targets": len(results),
                    "selected_targets": len(selected) * 8,
                    **dict(totals),
                    "reused_targets": reused,
                    "updated_at": now(),
                },
            )
        if max_patches is None and any(digests[k].hexdigest() != receipts[k] for k in required):
            raise ValueError("negative evidence bytes changed after completed value audit")
    if any(sha256(Path(paths[f]) / ".zattrs") != metadata[f] for f in roots):
        raise ValueError("negative evidence metadata changed during audit")
    atomic_parquet(pd.DataFrame(results), directory / "observations.parquet")
    summary = {
        "status": "negative_rule_audit_finished",
        "scope": scope,
        "processed_targets": len(results),
        "selected_targets": len(selected) * 8,
        **dict(totals),
        "reused_targets": reused,
        "fingerprint": fingerprint,
        "output": str(directory / "observations.parquet"),
        "training_authorized": False,
        "limitation": "Computational provenance of recorded heuristics; "
        "not independent negative-label accuracy or source geometry approval.",
        "finished_at": now(),
    }
    write_json(progress, summary)
    return summary
