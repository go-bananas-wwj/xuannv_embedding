"""Reconstruct sparse OSM corrections against dated sources without overwriting raw labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import zarr

from xuannv_embedding.data_process import v5_osm_geometry as geometry
from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_jilin_quality import _digest
from xuannv_embedding.data_process.v5_osm_geometry import FIELDS, _patch_coverages
from xuannv_embedding.data_process.v5_osm_projection import RecoveringSpatialIndex
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def array_digest(fields):
    h = hashlib.sha256()
    for key in FIELDS:
        a = fields[key]
        if a.dtype != np.dtype("u1") or a.ndim != 3:
            raise ValueError("invalid OSM correction array contract")
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def save_arrays(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(FIELDS) or any(
                not np.array_equal(old[k], arrays[k]) for k in FIELDS
            ):
                raise ValueError("OSM correction arrays changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)


def build_osm_corrections(dataset_root: Path, report_root: Path) -> dict:
    registry_path = dataset_root / "registry/national_62000.parquet"
    registry = pd.read_parquet(registry_path)
    audit_path = report_root / "osm_geometry_full.json"
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "osm_geometry_audit_finished"
        or audit.get("scope") != "full"
        or audit.get("processed_positions") != len(registry)
        or audit.get("selected_positions") != len(registry)
        or audit.get("compared_views") != 4 * len(registry)
    ):
        raise ValueError("complete OSM geometry scan required")
    original_root = Path(audit["output"]).parent
    fp = audit["fingerprint"]
    current_runtime = {
        "numpy": np.__version__,
        "shapely": shapely.__version__,
        "rasterio": geometry.rasterio.__version__,
        "gdal": geometry.rasterio.__gdal_version__,
        "pyproj": geometry.pyproj.__version__,
        "proj": geometry.pyproj.proj_version_str,
    }
    if fp["runtime"] != current_runtime:
        raise ValueError("OSM geometry audit runtime changed")
    inputs = {
        str(audit_path): sha256(audit_path),
        str(original_root / "observations.parquet"): sha256(original_root / "observations.parquet"),
        str(original_root / "input.lock.json"): sha256(original_root / "input.lock.json"),
    }
    if json.loads((original_root / "input.lock.json").read_text()) != fp:
        raise ValueError("OSM geometry audit input lock changed")
    for key, path in [
        ("registry_sha256", registry_path),
        ("manifest_sha256", dataset_root / "targets/manifest.parquet"),
        ("value_audit_sha256", report_root / "target_value_audit.parquet"),
        ("temporal_audit_sha256", report_root / "osm_temporal_progress.json"),
        ("code_sha256", Path(__file__).with_name("v5_osm_geometry.py")),
    ]:
        digest = sha256(path)
        if digest != fp[key]:
            raise ValueError("OSM audit source evidence changed")
        inputs[str(path)] = digest
    manifest = pd.read_parquet(dataset_root / "targets/manifest.parquet")
    entries = manifest.loc[manifest.family == "osm"]
    if entries.path.nunique() != 1:
        raise ValueError("one original OSM label source required")
    base_path = Path(entries.path.iloc[0])
    metadata = base_path / ".zattrs"
    if sha256(metadata) != fp["metadata_sha256"]:
        raise ValueError("original OSM label metadata changed")
    inputs[str(metadata)] = fp["metadata_sha256"]
    table = pd.read_parquet(audit["output"])
    keys = ["patch_id", "year", "resolution"]
    if (
        table.duplicated(keys).any()
        or len(table) != len(registry) * 4
        or not table.groupby("patch_id").size().eq(4).all()
        or not table.year.isin([2020, 2021]).all()
        or not table.resolution.isin(["10m", "2p5m"]).all()
        or not table.status.isin(["passed", "failed"]).all()
        or table.split.tolist()
        != registry.set_index("patch_id").loc[table.patch_id, "split"].tolist()
        or table.patch_id.tolist() != registry.iloc[table["index"]].patch_id.tolist()
    ):
        raise ValueError("OSM audit identity or spatial/year coverage changed")
    if int(table.status.eq("failed").sum()) != audit["failed_views"]:
        raise ValueError("OSM failure inventory changed")
    boundary = table.loc[table.candidate_file.ne("")]
    if len(boundary) != audit["boundary_changed_views"]:
        raise ValueError("OSM boundary inventory changed")
    for row in boundary.itertuples(index=False):
        path = original_root / row.candidate_file
        if sha256(path) != row.candidate_sha256:
            raise ValueError("original boundary candidate changed")
        inputs[str(path)] = row.candidate_sha256
    selected = sorted(
        table.loc[table.status.eq("failed") | table.candidate_file.ne(""), "index"].unique()
    )
    lock = {
        "inputs_sha256": inputs,
        "selected_positions": [int(i) for i in selected],
        "source_fingerprint": fp,
        "runtime": {"numpy": np.__version__, "shapely": shapely.__version__},
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ["v5_osm_corrections.py", "v5_osm_projection.py"]
        },
        "policy": (
            "reconstruct_legacy_exactly; verify_expanded_sources; "
            "no_raw_label_overwrite; negative_reconciliation_required"
        ),
    }
    directory = dataset_root / "targets/corrections/osm" / _digest(lock)[:20]
    directory.mkdir(parents=True, exist_ok=True)
    input_lock, output_lock = directory / "input.lock.json", directory / "output.lock.json"
    if input_lock.exists() and json.loads(input_lock.read_text()) != lock:
        raise ValueError("OSM correction inputs changed")
    if not input_lock.exists():
        write_json(input_lock, lock)
    reused = output_lock.exists()
    if reused:
        seal = json.loads(output_lock.read_text())
        if seal["input_lock_sha256"] != sha256(input_lock) or any(
            sha256(directory / name) != digest for name, digest in seal["files_sha256"].items()
        ):
            raise ValueError("OSM correction publication changed")
    base = zarr.open_group(str(base_path), mode="r")
    lookup = table.set_index(keys)
    indexes = {}
    rows, repairs, output_files = [], [], {}
    progress = report_root / "osm_corrections_progress.json"
    try:
        for key, source in fp["index_sources"].items():
            indexes[key] = RecoveringSpatialIndex(
                source, year=None if key == "current" else int(key)
            )
        for n, index in enumerate(selected):
            row = registry.iloc[index]
            for year, resolution, channels, legacy, fresh, query in _patch_coverages(indexes, row):
                prior = lookup.loc[(row.patch_id, year, resolution)]
                branch = "" if resolution == "10m" else "structure_2p5m/"
                old = {
                    field: np.stack(
                        [np.asarray(base[f"{year}/{branch}{field}/{c}"][index]) for c in channels]
                    )
                    for field in FIELDS
                }
                if any(not np.array_equal(old[key], legacy[key]) for key in FIELDS):
                    raise ValueError("original labels differ from dated source reconstruction")
                if prior.candidate_file:
                    candidate_path = original_root / prior.candidate_file
                    if sha256(candidate_path) != prior.candidate_sha256:
                        raise ValueError("original boundary candidate changed")
                    with np.load(candidate_path, allow_pickle=False) as candidate:
                        if any(not np.array_equal(candidate[key], fresh[key]) for key in FIELDS):
                            raise ValueError(
                                "expanded source reconstruction differs from boundary audit"
                            )
                elif prior.status == "passed" and any(
                    not np.array_equal(old[k], fresh[k]) for k in FIELDS
                ):
                    raise ValueError("unrecorded OSM boundary change")
                differences = {key: int(np.count_nonzero(old[key] != fresh[key])) for key in FIELDS}
                name = ""
                if any(differences.values()):
                    name = f"patches/{index:06d}_{year}_{resolution}.npz"
                    save_arrays(directory / name, fresh)
                    output_files[name] = sha256(directory / name)
                rows.append(
                    {
                        "patch_id": row.patch_id,
                        "index": int(index),
                        "split": row.split,
                        "year": year,
                        "resolution": resolution,
                        "channels": list(channels),
                        "original_fields_sha256": array_digest(old),
                        "corrected_fields_sha256": array_digest(fresh),
                        "source_reconstruction_passed": True,
                        "original_status": prior.status,
                        "correction_file": name,
                        "changed_pixels": json.dumps(differences),
                        "added_positive_pixels": int(
                            ((fresh["states"] == 1) & (old["states"] == 0)).sum()
                        ),
                        "negative_reconciliation_required": bool(name),
                        "training_authorized": False,
                    }
                )
            write_json(
                progress,
                {
                    "status": "running",
                    "processed_positions": n + 1,
                    "selected_positions": len(selected),
                    "verified_views": len(rows),
                    "updated_at": now(),
                    "training_authorized": False,
                },
            )
        if not all(source.unchanged() for source in indexes.values()):
            raise ValueError("OSM indexed sources changed during corrections")
        for key, source in indexes.items():
            repairs.extend({"source": key, **record} for record in source.repair_records)
    finally:
        for source in indexes.values():
            source.close()
    if any(sha256(Path(p)) != h for p, h in inputs.items()) or any(
        sha256(Path(__file__).with_name(p)) != h for p, h in lock["code_sha256"].items()
    ):
        raise ValueError("OSM correction evidence changed during processing")
    columns = [
        "patch_id",
        "index",
        "split",
        "year",
        "resolution",
        "channels",
        "original_fields_sha256",
        "corrected_fields_sha256",
        "source_reconstruction_passed",
        "original_status",
        "correction_file",
        "changed_pixels",
        "added_positive_pixels",
        "negative_reconciliation_required",
        "training_authorized",
    ]
    records = pd.DataFrame(rows, columns=columns)
    if sum(row["original_status"] == "failed" for row in rows) != audit["failed_views"]:
        raise ValueError("OSM correction did not resolve every original failed view")
    summary = {
        "status": "source_verified_corrections_pending_negative_reconciliation",
        "selected_positions": len(selected),
        "verified_views": len(rows),
        "corrected_views": sum(bool(row["correction_file"]) for row in rows),
        "recovered_failed_views": sum(row["original_status"] == "failed" for row in rows),
        "failed_views_remaining": 0,
        "projection_repair_queries": len(repairs),
        "added_positive_pixels": sum(row["added_positive_pixels"] for row in rows),
        "scope": "all_failed_and_boundary_changed_views_from_complete_OSM_scan",
        "raw_labels_modified": False,
        "training_authorized": False,
    }
    if reused:
        cached = pd.read_parquet(directory / "manifest.parquet")
        if (
            cached.to_json(orient="records") != records.to_json(orient="records")
            or json.loads((directory / "summary.json").read_text()) != summary
        ):
            raise ValueError("OSM correction replay changed")
    else:
        atomic_parquet(records, directory / "manifest.parquet")
        write_json(directory / "projection_repairs.json", repairs)
        write_json(directory / "summary.json", summary)
        for name in ["manifest.parquet", "projection_repairs.json", "summary.json"]:
            output_files[name] = sha256(directory / name)
        write_json(
            output_lock, {"input_lock_sha256": sha256(input_lock), "files_sha256": output_files}
        )
    result = {
        **summary,
        "output": str(directory),
        "output_lock_sha256": sha256(output_lock),
        "reused": reused,
        "finished_at": now(),
    }
    write_json(report_root / "osm_corrections_full.json", result)
    write_json(progress, result)
    return result
