"""Verify original source references while distinguishing new locks from historical proof."""

from __future__ import annotations

import json
from pathlib import Path

import zarr

from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def inspect_source_reference(reference: dict) -> dict:
    path = Path(reference["path"])
    result = {**reference, "issues": [], "checked_at": now()}
    if not path.is_file():
        return {**result, "status": "failed", "issues": ["source_missing"]}
    before = path.stat()
    result.update(actual_bytes=before.st_size, actual_mtime_ns=before.st_mtime_ns)
    if "size_bytes" in reference and before.st_size != reference["size_bytes"]:
        result["issues"].append("size_mismatch")
    if "mtime_ns" in reference and before.st_mtime_ns != reference["mtime_ns"]:
        result["issues"].append("mtime_mismatch")
    result["actual_sha256"] = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        result["issues"].append("source_changed_during_hash")
    if "sha256" in reference and result["actual_sha256"] != reference["sha256"]:
        result["issues"].append("sha256_mismatch")
    result["status"] = (
        "failed"
        if result["issues"]
        else (
            "matches_original_sha256"
            if "sha256" in reference
            else "metadata_matches_current_hash_locked"
        )
    )
    return result


def audit_target_sources(base_root: Path, report_root: Path) -> dict:
    static = zarr.open_group(str(base_root / "targets/static_10m/full62000_v1.zarr"), mode="r")
    descriptors = []
    manifests = []
    for shard in static.attrs["merged_shards"]:
        path = Path(shard).with_name(Path(shard).stem + "_quality.sources.json")
        entry = json.loads(path.read_text())
        descriptors.append({"path": str(path), "sha256": sha256(path)})
        manifests.append(entry["sources"])
    if not manifests or any(value != manifests[0] for value in manifests):
        raise ValueError("static shards disagree about original input sources")
    osm = zarr.open_group(str(base_root / "targets/osm30_10m/full62000_base_v1.zarr"), mode="r")
    indexes = dict(osm.attrs["indexes"])
    if set(indexes) != {"2020", "2021"}:
        raise ValueError("OSM annual source index contract differs")
    references = [{**row, "family": "static"} for row in manifests[0]]
    references += [
        {**row, "family": "osm_historical", "year": int(year)} for year, row in indexes.items()
    ]
    references.append(
        {**osm.attrs["current_index"], "family": "osm_current", "annual_truth_authorized": False}
    )
    results = []
    for reference in references:
        results.append(inspect_source_reference(reference))
        write_json(
            report_root / "target_source_audit.json",
            {
                "status": "running",
                "processed_sources": len(results),
                "selected_sources": len(references),
                "sources": results,
                "source_descriptors": descriptors,
                "updated_at": now(),
            },
        )
    result = {
        "status": "source_audit_finished",
        "processed_sources": len(results),
        "selected_sources": len(references),
        "failed_sources": sum(row["status"] == "failed" for row in results),
        "sources": results,
        "source_descriptors": descriptors,
        "limitation": "Current SHA locks do not retrospectively prove absent original SHA; "
        "spatial/value audits and year-specific evidence remain separate gates.",
        "finished_at": now(),
    }
    write_json(report_root / "target_source_audit.json", result)
    return result
