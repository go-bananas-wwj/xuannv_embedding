"""Cross-array audit of dated OSM positives, current-only unknowns, and their receipts."""

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

FIELDS = ("states", "confidence", "targets", "source_bits")


def inspect_osm_temporal_block(*, states, confidence, targets, source_bits) -> dict:
    arrays = [np.asarray(a) for a in (states, confidence, targets, source_bits)]
    if any(a.shape != arrays[0].shape or a.dtype != np.dtype("u1") for a in arrays):
        raise ValueError("OSM temporal arrays must have matching shapes and uint8 dtype")
    if not np.isin(states, [0, 1]).all() or not np.isin(source_bits, [0, 1, 2, 3]).all():
        raise ValueError("undated external or negative evidence in historical-only base contract")
    historical = (source_bits & 1) != 0
    current_only = ((source_bits & 2) != 0) & ~historical
    strong, unknown = states == 1, states == 0
    if np.any(strong & (~historical | (confidence != 255) | (targets == 0))):
        raise ValueError("historical positive lacks dated source, confidence or coverage")
    if np.any(unknown & ((confidence != 0) | (targets != 0))):
        raise ValueError("unknown pixels carry supervision")
    return {
        "pixels": int(states.size),
        "historical_positive_pixels": int(strong.sum()),
        "unknown_pixels": int(unknown.sum()),
        "current_only_unknown_pixels": int((current_only & unknown).sum()),
        "historical_boundary_unknown_pixels": int((historical & unknown).sum()),
    }


def audit_osm_temporal(dataset_root: Path, report_root: Path) -> dict:
    manifest_path = dataset_root / "targets/manifest.parquet"
    value_path = report_root / "target_value_audit.parquet"
    source_path = report_root / "target_source_audit.json"
    registry_path = dataset_root / "registry/national_62000.parquet"
    manifest_hash = sha256(manifest_path)
    manifest = pd.read_parquet(manifest_path)
    manifest = manifest.loc[manifest.family == "osm"]
    if manifest.empty or not manifest.registry_order_verified.all() or manifest.path.nunique() != 1:
        raise ValueError("one OSM base with verified registry order is required")
    root_path = Path(manifest.path.iloc[0])
    metadata_hash = sha256(root_path / ".zattrs")
    if set(manifest.source_metadata_sha256) != {metadata_hash}:
        raise ValueError("OSM source metadata changed after audit")
    root = zarr.open_group(str(root_path), mode="r")
    registry = pd.read_parquet(registry_path)
    if list(root.attrs.get("patch_ids", [])) != registry.patch_id.tolist():
        raise ValueError("OSM source registry ordering changed")
    if "external_evidence" not in root.attrs or root.attrs["external_evidence"] is not None:
        raise ValueError("external evidence requires independent year contracts")
    source_audit = json.loads(source_path.read_text())
    if source_audit.get("status") != "source_audit_finished":
        raise ValueError("completed historical source hash audit is required")
    indexes = root.attrs.get("indexes", {})
    if set(indexes) != {"2020", "2021"}:
        raise ValueError("both historical year indexes are required")
    for year, ref in indexes.items():
        matches = [
            r
            for r in source_audit["sources"]
            if r.get("family") == "osm_historical"
            and r.get("year") == int(year)
            and r.get("path") == ref["path"]
            and r.get("actual_sha256") == ref["sha256"]
            and r.get("status") == "matches_original_sha256"
        ]
        if len(matches) != 1:
            raise ValueError("historical year source checksum has no matching receipt")
    values = pd.read_parquet(value_path)
    values = values.loc[(values.family == "osm") & (values.path == str(root_path))]
    if values.array.duplicated().any() or set(values.source_manifest_sha256) != {manifest_hash}:
        raise ValueError("value receipts do not match the current target manifest")
    receipts = values.set_index("array").to_dict("index")
    names = sorted(name for name in manifest.array if "/states/" in name)
    expected = {
        f"{year}/{branch}states/{channel}"
        for year in indexes
        for branch, channels in [
            ("", root.attrs.get("target_names", [])),
            ("structure_2p5m/", root.attrs.get("structure_names", [])),
        ]
        for channel in channels
    }
    if not names or len(set(names)) != len(names) or set(names) != expected:
        raise ValueError("OSM temporal state groups disagree with declared channels and years")
    fingerprint = {
        "manifest_sha256": manifest_hash,
        "value_audit_sha256": sha256(value_path),
        "source_audit_sha256": sha256(source_path),
        "registry_sha256": sha256(registry_path),
        "metadata_sha256": metadata_hash,
        "code_sha256": sha256(Path(__file__)),
    }
    rows, totals = [], Counter()
    progress_path = report_root / "osm_temporal_progress.json"
    for name in names:
        prefix, channel = name.split("/states/")
        year = int(prefix.split("/")[0])
        if str(year) not in indexes:
            raise ValueError("state group year has no historical index")
        paths = {field: f"{prefix}/{field}/{channel}" for field in FIELDS}
        if any(p not in receipts for p in paths.values()):
            raise ValueError("missing cross-array value receipt")
        if any(
            receipts[p]["status"] != "values_checked_provenance_pending" for p in paths.values()
        ):
            raise ValueError("failed values cannot enter temporal audit")
        arrays = {field: root[path] for field, path in paths.items()}
        shape = arrays["states"].shape
        if shape[0] != len(registry) or any(a.shape != shape for a in arrays.values()):
            raise ValueError("OSM temporal array dimensions disagree")
        digests = {field: hashlib.sha256() for field in FIELDS}
        counts, issues = Counter(), Counter()
        for start in range(0, len(registry), arrays["states"].chunks[0]):
            stop = min(len(registry), start + arrays["states"].chunks[0])
            try:
                block = {field: np.asarray(a[start:stop]) for field, a in arrays.items()}
                for field, array in block.items():
                    digests[field].update(array.tobytes(order="C"))
                counts.update(inspect_osm_temporal_block(**block))
            except ValueError as exc:
                issues[str(exc)] += stop - start
        hashes = {field: digest.hexdigest() for field, digest in digests.items()}
        if any(hashes[f] != receipts[p]["decoded_values_sha256"] for f, p in paths.items()):
            issues["value_fingerprint_changed"] += 1
        row = {
            "year": year,
            "array": name,
            "source_index_sha256": indexes[str(year)]["sha256"],
            **counts,
            "status": "failed" if issues else "cross_array_temporal_checked",
            "issues": json.dumps(dict(issues)),
            "decoded_sha256": json.dumps(hashes),
            "finished_at": now(),
        }
        rows.append(row)
        totals.update(counts)
        key = hashlib.sha256(name.encode()).hexdigest()
        write_json(report_root / "osm_temporal_shards" / f"{key}.json", {**fingerprint, **row})
        write_json(
            progress_path,
            {
                "status": "running",
                "checked_groups": len(rows),
                "selected_groups": len(names),
                "failed_groups": sum(r["status"] == "failed" for r in rows),
                "updated_at": now(),
            },
        )
    if metadata_hash != sha256(root_path / ".zattrs"):
        raise ValueError("OSM metadata changed during temporal audit")
    atomic_parquet(pd.DataFrame(rows), report_root / "osm_temporal_audit.parquet")
    result = {
        **fingerprint,
        **totals,
        "status": "temporal_cross_checks_finished",
        "checked_groups": len(rows),
        "selected_groups": len(names),
        "failed_groups": sum(r["status"] == "failed" for r in rows),
        "limitation": "Checks dated source attribution and cross-array values; does not certify "
        "rasterization geometry, unchanged land use within year, or negative overlay rules.",
        "training_authorized": False,
        "finished_at": now(),
    }
    write_json(progress_path, result)
    return result
