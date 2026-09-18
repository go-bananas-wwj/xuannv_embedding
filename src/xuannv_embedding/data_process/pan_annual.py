"""Audit native 2m PAN and attach it to an annual candidate without granting QA approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from xuannv_embedding.data_process.annual_quality import add_moments, relative_file, select_seasons
from xuannv_embedding.data_process.observation_raster import parent_geometry, sha256_file
from xuannv_embedding.data_process.prepare_observations import atomic_json, rows
from xuannv_embedding.utils.manifest import load_manifest, write_manifest

VERSION = "annual-pan2m-candidate-v4"
SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}


def scene_identity(record: dict) -> str:
    """Use the original product name, not a parent identifier, as the scene identity."""
    name = Path(record["archive_member"]).name
    prefix, separator, original = name.partition("_PAN_2m_")
    platform = record["platform"]
    if not separator or not original.startswith(platform + "_PMS"):
        raise ValueError("unrecognized_pan_scene_identity")
    match = re.fullmatch(
        r"(GF(?:1[B-D]?|6)_PMS\d?_E[-\d.]+_N[-\d.]+_" r"(\d{8})_L\w+?)-PAN\d?\.tif", original
    )
    if match is None:
        raise ValueError("unrecognized_pan_scene_identity")
    observed = date.fromisoformat(record["acquisition_time"])
    if observed.strftime("%Y%m%d") != match[2] or not prefix.startswith(str(observed.year)):
        raise ValueError("scene_date_conflict")
    return match[1]


def audit_pan(record: dict, catalog: Path, output: Path, root: Path) -> dict:
    """Re-read every PAN payload; valid means numeric support, never clear sky."""
    result = {
        "observation_id": record["observation_id"],
        "parent_key": record["parent_key"],
        "source": record.get("source_signature"),
        "date": record.get("acquisition_time"),
        "split": record["split"],
        "archive_member": record["archive_member"],
        "status": "excluded",
        "reasons": list(record.get("issues", [])),
    }
    if result["reasons"]:
        return result
    try:
        scene = scene_identity(record)
        if date.fromisoformat(result["date"]).year not in (2020, 2021):
            raise ValueError("outside_annual_years")
        if not record.get("materialized_path"):
            raise ValueError("missing_materialized_payload")
        path = relative_file(catalog, record["materialized_path"])
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record["sha256"]:
            raise ValueError("payload_checksum_mismatch")
        with MemoryFile(payload) as memory, memory.open() as raster:
            epsg, bounds = parent_geometry(record["parent_key"])
            expected = [2, 0, bounds[0], 0, -2, bounds[3]]
            if (
                raster.crs is None
                or raster.crs.to_epsg() != epsg
                or (raster.count, raster.height, raster.width) != (1, 640, 640)
                or not np.allclose(list(raster.transform)[:6], expected, rtol=0, atol=1e-8)
            ):
                raise ValueError("pan_native_grid_mismatch")
            if (
                raster.dtypes != ("uint16",)
                or raster.scales != (1.0,)
                or raster.offsets != (0.0,)
                or raster.units != (None,)
            ):
                raise ValueError("unregistered_radiometric_schema")
            values = raster.read(1)
            valid = (raster.read_masks(1) > 0) & (values > 0) & (values < 65535)
            fraction = float(valid.mean())
            result["numeric_valid_fraction"] = fraction
            if fraction < 0.9:
                raise ValueError("insufficient_numeric_valid_area")
            accepted = values[valid].astype(np.float64)
            if accepted.var() <= 0:
                raise ValueError("constant_pan")
            mask = output / "pan_masks" / digest[:2] / (digest + ".tif")
            mask.parent.mkdir(parents=True, exist_ok=True)
            profile = raster.profile.copy()
            profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
            with MemoryFile() as mask_memory:
                with mask_memory.open(**profile) as target:
                    target.write(valid.astype(np.uint8), 1)
                    target.set_band_description(1, "numeric_valid_not_cloud_QA")
                mask_payload = mask_memory.read()
            # Separate parts can contain identical payloads; the mask is deterministic.
            temporary = mask.with_suffix("." + record["observation_id"] + ".tmp")
            temporary.write_bytes(mask_payload)
            temporary.replace(mask)
        result.update(
            status="numeric_candidate",
            scene_id=scene,
            year=int(result["date"][:4]),
            time_precision="day",
            date_evidence="consistent_original_product_and_export_filename",
            path=path.relative_to(root).as_posix(),
            mask=mask.relative_to(root).as_posix(),
            sha256=digest,
            mask_sha256=hashlib.sha256(mask_payload).hexdigest(),
            valid_fraction=fraction,
            band_counts=[int(valid.sum())],
            band_mean=[float(accepted.mean())],
            band_variance=[float(accepted.var())],
            band_min_max=[[float(accepted.min()), float(accepted.max())]],
            channels=1,
            height=640,
            width=640,
            grid_spacing_m=[2.0, 2.0],
            platform=record["platform"],
            product_type="PAN",
            radiometry="stored_uint16_values_not_calibrated_reflectance",
            cloud_shadow_QA="unverified",
            landmark_registration="unverified",
        )
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        result["reasons"].append(str(error))
    return result


def audit_part(task: tuple[str, str, str, str]) -> dict:
    part, catalog, output, root = map(Path, task)
    destination = output / "pan_parts" / part.name
    summary_path = destination.with_suffix(".summary.json")
    digest = sha256_file(part)
    upstream = json.loads(part.with_suffix(".summary.json").read_text())
    if digest != upstream["sha256"]:
        raise ValueError(f"Changed input catalog: {part}")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["input_sha256"] != digest or summary["sha256"] != sha256_file(destination):
            raise ValueError(f"Changed audit part: {part}")
        return summary
    counts, reasons, sources = Counter(), Counter(), Counter()
    temporary = destination.with_suffix(".tmp")
    candidates = (
        record
        for record in rows(part)
        if record.get("product_type") == "PAN" and record.get("declared_grid_spacing_m") == 2
    )
    with temporary.open("w") as handle, ThreadPoolExecutor(max_workers=4) as readers:
        for result in readers.map(
            lambda record: audit_pan(record, catalog, output, root), candidates
        ):
            handle.write(json.dumps(result, allow_nan=False) + "\n")
            counts[result["status"]] += 1
            reasons.update(result["reasons"])
            sources[result["source"]] += 1
    temporary.replace(destination)
    summary = {
        "part": part.name,
        "input_sha256": digest,
        "sha256": sha256_file(destination),
        "counts": dict(counts),
        "reasons": dict(reasons),
        "sources": dict(sources),
    }
    atomic_json(summary_path, summary)
    return summary


def scene_owners(records: list[tuple[str, str]]) -> dict[str, str]:
    """A source scene is owned by test, else validation, else train."""
    owners = {}
    for scene, split in records:
        if split not in SPLIT_PRIORITY:
            raise ValueError("Unknown spatial split")
        if scene not in owners or SPLIT_PRIORITY[split] > SPLIT_PRIORITY[owners[scene]]:
            owners[scene] = split
    return owners


def merge_annual(
    base: Path, output: Path, summaries: list[dict], review_exclusions: Path | None = None
) -> dict:
    base_summary = json.loads((base / "summary.json").read_text())
    schemas = json.loads((base / "sources.json").read_text())
    review = json.loads(review_exclusions.read_text()) if review_exclusions else {}
    excluded_scenes = set(review.get("scene_ids", []))
    excluded_ids = set(review.get("observation_ids", []))
    if (excluded_scenes or excluded_ids) and not review.get("reason"):
        raise ValueError("Review exclusions require a documented reason")
    database_path = output / "pan_index.sqlite"
    database_path.unlink(missing_ok=True)
    database = sqlite3.connect(database_path)
    database.execute("PRAGMA cache_size=-262144")
    database.execute("PRAGMA temp_store=MEMORY")
    database.execute(
        "CREATE TABLE obs (id TEXT PRIMARY KEY, parent TEXT, year INTEGER, "
        "source TEXT, split TEXT, scene TEXT, digest TEXT, payload TEXT)"
    )
    for part in sorted((output / "pan_parts").glob("*.jsonl")):
        for item in rows(part):
            if item["status"] != "numeric_candidate":
                continue
            database.execute(
                "INSERT INTO obs VALUES (?,?,?,?,?,?,?,?)",
                (
                    item["observation_id"],
                    item["parent_key"],
                    item["year"],
                    item["source"],
                    item["split"],
                    item["scene_id"],
                    item["sha256"],
                    json.dumps(item),
                ),
            )
        database.commit()
    database.execute("CREATE INDEX parent_year ON obs(parent,year)")
    owners = scene_owners(list(database.execute("SELECT DISTINCT scene,split FROM obs")))
    digest_owners = scene_owners(list(database.execute("SELECT DISTINCT digest,split FROM obs")))
    mixed_scenes = dict(
        database.execute(
            "SELECT scene,count(DISTINCT split) FROM obs GROUP BY scene "
            "HAVING count(DISTINCT split)>1"
        )
    )
    atomic_json(
        output / "pan_scene_split_audit.json",
        {
            "policy": "same_original_scene_and_identical_payload_kept_in_highest_priority_split",
            "priority": ["test", "validation", "train"],
            "mixed_scene_count": len(mixed_scenes),
            "scene_owners": owners,
            "scope": "PAN_2m_only; inherited_lowres_and_5m_lineage_not_certified",
        },
    )
    stats = defaultdict(dict)
    manifests, supported, selected_counts, exclusions = {}, {}, Counter(), Counter()
    selected_ids = set()
    for manifest in sorted(base.glob("*.manifest.jsonl")):
        document = load_manifest(manifest)
        support = 0
        for record in document.records:
            parent, year = record.grid["parent_key"], record.provenance["year"]
            split = record.provenance["split"]
            grouped = defaultdict(list)
            for (payload,) in database.execute(
                "SELECT payload FROM obs WHERE parent=? AND year=? ORDER BY id", (parent, year)
            ):
                item = json.loads(payload)
                if item["split"] != split:
                    raise ValueError("PAN / base parent split mismatch")
                if item["scene_id"] in excluded_scenes or item["observation_id"] in excluded_ids:
                    exclusions["review_quarantine"] += 1
                    continue
                if owners[item["scene_id"]] != split or digest_owners[item["sha256"]] != split:
                    exclusions["source_scene_or_payload_split_overlap"] += 1
                    continue
                grouped[item["source"]].append(item)
            selected = {source: select_seasons(items) for source, items in grouped.items()}
            for source, items in selected.items():
                if source in schemas and schemas[source].get("product_group") != "PAN_2m":
                    raise ValueError("PAN source collides with inherited schema")
                schemas[source] = {
                    "role": "highres",
                    "channels": 1,
                    "product_group": "PAN_2m",
                    "branch": "pan",
                    "platform": items[0]["platform"],
                    "grid_spacing_m": [2.0, 2.0],
                    "shape": [1, 640, 640],
                    "band_semantics": "PAN_from_original_product_name",
                    "radiometry": "stored_uint16_values_not_calibrated_reflectance",
                    "cloud_shadow_QA": "unverified",
                }
                record.sources[source] = [item["path"] for item in items]
                record.provenance["observations"][source] = items
                for item in items:
                    selected_ids.add(item["observation_id"])
                    selected_counts[source] += 1
                    if split == "train":
                        add_moments(stats[source], item)
            record.quality["pan2m_supported"] = bool(selected)
            record.quality["pan2m_cloud_shadow"] = "unverified" if selected else "not_available"
            record.quality["highres_supported"] = record.quality.get(
                "highres_supported", False
            ) or bool(selected)
            support += bool(selected)
        write_manifest(
            output / manifest.name,
            document.records,
            months=document.meta.months,
            generator_version=VERSION,
        )
        manifests[manifest.name] = len(document.records)
        supported[manifest.name] = support
        print(json.dumps({"merged": manifest.name, "pan_supported": support}), flush=True)
        del document
    statistics = output / "statistics"
    statistics.mkdir(exist_ok=True)
    for path in sorted((base / "statistics").glob("*.json")):
        shutil.copy2(path, statistics / path.name)
    missing_stats = []
    for source, schema in schemas.items():
        if schema.get("product_group") != "PAN_2m":
            continue
        if source not in stats:
            missing_stats.append(source)
            continue
        state = stats[source]
        std = np.sqrt(state["moment"] / state["counts"])
        if not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Invalid PAN train statistics")
        atomic_json(
            statistics / f"{source}_stats.json",
            {
                "mean": state["mean"].tolist(),
                "std": std.tolist(),
                "band_counts": state["counts"].astype(np.int64).tolist(),
                "num_files": state["files"],
                "fit_split": "train",
                "fit_years": [2020, 2021],
                "units": "stored_values",
                "basis": "selected_pan_numeric_valid_pixels",
                "cloud_QA": "unverified",
            },
        )
    if missing_stats:
        raise ValueError(f"PAN sources without train-only statistics: {missing_stats}")
    atomic_json(output / "sources.json", schemas)
    with (output / "pan_selection.jsonl").open("w") as handle:
        for (payload,) in database.execute("SELECT payload FROM obs ORDER BY id"):
            item = json.loads(payload)
            identity = item["observation_id"]
            if identity in selected_ids:
                reason = "selected"
            elif item["scene_id"] in excluded_scenes or identity in excluded_ids:
                reason = "review_quarantine"
            elif (
                owners[item["scene_id"]] != item["split"]
                or digest_owners[item["sha256"]] != item["split"]
            ):
                reason = "source_scene_or_payload_split_overlap"
            else:
                reason = "outside_base_samples_or_seasonal_budget_or_duplicate"
            handle.write(json.dumps({"id": identity, "reason": reason}) + "\n")
    database.close()
    counts, reasons = Counter(), Counter()
    for summary in summaries:
        counts.update(summary["counts"])
        reasons.update(summary["reasons"])
    report = {
        **base_summary,
        "version": VERSION,
        "previous_version": str(base),
        "status": "pan_numeric_audited_candidate_awaiting_scientific_validation",
        "training_ready": False,
        "scientific_quality_validated": False,
        "annual_model_ready": False,
        "manifests": manifests,
        "pan2m_supported_samples": supported,
        "pan2m_selected_observations": dict(selected_counts),
        "pan2m_full_payload_audit": dict(counts),
        "pan2m_exclusions_nonexclusive": dict(reasons + exclusions),
        "remaining_gates": sorted(
            set(base_summary.get("remaining_gates", []))
            | {
                "PAN_cloud_shadow_QA",
                "PAN_radiometric_processing_provenance",
                "PAN_landmark_registration",
                "annual_three_branch_runtime",
            }
        ),
    }
    # The inherited highres support count describes 5m only; label it explicitly.
    report["inherited_5m_supported_samples"] = report.pop("highres_supported_samples", {})
    if review_exclusions:
        report["pan_review_exclusions"] = {
            "path": str(review_exclusions),
            "sha256": sha256_file(review_exclusions),
            "scope": "explicit_review_only_not_full_cloud_validation",
        }
    atomic_json(output / "summary.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv data pan-annual")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--review-exclusions", type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    catalog, base, output = (p.resolve() for p in (args.catalog, args.base, args.output))
    root = Path(json.loads((base / "summary.json").read_text())["data_root"]).resolve()
    if not output.is_relative_to(root) or not catalog.is_relative_to(root):
        raise ValueError("Catalog and output must be within base data_root")
    if output in (base, catalog) or base.is_relative_to(output) or catalog.is_relative_to(output):
        raise ValueError("Output must not overwrite an input")
    parts = sorted((catalog / "catalog_parts").glob("*.jsonl"))
    manifests = sorted(base.glob("*.manifest.jsonl"))
    if not parts or not manifests:
        raise ValueError("Missing catalog parts or annual manifests")
    contract = {
        "version": VERSION,
        "catalog": str(catalog),
        "base": str(base),
        "data_root": str(root),
        "processor_sha256": sha256_file(Path(__file__)),
        "input_files": {
            str(path): sha256_file(path)
            for path in [
                base / "summary.json",
                base / "sources.json",
                *manifests,
                *sorted((base / "statistics").glob("*.json")),
            ]
        },
        "catalog_parts": [part.name for part in parts],
    }
    if output.exists():
        previous = json.loads((output / "pan_run.json").read_text())
        compared = dict(contract)
        if args.merge_only:
            compared["processor_sha256"] = previous["processor_sha256"]
        if previous != compared:
            raise ValueError("PAN resume contract mismatch")
    else:
        if args.merge_only:
            raise ValueError("Merge-only requires a completed payload audit")
        output.mkdir(parents=True)
        atomic_json(output / "pan_run.json", contract)
    (output / "pan_parts").mkdir(exist_ok=True)
    if args.merge_only:
        summaries = []
        for part in parts:
            audited = output / "pan_parts" / part.name
            summary = json.loads(audited.with_suffix(".summary.json").read_text())
            if (
                sha256_file(audited) != summary["sha256"]
                or sha256_file(part) != summary["input_sha256"]
            ):
                raise ValueError("Changed audit part before merge")
            summaries.append(summary)
        shutil.copy2(Path(__file__), output / "pan_merge_source.py")
        atomic_json(
            output / "pan_merge_run.json",
            {
                "processor_sha256": contract["processor_sha256"],
                "audit_processor_sha256": previous["processor_sha256"],
                "review_exclusions_sha256": (
                    sha256_file(args.review_exclusions) if args.review_exclusions else None
                ),
            },
        )
        merge_annual(base, output, summaries, args.review_exclusions)
        return 0
    shutil.copy2(Path(__file__), output / "pan_annual_source.py")
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(audit_part, tuple(map(str, (part, catalog, output, root))))
            for part in parts
        ]
        for future in as_completed(futures):
            result = future.result()
            summaries.append(result)
            print(json.dumps({"audited_parts": len(summaries), **result}), flush=True)
    report = merge_annual(base, output, summaries, args.review_exclusions)
    print(json.dumps({"output": str(output), "status": report["status"]}), flush=True)
    return 0
