"""Shard-by-archive Jilin inventory with geometry ownership and explicit exclusions."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_rasters import BRANCH_BANDS, inspect_jilin
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def grid_lookup(registry: pd.DataFrame) -> dict:
    lookup = {}
    for row in registry.itertuples():
        key = (int(row.grid_epsg), *(round(float(v), 3) for v in row.utm_bounds))
        if key in lookup:
            raise ValueError("ambiguous grid geometry")
        lookup[key] = (str(row.patch_id), str(row.split))
    return lookup


def catalog_file(path: Path, lookup: dict) -> dict:
    record = inspect_jilin(path)
    key = (record["epsg"], *(round(v, 3) for v in record["bounds"]))
    if key not in lookup:
        raise ValueError("no unique exact 1280m grid match")
    record["patch_id"], record["split"] = lookup[key]
    group = "|".join([record["patch_id"], record["sensor"], record["scene_id"]])
    record["scene_group_id"] = hashlib.sha256(group.encode()).hexdigest()
    record["observation_id"] = record["scene_group_id"] + ":" + record["product_id"]
    return record


def build_catalog(source_root: Path, dataset_root: Path, report_root: Path, *, workers=4) -> dict:
    source = json.loads((source_root / "manifests/source.lock.json").read_text())
    registry_path = dataset_root / "registry/national_62000.parquet"
    registry = pd.read_parquet(registry_path)
    lookup = grid_lookup(registry)
    index = pd.read_csv(source_root / "manifests/ARCHIVE_INDEX.tsv", sep="\t")
    out = dataset_root / "observations/highres/jilin1"
    out.mkdir(parents=True, exist_ok=True)
    shards = []
    rejected = []
    processed = []
    fingerprint = {
        "registry_sha256": sha256(registry_path),
        "revision": source["revision"],
        "reader_sha256": sha256(Path(__file__).with_name("v5_rasters.py")),
        "catalog_sha256": sha256(Path(__file__)),
    }

    def inspect(path):
        try:
            return catalog_file(path, lookup), None
        except Exception as exc:
            return None, {"path": str(path), "error_type": type(exc).__name__, "reason": str(exc)}

    for archive in source["archives"]:
        name = archive["archive"]
        marker = source_root / "manifests/extracted" / (name + ".json")
        integrity = report_root / "integrity_shards" / (name + ".json")
        if not marker.exists() or not integrity.exists():
            continue
        if json.loads(integrity.read_text())["status"] != "complete":
            continue
        shard = out / "shards" / (name + ".parquet")
        failed = out / "shards" / (name + ".rejected.parquet")
        lock_path = out / "shards" / (name + ".lock.json")
        expected = {**fingerprint, "archive_sha256": archive["sha256"]}
        if lock_path.exists():
            previous = json.loads(lock_path.read_text())
            if previous["fingerprint"] != expected:
                raise ValueError("catalog shard fingerprint changed; explicit new version required")
            if not shard.exists() or sha256(shard) != previous["output_sha256"]:
                raise ValueError("catalog shard changed")
            rows = pd.read_parquet(shard)
            errors = pd.read_parquet(failed)
        else:
            paths = []
            for patch in index.loc[index.archive == name, "patchid"]:
                paths.extend(sorted((source_root / "extracted" / patch).rglob("*.tif")))
            if len(paths) != archive["tiff_count"]:
                raise ValueError("extracted file count changed before catalog")
            records, failures = [], []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for result, error in pool.map(inspect, paths):
                    if error:
                        failures.append(error)
                    else:
                        records.append(result)
            rows = pd.DataFrame(records)
            errors = pd.DataFrame(failures, columns=["path", "error_type", "reason"])
            atomic_parquet(rows, shard)
            atomic_parquet(errors, failed)
            write_json(
                lock_path,
                {
                    "fingerprint": expected,
                    "output_sha256": sha256(shard),
                    "accepted": len(rows),
                    "rejected": len(errors),
                    "finished_at": now(),
                },
            )
        if not rows.empty:
            shards.append(rows)
        rejected.append(errors)
        processed.append(name)
        print(
            json.dumps({"catalog_archive": name, "accepted": len(rows), "rejected": len(errors)}),
            flush=True,
        )
    if not processed:
        raise RuntimeError("no fully extracted and pixel-verified archives available")
    errors = pd.concat(rejected, ignore_index=True)
    files = pd.concat(shards, ignore_index=True) if shards else pd.DataFrame()
    if files.empty:
        atomic_parquet(errors, report_root / "rejected_files.parquet")
        raise ValueError("no Jilin files passed the grid and band contracts")
    conflict = files.groupby("observation_id").file_sha256.transform("nunique") > 1
    conflicts = files.loc[conflict, ["path"]].assign(
        error_type="Conflict", reason="same identity with different pixels"
    )
    errors = pd.concat([errors, conflicts], ignore_index=True)
    deduplicated = files.loc[~conflict].drop_duplicates(["observation_id", "file_sha256"])
    atomic_parquet(files, report_root / "raw_file_inventory.parquet")
    atomic_parquet(errors, report_root / "rejected_files.parquet")
    atomic_parquet(deduplicated, out / "files.parquet")
    scene_rows = []
    expected_products = set(BRANCH_BANDS)
    for group_id, group in deduplicated.groupby("scene_group_id", sort=True):
        first = group.iloc[0]
        products = sorted(set(group.product_id))
        scene_rows.append(
            {
                key: first[key]
                for key in [
                    "patch_id",
                    "source_patch_id",
                    "sensor",
                    "scene_id",
                    "acquired_at",
                    "year",
                    "split",
                ]
            }
            | {
                "scene_group_id": group_id,
                "products": products,
                "missing_products": sorted(expected_products - set(products)),
                "complete_group": set(products) == expected_products,
            }
        )
    scenes = pd.DataFrame(scene_rows)
    atomic_parquet(scenes, out / "scene_groups.parquet")
    summary = {
        "scope": "available_verified_archives",
        "processed_archives": len(processed),
        "total_archives": len(source["archives"]),
        "accepted_files": len(deduplicated),
        "rejected_files": len(errors),
        "scenes": len(scenes),
        "unique_patches": int(scenes.patch_id.nunique()),
        "years": sorted(int(y) for y in scenes.year.unique()),
        "finished_at": now(),
    }
    write_json(report_root / "grid_match_report.json", summary)
    coverage = (
        scenes.groupby(["year", "sensor", "split"])
        .agg(scenes=("scene_group_id", "count"), patches=("patch_id", "nunique"))
        .reset_index()
    )
    coverage.to_csv(report_root / "coverage_by_year_sensor_split.csv", index=False)
    # Metadata is audited, not every downstream quality/normalization gate.
    write_json(
        dataset_root / "observations/product_contracts.json",
        {
            "jilin": {"verified_per_file": True, "selected_bands": BRANCH_BANDS},
            "dense": {"status": "requires_source_radiometry_audit"},
            "source_fingerprint": fingerprint,
        },
    )
    return summary
