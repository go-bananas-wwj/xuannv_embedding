"""Prepare extracted observation archives with resumable, bounded-memory QA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from xuannv_embedding.data_process.highres_catalog import parent_key, safe_member
from xuannv_embedding.data_process.observation_raster import (
    inspect_raster,
    merge_statistics,
    sha256_file,
)
from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest

VERSION = "extracted-observations-v1"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def rows(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def acquisition_date(filename: str) -> str:
    match = re.match(r"^(\d{8})_", filename)
    if not match:
        raise ValueError("missing_leading_acquisition_date")
    date = datetime.strptime(match.group(1), "%Y%m%d").date()
    for token in filename.split("_"):
        if re.fullmatch(r"\d{14}", token):
            if datetime.strptime(token, "%Y%m%d%H%M%S").date() != date:
                raise ValueError("conflicting_acquisition_date")
    return date.isoformat()


def product_schema(member: str, metadata: dict) -> tuple[str, dict]:
    parts = Path(member).parts
    match = re.search(r"_(PMS\d+)_", parts[-1])
    group = re.search(r"_(5m|10m|20m|B0)_[a-fA-F0-9]+\.tif$", parts[-1])
    if len(parts) != 3 or not match or not group:
        raise ValueError("unrecognized_product_layout")
    schema = {
        "platform": parts[1],
        "instrument": match.group(1),
        "product_group": group.group(1),
        **{
            key: metadata[key]
            for key in (
                "channels",
                "width",
                "height",
                "grid_spacing_m",
                "band_names",
                "stored_scales",
                "stored_offsets",
                "dtypes",
                "radiometric_units",
            )
        },
        "semantics": "TIFF_descriptions_preserved_not_externally_calibrated",
        "normalization": "stored_values_standardization",
    }
    if group.group(1) != "B0":
        expected = float(group.group(1).removesuffix("m"))
        if not all(abs(spacing - expected) < 1e-6 for spacing in metadata["grid_spacing_m"]):
            raise ValueError("declared_resolution_mismatch")
    if not re.fullmatch(r"[A-Za-z0-9]+", parts[1]):
        raise ValueError("invalid_platform")
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12]
    name = f"{parts[1]}_{match.group(1)}_{group.group(1)}_c{metadata['channels']}_{digest}"
    return name, schema


def prepare_one(raw: Path, output: Path, original: dict) -> dict:
    member = safe_member(original["archive_member"])
    result = {
        "archive_member": member,
        "archive": original["archive"],
        "parent_key": original["parent_key"],
        "split": original["split"],
        "status": "quarantined",
    }
    source = raw / member
    if not source.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Observation escapes raw root")
    if parent_key(Path(member).parts[0]) != original["parent_key"]:
        raise ValueError("Observation parent identity mismatch")
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = source.read_bytes()
    try:
        date = acquisition_date(Path(member).name)
        metadata = inspect_raster(payload, original["parent_key"], pixels=False)
        result.update(metadata, acquisition_time=date, year=int(date[:4]), month=date[:7])
        signature, schema = product_schema(member, metadata)
        result.update(source_signature=signature, source_schema=schema)
        if not metadata["grid_matches_parent"]:
            raise ValueError("grid_mismatch")
        with rasterio.io.MemoryFile(payload) as memory, memory.open() as dataset:
            values = dataset.read()
            valid = (dataset.read_masks() > 0).all(axis=0) & np.isfinite(values).all(axis=0)
            if np.issubdtype(values.dtype, np.integer):
                valid &= ~(values == np.iinfo(values.dtype).max).any(axis=0)
            valid &= ~(values == 0).all(axis=0)
            result["valid_fraction"] = float(valid.mean())
            if not valid.any():
                raise ValueError("all_pixels_invalid")
            accepted = values[:, valid].astype(np.float64)
            result.update(
                band_counts=[int(valid.sum())] * dataset.count,
                band_mean=accepted.mean(axis=1).tolist(),
                band_variance=accepted.var(axis=1).tolist(),
                mask_policy="all_band_GDAL_finite_exclude_integer_max_and_all_band_zero",
                quality_status="numeric_QA_only_cloud_shadow_and_landmark_alignment_unverified",
            )
            destination = output / "rasters" / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256_file(destination) != result["sha256"]:
                    raise ValueError("existing_materialized_file_changed")
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    temporary = destination.with_suffix(".tif.tmp")
                    temporary.write_bytes(payload)
                    temporary.replace(destination)
            mask_path = destination.with_name(destination.stem + "_mask.tif")
            temporary = mask_path.with_suffix(".tif.tmp")
            with rasterio.open(
                temporary,
                "w",
                driver="GTiff",
                count=1,
                dtype="uint8",
                nodata=0,
                width=dataset.width,
                height=dataset.height,
                crs=dataset.crs,
                transform=dataset.transform,
                compress="deflate",
            ) as mask_dataset:
                mask_dataset.write(valid.astype(np.uint8), 1)
            temporary.replace(mask_path)
            result.update(
                status="usable",
                materialized_path=destination.relative_to(output).as_posix(),
                mask_path=mask_path.relative_to(output).as_posix(),
                mask_sha256=sha256_file(mask_path),
            )
    except (ValueError, rasterio.errors.RasterioError) as error:
        if str(error) == "existing_materialized_file_changed":
            raise
        result["issue"] = str(error)
    return result


def prepare_archive(task: dict) -> dict:
    output = Path(task["output"])
    archive = task["archive"]
    name = archive["archive"]
    part = output / "parts" / f"{name}.jsonl"
    summary_path = part.with_suffix(".summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if sha256_file(part) != summary["sha256"]:
            raise ValueError(f"Changed completed part: {part}")
        return summary
    started = time.monotonic()
    archive_path = Path(task["packages"]) / name
    if archive_path.stat().st_size != int(archive["bytes"]):
        raise ValueError(f"Archive size mismatch: {name}")
    if task["verify_archives"] and sha256_file(archive_path) != archive["sha256"]:
        raise ValueError(f"Archive SHA256 mismatch: {name}")
    statuses = Counter()
    temporary = part.with_suffix(".jsonl.tmp")
    count = 0
    with temporary.open("w") as handle:
        for original in rows(Path(task["catalog"]) / "catalog_parts" / f"{name}.jsonl"):
            if task["per_archive_limit"] and count >= task["per_archive_limit"]:
                break
            if original["archive"] != name:
                raise ValueError("Catalog archive identity mismatch")
            result = prepare_one(Path(task["raw"]), output, original)
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            statuses[result["status"]] += 1
            count += 1
            if count % 500 == 0:
                handle.flush()
                atomic_json(
                    output / "parts" / f"{name}.progress.json",
                    {
                        "observations": count,
                        "statuses": dict(statuses),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
    expected = int(archive["tiff_count"])
    if task["per_archive_limit"]:
        expected = min(expected, task["per_archive_limit"])
    if count != expected:
        raise ValueError(f"Catalog count mismatch for {name}: {count} != {expected}")
    temporary.replace(part)
    summary = {
        "archive": name,
        "observations": count,
        "statuses": dict(statuses),
        "elapsed_seconds": time.monotonic() - started,
        "sha256": sha256_file(part),
        "archive_sha256_verified": task["verify_archives"],
    }
    atomic_json(summary_path, summary)
    return summary


def merge_online(previous: dict | None, record: dict) -> dict:
    merged = merge_statistics([previous, record] if previous else [record])
    return {
        "channels": record["channels"],
        "band_counts": merged["band_counts"],
        "band_mean": merged["mean"],
        "band_variance": [value**2 for value in merged["raw_std"]],
        "files": (previous["files"] if previous else 0) + 1,
    }


def finalize(output: Path, names: list[str], max_observations: int) -> dict:
    statistics, schemas = {}, {}
    candidates = defaultdict(lambda: defaultdict(list))
    statuses, issues, years = Counter(), Counter(), Counter()
    splits = {}
    for name in names:
        for record in rows(output / "parts" / f"{name}.jsonl"):
            statuses[record["status"]] += 1
            if record["status"] != "usable":
                issues[record["issue"]] += 1
                continue
            signature = record["source_signature"]
            schemas[signature] = record["source_schema"]
            statistics_key = (record["year"], signature)
            if record["split"] == "train":
                statistics[statistics_key] = merge_online(statistics.get(statistics_key), record)
            identity = (record["year"], record["parent_key"])
            if splits.setdefault(identity, record["split"]) != record["split"]:
                raise ValueError("Conflicting spatial split")
            years[record["year"]] += 1
            selected = candidates[identity][signature]
            selected.append(
                (-record["valid_fraction"], record["acquisition_time"], record["materialized_path"])
            )
            selected.sort()
            del selected[max_observations:]
    (output / "statistics").mkdir(exist_ok=True)
    for (year, signature), stats in statistics.items():
        result = merge_statistics([stats])
        result.update(
            num_files=stats["files"],
            purpose="numeric_QA_training_inputs",
            source_schema=schemas[signature],
        )
        directory = output / "statistics" / str(year)
        directory.mkdir(exist_ok=True)
        atomic_json(directory / f"{signature}_stats.json", result)
    manifests = defaultdict(list)
    for (year, key), sources in sorted(candidates.items()):
        manifests[(year, splits[(year, key)])].append(
            ManifestRecord(
                patch_id="parent_" + key,
                region=f"observations_{year}",
                sources={
                    source: [item[2] for item in selected] for source, selected in sources.items()
                },
                grid={"parent_key": key},
                quality={"QA": "numeric_only_cloud_unverified"},
                provenance={
                    "year": year,
                    "split": splits[(year, key)],
                    "selection": "highest_valid_fraction_then_date_then_path",
                    "max_observations_per_source_year": max_observations,
                },
            )
        )
    manifest_counts = {}
    for (year, split), records in manifests.items():
        name = f"{year}.{split}.manifest.jsonl"
        write_manifest(
            output / name,
            records,
            months=[f"{year}-{month:02d}" for month in range(1, 13)],
            generator_version=VERSION,
        )
        manifest_counts[name] = len(records)
    atomic_json(output / "sources.json", schemas)
    summary = {
        "version": VERSION,
        "status": "highres_prepared",
        "observations": dict(statuses),
        "quarantine_reasons": dict(issues),
        "usable_by_year": dict(years),
        "source_count": len(schemas),
        "manifests": manifest_counts,
        "sources_without_train_statistics": sorted(
            f"{year}/{signature}"
            for year, signature in {
                (year, signature)
                for (year, _), sources in candidates.items()
                for signature in sources
            }
            - set(statistics)
        ),
        "all_usable_observations_retained_in_parts": True,
        "paired_lowres_ready": False,
        "annual_model_ready": False,
        "scientific_quality_validated": False,
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv data prepare-observations")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--archive-limit", type=int, default=0)
    parser.add_argument("--per-archive-limit", type=int, default=0)
    parser.add_argument("--max-observations", type=int, default=4)
    parser.add_argument("--skip-archive-hash", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(args.workers, args.max_observations) < 1
        or min(args.archive_limit, args.per_archive_limit) < 0
    ):
        parser.error("Counts must be positive; limits can be zero for all")
    if args.output.exists():
        parser.error("Output exists; choose a new version")
    with args.archives.open() as handle:
        archives = list(csv.DictReader(handle, delimiter="\t"))
    if args.archive_limit:
        archives = archives[: args.archive_limit]
    for archive in archives:
        safe_member(archive["archive"])
    fingerprint = {
        "version": VERSION,
        "code_sha256": sha256_file(Path(__file__)),
        "raw_root": str(args.raw_root.resolve()),
        "packages": str(args.packages.resolve()),
        "catalog": str(args.catalog.resolve()),
        "archives": archives,
        "catalog_parts": {
            item["archive"]: sha256_file(
                args.catalog / "catalog_parts" / (item["archive"] + ".jsonl")
            )
            for item in archives
        },
        "per_archive_limit": args.per_archive_limit,
        "max_observations": args.max_observations,
        "verify_archives": not args.skip_archive_hash,
    }
    stage = args.output.with_name(args.output.name + ".partial")
    if stage.exists():
        if not args.resume or json.loads((stage / "fingerprint.json").read_text()) != fingerprint:
            parser.error("Partial output requires --resume and unchanged inputs/code")
    else:
        (stage / "parts").mkdir(parents=True)
        atomic_json(stage / "fingerprint.json", fingerprint)
    tasks = [
        {
            "output": str(stage),
            "raw": str(args.raw_root),
            "packages": str(args.packages),
            "catalog": str(args.catalog),
            "archive": archive,
            "verify_archives": not args.skip_archive_hash,
            "per_archive_limit": args.per_archive_limit,
        }
        for archive in archives
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, future in enumerate(
            as_completed([pool.submit(prepare_archive, task) for task in tasks]), 1
        ):
            result = future.result()
            print(
                json.dumps({"completed_archives": index, "total_archives": len(tasks), **result}),
                flush=True,
            )
    summary = finalize(stage, [archive["archive"] for archive in archives], args.max_observations)
    stage.rename(args.output)
    print(json.dumps(summary), flush=True)
    return 0
