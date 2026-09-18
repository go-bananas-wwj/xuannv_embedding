"""Prepare a bounded real-data experiment from the audited observation pilot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import yaml

from xuannv_embedding.config import Config
from xuannv_embedding.data_process.highres_catalog import write_json, write_jsonl
from xuannv_embedding.data_process.observation_raster import merge_statistics, sha256_file
from xuannv_embedding.data_process.pilot_cache import MONTHS, _smoke_config
from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest

TARGET_MONTHS = ["2020-01", "2020-07", "2021-01", "2021-07"]
HIGHRES_SOURCE = "GF6_PAN_c1"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sanitize_observation(task: tuple[Path, Path, dict, str]) -> dict:
    pilot, output, original, relative = task
    source = pilot / original["materialized_path"]
    if sha256_file(source) != original["sha256"]:
        raise ValueError(f"Pilot observation changed: {source}")
    result = dict(original)
    result.pop("materialized_path", None)
    result.pop("mask_path", None)
    with rasterio.open(source) as raster:
        values = raster.read()
        valid = (raster.read_masks() > 0).all(axis=0) & np.isfinite(values).all(axis=0)
        if np.issubdtype(values.dtype, np.integer):
            valid &= ~(values == np.iinfo(values.dtype).max).any(axis=0)
            valid &= ~(values == 0).all(axis=0)
        elif not np.any(values):
            valid[:] = False
        result["valid_fraction"] = float(valid.mean())
        result["p0_quality_policy"] = "finite_gdal_masks_exclude_integer_max_and_all_band_zero"
        result["quality_status"] = "conservative_numeric_QA; cloud_and_radiometry_unverified"
        if not valid.any():
            result["p0_status"] = "all_pixels_rejected"
            return result
        counts, means, variances, extrema = [], [], [], []
        for band in values:
            accepted = band[valid].astype(np.float64)
            counts.append(int(accepted.size))
            means.append(float(accepted.mean()))
            variances.append(float(accepted.var()))
            extrema.append([float(accepted.min()), float(accepted.max())])
        result.update(
            band_counts=counts,
            band_mean=means,
            band_variance=variances,
            band_min_max=extrema,
            materialized_path=relative,
            p0_status="usable_for_diagnostics",
        )
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        mask_path = destination.with_name(destination.stem + "_mask.tif")
        profile = raster.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
        with rasterio.open(mask_path, "w", **profile) as target:
            target.write(valid.astype(np.uint8), 1)
        result["mask_path"] = mask_path.relative_to(output).as_posix()
    return result


def prepare(
    pilot: Path,
    output: Path,
    workers: int = 4,
    target_months: list[str] | None = None,
    version: str = "national-p0-v1",
    highres_per_month: int = 1,
) -> dict:
    if output.exists():
        raise ValueError("P0 output already exists; choose a new version")
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True)
    months = list(target_months or TARGET_MONTHS)
    if months != sorted(set(months)) or any(month not in MONTHS for month in months):
        raise ValueError("target_months must be an ordered subset of the pilot months")
    if highres_per_month <= 0:
        raise ValueError("highres_per_month must be positive")
    parents = [row for row in read_rows(pilot / "selected_parents.jsonl") if row["materialize"]]
    lowres = [
        row for row in read_rows(pilot / "lowres_observations.jsonl") if "materialized_path" in row
    ]
    observations = {
        row["observation_id"]: row
        for row in read_rows(pilot / "observations.jsonl")
        if "materialized_path" in row
    }
    candidates = read_rows(pilot / "highres_month_candidates.jsonl")
    tasks = [(pilot, stage, row, row["materialized_path"]) for row in lowres]
    selected = []
    for row in candidates:
        if row["month"] not in months:
            continue
        available = [
            candidate
            for candidate in row["candidates"]
            if candidate["source_signature"] == HIGHRES_SOURCE
        ]
        for candidate in available[:highres_per_month]:
            original = dict(observations[candidate["observation_id"]])
            original["month"] = row["month"]
            original["assignment_gap_days"] = candidate["gap_days_to_month_interval"]
            year, month = row["month"].split("-")
            relative = (
                f"highres/{HIGHRES_SOURCE}/{year}/{month}/{original['parent_key']}/"
                f"frame-{candidate['observation_id'][:12]}.tif"
            )
            tasks.append((pilot, stage, original, relative))
            selected.append(original)
    processed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, record in enumerate(pool.map(sanitize_observation, tasks), 1):
            processed.append(record)
            if index % 500 == 0 or index == len(tasks):
                print(f"P0 numeric QA: {index}/{len(tasks)}", flush=True)
    write_jsonl(stage / "observations.jsonl", processed)
    write_jsonl(stage / "selected_parents.jsonl", parents)
    statistics = defaultdict(list)
    paths = defaultdict(lambda: defaultdict(list))
    for record in processed:
        if "materialized_path" not in record:
            continue
        signature = record["source_signature"]
        paths[record["parent_key"]][signature].append(record["materialized_path"])
        if record["split"] == "train" and record["month"] in months:
            statistics[signature].append(record)
    (stage / "statistics").mkdir()
    for signature, records in statistics.items():
        result = merge_statistics(records)
        result.update(
            purpose="P0_numeric_interface_diagnostics_only",
            target_months=months,
            radiometry="stored_values_per_source_standardization; not_calibrated",
        )
        write_json(stage / "statistics" / f"{signature}_stats.json", result)
    sources = ["s2", "s1", "landsat", HIGHRES_SOURCE]
    if set(statistics) != set(sources):
        raise ValueError("Missing train statistics for an enabled P0 source")
    for split in ("train", "validation", "test"):
        records = [
            ManifestRecord(
                patch_id="parent_" + parent["parent_key"],
                region=version.replace("-", "_"),
                sources={
                    source: sorted(paths[parent["parent_key"]][source]) or None
                    for source in sources
                },
                grid={"parent_key": parent["parent_key"]},
                quality={"purpose": "P0_only", "cloud_QA": "unverified"},
                provenance={"split": split, "observations": "observations.jsonl"},
            )
            for parent in parents
            if parent["split"] == split
        ]
        write_manifest(
            stage / f"{split}.manifest.jsonl",
            records,
            months=MONTHS,
            generator_version=version,
        )
        for variant in ("lowres", "highres"):
            config = _smoke_config(output, sources[:3], split)
            config["experiment"]["name"] = f"{version.replace('-', '_')}_{variant}_{split}"
            config["model"]["stem_dim"] = 16
            config["model"]["stp"].update(
                space_dim=64,
                time_dim=32,
                precision_dim=32,
                num_heads=4,
                num_blocks=2,
                time_attention_mode="full",
            )
            config["data"].update(
                target_months=months,
                highres_mode="observations",
                highres_max_observations=4,
                batch_size=2,
                num_workers=2,
            )
            config["training"].update(
                epochs=2,
                lr=0.0001,
                amp=True,
                uniformity_weight=0.01,
                gradient_checkpointing=False,
            )
            dataset = config["data"]["datasets"][0]
            dataset["region"] = version.replace("-", "_")
            if variant == "highres":
                config["model"]["input_sources"][HIGHRES_SOURCE] = {
                    "channels": 1,
                    "role": "highres",
                }
                dataset["source_map"][HIGHRES_SOURCE] = HIGHRES_SOURCE
            path = stage / f"{split}-{variant}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            Config.from_yaml(path)
    summary = {
        "parent_count": len(parents),
        "training_split": dict(Counter(row["split"] for row in parents)),
        "target_months": months,
        "highres_source": HIGHRES_SOURCE,
        "highres_candidate_parent_months": len(selected),
        "highres_per_month": highres_per_month,
        "usable_counts": dict(
            Counter(row["source_signature"] for row in processed if "materialized_path" in row)
        ),
        "numeric_QA_rejected": sum("materialized_path" not in row for row in processed),
        "purpose": (
            "P0 engineering diagnostics, not algorithm accuracy validation"
            if version.startswith("national-p0")
            else "P1 algorithm comparison pilot; not national accuracy validation"
        ),
        "p1_ready": False,
        "remaining": ["cloud_QA", "radiometric_semantics", "landmark_coregistration"],
        "pilot_summary_sha256": sha256_file(pilot / "summary.json"),
    }
    write_json(stage / "summary.json", summary)
    stage.rename(output)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv data p0")
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-months", nargs="+", default=None)
    parser.add_argument("--version", default="national-p0-v1")
    parser.add_argument("--highres-per-month", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("workers must be positive")
    print(
        json.dumps(
            prepare(
                args.pilot_root,
                args.output_root,
                args.workers,
                target_months=args.target_months,
                version=args.version,
                highres_per_month=args.highres_per_month,
            ),
            indent=2,
        )
    )
    return 0
