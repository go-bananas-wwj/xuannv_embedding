"""Conservative annual selection from pixel-audited observation catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio

from xuannv_embedding.data_process.observation_raster import parent_geometry, sha256_file
from xuannv_embedding.data_process.prepare_observations import atomic_json, rows
from xuannv_embedding.utils.manifest import ManifestRecord, load_manifest, write_manifest

VERSION = "annual-quality-candidate-v2"
LOWRES_CHANNELS = {"s2": 10, "s1": 2, "landsat": 6}


@lru_cache(maxsize=32)
def _resolved_root(root: Path) -> Path:
    """数据根在单个进程内是常量；逐次重解析会在网络 FS 上产生可观的 stat 开销。"""
    return root.resolve()


def relative_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or "://" in relative:
        raise ValueError("Unsafe observation path")
    result = root / path
    # 结果路径仍逐次完整解析，符号链接逃逸检查的语义与缓存前完全一致。
    if not result.resolve().is_relative_to(_resolved_root(root)):
        raise ValueError("Observation path escapes data root")
    return result


def numeric_reasons(record: dict, *, highres: bool) -> list[str]:
    reasons = []
    if not record.get("grid_matches_parent"):
        reasons.append("grid_mismatch")
    if highres and record.get("status") != "usable":
        reasons.append("upstream_quarantine")
    if not highres and record.get("pixel_check") != "decoded":
        reasons.append("pixels_not_previously_decoded")
    channels = record.get("channels", 0)
    source = record.get("source_signature", "")
    if not highres and channels != LOWRES_CHANNELS.get(source):
        reasons.append("channel_mismatch")
    fraction = record.get("valid_fraction", 0)
    if not math.isfinite(fraction) or fraction < (0.9 if highres else 0.8):
        reasons.append("insufficient_valid_area")
    means = record.get("band_mean", [])
    variances = record.get("band_variance", [])
    counts = record.get("band_counts", [])
    if (
        not channels
        or any(len(values) != channels for values in (means, variances, counts))
        or any(value is None or not math.isfinite(value) for value in means + variances)
        or any(value <= 0 for value in counts)
        or any(value < 0 for value in variances)
    ):
        reasons.append("invalid_pixel_statistics")
    elif all(value == 0 for value in variances):
        reasons.append("constant_image_review")
    if not highres:
        extrema = record.get("band_min_max", [])
        if len(extrema) != channels or any(
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(value is None or not math.isfinite(value) for value in pair)
            for pair in extrema
        ):
            reasons.append("missing_numeric_extrema")
        elif any(pair[0] <= -32768 for pair in extrema):
            reasons.append("undeclared_negative_fill_review")
        zeros = record.get("zero_fractions", [])
        maxima = record.get("dtype_max_fractions", [])
        if len(zeros) != channels or len(maxima) != channels:
            reasons.append("missing_numeric_diagnostics")
        elif any(value > 0 for value in zeros):
            reasons.append("zero_value_review")
        if any(value is not None and value > 0 for value in maxima):
            reasons.append("integer_max_review")
    return reasons


def buffered_parents(parents: list[dict], distance_m: float = 4000) -> set[str]:
    """Drop lower-priority split members near another split, across UTM zones too."""
    priority = {"train": 0, "validation": 1, "test": 2}
    cells: dict[tuple[int, int], list[dict]] = defaultdict(list)
    removed = set()
    for parent in parents:
        longitude, latitude = parent["longitude"], parent["latitude"]
        cell = (math.floor(longitude / 0.1), math.floor(latitude / 0.1))
        radius = (
            math.ceil(distance_m / (111000 * max(0.01, math.cos(math.radians(latitude)))) / 0.1) + 1
        )
        for column in range(cell[0] - radius, cell[0] + radius + 1):
            for row in range(cell[1] - radius, cell[1] + radius + 1):
                for other in cells.get((column, row), []):
                    if other["split"] == parent["split"]:
                        continue
                    delta_lat = math.radians(latitude - other["latitude"])
                    delta_lon = math.radians(longitude - other["longitude"])
                    haversine = (
                        math.sin(delta_lat / 2) ** 2
                        + math.cos(math.radians(latitude))
                        * math.cos(math.radians(other["latitude"]))
                        * math.sin(delta_lon / 2) ** 2
                    )
                    distance = 6371000 * 2 * math.asin(min(1, math.sqrt(haversine)))
                    if distance < distance_m:
                        lower = (
                            parent
                            if priority[parent["split"]] < priority[other["split"]]
                            else other
                        )
                        removed.add(lower["parent_key"])
        cells[cell].append(parent)
    return removed


def select_seasons(observations: list[dict], limit: int = 4) -> list[dict]:
    ordered = sorted(
        observations, key=lambda item: (-item["valid_fraction"], item["date"], item["path"])
    )
    selected, quarters, digests = [], set(), set()
    for observation in ordered:
        quarter = (int(observation["date"][5:7]) - 1) // 3
        if quarter in quarters or observation["sha256"] in digests:
            continue
        selected.append(observation)
        quarters.add(quarter)
        digests.add(observation["sha256"])
        if len(selected) == limit:
            return sorted(selected, key=lambda item: (item["date"], item["path"]))
    for observation in ordered:
        if len(selected) == limit:
            break
        if observation["sha256"] not in digests:
            selected.append(observation)
            digests.add(observation["sha256"])
    return sorted(selected, key=lambda item: (item["date"], item["path"]))


def add_moments(state: dict, observation: dict) -> None:
    counts = np.asarray(observation["band_counts"], dtype=np.float64)
    means = np.asarray(observation["band_mean"], dtype=np.float64)
    variances = np.asarray(observation["band_variance"], dtype=np.float64)
    if not state:
        state.update(
            counts=np.zeros_like(counts),
            mean=np.zeros_like(means),
            moment=np.zeros_like(means),
            files=0,
        )
    total = state["counts"] + counts
    delta = means - state["mean"]
    state["moment"] += variances * counts + delta**2 * state["counts"] * counts / total
    state["mean"] += delta * counts / total
    state["counts"] = total
    state["files"] += 1


def verify_observation(root: Path, observation: dict) -> dict:
    path = relative_file(root, observation["path"])
    if sha256_file(path) != observation["sha256"]:
        raise ValueError(f"Payload checksum mismatch: {path}")
    with rasterio.open(path) as raster:
        values = raster.read()
        valid = (raster.read_masks() > 0).all(axis=0) & np.isfinite(values).all(axis=0)
        geometry = (raster.crs, raster.transform, raster.shape)
        epsg, bounds = parent_geometry(observation["parent_key"])
        if (
            raster.crs is None
            or raster.crs.to_epsg() != epsg
            or not np.allclose(raster.bounds, bounds, rtol=0, atol=0.01)
        ):
            raise ValueError(f"Changed observation geometry: {path}")
    if np.issubdtype(values.dtype, np.integer):
        valid &= ~(values == np.iinfo(values.dtype).max).any(axis=0)
    if observation["source"] in LOWRES_CHANNELS and (values[:, valid] <= -32768).any():
        raise ValueError(f"Undeclared negative fill in observation: {path}")
    valid &= ~(values == 0).all(axis=0)
    mask_path = relative_file(root, observation["mask"])
    with rasterio.open(mask_path) as raster:
        if (raster.crs, raster.transform, raster.shape) != geometry:
            raise ValueError(f"Mask geometry mismatch: {mask_path}")
        mask = raster.read(1) > 0
    if not np.array_equal(mask, valid):
        raise ValueError(f"Numeric mask mismatch: {mask_path}")
    accepted = values[:, valid].astype(np.float64)
    if (
        accepted.size == 0
        or not np.allclose(accepted.mean(axis=1), observation["band_mean"], rtol=1e-6, atol=1e-6)
        or not np.allclose(accepted.var(axis=1), observation["band_variance"], rtol=1e-6, atol=1e-6)
    ):
        raise ValueError(f"Pixel statistics mismatch: {path}")
    return {
        "path": observation["path"],
        "sha256": observation["sha256"],
        "mask_sha256": sha256_file(mask_path),
    }


def build(lowres: Path, highres: Path, output: Path, *, verify_per_source: int = 16) -> dict:
    if verify_per_source <= 0:
        raise ValueError("verify_per_source must be positive")
    stage = output.with_name(output.name + ".partial")
    if output.exists() or stage.exists():
        raise FileExistsError(output if output.exists() else stage)
    stage.mkdir(parents=True)
    root = Path(os.path.commonpath([lowres.resolve(), highres.resolve(), output.resolve()]))
    if root == Path("/"):
        raise ValueError("Input and output must share a dataset root")
    parents = list(rows(lowres / "selected_parents.jsonl"))
    parent_map = {item["parent_key"]: item for item in parents}
    if len(parent_map) != len(parents):
        raise ValueError("Duplicate parent identity")
    excluded = buffered_parents(parents)
    atomic_json(
        stage / "spatial_exclusions.json",
        {"minimum_center_distance_m": 4000, "parent_keys": sorted(excluded)},
    )
    schemas = json.loads((highres / "sources.json").read_text())
    fingerprints = {
        str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (lowres / "selected_parents.jsonl", highres / "sources.json")
    }
    inputs = [(lowres / "lowres_observations.jsonl", False)] + [
        (path, True) for path in sorted((highres / "parts").glob("*.jsonl"))
    ]
    database = sqlite3.connect(stage / "observations.sqlite")
    database.execute("PRAGMA journal_mode=DELETE")
    database.execute(
        "CREATE TABLE observations (parent TEXT, year INTEGER, source TEXT, "
        "date TEXT, highres INTEGER, payload TEXT)"
    )
    counter, reasons, by_source = Counter(), Counter(), defaultdict(Counter)
    with (stage / "excluded_observations.jsonl").open("w") as rejected:
        for input_path, is_highres in inputs:
            digest = hashlib.sha256()
            initial = input_path.stat()
            source_root = highres if is_highres else lowres
            with input_path.open("rb") as handle:
                for number, line in enumerate(handle, 1):
                    digest.update(line)
                    item = json.loads(line)
                    parent = item["parent_key"]
                    issues = numeric_reasons(item, highres=is_highres)
                    source = item.get("source_signature", "unknown")
                    observed_date = (
                        item.get("acquisition_time") if is_highres else item.get("month")
                    )
                    if observed_date is None:
                        issues.append("missing_date")
                        year = 0
                    else:
                        try:
                            year = date.fromisoformat(
                                observed_date if is_highres else observed_date + "-01"
                            ).year
                        except ValueError:
                            issues.append("invalid_date")
                            year = 0
                    if year not in (2020, 2021):
                        issues.append("outside_lowres_years")
                    if parent not in parent_map:
                        issues.append("parent_not_in_lowres")
                    elif item["split"] != parent_map[parent]["split"]:
                        raise ValueError(f"Split identity mismatch: {parent}")
                    if parent in excluded:
                        issues.append("spatial_buffer")
                    if is_highres and schemas.get(source, {}).get("product_group") != "5m":
                        issues.append("product_not_in_initial_5m_scope")
                    if not item.get("materialized_path") or not item.get("mask_path"):
                        issues.append("missing_materialized_path")
                    counter["catalog_records"] += 1
                    if issues:
                        counter["excluded_records"] += 1
                        reasons.update(issues)
                        by_source[source]["excluded"] += 1
                        rejected.write(
                            json.dumps(
                                {
                                    "parent_key": parent,
                                    "source": source,
                                    "date": observed_date,
                                    "path": item.get("materialized_path"),
                                    "reasons": issues,
                                }
                            )
                            + "\n"
                        )
                        continue
                    payload = {
                        "parent_key": parent,
                        "source": source,
                        "date": observed_date,
                        "time_precision": "day" if is_highres else "month",
                        "path": (source_root / item["materialized_path"])
                        .relative_to(root)
                        .as_posix(),
                        "mask": (source_root / item["mask_path"]).relative_to(root).as_posix(),
                        **{
                            name: item[name]
                            for name in (
                                "sha256",
                                "valid_fraction",
                                "band_mean",
                                "band_variance",
                                "band_counts",
                                "channels",
                                "height",
                                "width",
                                "grid_spacing_m",
                            )
                        },
                    }
                    database.execute(
                        "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)",
                        (parent, year, source, observed_date, int(is_highres), json.dumps(payload)),
                    )
                    counter["numeric_candidates"] += 1
                    by_source[source]["numeric_candidates"] += 1
                    if number % 50000 == 0:
                        database.commit()
                        print(
                            json.dumps({"input": input_path.name, "line": number, **counter}),
                            flush=True,
                        )
            if (initial.st_size, initial.st_mtime_ns) != (
                input_path.stat().st_size,
                input_path.stat().st_mtime_ns,
            ):
                raise ValueError("Catalog changed during scan")
            fingerprints[str(input_path)] = {"sha256": digest.hexdigest(), "bytes": initial.st_size}
            previous_summary = input_path.with_suffix(".summary.json")
            if is_highres and previous_summary.is_file():
                previous = json.loads(previous_summary.read_text())
                if previous["sha256"] != digest.hexdigest():
                    raise ValueError(f"Highres catalog checksum mismatch: {input_path}")
            database.commit()
            print(json.dumps({"finished": input_path.name, **counter}), flush=True)
    database.execute("CREATE INDEX sample_lookup ON observations(parent, year, source)")
    database.commit()
    stats: dict[str, dict] = defaultdict(dict)
    documents: dict[tuple[int, str], list[ManifestRecord]] = defaultdict(list)
    probes: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    sample_reasons = Counter()
    with (stage / "excluded_samples.jsonl").open("w") as rejected_samples:
        for parent in sorted(parent_map):
            if parent in excluded:
                continue
            split = parent_map[parent]["split"]
            for year in (2020, 2021):
                grouped = defaultdict(list)
                for source, payload in database.execute(
                    "SELECT source, payload FROM observations "
                    "WHERE parent=? AND year=? ORDER BY date",
                    (parent, year),
                ):
                    grouped[source].append(json.loads(payload))
                selected = {}
                for source, observations in grouped.items():
                    selected[source] = (
                        observations if source in LOWRES_CHANNELS else select_seasons(observations)
                    )
                    if source in LOWRES_CHANNELS and len(
                        {item["date"] for item in observations}
                    ) != len(observations):
                        raise ValueError(f"Duplicate lowres source/month: {parent}:{year}:{source}")
                low = [
                    item
                    for source, items in selected.items()
                    if source in LOWRES_CHANNELS
                    for item in items
                ]
                months = {item["date"] for item in low}
                quarters = {(int(item["date"][5:7]) - 1) // 3 for item in low}
                if len(months) < 6 or len(quarters) < 3:
                    sample_reasons["insufficient_annual_coverage"] += 1
                    rejected_samples.write(
                        json.dumps(
                            {
                                "parent_key": parent,
                                "year": year,
                                "split": split,
                                "reason": "insufficient_annual_coverage",
                                "months": len(months),
                                "quarters": len(quarters),
                            }
                        )
                        + "\n"
                    )
                    continue
                for source, observations in selected.items():
                    for observation in observations:
                        if split == "train":
                            add_moments(stats[source], observation)
                        rank = hashlib.sha256(observation["path"].encode()).hexdigest()
                        bucket = probes[(source, split)]
                        bucket.append((rank, observation))
                        bucket.sort(key=lambda pair: pair[0])
                        del bucket[verify_per_source:]
                documents[(year, split)].append(
                    ManifestRecord(
                        patch_id=f"parent_{parent}",
                        region=f"annual_{year}",
                        sources={
                            source: [item["path"] for item in items]
                            for source, items in selected.items()
                        },
                        grid={"parent_key": parent},
                        provenance={"year": year, "split": split, "observations": selected},
                        quality={
                            "numeric_catalog_filter": True,
                            "cloud_shadow": "unverified",
                            "landmark_registration": "unverified",
                            "highres_supported": any(
                                source not in LOWRES_CHANNELS for source in selected
                            ),
                        },
                    )
                )
    database.close()
    (stage / "statistics").mkdir()
    bad_sources = set()
    for source, state in stats.items():
        std = np.sqrt(state["moment"] / state["counts"])
        if np.any(std <= 0) or not np.isfinite(std).all():
            bad_sources.add(source)
            continue
        atomic_json(
            stage / "statistics" / f"{source}_stats.json",
            {
                "mean": state["mean"].tolist(),
                "std": std.tolist(),
                "band_counts": state["counts"].astype(np.int64).tolist(),
                "num_files": state["files"],
                "fit_split": "train",
                "fit_years": [2020, 2021],
                "units": "stored_values",
                "basis": "selected_catalog_valid_pixels",
                "cloud_QA": "unverified",
            },
        )
    manifest_counts, highres_counts = {}, {}
    for (year, split), records in documents.items():
        for record in records:
            for source in list(record.sources):
                if source not in stats or source in bad_sources:
                    record.sources.pop(source)
                    record.provenance["observations"].pop(source)
            record.quality["highres_supported"] = any(
                source not in LOWRES_CHANNELS for source in record.sources
            )
        path = stage / f"{year}.{split}.manifest.jsonl"
        write_manifest(
            path,
            records,
            months=[f"{year}-{month:02}" for month in range(1, 13)],
            generator_version=VERSION,
        )
        load_manifest(path)
        manifest_counts[path.name] = len(records)
        highres_counts[path.name] = sum(record.quality["highres_supported"] for record in records)
    checked = []
    for (source, split), bucket in sorted(probes.items()):
        if source not in stats or source in bad_sources:
            continue
        for _, observation in bucket:
            checked.append(
                {"source": source, "split": split, **verify_observation(root, observation)}
            )
        print(
            json.dumps({"payload_check_source": source, "split": split, "checked": len(bucket)}),
            flush=True,
        )
    atomic_json(
        stage / "payload_check.json",
        {"scope": "deterministic_source_split_hash_sample", "samples": checked},
    )
    atomic_json(stage / "input_fingerprints.json", fingerprints)
    atomic_json(
        stage / "sources.json",
        {
            **{
                source: {
                    "channels": channels,
                    "role": "temporal",
                    "band_semantics": "requires_provenance_review",
                }
                for source, channels in LOWRES_CHANNELS.items()
            },
            **{
                source: {**schema, "role": "highres"}
                for source, schema in schemas.items()
                if source in stats and source not in bad_sources
            },
        },
    )
    report = {
        "version": VERSION,
        "data_root": str(root),
        "status": "numeric_candidate_awaiting_scientific_QA",
        "training_ready": False,
        "annual_model_ready": False,
        "scientific_quality_validated": False,
        "catalog_counts": dict(counter),
        "exclusion_reasons_nonexclusive": dict(reasons),
        "source_counts": dict(by_source),
        "spatial_buffer_excluded_parents": len(excluded),
        "sample_exclusions": dict(sample_reasons),
        "manifests": manifest_counts,
        "highres_supported_samples": highres_counts,
        "payload_samples_verified": len(checked),
        "numeric_policy": {
            "lowres_min_valid_fraction": 0.8,
            "highres_min_valid_fraction": 0.9,
            "zero_or_integer_max_lowres": "whole_observation_review_exclusion",
            "lowres_minimum_le_negative_32768": "whole_observation_review_exclusion",
            "min_annual_months": 6,
            "min_annual_quarters": 3,
            "highres_limit_per_source_year": 4,
        },
        "remaining_gates": [
            "cloud_shadow_QA",
            "band_and_radiometry_provenance",
            "landmark_registration",
            "source_scene_split_audit",
            "annual_model_and_8_gpu_smoke",
        ],
        "source_payload_policy": (
            "immutable_external_rasters_referenced_no_copy; "
            "indexed_pixel_audits_plus_sample_reverification"
        ),
    }
    atomic_json(stage / "summary.json", report)
    atomic_json(
        stage / "checksums.json",
        {
            path.relative_to(stage).as_posix(): sha256_file(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        },
    )
    stage.rename(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv data quality-annual")
    parser.add_argument("--lowres", type=Path, required=True)
    parser.add_argument("--highres", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-per-source", type=int, default=16)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build(args.lowres, args.highres, args.output, verify_per_source=args.verify_per_source)
        ),
        flush=True,
    )
    return 0
