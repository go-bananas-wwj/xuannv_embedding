"""Pair a high-resolution pilot with monthly archives and loader smoke manifests."""

from __future__ import annotations

import calendar
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from rasterio.errors import RasterioError

from xuannv_embedding.data_process.highres_catalog import write_json, write_jsonl
from xuannv_embedding.data_process.observation_raster import inspect_raster, merge_statistics
from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest

MONTHS = [f"{year}-{month:02d}" for year in (2020, 2021) for month in range(1, 13)]
SOURCES = {"pc-s2": ("s2", 10), "pc-s1": ("s1", 2), "pc-ls": ("landsat", 6)}


def month_gap(acquisition: str, month: str) -> int:
    year, number = map(int, month.split("-"))
    first = date(year, number, 1)
    last = date(year, number, calendar.monthrange(year, number)[1])
    observed = date.fromisoformat(acquisition)
    return (
        (observed - first).days
        if observed < first
        else (observed - last).days if observed > last else 0
    )


def highres_candidates(parents: list[dict], records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        if "materialized_path" in record and not record["issues"]:
            grouped[record["parent_key"]].append(record)
    result = []
    for parent in parents:
        if not parent["materialize"]:
            continue
        for month in MONTHS:
            candidates = []
            for record in grouped[parent["parent_key"]]:
                gap = month_gap(record["acquisition_time"], month)
                if abs(gap) <= 30 and record["acquisition_time"][:4] == month[:4]:
                    candidates.append(
                        {
                            "observation_id": record["observation_id"],
                            "materialized_path": record["materialized_path"],
                            "source_signature": record["source_signature"],
                            "gap_days_to_month_interval": gap,
                            "valid_fraction": record["valid_fraction"],
                            "acquisition_time": record["acquisition_time"],
                        }
                    )
            candidates.sort(
                key=lambda item: (
                    abs(item["gap_days_to_month_interval"]),
                    -item["valid_fraction"],
                    item["observation_id"],
                )
            )
            result.append(
                {
                    "parent_key": parent["parent_key"],
                    "month": month,
                    "split": parent["split"],
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "policy": "same_year_gap_to_month_le_30_days; retrospective; QA_unverified",
                    "a1_candidate": candidates[0]["observation_id"] if candidates else None,
                    "a2_candidates": [item["observation_id"] for item in candidates[:4]],
                    "training_approved": False,
                }
            )
    return result


def _monthly_archive(
    stage: Path, root: Path, source: str, month: str, parents: list[dict]
) -> tuple[list, list]:
    canonical, channels = SOURCES[source]
    year, number = month.split("-")
    path = root / source / year / number / f"{source}_{year}_{number}.zip"
    availability, observations = [], []
    state = "archive_missing"
    archive = None
    members = {}
    if path.exists():
        try:
            archive = zipfile.ZipFile(path)
            for member in archive.infolist():
                if member.filename.lower().endswith((".tif", ".tiff")):
                    identity = Path(member.filename).stem
                    if identity in members:
                        raise ValueError(f"Duplicate parent in {path}")
                    members[identity] = member
            state = "not_in_archive"
        except zipfile.BadZipFile:
            state = "archive_unreadable"
    try:
        for parent in parents:
            identity = "parent_" + parent["parent_key"]
            item = {
                "parent_key": parent["parent_key"],
                "source": canonical,
                "month": month,
                "status": state,
                "split": parent["split"],
                "materialized": False,
            }
            if identity not in members:
                availability.append(item)
                continue
            item["status"] = "present_unchecked"
            if parent["materialize"]:
                relative = Path("lowres") / canonical / year / number / f"{identity}.tif"
                record = {
                    "parent_key": parent["parent_key"],
                    "source_signature": canonical,
                    "split": parent["split"],
                    "month": month,
                    "time_precision": "month",
                    "archive": path.relative_to(root).as_posix(),
                    "archive_member": members[identity].filename,
                }
                try:
                    payload = archive.read(members[identity])
                    metadata = inspect_raster(payload, parent["parent_key"], pixels=True)
                    record.update(metadata)
                    if not metadata["grid_matches_parent"]:
                        item["status"] = "grid_mismatch"
                    elif metadata["channels"] != channels:
                        item["status"] = "channel_mismatch"
                    elif metadata["valid_fraction"] == 0:
                        item["status"] = "all_pixels_invalid"
                    else:
                        inspect_raster(
                            payload, parent["parent_key"], pixels=True, destination=stage / relative
                        )
                        record["materialized_path"] = relative.as_posix()
                        record["mask_path"] = relative.with_name(
                            relative.stem + "_mask.tif"
                        ).as_posix()
                        item.update(status="cached_QA_unverified", materialized=True)
                except (RasterioError, zipfile.BadZipFile, EOFError) as error:
                    item["status"] = "member_read_error"
                    record["read_error"] = type(error).__name__
                observations.append(record)
            availability.append(item)
    finally:
        if archive is not None:
            archive.close()
    return availability, observations


def _smoke_config(stage: Path, source_names: list[str], split: str) -> dict[str, Any]:
    channels = dict(SOURCES.values())
    return {
        "schema_version": "1",
        "paths": {
            "data_root": str(stage),
            "output_root": str(stage / "outputs"),
            "artifact_root": str(stage / "artifacts"),
        },
        "experiment": {"name": "national_raw_value_loader_smoke", "seed": 42},
        "model": {
            "embed_dim": 64,
            "stem_dim": 8,
            "num_months": len(MONTHS),
            "ref_year": 2020,
            "ref_month": 1,
            "input_sources": {
                source: {"channels": channels[source], "role": "temporal"}
                for source in source_names
            },
            "target_heads": {
                source
                + "_recon": {
                    "source": source,
                    "loss_type": "continuous",
                    "channels": channels[source],
                    "weight": 1.0,
                }
                for source in source_names
            },
            "stp": {
                "space_dim": 16,
                "time_dim": 16,
                "precision_dim": 16,
                "num_heads": 2,
                "num_blocks": 1,
                "temporal_fusion": "gated_sum",
                "time_attention_mode": "none",
            },
        },
        "training": {
            "epochs": 1,
            "lr": 0.0001,
            "weight_decay": 0.0,
            "warmup_epochs": 0,
            "gradient_accumulation_steps": 1,
            "save_every": 1,
            "amp": False,
            "semantic_probe_weight": 0.0,
        },
        "data": {
            "months": MONTHS,
            "batch_size": 1,
            "num_workers": 0,
            "patch_size": 128,
            "datasets": [
                {
                    "region": "national_pilot",
                    "manifest_path": str(stage / f"{split}.manifest.jsonl"),
                    "statistics_dir": str(stage / "statistics"),
                    "patch_grid_path": str(stage / "selected_parents.jsonl"),
                    "source_map": {source: source for source in source_names},
                    "supervised_label_roots": {},
                    "sampling_weight": 1.0,
                }
            ],
        },
    }


def prepare_lowres(
    stage: Path, lowres_root: Path, parents: list[dict], highres: list[dict]
) -> dict:
    import yaml

    availability, observations = [], []
    for source in SOURCES:
        for month in MONTHS:
            current_availability, current_observations = _monthly_archive(
                stage, lowres_root, source, month, parents
            )
            availability.extend(current_availability)
            observations.extend(current_observations)
            print(
                f"Lowres {source} {month}: "
                f"cached={sum(item['materialized'] for item in current_availability)}",
                flush=True,
            )
    write_jsonl(stage / "lowres_availability.jsonl", availability)
    write_jsonl(stage / "lowres_observations.jsonl", observations)
    candidates = highres_candidates(parents, highres)
    write_jsonl(stage / "highres_month_candidates.jsonl", candidates)
    statistics = defaultdict(list)
    paths = defaultdict(lambda: defaultdict(list))
    for record in observations + highres:
        if "materialized_path" not in record:
            continue
        if record["split"] == "train":
            statistics[record["source_signature"]].append(record)
    for record in observations:
        if "materialized_path" in record:
            paths[record["parent_key"]][record["source_signature"]].append(
                record["materialized_path"]
            )
    (stage / "statistics").mkdir(exist_ok=True)
    for source, values in sorted(statistics.items()):
        write_json(stage / "statistics" / f"{source}_stats.json", merge_statistics(values))
    source_names = [canonical for canonical, _ in SOURCES.values() if canonical in statistics]
    if len(source_names) != len(SOURCES):
        raise ValueError("Pilot has no training statistics for one or more low-resolution sources")
    manifests = defaultdict(list)
    for parent in parents:
        if not parent["materialize"]:
            continue
        manifests[parent["split"]].append(
            ManifestRecord(
                patch_id="parent_" + parent["parent_key"],
                region="national_pilot",
                source_patch_id=parent["patch_id"],
                sources={
                    source: sorted(paths[parent["parent_key"]].get(source, [])) or None
                    for source in source_names
                },
                grid={"parent_key": parent["parent_key"]},
                quality={"purpose": "IO_smoke_only", "QA": "unverified"},
                provenance={"split": parent["split"], "observations": "lowres_observations.jsonl"},
            )
        )
    for split in ("train", "validation", "test"):
        write_manifest(
            stage / f"{split}.manifest.jsonl",
            manifests[split],
            months=MONTHS,
            generator_version="observation-pilot-v1",
        )
    import torch

    from xuannv_embedding.config import Config
    from xuannv_embedding.data.raster_dataset import RegionRasterDataset

    torch.set_num_threads(2)
    checked, available_count = 0, 0
    expected_count = sum(item["materialized"] for item in availability)
    for split in ("train", "validation", "test"):
        if not manifests[split]:
            continue
        config_path = stage / f"{split}.loader-smoke.yaml"
        config_path.write_text(
            yaml.safe_dump(_smoke_config(stage, source_names, split), sort_keys=False),
            encoding="utf-8",
        )
        config = Config.from_yaml(config_path)
        dataset = RegionRasterDataset(config, config.data.datasets[0])
        for sample in dataset:
            for source in source_names:
                if not torch.isfinite(sample["source_frames"][source]).all():
                    raise ValueError("Nonfinite normalized loader values")
                available_count += int(sample["source_masks"][source].sum())
            checked += 1
    if available_count != expected_count:
        raise ValueError(f"Loader availability mismatch: {available_count} != {expected_count}")
    check = {
        "parents_checked": checked,
        "available_source_months": available_count,
        "expected_source_months": expected_count,
        "passed": True,
        "trained": False,
    }
    write_json(stage / "loader_check.json", check)
    final_root = stage.with_name(stage.name.removesuffix(".partial"))
    for split in manifests:
        (stage / f"{split}.loader-smoke.yaml").write_text(
            yaml.safe_dump(_smoke_config(final_root, source_names, split), sort_keys=False),
            encoding="utf-8",
        )
    return {
        "availability_statuses": dict(Counter(item["status"] for item in availability)),
        "cached_tiffs": expected_count,
        "loader": check,
        "highres_parent_months_with_candidates": sum(
            bool(item["candidates"]) for item in candidates
        ),
        "highres_training_approved": False,
        "labels_prepared": False,
    }
