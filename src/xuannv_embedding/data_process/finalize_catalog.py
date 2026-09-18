"""Finish an already materialized catalog with a bounded loader audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import yaml

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import RegionRasterDataset
from xuannv_embedding.data_process.highres_catalog import write_json
from xuannv_embedding.data_process.observation_raster import sha256_file
from xuannv_embedding.data_process.pilot_cache import SOURCES, _smoke_config
from xuannv_embedding.utils.manifest import load_manifest


def _sample_indices(dataset: RegionRasterDataset, limit: int, seed: int, split: str) -> list[int]:
    if limit <= 0 or limit >= len(dataset):
        return list(range(len(dataset)))
    ranked = sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{split}:{dataset.records[index].patch_id}".encode()
        ).digest(),
    )
    return sorted(ranked[:limit])


def _check_sample(dataset: RegionRasterDataset, index: int) -> tuple[str, int]:
    sample = dataset[index]
    available = 0
    for source, values in sample["source_frames"].items():
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"Nonfinite normalized loader values: {sample['patch_id']}:{source}")
        available += int(sample["source_masks"][source].sum())
    return sample["patch_id"], available


def _validate_samples(
    stage: Path,
    output: Path,
    limits: dict[str, int],
    *,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    source_names = [
        canonical
        for canonical, _ in SOURCES.values()
        if (stage / "statistics" / f"{canonical}_stats.json").is_file()
    ]
    if len(source_names) != len(SOURCES):
        raise ValueError("Prepared catalog lacks statistics for one or more low-resolution sources")
    torch.set_num_threads(1)
    splits: dict[str, Any] = {}
    total_available = 0
    total_checked = 0
    for split in ("train", "validation", "test"):
        manifest_path = stage / f"{split}.manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = load_manifest(manifest_path)
        if not manifest.records:
            splits[split] = {
                "manifest_records": 0,
                "sampled_records": 0,
                "available_source_months": 0,
                "sample_patch_ids_sha256": hashlib.sha256(b"").hexdigest(),
            }
            continue
        config_path = stage / f"{split}.loader-smoke.yaml"
        config_path.write_text(
            yaml.safe_dump(_smoke_config(stage, source_names, split), sort_keys=False),
            encoding="utf-8",
        )
        config = Config.from_yaml(config_path)
        dataset = RegionRasterDataset(config, config.data.datasets[0])
        indices = _sample_indices(dataset, limits[split], seed, split)
        checked_ids: list[str] = []
        available = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for checked, (patch_id, count) in enumerate(
                pool.map(lambda index: _check_sample(dataset, index), indices), start=1
            ):
                checked_ids.append(patch_id)
                available += count
                if checked % 64 == 0 or checked == len(indices):
                    print(
                        json.dumps(
                            {
                                "split": split,
                                "checked": checked,
                                "sample_size": len(indices),
                                "available_source_months": available,
                            }
                        ),
                        flush=True,
                    )
        digest = hashlib.sha256("\n".join(sorted(checked_ids)).encode()).hexdigest()
        splits[split] = {
            "manifest_records": len(dataset),
            "sampled_records": len(indices),
            "available_source_months": available,
            "sample_patch_ids_sha256": digest,
        }
        total_available += available
        total_checked += len(indices)
    check = {
        "mode": "deterministic_hash_sample",
        "seed": seed,
        "workers": workers,
        "splits": splits,
        "parents_checked": total_checked,
        "available_source_months_checked": total_available,
        "passed": True,
        "trained": False,
    }
    write_json(stage / "loader_check.json", check)
    for split in splits:
        config_path = stage / f"{split}.loader-smoke.yaml"
        if not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        stage_text = str(stage)
        output_text = str(output)
        serialized = yaml.safe_dump(config, sort_keys=False).replace(stage_text, output_text)
        config_path.write_text(serialized, encoding="utf-8")
    return check


def _availability_summary(path: Path) -> tuple[dict[str, int], int]:
    statuses: Counter[str] = Counter()
    materialized = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            statuses[record["status"]] += 1
            materialized += int(record["materialized"])
    return dict(statuses), materialized


def finalize_catalog(
    stage: Path,
    output: Path,
    *,
    train_samples: int = 2048,
    validation_samples: int = 512,
    test_samples: int = 512,
    seed: int = 42,
    workers: int = 4,
) -> dict[str, Any]:
    if not stage.is_dir() or stage.resolve() == output.resolve():
        raise ValueError("Stage must be an existing directory distinct from output")
    if output.exists():
        raise FileExistsError(output)
    required = [
        "fingerprint.json",
        "selected_parents.jsonl",
        "lowres_availability.jsonl",
        "lowres_observations.jsonl",
        "highres_month_candidates.jsonl",
    ]
    for relative in required:
        if not (stage / relative).is_file():
            raise FileNotFoundError(stage / relative)
    limits = {
        "train": train_samples,
        "validation": validation_samples,
        "test": test_samples,
    }
    if any(value <= 0 for value in limits.values()) or workers <= 0:
        raise ValueError("Sample counts and workers must be positive")
    check = _validate_samples(stage, output, limits, seed=seed, workers=workers)
    availability_statuses, cached_tiffs = _availability_summary(stage / "lowres_availability.jsonl")
    fingerprint = json.loads((stage / "fingerprint.json").read_text(encoding="utf-8"))
    manifests = {
        f"{split}.manifest.jsonl": load_manifest(
            stage / f"{split}.manifest.jsonl"
        ).meta.record_count
        for split in limits
    }
    part_summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((stage / "catalog_parts").glob("*.summary.json"))
    ]
    summary = {
        "version": fingerprint.get("version", "observation-pilot-v1"),
        "status": "training_data_prepared",
        "parent_count": fingerprint.get("parent_limit"),
        "observation_count": sum(item.get("observations", 0) for item in part_summaries),
        "materialized_highres_tiffs": sum(item.get("materialized", 0) for item in part_summaries),
        "manifests": manifests,
        "lowres": {
            "availability_statuses": availability_statuses,
            "cached_tiffs": cached_tiffs,
            "loader": check,
            "highres_training_approved": False,
            "labels_prepared": False,
        },
        "data_interface_ready": True,
        "scientific_quality_validated": False,
    }
    write_json(stage / "summary.json", summary)
    control_files = [
        stage / "fingerprint.json",
        stage / "selected_parents.jsonl",
        stage / "loader_check.json",
        stage / "summary.json",
        *(stage / f"{split}.manifest.jsonl" for split in limits),
        *(stage / f"{split}.manifest.jsonl.meta.json" for split in limits),
        *sorted((stage / "statistics").glob("*.json")),
    ]
    checksums = {
        "scope": "control_plane_only",
        "payload_policy": "GeoTIFF payloads were checked by deterministic loader sampling",
        "files": {
            path.relative_to(stage).as_posix(): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in control_files
        },
    }
    write_json(stage / "checksums.json", checksums)
    stage.rename(output)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv data finalize-catalog")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    report = finalize_catalog(
        args.stage,
        args.output,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        seed=args.seed,
        workers=args.workers,
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
