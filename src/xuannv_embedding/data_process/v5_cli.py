"""Data-only V5 stage dispatcher. No model training or acceptance approval operations."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import rasterio

from xuannv_embedding.data_process.v5_sources import (
    ArchiveSpec,
    extract_archive,
    lock_source,
    now,
    sha256,
    write_json,
)
from xuannv_embedding.data_process.v5_transfer import download_chunked


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def input_lock(args: argparse.Namespace, source: dict) -> None:
    registry = args.base_root / "registry/national_62000.parquet"
    frame = pd.read_parquet(registry)
    if frame.patch_id.duplicated().any() or not set(frame.split).issubset({"train", "val", "test"}):
        raise ValueError("invalid source grid registry")
    fingerprint = {
        "registry_sha256": sha256(registry),
        "source_revision": source["revision"],
        "source_manifest_sha256": source["manifest_sha256"],
    }
    path = args.dataset_root / "locks/input.lock.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("input fingerprint changed; choose a new dataset version")
        return
    target = args.dataset_root / "registry/national_62000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) != fingerprint["registry_sha256"]:
        raise ValueError("existing grid differs from source")
    shutil.copyfile(registry, target)
    write_json(
        path,
        {
            "schema": "xuannv.input-lock.v5",
            "fingerprint": fingerprint,
            "base_root": str(args.base_root.resolve()),
            "patches": len(frame),
            "split_counts": frame.split.value_counts().to_dict(),
            "created_at": now(),
        },
    )
    acceptance = args.dataset_root / "locks/acceptance.json"
    if not acceptance.exists():
        write_json(
            acceptance,
            {
                "status": "incomplete",
                "training_authorized": False,
                "user_accepted": False,
                "created_at": now(),
            },
        )


def validate_package(args: argparse.Namespace, spec: ArchiveSpec) -> dict:
    """Decode every TIFF in a package before releasing the pilot gate."""
    index = pd.read_csv(args.source_root / "manifests/ARCHIVE_INDEX.tsv", sep="\t")
    patch_ids = index.loc[index.archive == spec.archive, "patchid"]
    output = args.report_root / "integrity_shards" / (spec.archive + ".json")
    if output.exists():
        result = json.loads(output.read_text())
        if result["sha256"] != spec.sha256 or result["status"] != "complete":
            raise ValueError("invalid package verification marker")
        return result
    count = 0
    failures = []
    for patch_id in patch_ids:
        for path in sorted((args.source_root / "extracted" / patch_id).rglob("*.tif")):
            try:
                with rasterio.open(path) as dataset:
                    if dataset.crs is None or dataset.count <= 0:
                        raise ValueError("missing raster geometry or bands")
                    for _, window in dataset.block_windows(1):
                        dataset.read(window=window)
                        dataset.read_masks(window=window)
                count += 1
            except Exception as exc:
                failures.append(
                    {
                        "relative_path": str(path.relative_to(args.source_root)),
                        "error_type": type(exc).__name__,
                    }
                )
    result = {
        "archive": spec.archive,
        "sha256": spec.sha256,
        "decoded_tiffs": count,
        "expected_tiffs": spec.tiff_count,
        "failures": failures,
        "status": "complete" if count == spec.tiff_count and not failures else "failed",
        "finished_at": now(),
    }
    write_json(output, result)
    if result["status"] != "complete":
        raise ValueError("package pixel verification failed; inspect integrity report")
    return result


def acquire(args: argparse.Namespace, source: dict) -> None:
    specs = [ArchiveSpec(**item) for item in source["archives"]][: args.limit]
    status_path = args.source_root / "manifests/download_status.parquet"
    statuses = pd.read_parquet(status_path).to_dict("records") if status_path.exists() else []
    by_name = {row["archive"]: row for row in statuses}
    for offset in range(0, len(specs), 2):
        pair = specs[offset : offset + 2]
        if shutil.disk_usage(args.source_root).free < 30 * 1024**3:
            raise RuntimeError("less than 30 GiB free; download stopped before next pair")
        if args.stage in {"download", "ingest"}:
            for spec in pair:
                by_name[spec.archive] = {
                    **spec.__dict__,
                    "status": "running",
                    "actual_bytes": 0,
                    "retries": 0,
                    "started_at": now(),
                    "finished_at": None,
                }
            atomic_parquet(pd.DataFrame(by_name.values()), status_path)
            errors = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = {
                    pool.submit(
                        download_chunked, spec, args.source_root / "packages", source["revision"]
                    ): spec
                    for spec in pair
                }
                for future in as_completed(jobs):
                    spec = jobs[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            **spec.__dict__,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "finished_at": now(),
                        }
                        errors.append(exc)
                    by_name[spec.archive] = result
                    atomic_parquet(pd.DataFrame(by_name.values()), status_path)
                    print(
                        json.dumps({"archive": spec.archive, "status": result["status"]}),
                        flush=True,
                    )
            if errors:
                raise RuntimeError("archive download failed; inspect download_status") from None
        if args.stage in {"extract", "ingest"}:
            for spec in pair:
                result = extract_archive(
                    args.source_root / "packages" / spec.archive,
                    args.source_root / "extracted",
                    spec,
                )
                validate_package(args, spec)
                print(
                    json.dumps(
                        {
                            "archive": spec.archive,
                            "extracted": result["tiff_count"],
                            "pixel_verification": "passed",
                        }
                    ),
                    flush=True,
                )
    write_json(
        args.report_root / "archive_integrity.json",
        {
            "selected_archives": len(specs),
            "total_archives": len(source["archives"]),
            "scope": "selected_archives",
            "finished_at": now(),
            "fully_acquired": args.stage == "ingest" and len(specs) == len(source["archives"]),
        },
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["source-lock", "download", "extract", "ingest", "catalog"],
    )
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64, help="number of archive packages, 1..64")
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 64:
        parser.error("--limit must be between 1 and 64")
    args.source_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    lock_name = ".catalog.lock" if args.stage == "catalog" else ".prepare.lock"
    with (args.source_root / lock_name).open("a") as mutex:
        try:
            fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another V5 source stage is running") from None
        record = {
            "step": args.stage,
            "started_at": now(),
            "status": "running",
            "parameters": {"limit": args.limit},
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }
        record_path = args.report_root / "stages" / (args.stage + ".json")
        write_json(record_path, record)
        try:
            source = lock_source(args.source_root)
            input_lock(args, source)
            record["input_fingerprint"] = source["manifest_sha256"]
            write_json(record_path, record)
            if args.stage == "catalog":
                from xuannv_embedding.data_process.v5_catalog import build_catalog

                record["result"] = build_catalog(
                    args.source_root, args.dataset_root, args.report_root
                )
            elif args.stage != "source-lock":
                acquire(args, source)
            record["status"] = "complete"
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            record["finished_at"] = now()
            write_json(record_path, record)
    return 0
