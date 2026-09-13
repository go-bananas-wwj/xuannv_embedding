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
    target = args.dataset_root / "registry/national_62000.parquet"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("input fingerprint changed; choose a new dataset version")
        if not target.is_file() or sha256(target) != fingerprint["registry_sha256"]:
            raise ValueError("dataset registry is missing or differs from the locked source")
        return
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
        choices=[
            "source-lock",
            "download",
            "extract",
            "ingest",
            "catalog",
            "catalog-partial-bands",
            "radiometry",
            "dense-integrity",
            "quality",
            "gaofen-quality",
            "cloud-resolution-audit",
            "alignment-calibration",
            "band-alignment",
            "band-review",
            "targets",
            "target-values",
            "target-temporal",
            "target-negative-rules",
            "target-negative-corrections",
            "target-sources",
            "target-geometry",
            "dem-geometry",
            "osm-geometry",
            "dem-corrections",
            "visual-review",
            "followup",
            "report",
        ],
    )
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64, help="number of archive packages, 1..64")
    parser.add_argument("--dense-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--gaofen-source-catalog", type=Path)
    parser.add_argument("--quality-root", type=Path)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--sensor-family", choices=["jilin1", "gaofen"])
    parser.add_argument("--alignment-version", default="v1")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--target-family", choices=["worldcover", "clcd", "nightlights"])
    parser.add_argument("--max-patches", type=int)
    parser.add_argument("--target-audit-version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args(argv)
    if (
        args.stage in {"alignment-calibration", "band-alignment", "band-review"}
        and args.sensor_family is None
    ):
        parser.error("--sensor-family is required for native-band alignment")
    if args.stage == "target-geometry" and args.target_family is None:
        parser.error("--target-family is required for target geometry audit")
    if args.max_patches is not None and args.max_patches <= 0:
        parser.error("--max-patches must be positive")
    if args.stage in {"radiometry", "dense-integrity"} and args.dense_root is None:
        parser.error("--dense-root is required for dense source audits")
    if args.stage in {"quality", "gaofen-quality", "followup"} and args.model_dir is None:
        parser.error("--model-dir is required for quality")
    if args.stage == "gaofen-quality" and args.gaofen_source_catalog is None:
        parser.error("--gaofen-source-catalog is required")
    if args.stage in {"visual-review", "cloud-resolution-audit"} and args.quality_root is None:
        parser.error("--quality-root is required for cloud review")
    if args.stage == "cloud-resolution-audit" and args.model_dir is None:
        parser.error("--model-dir is required for cloud-resolution-audit")
    if args.max_scenes is not None and args.max_scenes <= 0:
        parser.error("--max-scenes must be positive")
    if not 1 <= args.limit <= 64:
        parser.error("--limit must be between 1 and 64")
    args.source_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    lock_name = (
        ".prepare.lock"
        if args.stage in {"source-lock", "download", "extract", "ingest"}
        else (
            f".{args.stage}.{args.sensor_family}.lock"
            if args.stage in {"alignment-calibration", "band-alignment", "band-review"}
            else f".{args.stage}.lock"
        )
    )
    if args.stage == "target-geometry":
        lock_name = f".target-geometry.{args.target_family}.lock"
    with (args.source_root / lock_name).open("a") as mutex:
        try:
            fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another V5 source stage is running") from None
        record = {
            "step": args.stage,
            "started_at": now(),
            "status": "running",
            "parameters": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "stage"
            },
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }
        record["code_fingerprint"] = {
            path.name: sha256(path) for path in Path(__file__).parent.glob("v5_*.py")
        }
        record_name = (
            f"{args.stage}_{args.sensor_family}"
            if args.stage in {"alignment-calibration", "band-alignment", "band-review"}
            else args.stage
        )
        if args.stage == "target-geometry":
            record_name = f"target-geometry_{args.target_family}"
        record_path = args.report_root / "stages" / (record_name + ".json")
        write_json(record_path, record)
        try:
            source = lock_source(args.source_root)
            input_lock(args, source)
            record["input_fingerprint"] = source["manifest_sha256"]
            write_json(record_path, record)
            if args.stage == "dense-integrity":
                from xuannv_embedding.data_process.v5_dense_integrity import audit_dense_integrity

                record["result"] = audit_dense_integrity(
                    args.dense_root, args.dataset_root, args.report_root
                )
            elif args.stage == "followup":
                from xuannv_embedding.data_process.v5_followup import follow_started_jobs

                record["result"] = follow_started_jobs(args)
            elif args.stage == "dem-corrections":
                from xuannv_embedding.data_process.v5_dem_corrections import build_dem_corrections

                if args.max_patches is not None:
                    raise ValueError("DEM corrections require the complete audit and grid")
                record["result"] = build_dem_corrections(args.dataset_root, args.report_root)
            elif args.stage == "osm-geometry":
                from xuannv_embedding.data_process.v5_osm_geometry import audit_osm_geometry

                record["result"] = audit_osm_geometry(
                    args.dataset_root, args.report_root, max_patches=args.max_patches
                )
            elif args.stage == "dem-geometry":
                from xuannv_embedding.data_process.v5_dem_geometry import audit_dem_geometry

                record["result"] = audit_dem_geometry(
                    args.dataset_root,
                    args.report_root,
                    max_patches=args.max_patches,
                )
            elif args.stage == "target-geometry":
                from xuannv_embedding.data_process.v5_target_geometry import audit_target_geometry

                options = {"max_patches": args.max_patches}
                if args.target_audit_version != "v1":
                    options["audit_version"] = args.target_audit_version
                record["result"] = audit_target_geometry(
                    args.dataset_root,
                    args.report_root,
                    args.target_family,
                    **options,
                )
            elif args.stage == "target-sources":
                from xuannv_embedding.data_process.v5_provenance import audit_target_sources

                record["result"] = audit_target_sources(args.base_root, args.report_root)
            elif args.stage == "band-review":
                from xuannv_embedding.data_process.v5_band_review import review_native_bands

                record["result"] = review_native_bands(
                    args.dataset_root,
                    args.report_root,
                    args.sensor_family,
                    version=args.alignment_version,
                )
            elif args.stage in {"alignment-calibration", "band-alignment"}:
                from xuannv_embedding.data_process.v5_intraband import (
                    calibrate_family,
                    run_intraband,
                )

                runner = (
                    calibrate_family if args.stage == "alignment-calibration" else run_intraband
                )
                options = {"version": args.alignment_version}
                if args.stage == "band-alignment":
                    options["workers"] = args.workers
                record["result"] = runner(
                    args.dataset_root,
                    args.report_root,
                    args.sensor_family,
                    **options,
                )
            elif args.stage == "cloud-resolution-audit":
                from xuannv_embedding.data_process.v5_resolution import compare_jilin_resolution

                record["result"] = compare_jilin_resolution(
                    args.dataset_root,
                    args.report_root,
                    args.quality_root,
                    args.model_dir,
                    device_id=args.device_id,
                )
            elif args.stage == "target-negative-corrections":
                from xuannv_embedding.data_process.v5_negative_corrections import (
                    build_negative_corrections,
                )

                if args.max_patches is not None:
                    raise ValueError("Negative corrections require the complete audited data view")
                record["result"] = build_negative_corrections(args.dataset_root, args.report_root)
            elif args.stage == "target-negative-rules":
                from xuannv_embedding.data_process.v5_negative_rules import audit_negative_rules

                record["result"] = audit_negative_rules(
                    args.dataset_root,
                    args.report_root,
                    max_patches=args.max_patches,
                )
            elif args.stage == "target-temporal":
                from xuannv_embedding.data_process.v5_temporal import audit_osm_temporal

                record["result"] = audit_osm_temporal(args.dataset_root, args.report_root)
            elif args.stage == "target-values":
                from xuannv_embedding.data_process.v5_targets import audit_target_values

                record["result"] = audit_target_values(args.dataset_root, args.report_root)
            elif args.stage == "visual-review":
                from xuannv_embedding.data_process.v5_visual import review_gaofen

                record["result"] = review_gaofen(args.quality_root, args.report_root)
            elif args.stage == "catalog-partial-bands":
                from xuannv_embedding.data_process.v5_partial_bands import catalog_partial_bands

                record["result"] = catalog_partial_bands(
                    args.source_root, args.dataset_root, args.report_root
                )
            elif args.stage == "catalog":
                from xuannv_embedding.data_process.v5_catalog import build_catalog

                record["result"] = build_catalog(
                    args.source_root, args.dataset_root, args.report_root
                )
            elif args.stage == "gaofen-quality":
                from xuannv_embedding.data_process.v5_gaofen import process_gaofen

                record["result"] = process_gaofen(
                    args.dataset_root,
                    args.report_root,
                    args.gaofen_source_catalog,
                    args.model_dir,
                    device_id=args.device_id,
                    limit=args.max_scenes,
                )
            elif args.stage == "quality":
                from xuannv_embedding.data_process.v5_cloud import process_jilin_cloud

                record["result"] = process_jilin_cloud(
                    args.dataset_root,
                    args.report_root,
                    args.model_dir,
                    device_id=args.device_id,
                    limit=args.max_scenes,
                )
            elif args.stage in {"radiometry", "targets", "report"}:
                from xuannv_embedding.data_process.v5_audit import (
                    audit_radiometry,
                    audit_targets,
                    report_progress,
                )

                if args.stage == "radiometry":
                    record["result"] = audit_radiometry(args.dense_root, args.report_root)
                elif args.stage == "targets":
                    record["result"] = audit_targets(
                        args.base_root, args.dataset_root, args.report_root
                    )
                else:
                    record["result"] = report_progress(
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
