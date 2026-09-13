"""Finite follow-through for already started data jobs; no scheduler or training operations."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path

from xuannv_embedding.data_process.v5_audit import report_progress
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def next_source_action(
    *,
    total: int,
    verified: int,
    cataloged: int,
    quality_status: str | None,
    band_ready: bool = True,
    band_running: bool = False,
    partial_ready: bool = True,
) -> str | None:
    if total > 0 and verified > cataloged:
        return "catalog"
    if total > 0 and verified > 0 and verified == cataloged and not partial_ready:
        return "catalog-partial-bands"
    if total > 0 and verified > 0 and verified == cataloged and not band_ready:
        return None if band_running else "band-alignment"
    if total > 0 and verified == total and cataloged == total and quality_status is None:
        return "quality"
    return None


def partial_catalog_finished(source_root: Path, dataset_root: Path, report_root: Path) -> bool:
    """Check the current partial snapshot, without implying cloud or alignment approval."""
    import numpy as np
    import pandas as pd
    import rasterio

    root = dataset_root / "observations/highres/jilin1"
    try:
        pointer = json.loads((root / "partial_bands/current.json").read_text())
        lock_path = Path(pointer["lock_path"])
        if sha256(lock_path) != pointer["lock_sha256"]:
            return False
        locked = json.loads(lock_path.read_text())
        fingerprint = locked["fingerprint"]
        version = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
        directory = root / "partial_bands" / version
        if (
            locked.get("status") != "partial_band_catalog_finished"
            or pointer["version"] != version
            or lock_path.resolve() != (directory / "catalog.lock.json").resolve()
            or Path(locked["output"]).resolve() != directory.resolve()
        ):
            return False
        inputs = {
            "registry_sha256": dataset_root / "registry/national_62000.parquet",
            "rejected_inventory_sha256": report_root / "rejected_files.parquet",
            "complete_catalog_sha256": root / "files.parquet",
            "source_lock_sha256": source_root / "manifests/source.lock.json",
            "code_sha256": Path(__file__).with_name("v5_partial_bands.py"),
            "reader_sha256": Path(__file__).with_name("v5_rasters.py"),
            "grid_matcher_sha256": Path(__file__).with_name("v5_catalog.py"),
        }
        if any(sha256(path) != fingerprint[key] for key, path in inputs.items()):
            return False
        runtime = {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rasterio": rasterio.__version__,
            "gdal": rasterio.__gdal_version__,
            "pyarrow": package_version("pyarrow"),
        }
        return (
            fingerprint.get("runtime") == runtime
            and sha256(directory / "files.parquet") == locked["output_sha256"]
            and sha256(directory / "files_with_partial_bands.parquet")
            == locked["augmented_catalog_sha256"]
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def band_inventory_finished(
    dataset_root: Path, summary: dict, inventory_hash: str | None, version: str
) -> bool:
    """Audit completion is distinct from passed observations or fusion authorization."""
    root = dataset_root / "quality/alignment/intraband/jilin1" / version
    calibration = root / "calibration.json"
    if (
        not inventory_hash
        or summary.get("status") != "intraband_audit_finished"
        or summary.get("family") != "jilin1"
        or summary.get("version") != version
        or summary.get("input_inventory_sha256") != inventory_hash
        or not summary.get("selected_observations")
        or summary.get("processed_observations") != summary.get("selected_observations")
        or not (root / "observations.parquet").is_file()
        or not calibration.is_file()
        or summary.get("calibration_sha256") != sha256(calibration)
        or json.loads(calibration.read_text()).get("status") != "passed"
        or not summary.get("code_sha256")
    ):
        return False
    return all(
        Path(__file__).with_name(name).is_file()
        and sha256(Path(__file__).with_name(name)) == expected
        for name, expected in summary["code_sha256"].items()
    )


def process_identity(pid: int) -> str | None:
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if parts[0] == "Z" else f"{pid}:{parts[19]}"
    except (FileNotFoundError, ProcessLookupError):
        return None


def follow_started_jobs(args) -> dict:
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}

    jobs = {}
    for name in (
        "download",
        "gaofen",
        "target_value",
        "target_source",
        "target_temporal",
        "dense_integrity",
        "band_gaofen",
        "band_jilin1",
    ):
        info = read(args.report_root / f"{name}_worker.json")
        if info:
            pid = int(info["pid"])
            current = process_identity(pid)
            identity = current if info.get("identity", current) == current else None
            jobs[name] = {"pid": pid, "identity": identity}
    if not jobs:
        raise ValueError("no started data jobs to follow")
    failures = []
    attempted_failures = set()
    output = args.report_root / "followup_progress.json"
    inventory_cache = {}
    alignment_version = args.alignment_version
    while True:
        alive = {
            name: bool(info["identity"]) and process_identity(info["pid"]) == info["identity"]
            for name, info in jobs.items()
        }
        source = read(args.source_root / "manifests/source.lock.json")
        specs = source.get("archives", [])
        verified = 0
        for spec in specs:
            marker = read(args.report_root / "integrity_shards" / (spec["archive"] + ".json"))
            if marker.get("status") == "complete" and marker.get("sha256") == spec["sha256"]:
                verified += 1
        catalog = read(args.report_root / "grid_match_report.json")
        cataloged = int(catalog.get("processed_archives", 0))
        quality = read(args.report_root / "jilin_cloud_summary.json")
        quality_lock = read(args.dataset_root / "quality/cloud/jilin1/source.lock.json")
        catalog_path = args.dataset_root / "observations/highres/jilin1/files.parquet"
        catalog_hash = sha256(catalog_path) if catalog_path.exists() else None
        if catalog_hash and catalog_hash not in inventory_cache:
            from xuannv_embedding.data_process.v5_intraband import current_inventory_fingerprint

            inventory_cache[catalog_hash] = current_inventory_fingerprint(
                args.dataset_root, "jilin1"
            )
        inventory_hash = inventory_cache.get(catalog_hash)
        band_ready = band_inventory_finished(
            args.dataset_root,
            read(args.report_root / "intraband_progress_jilin1.json"),
            inventory_hash,
            alignment_version,
        )
        full_quality_status = (
            quality.get("status")
            if quality_lock
            and quality_lock.get("limit") is None
            and catalog_path.exists()
            and quality_lock.get("catalog_sha256") == catalog_hash
            and quality.get("processed_scenes") == quality.get("selected_scenes")
            else None
        )
        partial_ready = partial_catalog_finished(
            args.source_root, args.dataset_root, args.report_root
        )
        action = next_source_action(
            total=len(specs),
            verified=verified,
            cataloged=cataloged,
            quality_status=full_quality_status,
            band_ready=band_ready,
            partial_ready=partial_ready,
            band_running=alive.get("band_jilin1", False),
        )
        if action == "quality" and alive.get("gaofen", False):
            action = None  # The same data inference device must have one owner.
        identity = (action, verified, inventory_hash, alignment_version)
        if action and identity not in attempted_failures:
            command = [
                sys.executable,
                "-m",
                "xuannv_embedding.cli",
                "data",
                "prepare-v5",
                "--stage",
                action,
            ]
            for field in ("source_root", "dataset_root", "report_root", "base_root"):
                command += ["--" + field.replace("_", "-"), str(getattr(args, field))]
            if action == "band-alignment":
                command += [
                    "--sensor-family",
                    "jilin1",
                    "--alignment-version",
                    alignment_version,
                    "--workers",
                    str(args.workers),
                ]
            if action == "quality":
                command += ["--model-dir", str(args.model_dir), "--device-id", str(args.device_id)]
            write_json(
                output,
                {
                    "status": "running",
                    "action": action,
                    "verified_archives": verified,
                    "jobs_alive": alive,
                    "failures": failures,
                    "updated_at": now(),
                    "training_authorized": False,
                },
            )
            with (args.report_root / f"followup_{action}.log").open("ab") as log:
                finished = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if finished.returncode:
                attempted_failures.add(identity)
                failures.append(
                    {
                        "stage": action,
                        "verified_archives": verified,
                        "exit_code": finished.returncode,
                        "failed_at": now(),
                    }
                )
            report_progress(args.source_root, args.dataset_root, args.report_root)
            continue
        report_progress(args.source_root, args.dataset_root, args.report_root)
        result = {
            "status": "running" if any(alive.values()) else "stopped_for_remaining_data_gates",
            "jobs_alive": alive,
            "verified_archives": verified,
            "cataloged_archives": cataloged,
            "band_inventory_finished": band_ready,
            "partial_catalog_finished": partial_ready,
            "alignment_version": alignment_version,
            "failures": failures,
            "training_authorized": False,
            "updated_at": now(),
        }
        write_json(output, result)
        if not any(alive.values()):
            return result
        time.sleep(30)
