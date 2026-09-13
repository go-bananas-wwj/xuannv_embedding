"""Finite follow-through for already started data jobs; no scheduler or training operations."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from xuannv_embedding.data_process.v5_audit import report_progress
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def next_source_action(*, total: int, verified: int, cataloged: int, quality_status: str | None):
    if total > 0 and verified > cataloged:
        return "catalog"
    if total > 0 and verified == total and cataloged == total and quality_status is None:
        return "quality"
    return None


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
    for name in ("download", "gaofen", "target_value", "target_source", "dense_integrity"):
        info = read(args.report_root / f"{name}_worker.json")
        if info:
            pid = int(info["pid"])
            jobs[name] = {"pid": pid, "identity": process_identity(pid)}
    if not jobs:
        raise ValueError("no started data jobs to follow")
    failures = []
    attempted_failures = set()
    output = args.report_root / "followup_progress.json"
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
        full_quality_status = (
            quality.get("status")
            if quality_lock
            and quality_lock.get("limit") is None
            and catalog_path.exists()
            and quality_lock.get("catalog_sha256") == sha256(catalog_path)
            and quality.get("processed_scenes") == quality.get("selected_scenes")
            else None
        )
        action = next_source_action(
            total=len(specs),
            verified=verified,
            cataloged=cataloged,
            quality_status=full_quality_status,
        )
        if action == "quality" and alive.get("gaofen", False):
            action = None  # The same data inference device must have one owner.
        identity = (action, verified)
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
            "failures": failures,
            "training_authorized": False,
            "updated_at": now(),
        }
        write_json(output, result)
        if not any(alive.values()):
            return result
        time.sleep(30)
