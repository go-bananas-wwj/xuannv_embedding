"""One bounded prefetch pair overlaps validated package extraction; no training operations."""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event, RLock

import pandas as pd

from xuannv_embedding.data_process.v5_sources import ArchiveSpec, now, sha256, write_json


def acquire_overlapped(args, source, download, extract, validate):
    specs = [ArchiveSpec(**row) for row in source["archives"]][: args.limit]
    if not specs:
        raise ValueError("no pinned packages selected")
    pairs = [specs[i : i + 2] for i in range(0, len(specs), 2)]
    status_path = args.source_root / "manifests/download_status.parquet"
    existing = pd.read_parquet(status_path).to_dict("records") if status_path.exists() else []
    statuses = {row["archive"]: row for row in existing}
    progress_path = args.report_root / "acquisition_progress.json"
    mutex, stopped = RLock(), Event()
    state = {
        "status": "running",
        "mode": "single_prefetch_pair_after_validated_pilot",
        "revision": source["revision"],
        "manifest_sha256": source["manifest_sha256"],
        "selected_archives": len(specs),
        "total_archives": len(source["archives"]),
        "maximum_download_packages": 2,
        "maximum_prefetch_pairs": 1,
        "download_route": args.download_route,
        "downloaded_archives": 0,
        "validated_archives": 0,
        "downloading_pair": [],
        "processing_pair": [],
        "events": [],
        "failures": [],
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ["v5_acquisition.py", "v5_cli.py", "v5_transfer.py", "v5_sources.py"]
        },
        "training_authorized": False,
        "started_at": now(),
    }

    def event(kind, pair, **updates):
        with mutex:
            state.update(updates)
            state["events"].append(
                {
                    "event": kind,
                    "archives": [s.archive for s in pair],
                    "at": now(),
                    "monotonic": time.monotonic(),
                }
            )
            state["updated_at"] = now()
            write_json(progress_path, state)

    def publish_status(spec, result):
        with mutex:
            statuses[spec.archive] = result
            status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = status_path.with_name(status_path.name + ".tmp")
            pd.DataFrame(statuses.values()).to_parquet(temporary, index=False)
            temporary.replace(status_path)

    def failed(stage, pair, exc):
        stopped.set()
        with mutex:
            state["failures"].append(
                {
                    "stage": stage,
                    "archives": [s.archive for s in pair],
                    "error_type": type(exc).__name__,
                }
            )
            event("failed", pair, status="failed")

    def download_pair(pair):
        if stopped.is_set():
            raise RuntimeError("acquisition stopped after a data stage failure")
        if shutil.disk_usage(args.source_root).free < 30 * 1024**3:
            raise RuntimeError("less than 30 GiB free before next download pair")
        event("download_started", pair, downloading_pair=[s.archive for s in pair])
        errors = []
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = {}
                for spec in pair:
                    publish_status(
                        spec,
                        {
                            **spec.__dict__,
                            "status": "running",
                            "actual_bytes": 0,
                            "retries": 0,
                            "started_at": now(),
                            "finished_at": None,
                        },
                    )
                    jobs[
                        pool.submit(
                            download,
                            spec,
                            args.source_root / "packages",
                            source["revision"],
                            route=args.download_route,
                        )
                    ] = spec
                for future in as_completed(jobs):
                    spec = jobs[future]
                    try:
                        result = future.result()
                        if result["status"] != "complete" or result["actual_bytes"] != spec.bytes:
                            raise ValueError("download did not publish a complete pinned archive")
                    except Exception as exc:
                        errors.append(exc)
                        failed("download", [spec], exc)
                        result = {
                            **spec.__dict__,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "finished_at": now(),
                        }
                    else:
                        with mutex:
                            state["downloaded_archives"] += 1
                    publish_status(spec, result)
                    print(
                        json.dumps({"archive": spec.archive, "status": result["status"]}),
                        flush=True,
                    )
            if errors:
                raise RuntimeError("archive download failed; inspect download_status") from None
        finally:
            event("download_finished", pair, downloading_pair=[])

    def process_pair(pair):
        event("processing_started", pair, processing_pair=[s.archive for s in pair])
        try:
            for spec in pair:
                if stopped.is_set():
                    raise RuntimeError("acquisition stopped after a data stage failure")
                result = extract(
                    args.source_root / "packages" / spec.archive,
                    args.source_root / "extracted",
                    spec,
                )
                validate(args, spec)
                with mutex:
                    state["validated_archives"] += 1
                event("package_validated", [spec])
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
        except Exception as exc:
            failed("extract_and_validate", pair, exc)
            raise
        finally:
            event("processing_finished", pair, processing_pair=[])

    previous = args.report_root / "archive_integrity.json"
    if previous.exists():
        digest = sha256(previous)
        history = args.report_root / "acquisition_history" / f"{digest}.json"
        history.parent.mkdir(parents=True, exist_ok=True)
        if history.exists():
            if sha256(history) != digest:
                raise ValueError("prior acquisition history changed")
        else:
            shutil.copyfile(previous, history)
        state["previous_integrity_sha256"] = digest
        previous.unlink()
    event("started", [])
    try:
        # The first one/two packages must pass extraction and decoding before later downloads.
        download_pair(pairs[0])
        process_pair(pairs[0])
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            pending = prefetch.submit(download_pair, pairs[1]) if len(pairs) > 1 else None
            for i in range(1, len(pairs)):
                pending.result()
                pending = (
                    prefetch.submit(download_pair, pairs[i + 1]) if i + 1 < len(pairs) else None
                )
                process_pair(pairs[i])
    except Exception as exc:
        if not stopped.is_set():
            failed("acquisition", [], exc)
        raise
    if state["downloaded_archives"] != len(specs) or state["validated_archives"] != len(specs):
        raise RuntimeError("acquisition did not reconcile every selected archive")
    write_json(
        args.report_root / "archive_integrity.json",
        {
            "selected_archives": len(specs),
            "total_archives": len(source["archives"]),
            "scope": "selected_archives",
            "finished_at": now(),
            "fully_acquired": len(specs) == len(source["archives"]),
        },
    )
    event("complete", [], status="complete", finished_at=now())
    return state
