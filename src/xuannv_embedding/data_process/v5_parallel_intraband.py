"""Process execution of the unchanged calibrated native-band audit, reusing legacy receipts."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_intraband import (
    _code_fingerprint,
    _inventory,
    _read_row,
    _root,
    calibrate_family,
    inspect_intraband,
    inventory_fingerprint,
)
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

THREAD_LIMITS = {
    name: "1" for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
}


def inspect_job(job):
    row, family, directory, fingerprint = job
    row = SimpleNamespace(**row)
    key = hashlib.sha256(row.observation_id.encode()).hexdigest()
    receipt = Path(directory) / "receipts" / key[:2] / f"{key}.json"
    expected = {
        **fingerprint,
        "file_sha256": row.file_sha256,
        "observation_id": row.observation_id,
        "patch_id": row.patch_id,
        "sensor": row.sensor,
        "split": row.split,
        "year": int(row.year),
    }
    try:
        if sha256(Path(row.path)) != row.file_sha256:
            raise ValueError("source file changed after catalog")
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous["fingerprint"] == expected:
                return previous["result"], True
        frame, gsd, reference = _read_row(row, family)
        result = {
            "observation_id": row.observation_id,
            "patch_id": row.patch_id,
            "sensor": row.sensor,
            "year": int(row.year),
            "split": row.split,
            **inspect_intraband(frame, reference_band=reference, gsd=gsd),
            "finished_at": now(),
        }
        write_json(receipt, {"fingerprint": expected, "result": result})
        return result, False
    except (ValueError, OSError) as exc:
        return {
            "observation_id": row.observation_id,
            "patch_id": row.patch_id,
            "sensor": row.sensor,
            "year": int(row.year),
            "split": row.split,
            "status": "rejected",
            "pairs": [],
            "reason": str(exc),
            "pixel_fusion_authorized": False,
            "finished_at": now(),
        }, False


def run_parallel_intraband(
    dataset_root: Path, report_root: Path, family: str, *, version="v5", workers=8
):
    if not 1 <= workers <= 16:
        raise ValueError("process audit workers must be 1..16")
    root = _root(dataset_root, family, version)
    calibration_path = root / "calibration.json"
    if not calibration_path.is_file():
        raise FileNotFoundError("passed calibration file is required")
    # This revalidates original code, runtime, parameters, frozen samples and actual source bytes.
    calibration = calibrate_family(dataset_root, report_root, family, version=version)
    if calibration["status"] != "passed":
        raise ValueError("passed calibration for this exact algorithm is required")
    rows = _inventory(dataset_root, family)
    if not set(rows.sensor).issubset(calibration["sensors"]):
        raise ValueError("uncalibrated sensor variant present")
    input_hash = inventory_fingerprint(rows)
    atomic_parquet(rows, root / "inputs" / f"{input_hash}.parquet")
    fingerprint = {
        "calibration_sha256": sha256(calibration_path),
        "code_sha256": _code_fingerprint(),
        "version": version,
        "family": family,
    }
    execution = {
        "backend": "processes",
        "workers": workers,
        "start_method": "spawn",
        "native_thread_limits": THREAD_LIMITS,
        "executor_sha256": sha256(Path(__file__)),
    }
    progress = report_root / f"intraband_progress_{family}.json"
    counts = Counter()
    output = []
    original = {key: os.environ.get(key) for key in THREAD_LIMITS}
    os.environ.update(THREAD_LIMITS)
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for start in range(0, len(rows), 128):
                jobs = [
                    (row, family, str(root), fingerprint)
                    for row in rows.iloc[start : start + 128].to_dict("records")
                ]
                for result, cached in pool.map(inspect_job, jobs, chunksize=1):
                    counts[result["status"]] += 1
                    counts["reused"] += int(cached)
                    output.append({**result, "pairs": json.dumps(result["pairs"])})
                write_json(
                    progress,
                    {
                        "status": "running",
                        "processed_observations": len(output),
                        "selected_observations": len(rows),
                        "counts": dict(counts),
                        "execution": execution,
                        "updated_at": now(),
                    },
                )
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    # Never publish completion under a source or algorithm changed while children were running.
    if (
        sha256(calibration_path) != fingerprint["calibration_sha256"]
        or _code_fingerprint() != fingerprint["code_sha256"]
        or sha256(Path(__file__)) != execution["executor_sha256"]
    ):
        raise ValueError("alignment implementation changed during execution")
    atomic_parquet(pd.DataFrame(output), root / "observations.parquet")
    summary = {
        **fingerprint,
        "status": "intraband_audit_finished",
        "input_inventory_sha256": input_hash,
        "processed_observations": len(output),
        "selected_observations": len(rows),
        "counts": dict(counts),
        "output": str(root / "observations.parquet"),
        "execution": execution,
        "pixel_fusion_authorized": False,
        "finished_at": now(),
    }
    write_json(progress, summary)
    return summary
