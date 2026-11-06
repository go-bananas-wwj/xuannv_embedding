"""Calibrated native-band registration audit; independent of absolute georegistration."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import scipy
from scipy.ndimage import shift

from xuannv_embedding.data_process.v5_alignment import PARAMETERS, audit_translation
from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_rasters import BRANCH_BANDS, NativeRaster, read_native
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

CALIBRATION = {
    "shifts_yx_pixels": [[0, 0], [0, 0.5], [0, 1.5], [2, -1]],
    "maximum_error_pixels": 0.35,
    "selected_positions_per_sensor": 8,
    "minimum_successful_positions_per_sensor": 4,
}


def inspect_intraband(frame: NativeRaster, *, reference_band: str, gsd: float) -> dict:
    if reference_band not in frame.band_ids:
        raise ValueError("reference band missing")
    ref = frame.band_ids.index(reference_band)
    pairs = []
    for i, name in enumerate(frame.band_ids):
        if i == ref:
            continue
        result = audit_translation(
            frame.values[ref], frame.values[i], frame.valid[ref] & frame.valid[i], gsd=gsd
        )
        pairs.append({"reference_band": reference_band, "moving_band": name, **result})
    if not pairs:
        raise ValueError("at least two bands are required")
    statuses = {p["status"] for p in pairs}
    status = (
        "over_limit"
        if "over_limit" in statuses
        else "uncertain" if "uncertain" in statuses else "passed"
    )
    return {
        "status": status,
        "pairs": pairs,
        "pixel_fusion_authorized": False,
        "scope": "relative native spectral bands; not alignment to the base reference",
    }


def calibrate_texture(reference: np.ndarray, valid: np.ndarray, *, gsd: float) -> dict:
    cases = []
    for delta in CALIBRATION["shifts_yx_pixels"]:
        moving = shift(reference, delta, order=1, mode="constant", cval=0)
        moved_valid = shift(valid.astype("f4"), delta, order=1, mode="constant", cval=0) >= 1 - 1e-6
        measured = audit_translation(reference, moving, valid & moved_valid, gsd=gsd)
        expected = -np.asarray(delta, dtype="f8")
        error = (
            float(np.linalg.norm(np.asarray(measured["translation_yx_m"]) / gsd - expected))
            if measured["status"] != "uncertain"
            else None
        )
        wanted = (
            "passed"
            if np.linalg.norm(expected) * gsd <= PARAMETERS["maximum_residual_m"]
            else "over_limit"
        )
        cases.append(
            {
                "injected_yx_pixels": delta,
                "expected_yx_pixels": expected.tolist(),
                "expected_status": wanted,
                "error_pixels": error,
                "measurement": measured,
            }
        )
    confident = [c for c in cases if c["error_pixels"] is not None]
    wrong = [
        c
        for c in confident
        if c["error_pixels"] > CALIBRATION["maximum_error_pixels"]
        or c["measurement"]["status"] != c["expected_status"]
    ]
    status = (
        "failed" if wrong else "passed" if len(confident) == len(cases) else "insufficient_texture"
    )
    return {"status": status, "cases": cases}


def _runtime_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "rasterio": rasterio.__version__,
        "gdal": rasterio.__gdal_version__,
    }


def inventory_fingerprint(rows: pd.DataFrame) -> str:
    return hashlib.sha256(rows.to_json(orient="records", double_precision=15).encode()).hexdigest()


def current_inventory_fingerprint(dataset_root: Path, family: str) -> str:
    return inventory_fingerprint(_inventory(dataset_root, family))


def _code_fingerprint() -> dict[str, str]:
    return {
        name: sha256(Path(__file__).with_name(name))
        for name in ["v5_intraband.py", "v5_alignment.py", "v5_rasters.py"]
    }


def _root(dataset_root: Path, family: str, version: str) -> Path:
    if family not in {"jilin1", "gaofen"} or not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("invalid alignment family or version")
    return dataset_root / "quality/alignment/intraband" / family / version


def _inventory(dataset_root: Path, family: str) -> pd.DataFrame:
    if family == "jilin1":
        frame = pd.read_parquet(dataset_root / "observations/highres/jilin1/files.parquet")
        frame = frame.loc[(frame.product_id == "jilin1_ms_5m") & frame.year.isin([2020, 2021])]
    elif family == "gaofen":
        frame = pd.read_parquet(dataset_root / "quality/cloud/gaofen/observation_quality.parquet")
        frame = frame.rename(
            columns={"pair_id": "observation_id", "ms_path": "path", "ms_sha256": "file_sha256"}
        )
    else:
        raise ValueError("unknown native-band family")
    if not frame.year.isin([2020, 2021]).all():
        raise ValueError("native-band inventory contains an unsupported year")
    cols = ["observation_id", "patch_id", "path", "file_sha256", "sensor", "split", "year"]
    frame = frame[cols].sort_values("observation_id").reset_index(drop=True)
    if frame.empty or frame.observation_id.duplicated().any():
        raise ValueError("empty or duplicate native-band inventory")
    owners = pd.read_parquet(dataset_root / "registry/national_62000.parquet").set_index("patch_id")
    if not frame.patch_id.isin(owners.index).all():
        raise ValueError("observation outside registry")
    if not (frame.split.to_numpy() == owners.loc[frame.patch_id, "split"].to_numpy()).all():
        raise ValueError("observation split disagrees with registry")
    return frame


def _read_row(row: Any, family: str) -> tuple[NativeRaster, float, str]:
    if sha256(Path(row.path)) != row.file_sha256:
        raise ValueError("source file changed after catalog")
    if family == "jilin1":
        frame = read_native(Path(row.path), BRANCH_BANDS["jilin1_ms_5m"])
        expected, gsd, ref = (6, 256, 256), 5, "B4"
    else:
        bands = ("blue", "green", "red", "nir")
        # Explicit stored DN contract, identical to the verified GF cloud source path.
        frame = read_native(
            Path(row.path),
            bands,
            contract={"verified": True, "band_ids": bands, "scales": [1] * 4, "offsets": [0] * 4},
        )
        expected, gsd, ref = (4, 160, 160), 8, "green"
    if frame.values.shape != expected or not np.allclose(
        [frame.transform[0], frame.transform[1], frame.transform[3], frame.transform[4]],
        [gsd, 0, 0, -gsd],
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("native spectral grid differs from source contract")
    return frame, gsd, ref


def calibrate_family(
    dataset_root: Path, report_root: Path, family: str, *, version: str = "v1"
) -> dict:
    root = _root(dataset_root, family, version)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "calibration_inputs.parquet"
    if snapshot.exists():
        rows = pd.read_parquet(snapshot)
    else:
        source = _inventory(dataset_root, family)
        candidates = source.loc[source.split == "train"].drop_duplicates(["sensor", "patch_id"])
        rows = candidates.groupby("sensor", sort=True).head(
            CALIBRATION["selected_positions_per_sensor"]
        )
        atomic_parquet(rows, snapshot)
    fingerprint = {
        "code_sha256": _code_fingerprint(),
        "parameters": PARAMETERS,
        "calibration": CALIBRATION,
        "runtime_versions": _runtime_versions(),
        "inputs_sha256": sha256(snapshot),
        "family": family,
        "version": version,
    }
    path = root / "calibration.json"
    if path.exists() and json.loads(path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("calibration contract changed; select a new alignment version")
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["status"] == "passed":
            for row in rows.itertuples():
                if sha256(Path(row.path)) != row.file_sha256:
                    raise ValueError("calibration source pixels changed")
            result = {**previous, "reused": True, "verified_at": now()}
            write_json(report_root / f"intraband_calibration_{family}.json", result)
            return result
    results = []
    for row in rows.itertuples():
        frame, gsd, ref = _read_row(row, family)
        i = frame.band_ids.index(ref)
        result = calibrate_texture(frame.values[i], frame.valid[i], gsd=gsd)
        results.append(
            {
                "observation_id": row.observation_id,
                "sensor": row.sensor,
                "patch_id": row.patch_id,
                "file_sha256": row.file_sha256,
                **result,
            }
        )
    sensors = {
        sensor: Counter(r["status"] for r in results if r["sensor"] == sensor)
        for sensor in sorted(rows.sensor.unique())
    }
    passed = bool(sensors) and all(
        c["passed"] >= CALIBRATION["minimum_successful_positions_per_sensor"] and c["failed"] == 0
        for c in sensors.values()
    )
    result = {
        "fingerprint": fingerprint,
        "status": "passed" if passed else "failed",
        "sensors": {k: dict(v) for k, v in sensors.items()},
        "results": results,
        "scope": "same-band known artificial shifts on train textures; not cross-spectral truth",
        "pixel_fusion_authorized": False,
        "finished_at": now(),
    }
    write_json(path, result)
    write_json(report_root / f"intraband_calibration_{family}.json", result)
    return result


def run_intraband(
    dataset_root: Path, report_root: Path, family: str, *, version: str = "v1", workers: int = 2
) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError("native-band audit workers must be 1..4")
    root = _root(dataset_root, family, version)
    calibration_path = root / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    if (
        calibration["status"] != "passed"
        or calibration["fingerprint"]["code_sha256"] != _code_fingerprint()
    ):
        raise ValueError("passed calibration for this exact algorithm is required")
    if calibration["fingerprint"].get("runtime_versions") != _runtime_versions():
        raise ValueError("calibration runtime versions changed")
    if calibration["fingerprint"]["parameters"] != PARAMETERS:
        raise ValueError("alignment parameters changed after calibration")
    calibration_inputs = root / "calibration_inputs.parquet"
    if sha256(calibration_inputs) != calibration["fingerprint"]["inputs_sha256"]:
        raise ValueError("calibration source inventory changed")
    for sample in pd.read_parquet(calibration_inputs).itertuples():
        if sha256(Path(sample.path)) != sample.file_sha256:
            raise ValueError("calibration source pixels changed")
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
    progress = report_root / f"intraband_progress_{family}.json"
    counts = Counter()
    output = []

    def inspect(row):
        key = hashlib.sha256(row.observation_id.encode()).hexdigest()
        receipt = root / "receipts" / key[:2] / f"{key}.json"
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
            # Rehash actual bytes even on a cached run: metadata alone is insufficient.
            if sha256(Path(row.path)) != row.file_sha256:
                raise ValueError("source file changed after catalog")
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if record["fingerprint"] == expected:
                    return record["result"], True
            frame, gsd, ref = _read_row(row, family)
            result = {
                "observation_id": row.observation_id,
                "patch_id": row.patch_id,
                "sensor": row.sensor,
                "year": int(row.year),
                "split": row.split,
                **inspect_intraband(frame, reference_band=ref, gsd=gsd),
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

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(rows), 128):
            for result, cached in pool.map(inspect, rows.iloc[start : start + 128].itertuples()):
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
                    "updated_at": now(),
                },
            )
    atomic_parquet(pd.DataFrame(output), root / "observations.parquet")
    summary = {
        **fingerprint,
        "status": "intraband_audit_finished",
        "input_inventory_sha256": input_hash,
        "processed_observations": len(output),
        "selected_observations": len(rows),
        "counts": dict(counts),
        "output": str(root / "observations.parquet"),
        "pixel_fusion_authorized": False,
        "finished_at": now(),
    }
    write_json(progress, summary)
    return summary
