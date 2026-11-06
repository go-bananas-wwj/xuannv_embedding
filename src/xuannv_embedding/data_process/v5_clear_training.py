"""Freeze year-balanced clear training samples before evaluating native registration."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from xuannv_embedding.data_process.v5_clear_intraband import NativeQualityReader
from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_intraband import (
    CALIBRATION,
    _code_fingerprint,
    _root,
    _runtime_versions,
    calibrate_family,
    calibrate_texture,
    inspect_intraband,
)
from xuannv_embedding.data_process.v5_jilin_quality import _digest
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

SAMPLING = {
    "minimum_reference_clear_fraction": 0.95,
    "split": "train",
    "years": [2020, 2021],
    "positions_per_sensor_year": 4,
    "distinct_patches_per_sensor": True,
    "seed": "clear-native-calibration-v1",
    "uses_measured_alignment": False,
}
COLUMNS = [
    "observation_id",
    "patch_id",
    "sensor",
    "year",
    "split",
    "path",
    "file_sha256",
    "reference_clear_fraction",
]


def select_training_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or rows.observation_id.duplicated().any():
        raise ValueError("empty or duplicate clear training observations")
    if (
        not np.isfinite(rows.reference_clear_fraction).all()
        or not rows.reference_clear_fraction.between(0, 1).all()
    ):
        raise ValueError("invalid reference quality fraction")
    eligible = rows.loc[
        rows.split.eq(SAMPLING["split"])
        & rows.year.isin(SAMPLING["years"])
        & rows.reference_clear_fraction.ge(SAMPLING["minimum_reference_clear_fraction"]),
        COLUMNS,
    ].copy()
    eligible["selection_key"] = eligible.observation_id.map(
        lambda identity: hashlib.sha256((SAMPLING["seed"] + "|" + identity).encode()).hexdigest()
    )
    selected = []
    for sensor in sorted(rows.sensor.unique()):
        used = set()
        for year in SAMPLING["years"]:
            candidates = eligible.loc[(eligible.sensor == sensor) & (eligible.year == year)]
            count = 0
            for _, row in candidates.sort_values(["selection_key", "observation_id"]).iterrows():
                if row.patch_id in used:
                    continue
                selected.append(row)
                used.add(row.patch_id)
                count += 1
                if count == SAMPLING["positions_per_sensor_year"]:
                    break
            if count != SAMPLING["positions_per_sensor_year"]:
                raise ValueError(f"four distinct training positions required for {sensor}/{year}")
    return pd.DataFrame(selected)[COLUMNS + ["selection_key"]].reset_index(drop=True)


def training_inventory(reader: NativeQualityReader) -> pd.DataFrame:
    rows = reader.table.reset_index()
    if reader.family == "gaofen":
        rows = rows.rename(
            columns={"pair_id": "observation_id", "ms_path": "path", "ms_sha256": "file_sha256"}
        )
        rows["reference_clear_fraction"] = rows.clear_fraction
    else:
        rows = rows.loc[(rows.product_id == "jilin1_ms_5m") & rows.selected_bands_complete].copy()
        expected = tuple(f"B{i}" for i in range(1, 7))
        if not rows.band_ids.map(lambda bands: tuple(bands) == expected).all():
            raise ValueError("clear training requires canonical complete Jilin native bands")
        rows["reference_clear_fraction"] = rows.clear_fraction_by_band.map(lambda values: values[3])
        catalog_paths = [p for p in reader.files if p.endswith("/files_with_partial_bands.parquet")]
        if len(catalog_paths) != 1:
            raise ValueError("one frozen Jilin source catalog is required")
        catalog = pd.read_parquet(catalog_paths[0])
        rows = rows.merge(
            catalog[["observation_id", "path", "file_sha256"]],
            on=["observation_id", "file_sha256"],
            how="left",
            validate="one_to_one",
        )
        if rows.path.isna().any():
            raise ValueError("Jilin QA source is missing from its frozen catalog")
    return rows[COLUMNS]


def calibrate_training(
    dataset_root: Path, report_root: Path, family: str, quality_root: Path, *, version="v5"
) -> dict:
    baseline = calibrate_family(dataset_root, report_root, family, version=version)
    if baseline["status"] != "passed":
        raise ValueError("passed original native calibration is required")
    baseline_path = _root(dataset_root, family, version) / "calibration.json"
    baseline_hash = sha256(baseline_path)
    reader = NativeQualityReader(dataset_root, family, quality_root)
    inventory = training_inventory(reader)
    rows = select_training_rows(inventory)
    if set(rows.sensor) != set(baseline["sensors"]):
        raise ValueError("clear training sensor coverage differs from original calibration")
    frames, proofs = [], []
    for row in rows.itertuples():
        raw, clear, gsd, reference, proof = reader.read(row)
        fraction = proof["clear_fraction_by_band"][clear.band_ids.index(reference)]
        if fraction < SAMPLING["minimum_reference_clear_fraction"] or not np.isclose(
            fraction, row.reference_clear_fraction, rtol=0, atol=1e-12
        ):
            raise ValueError("actual reference QA differs from frozen selection quality")
        frames.append((raw, clear, gsd, reference))
        proofs.append(proof)
    code = {
        **_code_fingerprint(),
        **{
            name: sha256(Path(__file__).with_name(name))
            for name in [
                "v5_clear_training.py",
                "v5_clear_intraband.py",
                "v5_jilin_quality.py",
            ]
        },
    }
    fingerprint = {
        "family": family,
        "sampling": SAMPLING,
        "calibration": CALIBRATION,
        "baseline_version": version,
        "baseline_calibration_sha256": baseline_hash,
        "code_sha256": code,
        "runtime": {**_runtime_versions(), "pandas": pd.__version__, "zarr": zarr.__version__},
        "quality_inputs_sha256": reader.files,
        "selected_rows_sha256": _digest(rows.to_dict("records")),
        "observations": proofs,
    }
    root = dataset_root / "quality/alignment/clear_training" / family / _digest(fingerprint)[:20]
    input_path, lock_path = root / "inputs.parquet", root / "inputs.lock.json"
    if input_path.exists():
        if (
            _digest(pd.read_parquet(input_path).to_dict("records"))
            != fingerprint["selected_rows_sha256"]
        ):
            raise ValueError("frozen clear training inputs changed")
    else:
        atomic_parquet(rows, input_path)
    input_lock = {"fingerprint": fingerprint, "inputs_sha256": sha256(input_path)}
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != input_lock:
            raise ValueError("frozen clear training input lock changed")
    else:
        write_json(lock_path, input_lock)
    locked_inputs_hash = sha256(lock_path)
    output = root / "calibration.json"
    publication_path = root / "output.lock.json"
    progress_path = report_root / f"clear_training_calibration_{family}_progress.json"
    publication = json.loads(publication_path.read_text()) if publication_path.exists() else None
    if publication is not None and (
        not output.exists()
        or publication.get("calibration_sha256") != sha256(output)
        or publication.get("inputs_lock_sha256") != locked_inputs_hash
    ):
        raise ValueError("published clear training output changed")
    reused = publication is not None
    if reused:
        locked = json.loads(output.read_text())
        if locked["inputs_lock_sha256"] != locked_inputs_hash:
            raise ValueError("published clear training calibration inputs changed")
    else:
        results = []
        for row, frame, proof in zip(rows.itertuples(), frames, proofs, strict=True):
            raw, clear, gsd, reference = frame
            ref = clear.band_ids.index(reference)
            results.append(
                {
                    **proof,
                    "patch_id": row.patch_id,
                    "sensor": row.sensor,
                    "year": int(row.year),
                    "split": row.split,
                    "known_shift_calibration": calibrate_texture(
                        clear.values[ref], clear.valid[ref], gsd=gsd
                    ),
                    "raw_alignment": inspect_intraband(raw, reference_band=reference, gsd=gsd),
                    "clear_alignment": inspect_intraband(clear, reference_band=reference, gsd=gsd),
                }
            )
            write_json(
                progress_path,
                {
                    "execution_status": "running",
                    "processed": len(results),
                    "selected": len(rows),
                    "at": now(),
                    "training_authorized": False,
                },
            )
        sensors = {
            sensor: dict(
                Counter(
                    r["known_shift_calibration"]["status"] for r in results if r["sensor"] == sensor
                )
            )
            for sensor in sorted(rows.sensor.unique())
        }
        failed = any(count.get("failed", 0) for count in sensors.values())
        enough = all(
            count.get("passed", 0) >= CALIBRATION["minimum_successful_positions_per_sensor"]
            for count in sensors.values()
        )
        locked = {
            "inputs_lock_sha256": locked_inputs_hash,
            "status": (
                "failed" if failed else "passed" if enough else "insufficient_clear_calibration"
            ),
            "sensors": sensors,
            "results": results,
            "scope": (
                "known synthetic shifts on QA-selected training textures; "
                "not absolute or cross-spectral truth"
            ),
            "pixel_fusion_authorized": False,
            "training_authorized": False,
        }
    # Recheck selected source pixels and actual masks after matching, including on cache hits.
    for row, expected in zip(rows.itertuples(), proofs, strict=True):
        _, _, _, _, actual = reader.read(row)
        if actual != expected:
            raise ValueError("selected clear training source or QA changed during calibration")
    reader.verify_unchanged()
    if (
        sha256(baseline_path) != baseline_hash
        or any(sha256(Path(__file__).with_name(name)) != value for name, value in code.items())
        or sha256(input_path) != input_lock["inputs_sha256"]
        or sha256(lock_path) != locked_inputs_hash
    ):
        raise ValueError("clear training code or frozen inputs changed during calibration")
    if not reused:
        if output.exists():
            if json.loads(output.read_text()) != locked:
                raise ValueError("unpublished clear training output differs from verified replay")
        else:
            write_json(output, locked)
        publication = {
            "calibration_sha256": sha256(output),
            "inputs_lock_sha256": locked_inputs_hash,
        }
        write_json(publication_path, publication)
    elif (
        sha256(output) != publication["calibration_sha256"]
        or json.loads(publication_path.read_text()) != publication
    ):
        raise ValueError("published clear training output changed during verification")
    result = {
        "status": locked["status"],
        "execution_status": "finished",
        "processed": len(rows),
        "sensors": locked["sensors"],
        "output": str(output),
        "sha256": sha256(output),
        "inputs_lock_sha256": locked_inputs_hash,
        "reused": reused,
        "finished_at": now(),
        "pixel_fusion_authorized": False,
        "training_authorized": False,
    }
    write_json(progress_path, result)
    write_json(report_root / f"clear_training_calibration_{family}.json", result)
    return result
