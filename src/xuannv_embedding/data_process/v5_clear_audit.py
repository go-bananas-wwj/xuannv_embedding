"""Audit every native multispectral observation with frozen QA and calibrated matching."""

from __future__ import annotations

import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from xuannv_embedding.data_process.v5_clear_intraband import NativeQualityReader, apply_clear_mask
from xuannv_embedding.data_process.v5_clear_training import calibrate_training
from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_intraband import _code_fingerprint, inspect_intraband
from xuannv_embedding.data_process.v5_jilin_quality import _array_digest, _digest
from xuannv_embedding.data_process.v5_parallel_intraband import THREAD_LIMITS
from xuannv_embedding.data_process.v5_partial_bands import read_jilin_branch
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

IDENTITY = ["observation_id", "patch_id", "sensor", "year", "split", "path", "file_sha256"]


class AuditReader(NativeQualityReader):
    """The complete QA reader plus explicitly missing native Jilin spectral channels."""

    def __init__(self, dataset_root, family, quality_root):
        super().__init__(dataset_root, family, quality_root)
        if family == "jilin1":
            paths = [p for p in self.files if p.endswith("/files_with_partial_bands.parquet")]
            if len(paths) != 1:
                raise ValueError("one frozen Jilin source catalog required")
            self.catalog = pd.read_parquet(paths[0]).set_index("observation_id")
            if not self.catalog.index.is_unique:
                raise ValueError("duplicate Jilin catalog observations")

    def read(self, row):
        if self.family == "gaofen":
            return super().read(row)
        source = self.catalog.loc[row.observation_id].to_dict()
        record = self.table.loc[row.observation_id]
        for field in IDENTITY[1:]:
            if source[field] != getattr(row, field):
                raise ValueError("frozen Jilin source identity changed")
        for field in ["patch_id", "sensor", "year", "split", "file_sha256"]:
            if record[field] != source[field]:
                raise ValueError("Jilin source and QA observation identity disagree")
        source["observation_id"] = row.observation_id
        if source["product_id"] != "jilin1_ms_5m":
            raise ValueError("only calibrated native 5m multispectral branches are supported")
        frame = read_jilin_branch(source)
        if frame.values.shape != (6, 256, 256) or not np.allclose(
            np.asarray(frame.transform)[[0, 1, 3, 4]], [5, 0, 0, -5], rtol=0, atol=1e-9
        ):
            raise ValueError("Jilin native spectral grid differs from calibrated contract")
        scene = record.scene_group_id
        receipt_path = self.root / "receipts" / f"{scene}.json"
        if sha256(receipt_path) != self.receipts[scene]:
            raise ValueError("Jilin QA receipt changed")
        receipt = json.loads(receipt_path.read_text())
        prefix = row.observation_id + "/"
        arrays = {}
        for name in ["data_valid", "valid", "qa_clear"]:
            array = np.asarray(self.masks[prefix + name])
            if _array_digest(array) != receipt["mask_arrays"][prefix + name]:
                raise ValueError("Jilin QA mask changed")
            arrays[name] = array
        mask = arrays["valid"]
        if (
            not np.array_equal(arrays["data_valid"], frame.valid)
            or not np.array_equal(mask, frame.valid & arrays["qa_clear"][None])
            or not np.array_equal(mask.sum(axis=(1, 2)), record.valid_pixels_by_band)
            or (record.quality_status == "qa_missing" and mask.any())
        ):
            raise ValueError("Jilin native validity differs from frozen QA")
        clear = apply_clear_mask(frame, mask, tuple(record.band_ids))
        proof = {
            "observation_id": row.observation_id,
            "file_sha256": row.file_sha256,
            "quality_mask_sha256": _array_digest(mask),
            "band_ids": list(frame.band_ids),
            "crs": frame.crs,
            "transform": list(frame.transform),
            "quality_status": record.quality_status,
            "clear_fraction_by_band": clear.valid.mean(axis=(1, 2)).tolist(),
            "missing_bands": [b for b in frame.band_ids if b not in source["band_ids"]],
        }
        return frame, clear, 5, "B4", proof


def audit_inventory(reader):
    rows = reader.table.reset_index()
    if reader.family == "gaofen":
        rows = rows.rename(
            columns={"pair_id": "observation_id", "ms_path": "path", "ms_sha256": "file_sha256"}
        )
    else:
        # No clear-fraction, availability, split, or complete-band filter belongs here.
        rows = rows.loc[rows.product_id.eq("jilin1_ms_5m")].merge(
            reader.catalog.reset_index()[["observation_id", "path", "file_sha256"]],
            on=["observation_id", "file_sha256"],
            how="left",
            validate="one_to_one",
        )
    if rows.empty or rows.observation_id.duplicated().any() or rows[IDENTITY].isna().any().any():
        raise ValueError("invalid or missing native QA observations")
    return rows[IDENTITY].sort_values("observation_id").reset_index(drop=True)


def inspect_clear(reader, row, *, loaded=None):
    _, clear, gsd, reference, proof = reader.read(row) if loaded is None else loaded
    result = inspect_intraband(clear, reference_band=reference, gsd=gsd)
    missing = proof.get("missing_bands", [])
    for pair in result["pairs"]:
        if reference in missing:
            pair["reason"] = "missing_reference_band"
        elif pair["moving_band"] in missing:
            pair["reason"] = "missing_spectral_band"
        elif not (
            clear.valid[clear.band_ids.index(reference)]
            & clear.valid[clear.band_ids.index(pair["moving_band"])]
        ).any():
            pair["reason"] = "no_joint_clear_pixels"
    return {
        **{k: getattr(row, k) for k in IDENTITY[:5]},
        **result,
        "missing_bands": missing,
        "quality_status": proof["quality_status"],
        "clear_fraction_by_band": proof["clear_fraction_by_band"],
        "quality_mask_sha256": proof["quality_mask_sha256"],
        "training_authorized": False,
        "reason": "",
    }


def audit_one(reader, row, directory, fingerprint):
    loaded = reader.read(row)  # Always recheck actual source and masks, even on cache hits.
    expected = {
        "algorithm": fingerprint,
        "identity": {k: getattr(row, k) for k in IDENTITY},
        "proof": loaded[-1],
    }
    key = _digest(row.observation_id)
    receipt = directory / "receipts" / key[:2] / f"{key}.json"
    if receipt.exists():
        cached = json.loads(receipt.read_text())
        if cached.get("result_sha256") != _digest(cached["result"]):
            raise ValueError("clear audit receipt result changed")
        if cached["fingerprint"] == expected:
            return cached["result"], True
    result = inspect_clear(reader, row, loaded=loaded)
    if reader.read(row)[-1] != loaded[-1]:
        raise ValueError("source or QA changed during native audit")
    write_json(
        receipt, {"fingerprint": expected, "result": result, "result_sha256": _digest(result)}
    )
    return result, False


def verify_calibration(dataset_root, report_root, family, calibration_root):
    locked = json.loads((calibration_root / "inputs.lock.json").read_text())
    output = calibration_root / "calibration.json"
    seal = json.loads((calibration_root / "output.lock.json").read_text())
    if seal != {
        "calibration_sha256": sha256(output),
        "inputs_lock_sha256": sha256(calibration_root / "inputs.lock.json"),
    }:
        raise ValueError("clear calibration publication seal changed")
    result = json.loads(output.read_text())
    fingerprint = locked["fingerprint"]
    if result["status"] != "passed" or fingerprint["family"] != family:
        raise ValueError("passed same-family clear calibration required")
    suffix = "/source.lock.json" if family == "gaofen" else "/quality.lock.json"
    paths = [Path(p).parent for p in fingerprint["quality_inputs_sha256"] if p.endswith(suffix)]
    if len(paths) != 1:
        raise ValueError("calibration QA source is ambiguous")
    replay = calibrate_training(
        dataset_root, report_root, family, paths[0], version=fingerprint["baseline_version"]
    )
    if (
        Path(replay["output"]).resolve() != output.resolve()
        or replay["sha256"] != seal["calibration_sha256"]
    ):
        raise ValueError("clear calibration differs from verified replay")
    original_reader = NativeQualityReader(dataset_root, family, paths[0])
    return result, original_reader.configuration, seal


_READER = None


def initialize_worker(dataset_root, family, quality_root, files):
    global _READER
    _READER = AuditReader(Path(dataset_root), family, Path(quality_root))
    if _READER.files != files:
        raise ValueError("QA inputs changed before worker initialization")


def inspect_job(job):
    row, directory, fingerprint = job
    row = SimpleNamespace(**row)
    try:
        return audit_one(_READER, row, Path(directory), fingerprint)
    except (OSError, ValueError, KeyError) as exc:
        return {
            **{k: getattr(row, k) for k in IDENTITY[:5]},
            "status": "rejected",
            "pairs": [],
            "missing_bands": [],
            "reason": str(exc),
            "quality_status": "unverified",
            "clear_fraction_by_band": [],
            "quality_mask_sha256": "",
            "scope": "relative native spectral bands; not alignment to the base reference",
            "pixel_fusion_authorized": False,
            "training_authorized": False,
        }, False


def run_clear_audit(
    dataset_root, report_root, family, quality_root, calibration_root, *, workers=8, limit=None
):
    if not 1 <= workers <= 16 or (limit is not None and limit <= 0):
        raise ValueError("workers must be 1..16 and pilot limit must be positive")
    calibration, configuration, seal = verify_calibration(
        dataset_root, report_root, family, calibration_root
    )
    reader = AuditReader(dataset_root, family, quality_root)
    if reader.configuration != configuration:
        raise ValueError("current QA policy differs from calibrated QA policy")
    rows = audit_inventory(reader)
    total = len(rows)
    if not set(rows.sensor).issubset(calibration["sensors"]):
        raise ValueError("uncalibrated sensor variant present")
    if limit is not None:
        rows = rows.head(limit)
    code = {
        **_code_fingerprint(),
        **{
            name: sha256(Path(__file__).with_name(name))
            for name in [
                "v5_clear_audit.py",
                "v5_clear_training.py",
                "v5_clear_intraband.py",
                "v5_partial_bands.py",
                "v5_jilin_quality.py",
            ]
        },
    }
    fingerprint = {
        "family": family,
        "calibration": seal,
        "configuration": configuration,
        "code_sha256": code,
    }
    root = dataset_root / "quality/alignment/clear_audit" / family / _digest(fingerprint)[:20]
    snapshot = {
        "quality_inputs_sha256": reader.files,
        "rows_sha256": _digest(rows.to_dict("records")),
        "limit": limit,
        "available_native_observations": total,
    }
    output_root = root / "catalogs" / _digest(snapshot)[:20]
    inputs = output_root / "inputs.parquet"
    if inputs.exists():
        if _digest(pd.read_parquet(inputs).to_dict("records")) != snapshot["rows_sha256"]:
            raise ValueError("frozen native audit inputs changed")
    else:
        atomic_parquet(rows, inputs)
    lock = {"fingerprint": fingerprint, "snapshot": snapshot, "inputs_sha256": sha256(inputs)}
    input_lock = output_root / "inputs.lock.json"
    if input_lock.exists():
        if json.loads(input_lock.read_text()) != lock:
            raise ValueError("native audit input lock changed")
    else:
        write_json(input_lock, lock)
    observations = output_root / "observations.parquet"
    output_lock = output_root / "output.lock.json"
    if output_lock.exists():
        prior = json.loads(output_lock.read_text())
        if prior != {
            "inputs_lock_sha256": sha256(input_lock),
            "observations_sha256": sha256(observations),
        }:
            raise ValueError("native audit output seal changed")
    progress = (
        report_root
        / f"clear_band_audit_{family}_{'full' if limit is None else 'pilot_' + str(limit)}.json"
    )
    counts, results = Counter(), []
    original = {k: os.environ.get(k) for k in THREAD_LIMITS}
    os.environ.update(THREAD_LIMITS)
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(str(dataset_root), family, str(quality_root), reader.files),
        ) as pool:
            for start in range(0, len(rows), 128):
                jobs = [
                    (row, str(root), fingerprint)
                    for row in rows.iloc[start : start + 128].to_dict("records")
                ]
                for result, cached in pool.map(inspect_job, jobs, chunksize=1):
                    counts[result["status"]] += 1
                    counts["reused"] += int(cached)
                    counts["missing_band_observations"] += bool(result["missing_bands"])
                    results.append(result)
                write_json(
                    progress,
                    {
                        "execution_status": "running",
                        "processed": len(results),
                        "selected": len(rows),
                        "counts": dict(counts),
                        "output": str(output_root),
                        "updated_at": now(),
                        "training_authorized": False,
                    },
                )
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    reader.verify_unchanged()
    if (
        any(sha256(Path(__file__).with_name(n)) != h for n, h in code.items())
        or json.loads(input_lock.read_text()) != lock
        or sha256(inputs) != lock["inputs_sha256"]
    ):
        raise ValueError("native audit implementation or inputs changed")
    # Repeat the sealed calibration/source check before publishing any completed snapshot.
    if verify_calibration(dataset_root, report_root, family, calibration_root)[2] != seal:
        raise ValueError("native audit calibration changed during execution")
    frame = pd.DataFrame(
        [
            {
                **r,
                "pairs": json.dumps(r["pairs"], sort_keys=True),
                "missing_bands": json.dumps(r["missing_bands"]),
                "clear_fraction_by_band": json.dumps(r["clear_fraction_by_band"]),
            }
            for r in results
        ]
    )
    if observations.exists():
        if _digest(pd.read_parquet(observations).to_dict("records")) != _digest(
            frame.to_dict("records")
        ):
            raise ValueError("published native audit differs from verified replay")
    else:
        atomic_parquet(frame, observations)
    publication = {
        "inputs_lock_sha256": sha256(input_lock),
        "observations_sha256": sha256(observations),
    }
    if not output_lock.exists():
        write_json(output_lock, publication)
    result = {
        "execution_status": "finished",
        "scope": "frozen_native_multispectral_QA_catalog" if limit is None else "pilot",
        "processed": len(results),
        "selected": len(rows),
        "available_native_observations": total,
        "counts": dict(counts),
        "output": str(output_root),
        "publication": publication,
        "workers": workers,
        "pixel_fusion_authorized": False,
        "training_authorized": False,
        "finished_at": now(),
    }
    write_json(progress, result)
    return result
