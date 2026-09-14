"""Calibrate masked structural matching on independently frozen training textures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import gaussian_filter, shift

from xuannv_embedding.data_process.v5_adaptive_alignment import LAYOUT
from xuannv_embedding.data_process.v5_alignment import audit_translation
from xuannv_embedding.data_process.v5_cfog_alignment import PARAMETERS, audit_cfog
from xuannv_embedding.data_process.v5_clear_intraband import NativeQualityReader
from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_intraband import CALIBRATION, _runtime_versions
from xuannv_embedding.data_process.v5_jilin_quality import _digest
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

EXPERIMENT = {
    "source_selection": "unchanged original clear-training calibration inputs",
    "masks": ["original_QA", "original_QA_intersect_center_116px"],
    "negative_controls": ["constant", "periodic_8px", "unrelated_seed_9041"],
    "known_shifts": CALIBRATION,
    "scope": "synthetic shifts and controls on frozen real textures, not cross-spectral truth",
    "descriptor_source": "https://arxiv.org/abs/1808.06194v8",
    "project_adaptations": "finite mask support, channel-centered NCC, independent masks",
    "threshold_policy": "0.80 NCC and 0.01 peak margin fixed before real calibration",
}


def evaluate_texture(reference, valid, *, gsd):
    cases = []
    center = np.zeros_like(valid)
    y, x = [(n - 116) // 2 for n in valid.shape]
    center[y : y + 116, x : x + 116] = True
    for mask_name, mask in [
        ("original_QA", valid),
        ("original_QA_intersect_center_116px", valid & center),
    ]:
        for delta in CALIBRATION["shifts_yx_pixels"]:
            moving = shift(reference, delta, order=1, mode="constant", cval=0)
            moved = shift(mask.astype(float), delta, order=1, mode="constant", cval=0) >= 1 - 1e-6
            baseline = audit_translation(reference, moving, mask & moved, gsd=gsd)
            candidate = audit_cfog(reference, moving, mask, moved, gsd=gsd)
            expected = -np.asarray(delta, dtype="f8")
            wanted = (
                "passed"
                if np.linalg.norm(expected) * gsd <= PARAMETERS["maximum_residual_m"]
                else "over_limit"
            )
            errors = {
                name: (
                    float(np.linalg.norm(np.array(result["translation_yx_m"]) / gsd - expected))
                    if result["status"] != "uncertain"
                    else None
                )
                for name, result in [("baseline", baseline), ("candidate", candidate)]
            }
            cases.append(
                {
                    "mask": mask_name,
                    "injected_yx_pixels": delta,
                    "expected_status": wanted,
                    "errors_pixels": errors,
                    "baseline": baseline,
                    "candidate": candidate,
                }
            )
    yy, xx = np.indices(reference.shape)
    periodic = np.sin(xx * np.pi / 4) + np.cos(yy * np.pi / 4)
    unrelated = gaussian_filter(np.random.default_rng(9041).normal(size=reference.shape), 1.5)
    all_valid = np.ones(reference.shape, bool)
    controls = [
        {"name": name, "measurement": audit_cfog(a, b, mask, mask, gsd=gsd)}
        for name, a, b, mask in [
            ("constant", np.ones_like(reference), np.ones_like(reference), all_valid),
            ("periodic_8px", periodic, periodic, all_valid),
            ("unrelated_seed_9041", reference, unrelated, valid),
        ]
    ]
    wrong = any(
        c["errors_pixels"]["candidate"] is not None
        and (
            c["errors_pixels"]["candidate"] > CALIBRATION["maximum_error_pixels"]
            or c["candidate"]["status"] != c["expected_status"]
        )
        for c in cases
    )
    lost = any(
        c["mask"] == "original_QA"
        and c["baseline"]["status"] != "uncertain"
        and c["candidate"]["status"] == "uncertain"
        for c in cases
    )
    false_control = any(c["measurement"]["status"] != "uncertain" for c in controls)
    complete = all(c["errors_pixels"]["candidate"] is not None for c in cases)
    return {
        "status": (
            "failed"
            if wrong or lost or false_control
            else "passed" if complete else "insufficient_texture"
        ),
        "cases": cases,
        "controls": controls,
    }


def calibrate_cfog(
    dataset_root: Path, report_root: Path, family: str, quality_root: Path, baseline_root: Path
) -> dict:
    paths = [
        baseline_root / name
        for name in ["inputs.parquet", "inputs.lock.json", "calibration.json", "output.lock.json"]
    ]
    source_files = {str(p): sha256(p) for p in paths}
    inputs, original, seal = [json.loads(p.read_text()) for p in paths[1:]]
    if (
        seal != {"inputs_lock_sha256": sha256(paths[1]), "calibration_sha256": sha256(paths[2])}
        or inputs["inputs_sha256"] != sha256(paths[0])
        or inputs["fingerprint"]["family"] != family
        or original["status"] != "passed"
    ):
        raise ValueError("passed frozen training calibration with intact seal required")
    for name, digest in inputs["fingerprint"]["code_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("original calibration implementation changed")
    runtime = {**_runtime_versions(), "pandas": pd.__version__, "zarr": zarr.__version__}
    if inputs["fingerprint"]["runtime"] != runtime:
        raise ValueError("original calibration runtime changed")
    reader = NativeQualityReader(dataset_root, family, quality_root)
    if reader.files != inputs["fingerprint"]["quality_inputs_sha256"]:
        raise ValueError("candidate QA differs from original calibration")
    rows = pd.read_parquet(paths[0])
    if (
        rows.empty
        or rows.observation_id.duplicated().any()
        or not rows.split.eq("train").all()
        or not rows.year.isin([2020, 2021]).all()
        or _digest(rows.to_dict("records")) != inputs["fingerprint"]["selected_rows_sha256"]
    ):
        raise ValueError("frozen training selection changed")
    implementation = {
        **inputs["fingerprint"]["code_sha256"],
        **{
            name: sha256(Path(__file__).with_name(name))
            for name in [
                "v5_adaptive_alignment.py",
                "v5_cfog_alignment.py",
                "v5_cfog_calibration.py",
            ]
        },
    }
    fingerprint = {
        "family": family,
        "baseline_files_sha256": source_files,
        "code_sha256": implementation,
        "runtime": runtime,
        "experiment": EXPERIMENT,
        "layout": LAYOUT,
        "matcher": PARAMETERS,
    }
    root = dataset_root / "quality/alignment/cfog_calibration" / family / _digest(fingerprint)[:20]
    frozen, frozen_lock = root / "inputs.parquet", root / "inputs.lock.json"
    if not frozen.exists():
        atomic_parquet(rows, frozen)
    elif (
        _digest(pd.read_parquet(frozen).to_dict("records"))
        != inputs["fingerprint"]["selected_rows_sha256"]
    ):
        raise ValueError("candidate frozen inputs changed")
    lock = {"fingerprint": fingerprint, "inputs_sha256": sha256(frozen)}
    if not frozen_lock.exists():
        write_json(frozen_lock, lock)
    elif json.loads(frozen_lock.read_text()) != lock:
        raise ValueError("candidate frozen inputs lock changed")
    output, publication = root / "calibration.json", root / "output.lock.json"
    reused = publication.exists()
    if reused and (
        not output.exists()
        or json.loads(publication.read_text())
        != {"inputs_lock_sha256": sha256(frozen_lock), "calibration_sha256": sha256(output)}
    ):
        raise ValueError("candidate output seal changed")
    results = json.loads(output.read_text())["results"] if reused else []
    expected_proofs = {p["observation_id"]: p for p in inputs["fingerprint"]["observations"]}
    if set(expected_proofs) != set(rows.observation_id):
        raise ValueError("source proof set differs from frozen inputs")
    if reused and [r["observation_id"] for r in results] != rows.observation_id.tolist():
        raise ValueError("candidate output rows changed")
    progress = report_root / f"cfog_calibration_{family}.json"
    for i, row in enumerate(rows.itertuples()):
        _, clear, gsd, ref_band, proof = reader.read(row)
        if proof != expected_proofs[row.observation_id]:
            raise ValueError("actual source or QA differs from frozen training proof")
        if not reused:
            ref = clear.band_ids.index(ref_band)
            measured = evaluate_texture(clear.values[ref], clear.valid[ref], gsd=gsd)
            results.append(
                {
                    "observation_id": row.observation_id,
                    "patch_id": row.patch_id,
                    "sensor": row.sensor,
                    "year": int(row.year),
                    "split": row.split,
                    "proof": proof,
                    **measured,
                }
            )
        elif results[i]["proof"] != proof:
            raise ValueError("candidate cached proof differs from actual source")
        write_json(
            progress,
            {
                "execution_status": "running",
                "processed": i + 1,
                "selected": len(rows),
                "output": str(root),
                "reused": reused,
                "updated_at": now(),
                "training_authorized": False,
            },
        )
    # Actual sources and QA are re-read before publication, including cache verification.
    for row in rows.itertuples():
        if reader.read(row)[-1] != expected_proofs[row.observation_id]:
            raise ValueError("source or QA changed during candidate calibration")
    reader.verify_unchanged()
    if (
        any(sha256(Path(p)) != h for p, h in source_files.items())
        or any(sha256(Path(__file__).with_name(n)) != h for n, h in implementation.items())
        or sha256(frozen) != lock["inputs_sha256"]
        or json.loads(frozen_lock.read_text()) != lock
    ):
        raise ValueError("candidate code or frozen inputs changed during calibration")
    sensors = {
        sensor: dict(Counter(r["status"] for r in results if r["sensor"] == sensor))
        for sensor in sorted(rows.sensor.unique())
    }
    failed = any(counts.get("failed", 0) for counts in sensors.values())
    enough = all(
        counts.get("passed", 0) >= CALIBRATION["minimum_successful_positions_per_sensor"]
        for counts in sensors.values()
    )
    status = "failed" if failed else "passed" if enough else "insufficient_calibration"
    document = {
        "status": status,
        "sensors": sensors,
        "results": results,
        "negative_controls_rejected": sum(
            c["measurement"]["status"] == "uncertain" for r in results for c in r["controls"]
        ),
        "experiment": EXPERIMENT,
        "pixel_fusion_authorized": False,
        "training_authorized": False,
    }
    if output.exists():
        if json.loads(output.read_text()) != document:
            raise ValueError("candidate output differs from sealed replay")
    else:
        write_json(output, document)
    expected_seal = {
        "inputs_lock_sha256": sha256(frozen_lock),
        "calibration_sha256": sha256(output),
    }
    if not publication.exists():
        write_json(publication, expected_seal)
    elif json.loads(publication.read_text()) != expected_seal:
        raise ValueError("candidate output seal changed during calibration")
    result = {
        "execution_status": "finished",
        "status": status,
        "processed": len(rows),
        "sensors": sensors,
        "output": str(root),
        "reused": reused,
        "finished_at": now(),
        "calibration_sha256": sha256(output),
        "training_authorized": False,
        "pixel_fusion_authorized": False,
    }
    write_json(progress, result)
    return result
