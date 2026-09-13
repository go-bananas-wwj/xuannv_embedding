"""Explain frozen native alignment failures without changing the matching algorithm."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

from xuannv_embedding.data_process.v5_alignment import (
    PARAMETERS,
    _edges,
    _masked_ncc,
    _match_window,
)
from xuannv_embedding.data_process.v5_clear_audit import AuditReader, inspect_clear
from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_intraband import _runtime_versions
from xuannv_embedding.data_process.v5_jilin_quality import _digest
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

SELECTION = {
    "split": "train",
    "status": "uncertain",
    "per_sensor_year_coverage_bin": 4,
    "seed": "native-failure-diagnostics-v1",
    "purpose": "explanation only; not a calibration or acceptance set",
}


def _window_counts(mask: np.ndarray, size: int) -> np.ndarray:
    integral = np.pad(mask.astype("int64"), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def coverage_evidence(native: np.ndarray, clear: np.ndarray) -> dict:
    if (
        native.shape != clear.shape
        or native.dtype != bool
        or clear.dtype != bool
        or native.ndim != 2
        or min(native.shape) < 128
        or (clear & ~native).any()
    ):
        raise ValueError("corresponding native and clear boolean masks required")
    mask = binary_erosion(clear, structure=np.ones((3, 3), bool), border_value=0)
    size = PARAMETERS["window_pixels"]
    pad = PARAMETERS["maximum_shift_pixels"]
    fractions = [
        float(mask[y : y + size, x : x + size].mean())
        for y in [pad, mask.shape[0] - size - pad]
        for x in [pad, mask.shape[1] - size - pad]
    ]
    counts = _window_counts(mask, size)
    # Keep the exact same search margin; this diagnostic does not claim a reliable match.
    allowed = counts[pad : mask.shape[0] - size - pad + 1, pad : mask.shape[1] - size - pad + 1]
    index = np.unravel_index(np.argmax(allowed), allowed.shape)
    return {
        "native_valid_fraction": float(native.mean()),
        "clear_valid_fraction": float(clear.mean()),
        "QA_retention_of_native": float(clear.sum() / native.sum()) if native.any() else None,
        "corner_clear_fractions": fractions,
        "corner_windows_meeting_validity": sum(
            f >= PARAMETERS["minimum_valid_fraction"] for f in fractions
        ),
        "best_64px_clear_fraction": float(allowed[index] / size**2),
        "best_window_origin_yx": [int(index[0] + pad), int(index[1] + pad)],
        "origins_meeting_validity": int(
            (allowed >= size**2 * PARAMETERS["minimum_valid_fraction"]).sum()
        ),
        "independent_reliable_windows_proven": False,
    }


def trace_windows(reference: np.ndarray, moving: np.ndarray, valid: np.ndarray) -> list[dict]:
    if (
        reference.shape != moving.shape
        or valid.shape != reference.shape
        or reference.ndim != 2
        or min(reference.shape) < 128
    ):
        raise ValueError("corresponding native grids required")
    reference = np.asarray(reference, dtype="f8")
    moving = np.asarray(moving, dtype="f8")
    valid = np.asarray(valid, dtype=bool) & np.isfinite(reference) & np.isfinite(moving)
    left, right = _edges(reference, valid), _edges(moving, valid)
    valid = binary_erosion(valid, structure=np.ones((3, 3), bool), border_value=0)
    size, pad = PARAMETERS["window_pixels"], PARAMETERS["maximum_shift_pixels"]
    windows = []
    for y in (pad, reference.shape[0] - size - pad):
        for x in (pad, reference.shape[1] - size - pad):
            mask = valid[y : y + size, x : x + size]
            template = left[y : y + size, x : x + size]
            search = right[y - pad : y + size + pad, x - pad : x + size + pad]
            svalid = valid[y - pad : y + size + pad, x - pad : x + size + pad]
            result = {
                "origin_yx": [y, x],
                "valid_fraction": float(mask.mean()),
                "template_edge_std": float(template[mask].std()) if mask.any() else None,
            }
            if mask.mean() < PARAMETERS["minimum_valid_fraction"]:
                reason = "insufficient_template_validity"
            elif float(template[mask].std()) < 1e-4:
                reason = "insufficient_template_texture"
            else:
                score = _masked_ncc(template, search, mask, svalid)
                if not np.isfinite(score).any():
                    reason = "no_valid_ncc_candidate"
                else:
                    peak = np.array(np.unravel_index(np.argmax(score), score.shape))
                    best = float(score[tuple(peak)])
                    yy, xx = np.indices(score.shape)
                    far = (
                        np.hypot(yy - peak[0], xx - peak[1])
                        > PARAMETERS["peak_exclusion_radius_pixels"]
                    )
                    competitors = score[far & np.isfinite(score)]
                    margin = best - float(competitors.max()) if competitors.size else 0.0
                    result.update(integer_peak_correlation=best, peak_margin=margin)
                    if best < PARAMETERS["minimum_correlation"]:
                        reason = "low_integer_peak_correlation"
                    elif margin < PARAMETERS["minimum_peak_margin"]:
                        reason = "ambiguous_peak"
                    else:
                        match = _match_window(template, search, mask, svalid)
                        reason = "accepted" if match else "refined_match_rejected"
                        if match:
                            result["match"] = match
            windows.append({**result, "reason": reason})
    return windows


def select_diagnostics(rows: pd.DataFrame, *, reference_index: int) -> pd.DataFrame:
    if rows.empty or rows.observation_id.duplicated().any():
        raise ValueError("empty or duplicate diagnostic observations")
    rows = rows.loc[rows.split.eq("train") & rows.status.eq("uncertain")].copy()
    rows["reference_clear_fraction"] = rows.clear_fraction_by_band.map(
        lambda x: json.loads(x)[reference_index]
    )
    if (
        not np.isfinite(rows.reference_clear_fraction).all()
        or not rows.reference_clear_fraction.between(0, 1).all()
    ):
        raise ValueError("invalid diagnostic clear fractions")

    def bin_name(x):
        return (
            "zero"
            if x == 0
            else (
                "0_to_25"
                if x <= 0.25
                else "25_to_60" if x <= 0.6 else "60_to_95" if x <= 0.95 else "above_95"
            )
        )

    rows["coverage_bin"] = rows.reference_clear_fraction.map(bin_name)
    rows["selection_key"] = rows.observation_id.map(lambda x: _digest([SELECTION["seed"], x]))
    keys = ["sensor", "year", "coverage_bin"]
    selected = (
        rows.sort_values(keys + ["selection_key", "observation_id"])
        .drop_duplicates(keys + ["patch_id"])
        .groupby(keys, sort=True)
        .head(4)
    )
    return selected.reset_index(drop=True)


def diagnose_alignment(
    dataset_root: Path, report_root: Path, family: str, quality_root: Path, audit_root: Path
) -> dict:
    output_path = audit_root / "observations.parquet"
    input_path = audit_root / "inputs.parquet"
    lock_path = audit_root / "inputs.lock.json"
    seal_path = audit_root / "output.lock.json"
    seal = json.loads(seal_path.read_text())
    locked = json.loads(lock_path.read_text())
    if (
        seal
        != {"inputs_lock_sha256": sha256(lock_path), "observations_sha256": sha256(output_path)}
        or sha256(input_path) != locked["inputs_sha256"]
        or locked["fingerprint"]["family"] != family
    ):
        raise ValueError("frozen native audit seal or inputs changed")
    audit_files = {str(p): sha256(p) for p in [output_path, input_path, lock_path, seal_path]}
    reader = AuditReader(dataset_root, family, quality_root)
    if reader.files != locked["snapshot"]["quality_inputs_sha256"]:
        raise ValueError("diagnostic QA differs from frozen audit")
    implementation = {
        name: sha256(Path(__file__).with_name(name))
        for name in [
            "v5_alignment_diagnostics.py",
            "v5_clear_audit.py",
            "v5_clear_intraband.py",
            "v5_intraband.py",
            "v5_alignment.py",
            "v5_rasters.py",
            "v5_partial_bands.py",
        ]
    }
    for name, value in locked["fingerprint"]["code_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != value:
            raise ValueError("original audit implementation changed")
    all_rows = pd.read_parquet(output_path)
    inputs = pd.read_parquet(input_path)
    if all_rows.observation_id.duplicated().any() or set(all_rows.observation_id) != set(
        inputs.observation_id
    ):
        raise ValueError("audit observations differ from frozen input set")
    selected = select_diagnostics(all_rows, reference_index=1 if family == "gaofen" else 3)
    selected = selected.merge(
        inputs[["observation_id", "path", "file_sha256"]],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    if selected.empty or selected.path.isna().any():
        raise ValueError("no matching diagnostic source observations")
    fingerprint = {
        "family": family,
        "audit_files_sha256": audit_files,
        "quality_files_sha256": reader.files,
        "implementation_sha256": implementation,
        "selection": SELECTION,
        "runtime": {**_runtime_versions(), "pandas": pd.__version__},
        "selected_rows_sha256": _digest(selected.to_dict("records")),
        "parameters": PARAMETERS,
    }
    root = report_root / "alignment_diagnostics" / family / _digest(fingerprint)[:20]
    frozen = root / "inputs.parquet"
    frozen_lock = root / "inputs.lock.json"
    if frozen.exists():
        if (
            _digest(pd.read_parquet(frozen).to_dict("records"))
            != fingerprint["selected_rows_sha256"]
        ):
            raise ValueError("diagnostic selection changed")
    else:
        atomic_parquet(selected, frozen)
    input_lock = {"fingerprint": fingerprint, "inputs_sha256": sha256(frozen)}
    if frozen_lock.exists():
        if json.loads(frozen_lock.read_text()) != input_lock:
            raise ValueError("diagnostic input lock changed")
    else:
        write_json(frozen_lock, input_lock)
    results = []
    progress = report_root / f"alignment_diagnostics_{family}.json"
    for row in selected.itertuples():
        loaded = reader.read(row)
        raw, clear, gsd, reference, proof = loaded
        actual = inspect_clear(reader, row, loaded=loaded)
        if (
            actual["pairs"] != json.loads(row.pairs)
            or actual["status"] != row.status
            or proof["quality_mask_sha256"] != row.quality_mask_sha256
        ):
            raise ValueError("diagnostic replay differs from published measurement")
        ref = clear.band_ids.index(reference)
        pairs = []
        for pair in actual["pairs"]:
            i = clear.band_ids.index(pair["moving_band"])
            native = raw.valid[ref] & raw.valid[i]
            valid = clear.valid[ref] & clear.valid[i]
            trace = trace_windows(clear.values[ref], clear.values[i], valid)
            if sum(x["reason"] == "accepted" for x in trace) != pair["valid_windows"]:
                raise ValueError("diagnostic trace differs from algorithm decisions")
            pairs.append(
                {
                    "moving_band": pair["moving_band"],
                    "status": pair["status"],
                    "original_reason": pair.get("reason", ""),
                    "coverage": coverage_evidence(native, valid),
                    "windows": trace,
                }
            )
        results.append(
            {
                "observation_id": row.observation_id,
                "patch_id": row.patch_id,
                "sensor": row.sensor,
                "year": int(row.year),
                "split": row.split,
                "coverage_bin": row.coverage_bin,
                "proof": proof,
                "pairs": pairs,
            }
        )
        write_json(
            progress,
            {
                "execution_status": "running",
                "processed": len(results),
                "selected": len(selected),
                "output": str(root),
                "updated_at": now(),
                "training_authorized": False,
            },
        )
    # Re-read the selected inputs before sealing the explanation, including a repeated run.
    for row, expected in zip(selected.itertuples(), results, strict=True):
        if reader.read(row)[-1] != expected["proof"]:
            raise ValueError("diagnostic source or masks changed during explanation")
    reader.verify_unchanged()
    if (
        any(sha256(Path(p)) != h for p, h in audit_files.items())
        or any(sha256(Path(__file__).with_name(n)) != h for n, h in implementation.items())
        or sha256(frozen) != input_lock["inputs_sha256"]
        or json.loads(frozen_lock.read_text()) != input_lock
    ):
        raise ValueError("diagnostic inputs or algorithm changed")
    reasons = Counter(w["reason"] for r in results for p in r["pairs"] for w in p["windows"])
    missed = sum(
        p["coverage"]["corner_windows_meeting_validity"] == 0
        and p["coverage"]["best_64px_clear_fraction"] >= PARAMETERS["minimum_valid_fraction"]
        for r in results
        for p in r["pairs"]
    )
    output = {
        "selection": SELECTION,
        "results": results,
        "window_reasons": dict(reasons),
        "pairs_with_no_usable_corner_but_usable_interior": missed,
        "selected": len(results),
        "scope": "fixed training diagnostic sample; no parameter changes or new alignment approval",
        "pixel_fusion_authorized": False,
        "training_authorized": False,
    }
    path = root / "diagnostics.json"
    publication = root / "output.lock.json"
    if publication.exists():
        old = json.loads(publication.read_text())
        if old != {"inputs_lock_sha256": sha256(frozen_lock), "diagnostics_sha256": sha256(path)}:
            raise ValueError("diagnostic output seal changed")
    if path.exists():
        if json.loads(path.read_text()) != output:
            raise ValueError("published diagnostic differs from replay")
    else:
        write_json(path, output)
    if not publication.exists():
        write_json(
            publication,
            {"inputs_lock_sha256": sha256(frozen_lock), "diagnostics_sha256": sha256(path)},
        )
    summary = {
        "execution_status": "finished",
        "selected": len(results),
        "window_reasons": dict(reasons),
        "pairs_with_no_usable_corner_but_usable_interior": missed,
        "output": str(root),
        "diagnostics_sha256": sha256(path),
        "finished_at": now(),
        "training_authorized": False,
    }
    write_json(progress, summary)
    return summary
