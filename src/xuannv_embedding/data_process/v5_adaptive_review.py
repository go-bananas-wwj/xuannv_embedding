"""Freeze unseen training locations and challenge real spectral alignment candidates."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import shift

from xuannv_embedding.data_process.v5_adaptive_alignment import LAYOUT, audit_adaptive
from xuannv_embedding.data_process.v5_alignment import PARAMETERS
from xuannv_embedding.data_process.v5_clear_audit import AuditReader, inspect_clear
from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_intraband import _runtime_versions
from xuannv_embedding.data_process.v5_jilin_quality import _digest
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

REVIEW = {
    "seed": "unseen-native-window-review-v1",
    "split": "train",
    "years": [2020, 2021],
    "per_sensor_year_coverage_bin": 4,
    "globally_distinct_patches": True,
    "selection_uses_measured_status": False,
    "perturbations_yx_pixels": [[0, 0.5], [0, 1.5], [2, -1]],
    "maximum_consistency_error_pixels": 0.35,
    "ground_truth_available": False,
}
BINS = ["zero", "0_to_25", "25_to_60", "60_to_95", "above_95"]


def select_review_rows(rows: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    if rows.observation_id.duplicated().any():
        raise ValueError("duplicate review observations")
    rows = rows.loc[
        rows.split.eq("train") & rows.year.isin(REVIEW["years"]) & ~rows.patch_id.isin(excluded)
    ].copy()
    if rows.empty:
        raise ValueError("no eligible training locations after exclusions")
    fractions = rows.reference_clear_fraction
    if not np.isfinite(fractions).all() or not fractions.between(0, 1).all():
        raise ValueError("invalid reference clear fraction")
    rows["coverage_bin"] = fractions.map(
        lambda x: (
            BINS[0]
            if x == 0
            else (
                BINS[1] if x <= 0.25 else BINS[2] if x <= 0.6 else BINS[3] if x <= 0.95 else BINS[4]
            )
        )
    )
    rows["selection_key"] = rows.observation_id.map(lambda x: _digest([REVIEW["seed"], x]))
    chosen, used = [], set()
    for _, candidates in rows.sort_values(
        ["sensor", "year", "coverage_bin", "selection_key", "observation_id"]
    ).groupby(["sensor", "year", "coverage_bin"], sort=True):
        count = 0
        for index, row in candidates.iterrows():
            if row.patch_id in used:
                continue
            chosen.append(index)
            used.add(row.patch_id)
            count += 1
            if count == REVIEW["per_sensor_year_coverage_bin"]:
                break
    return rows.loc[chosen].reset_index(drop=True)


def review_pair(reference, moving, reference_valid, moving_valid, *, gsd):
    forward = audit_adaptive(reference, moving, reference_valid, moving_valid, gsd=gsd)
    result = {
        "candidate": forward,
        "consistency": "forward_uncertain",
        "reverse": None,
        "reverse_cycle_error_pixels": None,
        "perturbations": [],
        "pixel_fusion_authorized": False,
    }
    if forward["status"] == "uncertain":
        return result
    base = np.asarray(forward["translation_yx_m"]) / gsd
    reverse = audit_adaptive(moving, reference, moving_valid, reference_valid, gsd=gsd)
    cycle = (
        float(np.linalg.norm(base + np.asarray(reverse["translation_yx_m"]) / gsd))
        if reverse["status"] != "uncertain"
        else None
    )
    result.update(reverse=reverse, reverse_cycle_error_pixels=cycle)
    for delta in REVIEW["perturbations_yx_pixels"]:
        perturbed = shift(moving, delta, order=1, mode="constant", cval=0)
        valid = (
            shift(moving_valid.astype(float), delta, order=1, mode="constant", cval=0) >= 1 - 1e-6
        )
        measured = audit_adaptive(reference, perturbed, reference_valid, valid, gsd=gsd)
        error = (
            float(np.linalg.norm(np.asarray(measured["translation_yx_m"]) / gsd + delta - base))
            if measured["status"] != "uncertain"
            else None
        )
        result["perturbations"].append(
            {"injected_yx_pixels": delta, "error_pixels": error, "measurement": measured}
        )
    errors = [cycle] + [c["error_pixels"] for c in result["perturbations"]]
    result["consistency"] = (
        "self_consistent_candidate"
        if all(e is not None and e <= REVIEW["maximum_consistency_error_pixels"] for e in errors)
        else "consistency_failed"
    )
    return result


def _sealed_inputs(root, output_name, hash_name):
    paths = [
        root / n for n in ["inputs.parquet", "inputs.lock.json", output_name, "output.lock.json"]
    ]
    lock = json.loads(paths[1].read_text())
    if json.loads(paths[3].read_text()) != {
        "inputs_lock_sha256": sha256(paths[1]),
        hash_name: sha256(paths[2]),
    } or lock["inputs_sha256"] != sha256(paths[0]):
        raise ValueError("review prerequisite seal changed")
    return lock, {str(p): sha256(p) for p in paths}


def run_adaptive_review(
    dataset_root: Path,
    report_root: Path,
    family: str,
    quality_root: Path,
    audit_root: Path,
    calibration_root: Path,
    exclusion_inputs: list[Path],
) -> dict:
    audit_lock, files = _sealed_inputs(audit_root, "observations.parquet", "observations_sha256")
    cal_lock, cal_files = _sealed_inputs(calibration_root, "calibration.json", "calibration_sha256")
    files.update(cal_files)
    calibrated = json.loads((calibration_root / "calibration.json").read_text())
    if (
        calibrated["status"] != "passed"
        or audit_lock["fingerprint"]["family"] != family
        or cal_lock["fingerprint"]["family"] != family
    ):
        raise ValueError("passed candidate calibration for the audited family required")
    if (
        cal_lock["fingerprint"]["matcher"] != PARAMETERS
        or cal_lock["fingerprint"]["layout"] != LAYOUT
    ):
        raise ValueError("candidate matching parameters changed")
    if cal_lock["fingerprint"]["runtime"] != {**_runtime_versions(), "pandas": pd.__version__}:
        raise ValueError("candidate calibration runtime changed")
    baseline_files = cal_lock["fingerprint"]["baseline_files_sha256"]
    baseline_calibrations = [
        h for p, h in baseline_files.items() if p.endswith("/calibration.json")
    ]
    if baseline_calibrations != [audit_lock["fingerprint"]["calibration"]["calibration_sha256"]]:
        raise ValueError("candidate and original audit use different calibration")
    files.update(baseline_files)
    if any(sha256(Path(p)) != h for p, h in files.items()):
        raise ValueError("calibration source seal changed")
    implementation = {}
    for source in [audit_lock, cal_lock]:
        for name, digest in source["fingerprint"]["code_sha256"].items():
            if sha256(Path(__file__).with_name(name)) != digest:
                raise ValueError("frozen algorithm implementation changed")
            implementation[name] = digest
    implementation[Path(__file__).name] = sha256(Path(__file__))
    reader = AuditReader(dataset_root, family, quality_root)
    if reader.files != audit_lock["snapshot"]["quality_inputs_sha256"]:
        raise ValueError("review QA differs from original audit")
    files.update(reader.files)
    # Candidate calibration locations are excluded even if the caller omits them.
    exclusions = set(pd.read_parquet(calibration_root / "inputs.parquet").patch_id)
    if not exclusion_inputs:
        raise ValueError("explicit prior calibration and diagnostic exclusion inputs required")
    for path in sorted(set(exclusion_inputs)):
        source = pd.read_parquet(path)
        if "patch_id" not in source or source.patch_id.isna().any():
            raise ValueError("exclusions require explicit patch identities")
        exclusions.update(source.patch_id)
        files[str(path)] = sha256(path)
    rows = pd.read_parquet(audit_root / "observations.parquet")
    inputs = pd.read_parquet(audit_root / "inputs.parquet")
    if rows.observation_id.duplicated().any() or set(rows.observation_id) != set(
        inputs.observation_id
    ):
        raise ValueError("original audit rows differ from its input set")
    ref_index = 1 if family == "gaofen" else 3
    rows["reference_clear_fraction"] = rows.clear_fraction_by_band.map(
        lambda x: json.loads(x)[ref_index]
    )
    selected = select_review_rows(rows, exclusions).merge(
        inputs[["observation_id", "path", "file_sha256"]],
        on="observation_id",
        validate="one_to_one",
    )
    if selected.path.isna().any():
        raise ValueError("review sources missing from original audit inputs")
    fingerprint = {
        "family": family,
        "files_sha256": files,
        "code_sha256": implementation,
        "runtime": {**_runtime_versions(), "pandas": pd.__version__},
        "review": REVIEW,
        "selected_rows_sha256": _digest(selected.to_dict("records")),
        "excluded_patch_ids": sorted(exclusions),
    }
    root = report_root / "adaptive_alignment_review" / family / _digest(fingerprint)[:20]
    frozen, frozen_lock = root / "inputs.parquet", root / "inputs.lock.json"
    if frozen.exists():
        if (
            _digest(pd.read_parquet(frozen).to_dict("records"))
            != fingerprint["selected_rows_sha256"]
        ):
            raise ValueError("review frozen selection changed")
    else:
        atomic_parquet(selected, frozen)
    locked = {"fingerprint": fingerprint, "inputs_sha256": sha256(frozen)}
    if frozen_lock.exists():
        if json.loads(frozen_lock.read_text()) != locked:
            raise ValueError("review frozen input lock changed")
    else:
        write_json(frozen_lock, locked)
    output, publication = root / "review.json", root / "output.lock.json"
    reused = publication.exists()
    if reused and (
        not output.exists()
        or json.loads(publication.read_text())
        != {"inputs_lock_sha256": sha256(frozen_lock), "review_sha256": sha256(output)}
    ):
        raise ValueError("review output seal changed")
    results = json.loads(output.read_text())["results"] if reused else []
    if reused and [r["observation_id"] for r in results] != selected.observation_id.tolist():
        raise ValueError("review output selection changed")
    progress = report_root / f"adaptive_alignment_review_{family}.json"
    for i, row in enumerate(selected.itertuples()):
        loaded = reader.read(row)
        _, clear, gsd, reference, proof = loaded
        if not reused:
            original = inspect_clear(reader, row, loaded=loaded)
            if (
                original["status"] != row.status
                or original["pairs"] != json.loads(row.pairs)
                or proof["quality_mask_sha256"] != row.quality_mask_sha256
            ):
                raise ValueError("original real alignment does not replay exactly")
            ref = clear.band_ids.index(reference)
            pairs = []
            for base in original["pairs"]:
                j = clear.band_ids.index(base["moving_band"])
                review = review_pair(
                    clear.values[ref], clear.values[j], clear.valid[ref], clear.valid[j], gsd=gsd
                )
                disagreement = (
                    float(
                        np.linalg.norm(
                            np.asarray(base["translation_yx_m"]) / gsd
                            - np.asarray(review["candidate"]["translation_yx_m"]) / gsd
                        )
                    )
                    if base["status"] != "uncertain"
                    and review["candidate"]["status"] != "uncertain"
                    else None
                )
                pairs.append(
                    {
                        "reference_band": reference,
                        "moving_band": base["moving_band"],
                        "baseline": base,
                        "baseline_candidate_disagreement_pixels": disagreement,
                        **review,
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
        elif proof != results[i]["proof"]:
            raise ValueError("review source or mask differs from cached proof")
        write_json(
            progress,
            {
                "execution_status": "running",
                "processed": i + 1,
                "selected": len(selected),
                "output": str(root),
                "reused": reused,
                "updated_at": now(),
                "training_authorized": False,
            },
        )
    for row, result in zip(selected.itertuples(), results, strict=True):
        if reader.read(row)[-1] != result["proof"]:
            raise ValueError("review sources or masks changed during measurement")
    reader.verify_unchanged()
    if (
        any(sha256(Path(p)) != h for p, h in files.items())
        or any(sha256(Path(__file__).with_name(n)) != h for n, h in implementation.items())
        or sha256(frozen) != locked["inputs_sha256"]
        or json.loads(frozen_lock.read_text()) != locked
    ):
        raise ValueError("review inputs or implementation changed during measurement")
    pairs = [p for row in results for p in row["pairs"]]
    strata = [
        {
            "sensor": sensor,
            "year": year,
            "coverage_bin": coverage,
            "selected": int(
                (
                    (selected.sensor == sensor)
                    & (selected.year == year)
                    & (selected.coverage_bin == coverage)
                ).sum()
            ),
            "requested": 4,
        }
        for sensor in sorted(rows.sensor.unique())
        for year in REVIEW["years"]
        for coverage in BINS
    ]
    document = {
        "review": REVIEW,
        "results": results,
        "strata": strata,
        "excluded_positions": len(exclusions),
        "transitions": dict(
            Counter(p["baseline"]["status"] + "->" + p["candidate"]["status"] for p in pairs)
        ),
        "consistency": dict(Counter(p["consistency"] for p in pairs)),
        "new_self_consistent_pairs": sum(
            p["baseline"]["status"] == "uncertain"
            and p["consistency"] == "self_consistent_candidate"
            for p in pairs
        ),
        "scope": (
            "real spectral consistency evidence without ground truth; "
            "no QA or alignment approval changes"
        ),
        "pixel_fusion_authorized": False,
        "training_authorized": False,
    }
    if output.exists():
        if json.loads(output.read_text()) != document:
            raise ValueError("review output differs from verified replay")
    else:
        write_json(output, document)
    expected = {"inputs_lock_sha256": sha256(frozen_lock), "review_sha256": sha256(output)}
    if publication.exists():
        if json.loads(publication.read_text()) != expected:
            raise ValueError("review output seal changed")
    else:
        write_json(publication, expected)
    summary = {
        "execution_status": "finished",
        "selected": len(selected),
        "pairs": len(pairs),
        "transitions": document["transitions"],
        "consistency": document["consistency"],
        "new_self_consistent_pairs": document["new_self_consistent_pairs"],
        "output": str(root),
        "review_sha256": sha256(output),
        "reused": reused,
        "finished_at": now(),
        "training_authorized": False,
        "pixel_fusion_authorized": False,
    }
    write_json(progress, summary)
    return summary
