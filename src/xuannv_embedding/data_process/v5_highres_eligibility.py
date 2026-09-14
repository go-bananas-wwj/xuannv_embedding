"""Versioned high-resolution candidate views; unknown alignment is never a pass."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_highres_reader import HighresQualityReader
from xuannv_embedding.data_process.v5_jilin_quality import _digest
from xuannv_embedding.data_process.v5_rasters import select_annual
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

POLICY = {
    "version": "highres_eligibility_v1",
    "years": [2020, 2021],
    "alignment_unknown_can_be_reconstruction_candidate": True,
    "native_spectral_over_limit_excludes_QA_dependents": True,
    "native_pass_is_not_absolute_pass": True,
    "strict_pixel_fusion_requires_absolute_and_cross_resolution_evidence": True,
    "absolute_evidence_available": False,
    "scene_ranking": "maximum_eligible_branch_clear_fraction",
    "selection": "same_year; cover_quarters_then_clear_fraction_date_scene_id; max_4_per_family",
    "source_validation": "sealed_QA_and_catalog_metadata; loader_revalidates_actual_pixels",
    "cloud_visual_review_complete": False,
    "training_authorized": False,
}
QA_OK = {"model_inferred_needs_visual_review", "recomputed_needs_visual_review"}


def normalize_branches(reader, family):
    records = []
    for row in reader.inventory.itertuples():
        qa = reader.qa.table.loc[row.scene_group_id if family == "gaofen" else row.observation_id]
        if family == "gaofen":
            pan = row.product_id == "gaofen_pan"
            bands = ["pan"] if pan else ["blue", "green", "red", "nir"]
            counts = [int(qa.pan_valid_pixels if pan else qa.ms_valid_pixels)] * len(bands)
            native_id = row.scene_group_id
            complete = True
        else:
            bands = list(qa.band_ids)
            counts = list(qa.valid_pixels_by_band)
            native_id = row.scene_group_id + ":jilin1_ms_5m"
            complete = bool(qa.selected_bands_complete)
        records.append(
            {
                **row._asdict(),
                "family": family,
                "native_observation_id": native_id,
                "acquired_at": qa.acquired_at,
                "band_ids": bands,
                "valid_pixels_by_band": [int(n) for n in counts],
                "clear_fraction": float(qa.clear_fraction),
                "quality_status": qa.quality_status,
                "selected_bands_complete": complete,
                "geometry_status": "verified_source_grid",
                "contract_status": "verified_native_product",
            }
        )
    return pd.DataFrame(records).drop(columns="Index", errors="ignore")


def build_views(registry, rows, audit):
    if (
        registry.patch_id.duplicated().any()
        or not registry.split.isin(["train", "val", "test"]).all()
    ):
        raise ValueError("invalid spatial registry")
    owners = registry.set_index("patch_id").split
    if rows.empty or rows.observation_id.duplicated().any():
        raise ValueError("empty or duplicate branch identity")
    if not rows.patch_id.isin(owners.index).all() or not np.array_equal(
        rows.split.to_numpy(), owners.loc[rows.patch_id].to_numpy()
    ):
        raise ValueError("branch spatial split disagreement")
    if not rows.year.isin(POLICY["years"]).all():
        raise ValueError("unsupported annual candidate year")
    if not np.isfinite(rows.clear_fraction).all() or not rows.clear_fraction.between(0, 1).all():
        raise ValueError("invalid quality fraction")
    if rows.groupby(["family", "sensor", "product_id", "file_sha256"]).split.nunique().gt(1).any():
        raise ValueError("same content crosses spatial split")
    native = {}
    if not audit.empty:
        if (
            audit.observation_id.duplicated().any()
            or not audit.status.isin(["passed", "over_limit", "uncertain"]).all()
        ):
            raise ValueError("invalid or duplicate native audit")
        if not audit.observation_id.isin(rows.native_observation_id).all():
            raise ValueError("audit identity outside candidate source snapshot")
        native = {row.observation_id: row for row in audit.itertuples()}
    result = []
    seen = set()
    for row in rows.sort_values("observation_id").itertuples(index=False):
        if datetime.fromisoformat(row.acquired_at).year != row.year:
            raise ValueError("acquisition date and candidate year disagree")
        counts = np.asarray(row.valid_pixels_by_band)
        if (
            len(row.band_ids) != len(counts)
            or counts.ndim != 1
            or not len(counts)
            or not np.isfinite(counts).all()
            or (counts < 0).any()
            or (counts != np.floor(counts)).any()
        ):
            raise ValueError("invalid per-band quality counts")
        if len(set(row.band_ids)) != len(row.band_ids):
            raise ValueError("duplicate band identity")
        if row.quality_status == "qa_missing" and counts.any():
            raise ValueError("missing QA has valid pixels")
        record = native.get(row.native_observation_id)
        status, pairs = "not_audited", []
        if record is not None:
            if any(
                getattr(record, key) != getattr(row, key)
                for key in ["patch_id", "split", "sensor", "year"]
            ):
                raise ValueError("native audit identity disagrees with branch")
            status = record.status
            pairs = json.loads(record.pairs)
            if not isinstance(pairs, list) or any(
                p.get("status") not in {"passed", "over_limit", "uncertain"} for p in pairs
            ):
                raise ValueError("invalid native pair evidence")
        over = status == "over_limit" or any(p["status"] == "over_limit" for p in pairs)
        reasons = []
        if row.geometry_status != "verified_source_grid":
            reasons.append("invalid_geometry")
        if row.contract_status != "verified_native_product":
            reasons.append("unknown_product_contract")
        if row.quality_status not in QA_OK:
            reasons.append("qa_missing" if row.quality_status == "qa_missing" else "unverified_qa")
        if not counts.any():
            reasons.append("no_valid_pixels")
        if over:
            reasons.append("native_spectral_over_limit")
        identity = (row.family, row.sensor, row.product_id, row.file_sha256)
        if identity in seen:
            reasons.append("duplicate_content")
        # Prefer the first eligible representative; an invalid alias must not suppress good data.
        if not reasons:
            seen.add(identity)
        result.append(
            {
                **row._asdict(),
                "usable_band_ids": [b for b, n in zip(row.band_ids, counts, strict=True) if n > 0],
                "native_alignment_status": status,
                "native_alignment_audit_id": (
                    row.native_observation_id if record is not None else None
                ),
                "native_alignment_scope": record.scope if record is not None else "not_audited",
                "alignment_status": "unknown",
                "alignment_scope": "absolute_and_cross_resolution_not_audited",
                "offset_m": None,
                "confidence": None,
                "tolerant_reconstruction_candidate": not reasons,
                "strict_pixel_fusion_candidate": False,
                "strict_exclusion_reason": "absolute_and_cross_resolution_evidence_missing",
                "exclusion_reasons": reasons,
                "training_authorized": False,
            }
        )
    branches = pd.DataFrame(result)
    scenes = []
    for identity, group in branches.groupby("scene_group_id", sort=True):
        keys = ["patch_id", "split", "year", "acquired_at", "family", "sensor"]
        if any(group[k].nunique(dropna=False) != 1 for k in keys):
            raise ValueError("scene branches disagree on spatial or temporal identity")
        eligible = group.loc[group.tolerant_reconstruction_candidate]
        scenes.append(
            {
                **{k: group.iloc[0][k] for k in keys},
                "scene_group_id": identity,
                "branch_ids": group.observation_id.tolist(),
                "eligible_branch_ids": eligible.observation_id.tolist(),
                "available": not eligible.empty,
                "clear_fraction": (
                    float(eligible.clear_fraction.max()) if not eligible.empty else 0.0
                ),
                "alignment_status": "unknown",
                "strict_pixel_fusion_candidate": False,
                "serves_quarters": [1, 2, 3, 4],
                "prior_mode": "same_calendar_year_retrospective",
                "training_authorized": False,
            }
        )
    scenes = pd.DataFrame(scenes)
    candidates = scenes.loc[scenes.available].reset_index(drop=True)
    selected = []
    for (_, year, _), group in candidates.groupby(["patch_id", "year", "family"], sort=True):
        selected.extend(select_annual(group.to_dict("records"), year=int(year), limit=4))
    return branches, scenes, candidates, pd.DataFrame(selected, columns=scenes.columns)


def read_audit(root, family, reader):
    if root is None:
        return pd.DataFrame(), {}
    paths = [
        root / n
        for n in ["inputs.parquet", "inputs.lock.json", "observations.parquet", "output.lock.json"]
    ]
    lock = json.loads(paths[1].read_text())
    if (
        json.loads(paths[3].read_text())
        != {"inputs_lock_sha256": sha256(paths[1]), "observations_sha256": sha256(paths[2])}
        or sha256(paths[0]) != lock["inputs_sha256"]
    ):
        raise ValueError("native audit publication seal changed")
    if (
        lock["fingerprint"]["family"] != family
        or lock["snapshot"]["quality_inputs_sha256"] != reader.files
    ):
        raise ValueError("native audit QA snapshot differs from eligibility snapshot")
    inputs, audit = pd.read_parquet(paths[0]), pd.read_parquet(paths[2])
    if inputs.observation_id.tolist() != audit.observation_id.tolist():
        raise ValueError("native audit output identities changed")
    return audit, {str(p): sha256(p) for p in paths}


def run_highres_eligibility(dataset_root, report_root, family, quality_root, audit_root=None):
    reader = HighresQualityReader(dataset_root, family, quality_root)
    rows = normalize_branches(reader, family)
    audit, audit_files = read_audit(audit_root, family, reader)
    files = {**reader.files, **audit_files}
    fingerprint = {
        "family": family,
        "policy": POLICY,
        "files_sha256": files,
        "quality_configuration": reader.configuration,
        "rows_sha256": _digest(rows.to_dict("records")),
        "runtime": {"numpy": np.__version__, "pandas": pd.__version__},
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in [
                "v5_highres_eligibility.py",
                "v5_highres_reader.py",
                "v5_clear_intraband.py",
                "v5_rasters.py",
            ]
        },
    }
    root = dataset_root / "observations/index/highres" / family / _digest(fingerprint)[:20]
    root.mkdir(parents=True, exist_ok=True)
    locked = {"fingerprint": fingerprint}
    input_lock, output_lock = root / "inputs.lock.json", root / "output.lock.json"
    if input_lock.exists() and json.loads(input_lock.read_text()) != locked:
        raise ValueError("eligibility input lock changed")
    if not input_lock.exists():
        write_json(input_lock, locked)
    reused = output_lock.exists()
    if reused:
        seal = json.loads(output_lock.read_text())
        if seal["inputs_lock_sha256"] != sha256(input_lock) or any(
            sha256(root / n) != h for n, h in seal["outputs_sha256"].items()
        ):
            raise ValueError("eligibility output seal changed")
    else:
        tables = build_views(reader.registry.reset_index(), rows, audit)
        outputs = {}
        for name, frame in zip(
            [
                "branches.parquet",
                "scene_groups.parquet",
                "annual_highres_candidates.parquet",
                "annual_highres_default_selection.parquet",
            ],
            tables,
            strict=True,
        ):
            atomic_parquet(frame, root / name)
            outputs[name] = sha256(root / name)
        branches, scenes, candidates, selected = tables
        summary = {
            "branches": len(branches),
            "scenes": len(scenes),
            "candidate_branches": int(branches.tolerant_reconstruction_candidate.sum()),
            "candidate_scenes": len(candidates),
            "default_selected_scenes": len(selected),
            "strict_pixel_fusion_candidates": 0,
            "exclusion_counts": dict(
                Counter(r for reasons in branches.exclusion_reasons for r in reasons)
            ),
            "native_alignment_status_counts": dict(Counter(branches.native_alignment_status)),
            "coverage": [
                {
                    "year": int(key[0]),
                    "split": key[1],
                    "sensor": key[2],
                    "scenes": len(g),
                    "positions": int(g.patch_id.nunique()),
                }
                for key, g in candidates.groupby(["year", "split", "sensor"], sort=True)
            ],
            "scope": "frozen_available_QA_snapshot_candidate_views",
            "policy": POLICY,
            "training_authorized": False,
            "normalization_authorized": False,
        }
        write_json(root / "summary.json", summary)
        outputs["summary.json"] = sha256(root / "summary.json")
    reader.verify_unchanged()
    if (
        any(sha256(Path(p)) != h for p, h in files.items())
        or any(
            sha256(Path(__file__).with_name(n)) != h for n, h in fingerprint["code_sha256"].items()
        )
        or json.loads(input_lock.read_text()) != locked
    ):
        raise ValueError("eligibility inputs changed during publication")
    if not reused:
        write_json(
            output_lock, {"inputs_lock_sha256": sha256(input_lock), "outputs_sha256": outputs}
        )
    result = {
        **json.loads((root / "summary.json").read_text()),
        "output": str(root),
        "output_lock_sha256": sha256(output_lock),
        "reused": reused,
        "finished_at": now(),
    }
    write_json(report_root / f"highres_eligibility_{family}.json", result)
    return result
