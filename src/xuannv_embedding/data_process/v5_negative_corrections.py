"""Rebuild dependent negative overlays from corrected, source-verified slope evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_dem_corrections import CorrectedDEMReader
from xuannv_embedding.data_process.v5_negative_rules import TASKS, expected_negatives
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def corrected_overlay(states, worldcover, worldcover_valid, slope, slope_valid, parameters):
    masks = expected_negatives(
        states=states,
        worldcover=worldcover,
        worldcover_valid=worldcover_valid,
        slope=slope,
        slope_valid=slope_valid,
        erosion_pixels=parameters["erosion_pixels"],
        steep_slope_degrees=parameters["steep_slope_degrees"],
    )
    valid = np.stack([masks[t] for t in TASKS])
    return {
        "states": np.where(valid, 3, 0).astype("u1"),
        "confidence": np.where(valid, parameters["negative_confidence"], 0).astype("u1"),
    }


class NegativeEvidence:
    """Re-read actual chunks against the completed full negative-rule audit."""

    def __init__(self, dataset_root: Path, report_root: Path):
        self.dataset_root, self.report_root = dataset_root, report_root
        self.audit_path = report_root / "negative_rule_audit_full.json"
        self.audit_sha = sha256(self.audit_path)
        audit = json.loads(self.audit_path.read_text())
        self.fp = audit.get("fingerprint", {})
        self.registry = pd.read_parquet(dataset_root / "registry/national_62000.parquet")
        if (
            audit.get("status") != "negative_rule_audit_finished"
            or audit.get("scope") != "full"
            or audit.get("failed_targets") != 0
            or audit.get("processed_targets") != len(self.registry) * 8
            or audit.get("selected_targets") != len(self.registry) * 8
        ):
            raise ValueError("complete successful negative-rule audit required")
        for key, path in [
            ("registry_sha256", dataset_root / "registry/national_62000.parquet"),
            ("manifest_sha256", dataset_root / "targets/manifest.parquet"),
            ("value_audit_sha256", report_root / "target_value_audit.parquet"),
            ("temporal_audit_sha256", report_root / "osm_temporal_progress.json"),
            ("code_sha256", Path(__file__).with_name("v5_negative_rules.py")),
        ]:
            if sha256(path) != self.fp.get(key):
                raise ValueError("negative audit evidence changed")
        manifest = pd.read_parquet(dataset_root / "targets/manifest.parquet")
        self.roots, self.paths = {}, {}
        for family in ["osm", "static", "reliable_negative"]:
            entries = manifest.loc[manifest.family == family]
            if entries.path.nunique() != 1:
                raise ValueError("ambiguous negative evidence family")
            path = Path(entries.path.iloc[0])
            if sha256(path / ".zattrs") != self.fp["metadata_sha256"][family]:
                raise ValueError("negative evidence metadata changed")
            self.paths[family] = path
            self.roots[family] = zarr.open_group(str(path), mode="r")
        self.parameters = self.fp["parameters"]

    def read(self, index: int, year: int) -> dict:
        if year not in [2020, 2021] or not 0 <= index < len(self.registry):
            raise ValueError("invalid annual negative evidence key")
        if sha256(self.audit_path) != self.audit_sha:
            raise ValueError("negative audit changed after reader initialization")
        for family, path in self.paths.items():
            if sha256(path / ".zattrs") != self.fp["metadata_sha256"][family]:
                raise ValueError("negative evidence metadata changed")
        required = [("osm", f"{year}/states/{task}") for task in TASKS]
        required += [
            ("static", f"{group}/{target}")
            for group in ["targets", "valid_masks"]
            for target in [f"worldcover_{year}", "dem_slope"]
        ]
        required += [
            ("reliable_negative", f"{year}/{group}/{task}")
            for group in ["states", "confidence"]
            for task in TASKS
        ]
        start = index // 32 * 32
        stop = min(start + 32, len(self.registry))
        receipt = json.loads(
            (
                self.dataset_root
                / "quality/targets/negative_rules/full/chunks"
                / f"{year}_{start:06d}.json"
            ).read_text()
        )
        block = {key: np.asarray(self.roots[key[0]][key[1]][start:stop]) for key in required}
        digest = hashlib.sha256()
        for key in required:
            digest.update(block[key].tobytes())
        expected = {
            **self.fp,
            "year": year,
            "start": start,
            "stop": stop,
            "decoded_chunk_sha256": digest.hexdigest(),
        }
        if receipt.get("fingerprint") != expected:
            raise ValueError("negative evidence pixels differ from full audit")
        offset = index - start
        return {
            "states": {t: block[("osm", f"{year}/states/{t}")][offset] for t in TASKS},
            "worldcover": block[("static", f"targets/worldcover_{year}")][offset],
            "worldcover_valid": block[("static", f"valid_masks/worldcover_{year}")][offset],
            "slope": block[("static", "targets/dem_slope")][offset],
            "slope_valid": block[("static", "valid_masks/dem_slope")][offset],
            "overlay": {
                group: np.stack(
                    [block[("reliable_negative", f"{year}/{group}/{t}")][offset] for t in TASKS]
                )
                for group in ["states", "confidence"]
            },
        }


def _directory(dataset_root: Path) -> Path:
    return dataset_root / "targets/corrections/reliable_negative/v1"


def build_negative_corrections(dataset_root: Path, report_root: Path) -> dict:
    evidence = NegativeEvidence(dataset_root, report_root)
    dem = CorrectedDEMReader(dataset_root)
    demdir = dataset_root / "targets/corrections/dem/v1"
    demlockpath = demdir / "corrections.lock.json"
    demlock = json.loads(demlockpath.read_text())
    if sha256(demdir / "manifest.parquet") != demlock["manifest_sha256"]:
        raise ValueError("DEM correction manifest changed")
    affected = pd.read_parquet(demdir / "manifest.parquet")
    affected = affected.loc[affected.correction_file != ""]
    fingerprint = {
        "negative_audit_sha256": evidence.audit_sha,
        "dem_correction_lock_sha256": sha256(demlockpath),
        "code_sha256": sha256(Path(__file__)),
        "negative_rules_code_sha256": evidence.fp["code_sha256"],
        "registry_sha256": evidence.fp["registry_sha256"],
        "parameters": evidence.parameters,
    }
    directory = _directory(dataset_root)
    directory.mkdir(parents=True, exist_ok=True)
    inputlock = directory / "input.lock.json"
    if inputlock.exists() and json.loads(inputlock.read_text()) != fingerprint:
        raise ValueError("negative correction inputs changed; use a new version")
    if not inputlock.exists():
        write_json(inputlock, fingerprint)
    rows = []
    added = removed = changed = 0
    for row in affected.itertuples(index=False):
        corrected = dem.read(row.patch_id)
        for year in [2020, 2021]:
            old = evidence.read(row.index, year)
            reconstructed = corrected_overlay(
                old["states"],
                old["worldcover"],
                old["worldcover_valid"],
                old["slope"],
                old["slope_valid"],
                evidence.parameters,
            )
            if any(not np.array_equal(reconstructed[k], old["overlay"][k]) for k in reconstructed):
                raise ValueError("original negative overlay does not match recorded rules")
            fresh = corrected_overlay(
                old["states"],
                old["worldcover"],
                old["worldcover_valid"],
                corrected["slope"],
                corrected["slope_valid"],
                evidence.parameters,
            )
            was = old["overlay"]["states"] == 3
            is_negative = fresh["states"] == 3
            additions = is_negative & ~was
            removals = was & ~is_negative
            if additions[[0, 1, 3]].any() or removals[[0, 1, 3]].any():
                raise ValueError("slope correction altered unrelated negative tasks")
            count_added = int(additions.sum())
            count_removed = int(removals.sum())
            entry = dict(
                patch_id=row.patch_id,
                index=row.index,
                year=year,
                split=row.split,
                added_negative_pixels=count_added,
                removed_negative_pixels=count_removed,
                correction_file="",
                correction_sha256="",
            )
            if count_added or count_removed:
                name = f"patches/{row.index:06d}_{year}.npz"
                path = directory / name
                if path.exists():
                    with np.load(path, allow_pickle=False) as cached:
                        if set(cached.files) != set(fresh) or any(
                            not np.array_equal(cached[k], fresh[k]) for k in fresh
                        ):
                            raise ValueError("published negative correction changed")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".partial")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, **fresh)
                    temporary.replace(path)
                entry.update(correction_file=name, correction_sha256=sha256(path))
                changed += 1
            added += count_added
            removed += count_removed
            rows.append(entry)
    if (
        sha256(demlockpath) != fingerprint["dem_correction_lock_sha256"]
        or sha256(evidence.audit_path) != evidence.audit_sha
    ):
        raise ValueError("negative correction dependencies changed during processing")
    manifest = directory / "manifest.parquet"
    atomic_parquet(
        pd.DataFrame(
            rows,
            columns=[
                "patch_id",
                "index",
                "year",
                "split",
                "added_negative_pixels",
                "removed_negative_pixels",
                "correction_file",
                "correction_sha256",
            ],
        ),
        manifest,
    )
    lock = {
        "status": "negative_corrections_materialized",
        "fingerprint": fingerprint,
        "manifest_sha256": sha256(manifest),
        "covered_positions": len(evidence.registry),
        "audited_affected_positions": len(affected),
        "audited_affected_years": len(rows),
        "corrected_position_years": changed,
        "added_negative_pixels": added,
        "removed_negative_pixels": removed,
        "training_authorized": False,
        "acceptance_status": "incomplete",
        "limitation": "Corrected computational evidence only; "
        "independent OSM geometry and label accuracy remain separate.",
    }
    lockpath = directory / "corrections.lock.json"
    if lockpath.exists() and json.loads(lockpath.read_text()) != lock:
        raise ValueError("published negative correction lock changed")
    if not lockpath.exists():
        write_json(lockpath, lock)
    summary = {**lock, "output": str(directory), "finished_at": now()}
    write_json(report_root / "negative_corrections_full.json", summary)
    return summary


class CorrectedNegativeReader:
    """Read an annual overlay with original evidence and correction bytes checked."""

    def __init__(self, dataset_root: Path, report_root: Path):
        self.directory = _directory(dataset_root)
        self.lock = json.loads((self.directory / "corrections.lock.json").read_text())
        self.evidence = NegativeEvidence(dataset_root, report_root)
        self.dem = CorrectedDEMReader(dataset_root)
        self.demlock = dataset_root / "targets/corrections/dem/v1/corrections.lock.json"
        manifest = self.directory / "manifest.parquet"
        if (
            self.lock.get("status") != "negative_corrections_materialized"
            or self.lock["fingerprint"]["negative_audit_sha256"] != self.evidence.audit_sha
            or sha256(manifest) != self.lock["manifest_sha256"]
        ):
            raise ValueError("negative correction manifest or evidence changed")
        self.rows = pd.read_parquet(manifest).set_index(["patch_id", "year"])
        self.indices = {
            row.patch_id: i for i, row in enumerate(self.evidence.registry.itertuples())
        }

    def read(self, patch_id: str, year: int) -> dict[str, np.ndarray]:
        if sha256(self.demlock) != self.lock["fingerprint"]["dem_correction_lock_sha256"]:
            raise ValueError("dependent DEM correction changed")
        result = self.evidence.read(self.indices[patch_id], year)["overlay"]
        if (patch_id, year) in self.rows.index:
            self.dem.read(patch_id)
            row = self.rows.loc[(patch_id, year)]
            if row.correction_file:
                path = (self.directory / row.correction_file).resolve()
                if (
                    not path.is_relative_to(self.directory.resolve())
                    or sha256(path) != row.correction_sha256
                ):
                    raise ValueError("negative correction file changed")
                with np.load(path, allow_pickle=False) as saved:
                    result = {key: saved[key] for key in saved.files}
        if (
            set(result) != {"states", "confidence"}
            or any(v.dtype != np.dtype("u1") or v.shape != (4, 128, 128) for v in result.values())
            or not np.isin(result["states"], [0, 3]).all()
            or not np.array_equal(
                result["confidence"],
                np.where(result["states"] == 3, self.evidence.parameters["negative_confidence"], 0),
            )
        ):
            raise ValueError("invalid corrected negative overlay")
        return result
