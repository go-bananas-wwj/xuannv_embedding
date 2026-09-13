"""Publish source-backed sparse DEM corrections without rewriting historical targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_dem_geometry import DEMSource, legacy_slope, slope_support
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json
from xuannv_embedding.data_process.v5_target_geometry import compare_target

FIELDS = {
    "elevation": "targets/dem_elevation",
    "slope": "targets/dem_slope",
    "elevation_valid": "valid_masks/dem_elevation",
    "slope_valid": "valid_masks/dem_slope",
}


def payload_sha256(payload: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in FIELDS:
        array = np.asarray(payload[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_payload(payload: dict[str, np.ndarray]) -> None:
    if set(payload) != set(FIELDS):
        raise ValueError("incomplete DEM payload")
    for name, array in payload.items():
        expected = np.dtype(bool if name.endswith("_valid") else "f4")
        if array.shape != (128, 128) or array.dtype != expected:
            raise ValueError("DEM payload shape or dtype mismatch")
    for name in ["elevation", "slope"]:
        values, valid = payload[name], payload[name + "_valid"]
        if not np.isfinite(values).all() or np.any(values[~valid] != 0):
            raise ValueError("invalid DEM values must be zero; valid values must be finite")
    if np.any(payload["slope_valid"] & ~slope_support(payload["elevation_valid"])):
        raise ValueError("slope has missing elevation support")


def correct_dem(elevation: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    if elevation.shape != (128, 128) or valid.shape != elevation.shape or valid.dtype != bool:
        raise ValueError("invalid native DEM grid")
    if np.any(valid & ~np.isfinite(elevation)):
        raise ValueError("nonfinite valid source elevation")
    values = np.where(valid, elevation, 0).astype("f4")
    supported = slope_support(valid)
    result = {
        "elevation": values,
        "slope": np.where(supported, legacy_slope(values, valid), 0).astype("f4"),
        "elevation_valid": valid.copy(),
        "slope_valid": supported,
    }
    validate_payload(result)
    return result


def _directory(dataset_root: Path) -> Path:
    return dataset_root / "targets/corrections/dem/v1"


def _save_payload(path: Path, payload: dict[str, np.ndarray]) -> str:
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if set(cached.files) != set(payload) or any(
                not np.array_equal(cached[key], value) for key, value in payload.items()
            ):
                raise ValueError("existing correction differs; use a new version")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(path)
    return sha256(path)


def build_dem_corrections(dataset_root: Path, report_root: Path) -> dict:
    audit_path = report_root / "target_geometry_dem_full.json"
    audit = json.loads(audit_path.read_text())
    regpath = dataset_root / "registry/national_62000.parquet"
    manifest_path = dataset_root / "targets/manifest.parquet"
    value_path = report_root / "target_value_audit.parquet"
    registry = pd.read_parquet(regpath)
    fp = audit.get("fingerprint", {})
    if (
        audit.get("status") != "dem_geometry_audit_finished"
        or audit.get("scope") != "full"
        or audit.get("processed_targets") != len(registry) * 2
        or audit.get("selected_targets") != len(registry) * 2
        or fp.get("registry_sha256") != sha256(regpath)
        or fp.get("manifest_sha256") != sha256(manifest_path)
        or registry.empty
        or registry.patch_id.duplicated().any()
    ):
        raise ValueError("completed matching full DEM geometry audit required")
    for name in ["v5_dem_geometry.py", "v5_target_geometry.py"]:
        if fp.get("code_sha256", {}).get(name) != sha256(Path(__file__).with_name(name)):
            raise ValueError("DEM audit algorithm changed")
    observation_path = Path(audit["output"])
    observations = pd.read_parquet(observation_path)
    pairs = {(p, t) for p in registry.patch_id for t in ["dem_elevation", "dem_slope"]}
    if (
        len(observations) != len(pairs)
        or set(zip(observations.patch_id, observations.target)) != pairs
        or not observations.status.isin(["passed", "failed"]).all()
        or int((observations.status == "failed").sum()) != audit["failed_targets"]
    ):
        raise ValueError("incomplete DEM observation audit")
    observed = observations.set_index(["patch_id", "target"])
    failed = set(observations.loc[observations.status == "failed", "patch_id"])
    entries = pd.read_parquet(manifest_path)
    entries = entries.loc[(entries.family == "static") & entries.array.isin(FIELDS.values())]
    if (
        len(entries) != 4
        or entries.path.nunique() != 1
        or not entries.registry_order_verified.all()
        or not (entries.temporal_mode == "static").all()
    ):
        raise ValueError("verified static DEM manifest required")
    baseline = Path(entries.path.iloc[0])
    meta = sha256(baseline / ".zattrs")
    root = zarr.open_group(str(baseline), mode="r")
    if (
        set(entries.source_metadata_sha256) != {meta}
        or list(root.attrs["patch_ids"]) != registry.patch_id.tolist()
    ):
        raise ValueError("baseline DEM metadata changed")
    arrays = {name: root[path] for name, path in FIELDS.items()}
    if any(a.shape != (len(registry), 128, 128) for a in arrays.values()):
        raise ValueError("baseline DEM shape changed")
    receipts = pd.read_parquet(value_path)
    hashes = {}
    for key, name in FIELDS.items():
        row = receipts.loc[
            (receipts.family == "static")
            & (receipts.array == name)
            & (receipts.path == str(baseline))
        ]
        if len(row) != 1:
            raise ValueError("missing baseline value audit")
        hashes[key] = row.iloc[0].decoded_values_sha256
    sources = fp.get("sources", [])
    if not sources:
        raise ValueError("locked DEM sources required")
    parts = [Path(item["path"]) for item in sources]
    stamps = [(p.stat().st_size, p.stat().st_mtime_ns) for p in parts]
    for part, item in zip(parts, sources, strict=True):
        if sha256(part) != item["sha256"]:
            raise ValueError("DEM source changed before correction")
    fingerprint = {
        "geometry_audit_sha256": sha256(audit_path),
        "geometry_observations_sha256": sha256(observation_path),
        "registry_sha256": sha256(regpath),
        "manifest_sha256": sha256(manifest_path),
        "value_audit_sha256": sha256(value_path),
        "baseline_metadata_sha256": meta,
        "baseline_path": str(baseline),
        "sources": sources,
        "geometry_code_sha256": fp["code_sha256"],
        "correction_code_sha256": sha256(Path(__file__)),
        "policy": "actual source bounds; slope valid only with all derivative neighbors observed",
    }
    directory = _directory(dataset_root)
    directory.mkdir(parents=True, exist_ok=True)
    runlock = directory / "input.lock.json"
    if runlock.exists() and json.loads(runlock.read_text()) != fingerprint:
        raise ValueError("correction inputs changed; use a new version")
    if not runlock.exists():
        write_json(runlock, fingerprint)
    source = DEMSource(parts, directory / "source.xml")
    rows, digests = [], {name: hashlib.sha256() for name in FIELDS}
    total = {
        "corrected_positions": 0,
        "original_unsupported_slope_pixels": 0,
        "added_elevation_pixels": 0,
        "removed_elevation_pixels": 0,
        "added_slope_pixels": 0,
        "removed_slope_pixels": 0,
        "changed_valid_slope_pixels": 0,
    }
    progress = report_root / "dem_corrections_full.json"
    try:
        for start in range(0, len(registry), 32):
            stop = min(start + 32, len(registry))
            block = {key: np.asarray(array[start:stop]) for key, array in arrays.items()}
            for key, values in block.items():
                digests[key].update(values.tobytes())
            for offset, row in enumerate(registry.iloc[start:stop].itertuples()):
                old = {key: value[offset] for key, value in block.items()}
                unsupported = int(
                    (old["slope_valid"] & ~slope_support(old["elevation_valid"])).sum()
                )
                total["original_unsupported_slope_pixels"] += unsupported
                result = dict(
                    patch_id=row.patch_id,
                    index=start + offset,
                    split=row.split,
                    baseline_sha256=payload_sha256(old),
                    correction_file="",
                    correction_sha256="",
                    original_unsupported_slope_pixels=unsupported,
                    source_members="[]",
                )
                if row.patch_id in failed or unsupported:
                    elevation, valid, members = source.reconstruct(row.grid_epsg, row.utm_bounds)
                    fresh = correct_dem(elevation, valid)
                    # Reproduce the audit difference before restricting derivative support.
                    for target, values in [
                        ("dem_elevation", elevation),
                        ("dem_slope", legacy_slope(elevation, valid)),
                    ]:
                        key = target.removeprefix("dem_")
                        checked = compare_target(
                            np.where(valid, values, 0),
                            valid,
                            old[key],
                            old[key + "_valid"],
                            categorical=False,
                        )
                        if checked["status"] != observed.loc[(row.patch_id, target), "status"]:
                            raise ValueError(
                                "source reconstruction no longer matches original failure evidence"
                            )
                    if payload_sha256(fresh) != result["baseline_sha256"]:
                        name = f"patches/{start+offset:06d}.npz"
                        result.update(
                            correction_file=name,
                            correction_sha256=_save_payload(directory / name, fresh),
                            source_members=json.dumps(members),
                        )
                        total["corrected_positions"] += 1
                        for target in ["elevation", "slope"]:
                            total["added_" + target + "_pixels"] += int(
                                (fresh[target + "_valid"] & ~old[target + "_valid"]).sum()
                            )
                            total["removed_" + target + "_pixels"] += int(
                                (~fresh[target + "_valid"] & old[target + "_valid"]).sum()
                            )
                        joint = fresh["slope_valid"] & old["slope_valid"]
                        total["changed_valid_slope_pixels"] += int(
                            (
                                joint
                                & ~np.isclose(fresh["slope"], old["slope"], rtol=1e-6, atol=1e-5)
                            ).sum()
                        )
                else:
                    validate_payload(old)
                rows.append(result)
            write_json(
                progress,
                {
                    "status": "running",
                    "processed_positions": len(rows),
                    "selected_positions": len(registry),
                    **total,
                    "updated_at": now(),
                },
            )
    finally:
        source.close()
    if any(digests[key].hexdigest() != hashes[key] for key in FIELDS):
        raise ValueError("baseline pixels changed after value audit")
    if (
        stamps != [(p.stat().st_size, p.stat().st_mtime_ns) for p in parts]
        or sha256(baseline / ".zattrs") != meta
    ):
        raise ValueError("DEM inputs changed during correction")
    if (
        sha256(audit_path) != fingerprint["geometry_audit_sha256"]
        or sha256(observation_path) != fingerprint["geometry_observations_sha256"]
    ):
        raise ValueError("geometry evidence changed during correction")
    table = directory / "manifest.parquet"
    atomic_parquet(pd.DataFrame(rows), table)
    lock = {
        "status": "corrections_materialized",
        "fingerprint": fingerprint,
        "manifest_sha256": sha256(table),
        "processed_positions": len(rows),
        **total,
        "training_authorized": False,
        "dependent_negative_overlay_status": "requires_corrected_slope_evidence",
        "acceptance_status": "incomplete",
    }
    lockpath = directory / "corrections.lock.json"
    if lockpath.exists() and json.loads(lockpath.read_text()) != lock:
        raise ValueError("published correction lock differs; use a new version")
    if not lockpath.exists():
        write_json(lockpath, lock)
    summary = {**lock, "output": str(directory), "finished_at": now()}
    write_json(progress, summary)
    return summary


class CorrectedDEMReader:
    """Read the corrected static target view, verifying baseline and sparse patch bytes."""

    def __init__(self, dataset_root: Path):
        self.directory = _directory(dataset_root)
        lock = json.loads((self.directory / "corrections.lock.json").read_text())
        self.fingerprint = lock["fingerprint"]
        manifest = self.directory / "manifest.parquet"
        if (
            lock.get("status") != "corrections_materialized"
            or sha256(manifest) != lock["manifest_sha256"]
        ):
            raise ValueError("DEM correction manifest is incomplete or changed")
        registry_path = dataset_root / "registry/national_62000.parquet"
        if sha256(registry_path) != self.fingerprint["registry_sha256"]:
            raise ValueError("correction registry changed")
        registry = pd.read_parquet(registry_path)
        self.rows = pd.read_parquet(manifest)
        if self.rows.patch_id.tolist() != registry.patch_id.tolist() or self.rows[
            "index"
        ].tolist() != list(range(len(registry))):
            raise ValueError("correction row order differs from registry")
        self.rows = self.rows.set_index("patch_id")
        self.baseline = Path(self.fingerprint["baseline_path"])
        self.root = zarr.open_group(str(self.baseline), mode="r")

    def read(self, patch_id: str) -> dict[str, np.ndarray]:
        if sha256(self.baseline / ".zattrs") != self.fingerprint["baseline_metadata_sha256"]:
            raise ValueError("baseline metadata changed")
        row = self.rows.loc[patch_id]
        original = {
            key: np.asarray(self.root[name][int(row["index"])]) for key, name in FIELDS.items()
        }
        if payload_sha256(original) != row.baseline_sha256:
            raise ValueError("baseline DEM pixels changed")
        if row.correction_file:
            path = (self.directory / row.correction_file).resolve()
            if (
                not path.is_relative_to(self.directory.resolve())
                or sha256(path) != row.correction_sha256
            ):
                raise ValueError("DEM correction file changed or escaped its directory")
            with np.load(path, allow_pickle=False) as saved:
                result = {key: saved[key] for key in saved.files}
        else:
            result = original
        validate_payload(result)
        return result
