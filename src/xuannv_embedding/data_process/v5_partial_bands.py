"""Recover explicitly named Jilin channels while keeping absent bands invalid."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from xuannv_embedding.data_process.v5_catalog import grid_lookup
from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_rasters import (
    BRANCH_BANDS,
    NativeRaster,
    band_metadata,
    read_native,
)
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def inspect_partial_jilin(path: Path) -> dict:
    with rasterio.open(path) as source:
        names, wavelengths = band_metadata(source.descriptions)
        tags = source.tags()
        gsd = float(source.res[0])
        if (
            source.crs is None
            or gsd not in [5, 10, 20]
            or source.res[1] != gsd
            or source.transform.a != gsd
            or source.transform.e != -gsd
            or source.transform.b != 0
            or source.transform.d != 0
            or source.width * gsd != 1280
            or source.height * gsd != 1280
        ):
            raise ValueError("unverified partial Jilin grid")
        expected = tuple(f"B{i}" for i in range(1, {5: 6, 10: 12, 20: 19}[int(gsd)] + 1))
        if not names or not set(names) < set(expected):
            raise ValueError("not a verified subset of native Jilin band identities")
        if (
            set(source.dtypes) != {"int16"}
            or source.nodata != -28672
            or not np.allclose(source.scales, 0.0001, rtol=0, atol=1e-12)
            or not np.allclose(source.offsets, 0, rtol=0, atol=1e-12)
            or tags.get("units") != "reflectance"
            or not np.isfinite(wavelengths).all()
            or min(wavelengths) <= 0
        ):
            raise ValueError("unverified partial Jilin radiometry")
        raw_time = tags.get("acquisition_time", "")
        acquired = datetime.fromisoformat(raw_time)
        if not all(tags.get(k) for k in ["source_product", "source_signature", "patch_id"]):
            raise ValueError("missing partial scene provenance")
        product = {5: "jilin1_ms_5m", 10: "jilin1_extra_10m", 20: "jilin1_extra_20m"}[int(gsd)]
        selected = BRANCH_BANDS[product]
        present = [b for b in selected if b in names]
        return {
            "path": str(path.resolve()),
            "file_sha256": sha256(path),
            "source_patch_id": tags["patch_id"],
            "sensor": path.parent.name,
            "scene_id": tags["source_product"],
            "source_signature": tags["source_signature"],
            "acquired_at": acquired.isoformat(),
            "acquired_at_raw": raw_time,
            "year": acquired.year,
            "time_precision": (
                "second" if raw_time.count(":") >= 2 else "day" if len(raw_time) == 10 else "minute"
            ),
            "time_zone": "unspecified" if acquired.tzinfo is None else str(acquired.tzinfo),
            "product_id": product,
            "crs": str(source.crs),
            "epsg": source.crs.to_epsg(),
            "transform": list(source.transform)[:6],
            "bounds": list(source.bounds),
            "band_ids": list(names),
            "wavelengths": list(wavelengths),
            "selected_band_ids": list(selected),
            "present_selected_band_ids": present,
            "missing_native_band_ids": [b for b in expected if b not in names],
            "missing_selected_band_ids": [b for b in selected if b not in names],
            "selected_bands_complete": len(present) == len(selected),
            "branch_has_selected_bands": bool(present),
            "cloud_input_band_ids_available": {"B5", "B4", "B6"}.issubset(names),
            "shape": [source.count, source.height, source.width],
            "dtype": source.dtypes[0],
            "native_gsd": gsd,
            "stored_gsd": gsd,
            "nodata": source.nodata,
            "scale": list(source.scales),
            "offset": list(source.offsets),
            "metadata_status": "verified_present_bands",
            "quality_status": "pending",
            "pixel_fusion_authorized": False,
        }


def read_jilin_branch(record: dict, *, quality=None, mean=None, std=None) -> NativeRaster:
    path = Path(record["path"])
    if sha256(path) != record["file_sha256"]:
        raise ValueError("Jilin source changed after contract audit")
    if record.get("metadata_status") not in ["verified", "verified_present_bands"]:
        raise ValueError("unverified Jilin branch")
    bands = BRANCH_BANDS[record["product_id"]]
    if tuple(record["selected_band_ids"]) != bands:
        raise ValueError("noncanonical Jilin branch bands")
    present = tuple(b for b in bands if b in record["band_ids"])
    positions = [bands.index(b) for b in present]
    shape = (len(bands), int(record["shape"][1]), int(record["shape"][2]))
    selected_quality = quality
    if quality is not None:
        quality = np.asarray(quality)
        if quality.dtype != bool or quality.shape not in [shape, shape[1:]]:
            raise ValueError("invalid canonical branch quality mask")
        selected_quality = quality[positions] if quality.ndim == 3 else quality
    if (mean is None) != (std is None):
        raise ValueError("both normalization statistics are required")
    if mean is not None:
        mean, std = np.asarray(mean, dtype="f4"), np.asarray(std, dtype="f4")
        if (
            mean.shape != (len(bands),)
            or std.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError("invalid canonical branch statistics")
    values = np.zeros(shape, "f4")
    valid = np.zeros(shape, bool)
    if present:
        native = read_native(
            path,
            present,
            quality=selected_quality,
            mean=None if mean is None else mean[positions],
            std=None if std is None else std[positions],
        )
        if (
            native.values.shape[1:] != shape[1:]
            or native.crs != record["crs"]
            or tuple(native.transform) != tuple(record["transform"])
        ):
            raise ValueError("Jilin native grid changed")
        values[positions] = native.values
        valid[positions] = native.valid
    return NativeRaster(values, valid, bands, tuple(record["transform"]), record["crs"])


def catalog_partial_bands(source_root: Path, dataset_root: Path, report_root: Path) -> dict:
    registry_path = dataset_root / "registry/national_62000.parquet"
    rejected_path = report_root / "rejected_files.parquet"
    complete_path = dataset_root / "observations/highres/jilin1/files.parquet"
    source_path = source_root / "manifests/source.lock.json"
    index_path = source_root / "manifests/ARCHIVE_INDEX.tsv"
    rejection_sha = sha256(rejected_path)
    complete_sha = sha256(complete_path)
    rejected = pd.read_parquet(rejected_path)
    complete = pd.read_parquet(complete_path)
    selected = rejected.loc[rejected.reason == "unverified Jilin band contract"].drop_duplicates(
        "path"
    )
    source = json.loads(source_path.read_text())
    index = pd.read_csv(index_path, sep="\t")
    if sha256(index_path) != source["manifest_sha256"]["ARCHIVE_INDEX.tsv"]:
        raise ValueError("published source patch index changed")
    if index.patchid.duplicated().any():
        raise ValueError("ambiguous source patch archive")
    archives = index.set_index("patchid").archive.to_dict()
    specs = {s["archive"]: s for s in source["archives"]}
    downloads = pd.read_parquet(source_root / "manifests/download_status.parquet").set_index(
        "archive"
    )
    lookup = grid_lookup(pd.read_parquet(registry_path))
    package_locks = {}
    candidate_hashes = {}
    for name in selected.path:
        path = Path(name).resolve()
        if not path.is_relative_to((source_root / "extracted").resolve()):
            raise ValueError("partial file outside extracted source")
        relative = path.relative_to((source_root / "extracted").resolve())
        if len(relative.parts) != 3 or relative.parts[0] not in archives:
            raise ValueError("unmapped extracted partial source")
        archive = archives[relative.parts[0]]
        if archive not in package_locks:
            spec = specs[archive]
            download = downloads.loc[archive]
            integrity_path = report_root / "integrity_shards" / (archive + ".json")
            extracted_path = source_root / "manifests/extracted" / (archive + ".json")
            integrity = json.loads(integrity_path.read_text())
            extracted = json.loads(extracted_path.read_text())
            if (
                download.status != "complete"
                or download.actual_bytes != spec["bytes"]
                or download.sha256 != spec["sha256"]
                or extracted.get("status") != "complete"
                or extracted.get("sha256") != spec["sha256"]
                or integrity.get("status") != "complete"
                or integrity.get("sha256") != spec["sha256"]
                or integrity.get("decoded_tiffs") != spec["tiff_count"]
                or integrity.get("failures") != []
            ):
                raise ValueError("partial channels require a decoded, verified source package")
            package = source_root / "packages" / archive
            if package.stat().st_size != spec["bytes"] or sha256(package) != spec["sha256"]:
                raise ValueError("source package changed before partial band catalog")
            package_locks[archive] = {
                "sha256": spec["sha256"],
                "integrity_sha256": sha256(integrity_path),
                "extracted_sha256": sha256(extracted_path),
            }
        candidate_hashes[str(path)] = sha256(path)
    fingerprint = {
        "registry_sha256": sha256(registry_path),
        "rejected_inventory_sha256": rejection_sha,
        "complete_catalog_sha256": complete_sha,
        "source_lock_sha256": sha256(source_path),
        "code_sha256": sha256(Path(__file__)),
        "reader_sha256": sha256(Path(__file__).with_name("v5_rasters.py")),
        "grid_matcher_sha256": sha256(Path(__file__).with_name("v5_catalog.py")),
        "source_packages": package_locks,
        "candidate_file_sha256": candidate_hashes,
        "runtime": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rasterio": rasterio.__version__,
            "gdal": rasterio.__gdal_version__,
            "pyarrow": package_version("pyarrow"),
        },
    }
    version = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
    directory = dataset_root / "observations/highres/jilin1/partial_bands" / version
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    complete_ids = set(complete.observation_id)
    for name in selected.path:
        try:
            record = inspect_partial_jilin(Path(name))
            path = Path(record["path"])
            if record["file_sha256"] != candidate_hashes[str(path)]:
                raise ValueError("partial source changed during catalog")
            source_id = path.relative_to((source_root / "extracted").resolve()).parts[0]
            if record["source_patch_id"] != source_id:
                raise ValueError("partial source patch tag differs from published package index")
            key = (record["epsg"], *(round(v, 3) for v in record["bounds"]))
            if key not in lookup:
                raise ValueError("no unique national grid match for partial bands")
            record["patch_id"], record["split"] = lookup[key]
            group = "|".join([record["patch_id"], record["sensor"], record["scene_id"]])
            record["scene_group_id"] = hashlib.sha256(group.encode()).hexdigest()
            record["observation_id"] = record["scene_group_id"] + ":" + record["product_id"]
            if record["observation_id"] in complete_ids:
                raise ValueError("partial branch conflicts with a complete observation")
            native = read_jilin_branch(record)
            record.update(
                source_archive=archives[source_id],
                source_revision=source["revision"],
                valid_pixels_by_band=[int(a.sum()) for a in native.valid],
                training_year_view=record["year"] in [2020, 2021],
            )
            rows.append(record)
        except (ValueError, OSError) as exc:
            failures.append(
                {"path": str(name), "reason": str(exc), "error_type": type(exc).__name__}
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        conflict = frame.observation_id.duplicated(keep=False)
        failures.extend(
            {
                "path": p,
                "reason": "duplicate partial observation identity",
                "error_type": "Conflict",
            }
            for p in frame.loc[conflict, "path"]
        )
        frame = frame.loc[~conflict].copy()
    if sha256(rejected_path) != rejection_sha or sha256(complete_path) != complete_sha:
        raise ValueError(
            "catalog advanced during partial band processing; retry against the new snapshot"
        )
    atomic_parquet(
        (
            frame
            if not frame.empty
            else pd.DataFrame(
                columns=["path", "observation_id", "product_id", "patch_id", "year", "split"]
            )
        ),
        directory / "files.parquet",
    )
    atomic_parquet(
        pd.DataFrame(failures, columns=["path", "reason", "error_type"]),
        directory / "rejected.parquet",
    )
    normalized_complete = complete.copy()
    if not normalized_complete.empty:
        normalized_complete["present_selected_band_ids"] = complete.selected_band_ids.map(list)
        for field in ["missing_native_band_ids", "missing_selected_band_ids"]:
            normalized_complete[field] = [[] for _ in range(len(complete))]
        normalized_complete["selected_bands_complete"] = True
        normalized_complete["branch_has_selected_bands"] = True
        normalized_complete["cloud_input_band_ids_available"] = complete.band_ids.map(
            lambda bands: {"B5", "B4", "B6"}.issubset(bands)
        )
        normalized_complete["training_year_view"] = complete.year.isin([2020, 2021])
        normalized_complete["quality_status"] = "pending"
        normalized_complete["pixel_fusion_authorized"] = False
    augmented = pd.concat([normalized_complete, frame], ignore_index=True)
    atomic_parquet(augmented, directory / "files_with_partial_bands.parquet")
    summary = {
        "status": "partial_band_catalog_finished",
        "scope": "current_verified_catalog_rejections",
        "fingerprint": fingerprint,
        "selected_files": len(selected),
        "verified_partial_files": len(frame),
        "rejected_files": len(failures),
        "selected_bands_complete_files": (
            int(frame.selected_bands_complete.sum()) if not frame.empty else 0
        ),
        "selected_bands_missing_files": (
            int((~frame.selected_bands_complete).sum()) if not frame.empty else 0
        ),
        "training_year_files": int(frame.training_year_view.sum()) if not frame.empty else 0,
        "cloud_inputs_missing_files": (
            int((~frame.cloud_input_band_ids_available).sum()) if not frame.empty else 0
        ),
        "output": str(directory),
        "output_sha256": sha256(directory / "files.parquet"),
        "augmented_catalog_sha256": sha256(directory / "files_with_partial_bands.parquet"),
        "quality_status": "pending",
        "pixel_fusion_authorized": False,
        "training_authorized": False,
        "finished_at": now(),
    }
    lock_path = directory / "catalog.lock.json"
    locked = {key: value for key, value in summary.items() if key != "finished_at"}
    if lock_path.exists() and json.loads(lock_path.read_text()) != locked:
        raise ValueError("published partial catalog changed; use a new version")
    if not lock_path.exists():
        write_json(lock_path, locked)
    pointer = {
        "version": version,
        "lock_path": str(lock_path),
        "lock_sha256": sha256(lock_path),
    }
    current = directory.parent / "current.json"
    if not current.exists() or json.loads(current.read_text()) != pointer:
        write_json(current, pointer)
    write_json(report_root / "partial_band_catalog.json", summary)
    return summary
