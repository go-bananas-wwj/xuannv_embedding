"""Versioned Jilin cloud inference with explicit per-band validity and missing QA."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import zarr
from affine import Affine

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_partial_bands import read_jilin_branch
from xuannv_embedding.data_process.v5_quality import quality_masks, transfer_invalid
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

CLOUD_BANDS = ("B5", "B4", "B6")


def infer_scene(records: list[dict], predictor) -> dict:
    if not records or len({r["product_id"] for r in records}) != len(records):
        raise ValueError("one observation per native branch is required")
    for key in ["scene_group_id", "patch_id", "split", "year", "acquired_at", "sensor"]:
        if len({r[key] for r in records}) != 1:
            raise ValueError("same-scene branch metadata disagree")
    frames = {r["observation_id"]: read_jilin_branch(r) for r in records}
    optical = [r for r in records if r["product_id"] == "jilin1_ms_5m"]
    reason = "no_same_scene_5m_ms" if not optical else ""
    cloud = {
        "classes": np.full((256, 256), 255, "u1"),
        "confidence": np.zeros((256, 256), "f4"),
        **{
            name: np.zeros((256, 256), bool)
            for name in ["before_buffer", "cloud_buffered", "data_valid", "valid"]
        },
    }
    reference = None
    if optical:
        source = optical[0]
        reference = frames[source["observation_id"]]
        if not set(CLOUD_BANDS).issubset(source["band_ids"]):
            reason = "missing_B5_B4_B6"
        else:
            indices = [reference.band_ids.index(b) for b in CLOUD_BANDS]
            data_valid = reference.valid[indices].all(axis=0)
            cloud["data_valid"] = data_valid
            if not data_valid.any():
                reason = "no_valid_cloud_input_pixels"
            else:
                values = reference.values[indices].copy()
                values[:, ~data_valid] = 0
                predicted, confidence = predictor.predict_batch([values])
                predicted, confidence = np.asarray(predicted), np.asarray(confidence)
                if (
                    predicted.shape != (1, 256, 256)
                    or confidence.shape != predicted.shape
                    or not set(np.unique(predicted)).issubset({0, 1, 2, 3})
                    or not np.isfinite(confidence).all()
                    or np.any((confidence < 0) | (confidence > 1))
                ):
                    raise ValueError("invalid cloud prediction")
                qa = quality_masks(predicted[0], data_valid, gsd=5)
                cloud.update(
                    {
                        name: qa[name]
                        for name in ["before_buffer", "cloud_buffered", "data_valid", "valid"]
                    }
                )
                cloud["classes"] = predicted[0].astype("u1")
                cloud["confidence"] = confidence[0].astype("f4")
    status = "qa_missing" if reason else "model_inferred_needs_visual_review"
    branches = {}
    for record in records:
        native = frames[record["observation_id"]]
        clear = np.zeros(native.values.shape[1:], bool)
        if not reason:
            clear = ~transfer_invalid(
                ~cloud["valid"],
                src_transform=Affine(*reference.transform),
                src_crs=reference.crs,
                dst_transform=Affine(*native.transform),
                dst_crs=native.crs,
                shape=native.values.shape[1:],
            )
        valid = native.valid & clear[None]
        any_valid = valid.any(axis=0)
        row = {
            key: record[key]
            for key in [
                "observation_id",
                "scene_group_id",
                "patch_id",
                "year",
                "sensor",
                "split",
                "acquired_at",
                "product_id",
                "file_sha256",
            ]
        }
        row.update(
            quality_status=status,
            qa_missing_reason=reason,
            qa_source_observation_id=optical[0]["observation_id"] if optical else "",
            band_ids=list(native.band_ids),
            valid_pixels_by_band=[int(a.sum()) for a in valid],
            data_pixels_by_band=[int(a.sum()) for a in native.valid],
            clear_fraction_by_band=[float(a.mean()) for a in valid],
            valid_pixels=int(any_valid.sum()),
            clear_fraction=float(any_valid.mean()),
            strict_scene_qualified=float(any_valid.mean()) >= 0.6,
            available=bool(any_valid.any()),
            selected_bands_complete=set(native.band_ids).issubset(record["band_ids"]),
            pixel_fusion_authorized=False,
        )
        branches[record["observation_id"]] = {
            "row": row,
            "data_valid": native.valid,
            "valid": valid,
            "qa_clear": clear,
        }
    return {"quality_status": status, "reason": reason, "cloud": cloud, "branches": branches}


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    return value.item() if isinstance(value, np.generic) else value


def _digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True).encode()).hexdigest()


def _array_digest(array):
    a = np.ascontiguousarray(array)
    return _digest(
        {"shape": a.shape, "dtype": a.dtype.str, "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    )


def process_jilin_quality(
    source_root: Path,
    dataset_root: Path,
    report_root: Path,
    model_dir: Path,
    *,
    device_id=7,
    limit=None,
    predictor_factory=None,
):
    from xuannv_embedding.data_process.v5_followup import partial_catalog_finished

    if limit is not None and limit <= 0:
        raise ValueError("positive scene limit required")
    if not partial_catalog_finished(source_root, dataset_root, report_root):
        raise ValueError("current verified partial-band catalog is required")
    pointer_path = dataset_root / "observations/highres/jilin1/partial_bands/current.json"
    pointer = json.loads(pointer_path.read_text())
    catalog_lock = Path(pointer["lock_path"])
    catalog_path = catalog_lock.parent / "files_with_partial_bands.parquet"
    catalog = pd.read_parquet(catalog_path)
    if catalog.observation_id.duplicated().any():
        raise ValueError("duplicate quality observation identity")
    files = catalog.loc[catalog.year.isin([2020, 2021])]
    groups = sorted(files.scene_group_id.unique())
    if limit is not None:
        partial_groups = set(
            files.loc[files.metadata_status == "verified_present_bands", "scene_group_id"]
        )
        groups = sorted(groups, key=lambda group: (group not in partial_groups, group))[:limit]
    if not groups:
        raise ValueError("no annual scenes for cloud processing")
    model_paths = [model_dir / f"ocm_v4_model_{i}_96_910b4.om" for i in (0, 1)]
    configuration = {
        "models_sha256": {p.name: sha256(p) for p in model_paths},
        "cloud_bands": list(CLOUD_BANDS),
        "native_gsd": 5,
        "buffer_m": 30,
        "strict_threshold": 0.6,
        "validity": "per_band",
        "missing_class": 255,
        "runtime": {
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "gdal": rasterio.__gdal_version__,
            "zarr": zarr.__version__,
            "pandas": pd.__version__,
            "pyarrow": package_version("pyarrow"),
        },
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in [
                "v5_jilin_quality.py",
                "v5_partial_bands.py",
                "v5_rasters.py",
                "v5_quality.py",
                "omnicloudmask_npu.py",
            ]
        },
    }
    config_id = _digest(configuration)[:20]
    root = dataset_root / "quality/cloud/jilin1/v2" / config_id
    root.mkdir(parents=True, exist_ok=True)
    configuration_path = root / "configuration.lock.json"
    if configuration_path.exists() and json.loads(configuration_path.read_text()) != configuration:
        raise ValueError("quality configuration changed")
    if not configuration_path.exists():
        write_json(configuration_path, configuration)
    snapshot = {
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256(catalog_path),
        "catalog_lock_sha256": sha256(catalog_lock),
        "configuration_id": config_id,
        "years": [2020, 2021],
        "limit": limit,
    }
    snapshot_id = _digest(snapshot)[:20]
    output = root / "catalogs" / snapshot_id
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "input.lock.json", snapshot)
    classes = zarr.open_group(str(root / "classes.zarr"), mode="a")
    masks = zarr.open_group(str(root / "valid_masks.zarr"), mode="a")
    classes.attrs.update(
        class_codes={"clear": 0, "thick_cloud": 1, "thin_cloud": 2, "shadow": 3, "unknown_qa": 255},
        configuration_id=config_id,
    )
    masks.attrs.update(validity="per_band", configuration_id=config_id)
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    factory = predictor_factory
    if factory is None:
        from xuannv_embedding.data_process.omnicloudmask_npu import AscendOmniCloudMaskV4Predictor

        factory = AscendOmniCloudMaskV4Predictor
    predictor = None
    rows = []
    receipt_hashes = {}
    reused = 0
    counts = {}
    grouped = files.groupby("scene_group_id")
    progress = report_root / (
        "jilin_quality_v2_full.json" if limit is None else f"jilin_quality_v2_pilot_{limit}.json"
    )

    def save_array(group, name, array):
        arr = group.require_dataset(
            name, shape=array.shape, chunks=array.shape, dtype=array.dtype, compressor=compressor
        )
        arr[:] = array
        return _array_digest(array)

    try:
        for position, scene in enumerate(groups):
            records = _jsonable(
                grouped.get_group(scene).sort_values("product_id").to_dict("records")
            )
            # Names are used as storage keys, never accept paths supplied as IDs.
            if (
                any("/" in r["observation_id"] or "\\" in r["observation_id"] for r in records)
                or "/" in scene
                or "\\" in scene
            ):
                raise ValueError("invalid observation storage identity")
            for record in records:
                if sha256(Path(record["path"])) != record["file_sha256"]:
                    raise ValueError("Jilin source changed before quality cache lookup")
            fingerprint = _digest({"configuration_id": config_id, "records": records})
            receipt_path = root / "receipts" / f"{scene}.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["fingerprint"] != fingerprint:
                    raise ValueError("same scene quality inputs changed; use a new data version")
                for path, expected in receipt["cloud_arrays"].items():
                    if _array_digest(np.asarray(classes[path])) != expected:
                        raise ValueError("cached cloud array changed")
                for path, expected in receipt["mask_arrays"].items():
                    if _array_digest(np.asarray(masks[path])) != expected:
                        raise ValueError("cached per-band quality mask changed")
                reused += 1
            else:
                if predictor is None:
                    predictor = factory(model_paths=model_paths, device_id=device_id)
                result = infer_scene(records, predictor)
                cloud_hashes = {
                    f"{scene}/{name}": save_array(classes, f"{scene}/{name}", array)
                    for name, array in result["cloud"].items()
                }
                mask_hashes = {}
                for observation, branch in result["branches"].items():
                    for name in ["data_valid", "valid", "qa_clear"]:
                        path = f"{observation}/{name}"
                        mask_hashes[path] = save_array(masks, path, branch[name])
                receipt = {
                    "fingerprint": fingerprint,
                    "quality_status": result["quality_status"],
                    "reason": result["reason"],
                    "rows": [b["row"] for b in result["branches"].values()],
                    "cloud_arrays": cloud_hashes,
                    "mask_arrays": mask_hashes,
                }
                write_json(receipt_path, receipt)
            rows.extend(receipt["rows"])
            receipt_hashes[scene] = sha256(receipt_path)
            status = receipt["quality_status"]
            counts[status] = counts.get(status, 0) + 1
            if (position + 1) % 32 == 0:
                write_json(
                    progress,
                    {
                        "status": "running",
                        "processed_scenes": position + 1,
                        "selected_scenes": len(groups),
                        "reused_scenes": reused,
                        "counts": counts,
                        "output": str(output),
                        "updated_at": now(),
                        "training_authorized": False,
                    },
                )
    finally:
        if predictor is not None:
            predictor.close()
    if (
        sha256(catalog_path) != snapshot["catalog_sha256"]
        or sha256(catalog_lock) != snapshot["catalog_lock_sha256"]
    ):
        raise ValueError("quality catalog snapshot changed during processing")
    table = output / "observation_quality.parquet"
    atomic_parquet(pd.DataFrame(rows), table)
    locked = {
        "snapshot": snapshot,
        "processed_scenes": len(groups),
        "branches": len(rows),
        "counts": counts,
        "receipts_sha256": receipt_hashes,
        "quality_table_sha256": sha256(table),
        "pixel_fusion_authorized": False,
        "training_authorized": False,
    }
    lock = output / "quality.lock.json"
    if lock.exists() and json.loads(lock.read_text()) != locked:
        raise ValueError("published quality snapshot changed")
    if not lock.exists():
        write_json(lock, locked)
    result = {
        "status": "inferred_needs_visual_review",
        "selected_scenes": len(groups),
        "processed_scenes": len(groups),
        "reused_scenes": reused,
        "counts": counts,
        "branches": len(rows),
        "output": str(output),
        "quality_root": str(root),
        "quality_lock_sha256": sha256(lock),
        "catalog_snapshot_id": snapshot_id,
        "partial_catalog_version": pointer["version"],
        "scope": "frozen_available_catalog_2020_2021",
        "acceptance_passed": False,
        "training_authorized": False,
        "finished_at": now(),
    }
    write_json(progress, result)
    return result
