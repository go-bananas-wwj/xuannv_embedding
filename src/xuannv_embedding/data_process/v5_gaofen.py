"""Recompute native Gaofen QA over the unfiltered scene inventory."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import zarr
from affine import Affine

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_quality import quality_masks, transfer_invalid
from xuannv_embedding.data_process.v5_rasters import read_native
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

SENSORS = {"GF1", "GF1B", "GF1C", "GF1D", "GF6"}


def process_gaofen(
    dataset_root: Path,
    report_root: Path,
    source_catalog: Path,
    model_dir: Path,
    *,
    device_id: int = 1,
    predictor_factory=None,
    limit=None,
) -> dict:
    from xuannv_embedding.data_process.omnicloudmask_npu import AscendOmniCloudMaskV4Predictor

    registry_path = dataset_root / "registry/national_62000.parquet"
    registry = pd.read_parquet(registry_path).set_index("patch_id")
    sources = pd.read_parquet(source_catalog).sort_values("pair_id").reset_index(drop=True)
    if sources.pair_id.duplicated().any():
        raise ValueError("duplicate Gaofen scene group")
    sources = sources.loc[
        sources.acquired_at.map(lambda v: datetime.fromisoformat(v).year).isin([2020, 2021])
    ]
    if limit:
        sources = sources.head(limit)
    sources = sources.reset_index(drop=True)
    if sources.empty:
        raise ValueError("no 2020/2021 Gaofen sources")
    models = [model_dir / f"ocm_v4_model_{i}_96_910b4.om" for i in (0, 1)]
    fingerprint = {
        "source_catalog_sha256": sha256(source_catalog),
        "registry_sha256": sha256(registry_path),
        "limit": limit,
        "model_sha256": {p.name: sha256(p) for p in models},
        "code_sha256": {
            n: sha256(Path(__file__).with_name(n))
            for n in ["v5_gaofen.py", "v5_quality.py", "v5_rasters.py", "omnicloudmask_npu.py"]
        },
        "input_bands": ["red", "green", "nir"],
        "radiometry": "native_stored_dn",
        "buffer_m": 30,
        "strict_threshold": 0.6,
    }
    root = dataset_root / "quality/cloud/gaofen"
    if limit:
        root = root / f"pilot_{limit}"
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "source.lock.json"
    if lock.exists() and json.loads(lock.read_text()) != fingerprint:
        raise ValueError("Gaofen QA source changed; a new version is required")
    write_json(lock, fingerprint)
    atomic_parquet(
        sources[["pair_id", "patch_id", "sensor", "acquired_at"]],
        root / "observation_order.parquet",
    )
    masks = zarr.open_group(str(root / "valid_masks.zarr"), mode="a")
    classes = zarr.open_group(str(root / "classes.zarr"), mode="a")
    n = len(sources)
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    for name in ["classes", "confidence_uint8"]:
        classes.require_dataset(
            name, shape=(n, 160, 160), chunks=(16, 160, 160), dtype="u1", compressor=compressor
        )
    shapes = {
        "data_valid_packed": (160, 20),
        "before_buffer_packed": (160, 20),
        "ms_valid_packed": (160, 20),
        "pan_valid_packed": (640, 80),
    }
    for name, shape in shapes.items():
        masks.require_dataset(
            name, shape=(n, *shape), chunks=(16, *shape), dtype="u1", compressor=compressor
        )
    completed = masks.require_dataset("completed", shape=(n,), chunks=(16,), dtype="bool")
    classes.attrs.update(fingerprint)
    masks.attrs.update({"packed_axis": -1, "bitorder": "little", **fingerprint})
    quality_path = root / "observation_quality.parquet"
    rows = pd.read_parquet(quality_path).to_dict("records") if quality_path.exists() else []
    by_id = {r["pair_id"]: r for r in rows}
    errors_path = root / "rejected_observations.parquet"
    errors = pd.read_parquet(errors_path).to_dict("records") if errors_path.exists() else []
    rejected = {r["pair_id"]: r for r in errors}
    factory = predictor_factory or AscendOmniCloudMaskV4Predictor
    predictor = factory(model_paths=models, device_id=device_id)
    try:
        for start in range(0, n, 16):
            batch = sources.iloc[start : start + 16]
            if all(bool(completed[i]) and r.pair_id in by_id for i, r in batch.iterrows()):
                continue
            output = {
                name: np.zeros((len(batch), *shape), dtype="u1") for name, shape in shapes.items()
            }
            labels = np.zeros((len(batch), 160, 160), dtype="u1")
            confidence = np.zeros_like(labels)
            done = np.zeros(len(batch), dtype=bool)
            for offset, (_, row) in enumerate(batch.iterrows()):
                try:
                    if row.sensor not in SENSORS or row.patch_id not in registry.index:
                        raise ValueError("unverified Gaofen sensor or grid owner")
                    owner = registry.loc[row.patch_id]
                    ms_path, pan_path = Path(row.ms_path), Path(row.pan_path)
                    for path, expected_shape, expected_gsd in [
                        (ms_path, (4, 160, 160), 8),
                        (pan_path, (1, 640, 640), 2),
                    ]:
                        with rasterio.open(path) as ds:
                            if (
                                (ds.count, ds.height, ds.width) != expected_shape
                                or ds.crs.to_epsg() != int(owner.grid_epsg)
                                or not np.allclose(ds.bounds, owner.utm_bounds, rtol=0, atol=0.001)
                                or not np.allclose(ds.res, expected_gsd)
                                or set(ds.dtypes) != {"uint16"}
                                or ds.nodata != 0
                            ):
                                raise ValueError(
                                    "Gaofen native grid or stored-DN contract disagrees"
                                )
                    bands = ["blue", "green", "red", "nir"]
                    ms = read_native(
                        ms_path,
                        bands,
                        contract={
                            "verified": True,
                            "band_ids": bands,
                            "scales": [1] * 4,
                            "offsets": [0] * 4,
                        },
                    )
                    pan = read_native(
                        pan_path,
                        ["pan"],
                        contract={
                            "verified": True,
                            "band_ids": ["pan"],
                            "scales": [1],
                            "offsets": [0],
                        },
                    )
                    ms_valid = np.all(ms.valid, axis=0)
                    rgb = ms.values[[2, 1, 3]].copy()
                    rgb[:, ~ms_valid] = 0
                    prediction, certainty = predictor.predict_batch([rgb])
                    qa = quality_masks(prediction[0], ms_valid, gsd=8)
                    pan_bad = transfer_invalid(
                        ~qa["valid"],
                        src_transform=Affine(*ms.transform),
                        src_crs=ms.crs,
                        dst_transform=Affine(*pan.transform),
                        dst_crs=pan.crs,
                        shape=pan.values.shape[1:],
                    )
                    pan_valid = pan.valid[0] & ~pan_bad
                    for name, value in [
                        ("data_valid_packed", ms_valid),
                        ("before_buffer_packed", qa["before_buffer"]),
                        ("ms_valid_packed", qa["valid"]),
                        ("pan_valid_packed", pan_valid),
                    ]:
                        output[name][offset] = np.packbits(value, axis=-1, bitorder="little")
                    labels[offset] = prediction[0]
                    confidence[offset] = np.round(np.clip(certainty[0], 0, 1) * 255).astype("u1")
                    by_id[row.pair_id] = {
                        "pair_id": row.pair_id,
                        "patch_id": row.patch_id,
                        "sensor": row.sensor,
                        "split": owner.split,
                        "acquired_at": row.acquired_at,
                        "year": datetime.fromisoformat(row.acquired_at).year,
                        "ms_path": str(ms_path),
                        "pan_path": str(pan_path),
                        "ms_sha256": sha256(ms_path),
                        "pan_sha256": sha256(pan_path),
                        "clear_fraction": qa["clear_fraction"],
                        "strict_scene_qualified": qa["strict_scene_qualified"],
                        "available": qa["available"],
                        "ms_valid_pixels": int(qa["valid"].sum()),
                        "pan_valid_pixels": int(pan_valid.sum()),
                        "quality_status": "recomputed_needs_visual_review",
                    }
                    rejected.pop(row.pair_id, None)
                    done[offset] = True
                except (ValueError, OSError) as exc:
                    by_id.pop(row.pair_id, None)
                    rejected[row.pair_id] = {
                        "pair_id": row.pair_id,
                        "reason": str(exc),
                        "error_type": type(exc).__name__,
                    }
            stop = start + len(batch)
            classes["classes"][start:stop] = labels
            classes["confidence_uint8"][start:stop] = confidence
            for name in shapes:
                masks[name][start:stop] = output[name]
            completed[start:stop] = done
            if start % 256 == 0:
                atomic_parquet(pd.DataFrame(by_id.values()), quality_path)
                atomic_parquet(
                    pd.DataFrame(rejected.values(), columns=["pair_id", "reason", "error_type"]),
                    errors_path,
                )
                progress = {
                    "processed_scenes": len(by_id),
                    "rejected_scenes": len(rejected),
                    "selected_scenes": n,
                    "status": "running",
                    "updated_at": now(),
                }
                write_json(report_root / "gaofen_cloud_summary.json", progress)
                print(json.dumps(progress), flush=True)
    finally:
        atomic_parquet(pd.DataFrame(by_id.values()), quality_path)
        atomic_parquet(
            pd.DataFrame(rejected.values(), columns=["pair_id", "reason", "error_type"]),
            errors_path,
        )
        predictor.close()
    result = {
        "processed_scenes": len(by_id),
        "rejected_scenes": len(rejected),
        "selected_scenes": n,
        "status": "inferred_needs_visual_review",
        "acceptance_passed": False,
        "finished_at": now(),
    }
    write_json(report_root / "gaofen_cloud_summary.json", result)
    return result
