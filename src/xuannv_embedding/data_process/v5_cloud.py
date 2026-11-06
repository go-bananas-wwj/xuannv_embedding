"""Frozen-model Jilin cloud inference; records QA without granting data acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from affine import Affine

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_quality import quality_masks, transfer_invalid
from xuannv_embedding.data_process.v5_rasters import BRANCH_BANDS, read_native
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def process_jilin_cloud(
    dataset_root: Path,
    report_root: Path,
    model_dir: Path,
    *,
    device_id: int = 1,
    limit: int | None = None,
    predictor_factory=None,
) -> dict:
    from xuannv_embedding.data_process.omnicloudmask_npu import AscendOmniCloudMaskV4Predictor

    catalog_path = dataset_root / "observations/highres/jilin1/files.parquet"
    files = pd.read_parquet(catalog_path)
    files = files.loc[files.year.isin([2020, 2021])]
    selected = files.loc[files.product_id == "jilin1_ms_5m"].sort_values("observation_id")
    if limit is not None:
        selected = selected.head(limit)
    if selected.empty:
        raise ValueError("no verified 2020/2021 Jilin multispectral observations")
    paths = [model_dir / f"ocm_v4_model_{i}_96_910b4.om" for i in (0, 1)]
    fingerprint = {
        "catalog_sha256": sha256(catalog_path),
        "models_sha256": {path.name: sha256(path) for path in paths},
        "input_bands": ["B5", "B4", "B6"],
        "cloud_buffer_m": 30,
        "mask_policy": "valid_pixels",
        "strict_threshold": 0.6,
        "limit": limit,
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ["v5_cloud.py", "v5_quality.py", "v5_rasters.py", "omnicloudmask_npu.py"]
        },
    }
    root = dataset_root / "quality/cloud/jilin1"
    if limit is not None:
        root = root / f"pilot_{limit}"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "source.lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != fingerprint:
        raise ValueError("cloud source fingerprint changed; choose a separate QA version")
    write_json(lock_path, fingerprint)
    selected_ids = selected.observation_id.tolist()
    atomic_parquet(
        selected[["observation_id", "scene_group_id"]], root / "observation_order.parquet"
    )
    classes = zarr.open_group(str(root / "classes.zarr"), mode="a")
    masks = zarr.open_group(str(root / "valid_masks.zarr"), mode="a")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    count = len(selected)
    for name, dtype in [("classes", "u1"), ("confidence", "f4")]:
        classes.require_dataset(
            name, shape=(count, 256, 256), chunks=(1, 256, 256), dtype=dtype, compressor=compressor
        )
    for name in ["before_buffer", "cloud_buffered", "data_valid", "valid"]:
        masks.require_dataset(
            name, shape=(count, 256, 256), chunks=(1, 256, 256), dtype="bool", compressor=compressor
        )
    completed = masks.require_dataset("completed", shape=(count,), chunks=(1,), dtype="bool")
    classes.attrs.update(fingerprint)
    masks.attrs.update({"mask_semantics": "pixel_validity_not_scene_threshold", **fingerprint})
    rows_path = root / "observation_quality.parquet"
    rows = pd.read_parquet(rows_path).to_dict("records") if rows_path.exists() else []
    by_id = {row["observation_id"]: row for row in rows}
    branch_groups = {group_id: group for group_id, group in files.groupby("scene_group_id")}
    factory = predictor_factory or AscendOmniCloudMaskV4Predictor
    predictor = factory(model_paths=paths, device_id=device_id)
    try:
        for position, (_, row) in enumerate(selected.iterrows()):
            if bool(completed[position]) and row.observation_id in by_id:
                continue
            frame = read_native(Path(row.path), BRANCH_BANDS["jilin1_ms_5m"])
            rgb = frame.values[[4, 3, 5]].copy()
            valid = np.all(frame.valid, axis=0)
            rgb[:, ~valid] = 0
            predicted, confidence = predictor.predict_batch([rgb])
            qa = quality_masks(predicted[0], valid, gsd=5)
            classes["classes"][position] = predicted[0]
            classes["confidence"][position] = confidence[0]
            for name in ["before_buffer", "cloud_buffered", "data_valid", "valid"]:
                masks[name][position] = qa[name]
            base = {
                "scene_group_id": row.scene_group_id,
                "patch_id": row.patch_id,
                "year": int(row.year),
                "sensor": row.sensor,
                "split": row.split,
                "acquired_at": row.acquired_at,
                "quality_status": "model_inferred_needs_visual_review",
            }
            for branch in branch_groups[row.scene_group_id].itertuples():
                native = read_native(Path(branch.path), BRANCH_BANDS[branch.product_id])
                invalid = transfer_invalid(
                    ~qa["valid"],
                    src_transform=Affine(*frame.transform),
                    src_crs=frame.crs,
                    dst_transform=Affine(*native.transform),
                    dst_crs=native.crs,
                    shape=native.values.shape[1:],
                )
                final = np.all(native.valid, axis=0) & ~invalid
                branch_root = zarr.open_group(str(root / "branches.zarr"), mode="a")
                # Stable observation IDs avoid dependence on changing DataFrame positions.
                branch_root.require_dataset(
                    branch.observation_id,
                    shape=final.shape,
                    chunks=final.shape,
                    dtype="bool",
                    compressor=compressor,
                )[:] = final
                fraction = float(final.mean())
                by_id[branch.observation_id] = {
                    **base,
                    "observation_id": branch.observation_id,
                    "product_id": branch.product_id,
                    "valid_pixels": int(final.sum()),
                    "clear_fraction": fraction,
                    "strict_scene_qualified": fraction >= 0.6,
                    "available": bool(final.any()),
                }
            completed[position] = True
            if (position + 1) % 100 == 0:
                atomic_parquet(pd.DataFrame(by_id.values()), rows_path)
                print(
                    json.dumps({"cloud_scenes": position + 1, "selected_scenes": count}), flush=True
                )
    finally:
        atomic_parquet(pd.DataFrame(by_id.values()), rows_path)
        predictor.close()
    result = {
        "status": "inferred_needs_visual_review",
        "selected_scenes": len(selected_ids),
        "processed_scenes": int(np.asarray(completed[:]).sum()),
        "branches": len(by_id),
        "scope": "selected_2020_2021_scenes",
        "acceptance_passed": False,
        "finished_at": now(),
    }
    write_json(report_root / "jilin_cloud_summary.json", result)
    return result
