"""Sequentially repack immutable local ZIP observations into a smoke-only Zarr cache."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import zarr

from xuannv_embedding.config import V2Config
from xuannv_embedding.data.v2_dataset import V2LocalZipDataset


def build_smoke_zarr_cache(
    config: V2Config,
    registry_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 Zarr cache: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    dataset = V2LocalZipDataset(
        config,
        registry_path,
        spatial_size=128,
        output_selection="all",
        include_targets=False,
        normalize=False,
    )
    try:
        root = zarr.open_group(str(temporary), mode="w")
        first = dataset[0]
        intervals = first["model_inputs"]["output_intervals"].numpy()
        rows = dataset.observations[(first["patch_id"], dataset.dense_products[0])]
        root.attrs.update(
            {
                "schema_version": "xuannv_v2_smoke_dense_cache_v1",
                "source": "local_zip_repack",
                "network_remote_pixels": False,
                "source_archive_lock_sha256": hashlib.sha256(
                    (config.paths.data_root / "locks" / "local_archive_sha256.jsonl").read_bytes()
                ).hexdigest(),
                "patch_ids": [str(row["patch_id"]) for row in dataset.records],
                "years": [int(row["year"]) for row in rows],
                "months": [int(row["month"]) for row in rows],
                "interval_start_days": intervals[:, 0].tolist(),
                "interval_end_days": intervals[:, 1].tolist(),
                "available_at_days": [
                    float(row["available_at"].timestamp() / 86400.0) for row in rows
                ],
            }
        )
        arrays = {}
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
        for product_id in dataset.dense_products:
            frames = first["model_inputs"]["source_frames"][product_id]
            product = config.products[product_id]
            group = root.create_group(product_id)
            arrays[product_id] = {
                "frames": group.create_dataset(
                    "frames",
                    shape=(len(dataset), *frames.shape),
                    chunks=(1, 1, *frames.shape[1:]),
                    dtype=np.dtype(product.dtype),
                    compressor=compressor,
                ),
                "masks": group.create_dataset(
                    "masks",
                    shape=(len(dataset), frames.shape[0], 1, *frames.shape[-2:]),
                    chunks=(1, 1, 1, *frames.shape[-2:]),
                    dtype="uint8",
                    compressor=compressor,
                ),
                "present": group.create_dataset(
                    "present",
                    shape=(len(dataset), frames.shape[0]),
                    chunks=(1, frames.shape[0]),
                    dtype="uint8",
                    compressor=compressor,
                ),
            }
        for index in range(len(dataset)):
            sample = first if index == 0 else dataset[index]
            inputs = sample["model_inputs"]
            for product_id in dataset.dense_products:
                arrays[product_id]["frames"][index] = inputs["source_frames"][product_id].numpy()
                arrays[product_id]["masks"][index] = (
                    inputs["source_pixel_masks"][product_id].numpy().astype(np.uint8)
                )
                arrays[product_id]["present"][index] = (
                    inputs["source_observation_masks"][product_id].numpy().astype(np.uint8)
                )
        zarr.consolidate_metadata(str(temporary))
        os.replace(temporary, output_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        dataset.close()
    return {
        "path": str(output_path),
        "records": len(dataset),
        "months": 24,
        "products": list(dataset.dense_products),
        "network_remote_pixels": False,
    }
