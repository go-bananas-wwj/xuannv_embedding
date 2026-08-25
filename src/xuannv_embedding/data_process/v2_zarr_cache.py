"""Sequentially repack immutable local ZIP observations into a smoke-only Zarr cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import zarr

from xuannv_embedding.config import V2Config
from xuannv_embedding.data.v2_dataset import V2LocalZipDataset


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_smoke_zarr_cache(cache_path: Path, *, full: bool) -> dict[str, object]:
    """Verify cache metadata and deterministic or complete stored-array digests."""
    root = zarr.open_group(str(cache_path), mode="r")
    manifest_path = cache_path / "cache_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Zarr cache manifest 不存在: {manifest_path}")
    expected_manifest_sha = root.attrs.get("cache_manifest_sha256")
    if expected_manifest_sha != _sha256_file(manifest_path):
        raise ValueError(f"Zarr cache manifest SHA256 不匹配: {cache_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contracts = manifest.get("array_contracts", {})
    verified_patches = 0
    for product_id, product_contract in contracts.items():
        if product_id not in root:
            raise ValueError(f"Zarr cache 缺少产品 group: {product_id}")
        group = root[product_id]
        patch_hashes = product_contract.get("patch_sha256", [])
        if full:
            indices = range(len(patch_hashes))
        else:
            indices = manifest.get("zip_zarr_audit_indices", [])
        content = {name: hashlib.sha256() for name in ("frames", "masks", "present")}
        for index in indices:
            patch_digest = hashlib.sha256()
            for name in ("frames", "masks", "present"):
                array = group[name]
                contract = product_contract[name]
                if list(array.shape) != contract["shape"] or str(array.dtype) != contract["dtype"]:
                    raise ValueError(f"Zarr array 合同不匹配: {product_id}/{name}")
                value = np.ascontiguousarray(array[index])
                payload = value.tobytes()
                patch_digest.update(name.encode("ascii") + b"\0" + payload)
                if full:
                    content[name].update(payload)
            if patch_digest.hexdigest() != patch_hashes[index]:
                raise ValueError(f"Zarr patch SHA256 不匹配: {product_id}/{index}")
            verified_patches += 1
        if full:
            for name, digest in content.items():
                if digest.hexdigest() != product_contract[name]["sha256"]:
                    raise ValueError(f"Zarr array SHA256 不匹配: {product_id}/{name}")
    return {
        "verified": True,
        "full": full,
        "verified_product_patches": verified_patches,
        "builder_git_sha": manifest.get("builder_git_sha"),
    }


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
    registry_sha256 = _sha256_file(registry_path)
    availability_path = config.paths.data_root / "observations" / "index" / "availability.parquet"
    availability_sha256 = _sha256_file(availability_path)
    try:
        root = zarr.open_group(str(temporary), mode="w")
        first = dataset[0]
        intervals = first["model_inputs"]["output_intervals"].numpy()
        rows = dataset.observations[(first["patch_id"], dataset.dense_products[0])]
        root.attrs.update(
            {
                "schema_version": "xuannv_v2_smoke_dense_cache_v2",
                "source": "local_zip_repack",
                "network_remote_pixels": False,
                "source_archive_lock_sha256": hashlib.sha256(
                    (config.paths.data_root / "locks" / "local_archive_sha256.jsonl").read_bytes()
                ).hexdigest(),
                "registry_sha256": registry_sha256,
                "availability_sha256": availability_sha256,
                "builder_git_sha": _git_sha(),
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
        content_digests: dict[str, dict[str, hashlib._Hash]] = {}
        patch_digests: dict[str, list[str]] = {}
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
        for product_id in dataset.dense_products:
            frames = first["model_inputs"]["source_frames"][product_id]
            product = config.products[product_id]
            group = root.create_group(product_id)
            content_digests[product_id] = {
                name: hashlib.sha256() for name in ("frames", "masks", "present")
            }
            patch_digests[product_id] = []
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
                values = {
                    "frames": inputs["source_frames"][product_id].numpy(),
                    "masks": inputs["source_pixel_masks"][product_id].numpy().astype(np.uint8),
                    "present": inputs["source_observation_masks"][product_id]
                    .numpy()
                    .astype(np.uint8),
                }
                patch_digest = hashlib.sha256()
                for name, value in values.items():
                    value = np.asarray(value, dtype=arrays[product_id][name].dtype)
                    arrays[product_id][name][index] = value
                    payload = np.ascontiguousarray(value).tobytes()
                    content_digests[product_id][name].update(payload)
                    patch_digest.update(name.encode("ascii") + b"\0" + payload)
                patch_digests[product_id].append(patch_digest.hexdigest())
        contracts = {}
        for product_id in dataset.dense_products:
            contracts[product_id] = {
                name: {
                    "shape": list(arrays[product_id][name].shape),
                    "chunks": list(arrays[product_id][name].chunks),
                    "dtype": str(arrays[product_id][name].dtype),
                    "sha256": content_digests[product_id][name].hexdigest(),
                }
                for name in ("frames", "masks", "present")
            }
            contracts[product_id]["patch_sha256"] = patch_digests[product_id]
        audit_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
        for index in audit_indices:
            for product_id in dataset.dense_products:
                expected = patch_digests[product_id][index]
                actual = hashlib.sha256()
                for name in ("frames", "masks", "present"):
                    value = np.ascontiguousarray(arrays[product_id][name][index])
                    actual.update(name.encode("ascii") + b"\0" + value.tobytes())
                if actual.hexdigest() != expected:
                    raise RuntimeError(f"ZIP↔Zarr audit 失败: {product_id}/{index}")
        manifest = {
            "schema_version": "xuannv_v2_smoke_dense_cache_manifest_v1",
            "registry_sha256": registry_sha256,
            "availability_sha256": availability_sha256,
            "builder_git_sha": root.attrs["builder_git_sha"],
            "array_contracts": contracts,
            "zip_zarr_audit_indices": audit_indices,
            "zip_zarr_audit_passed": True,
        }
        manifest_path = temporary / "cache_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        root.attrs["cache_manifest_sha256"] = _sha256_file(manifest_path)
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
        "registry_sha256": registry_sha256,
        "availability_sha256": availability_sha256,
        "zip_zarr_audit_passed": True,
    }
