"""V2 interval/UTM sharded Zarr export with a Parquet lineage catalog."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import zarr
from torch import nn


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type != "cpu")
    if isinstance(value, dict):
        return {key: _move(child, device) for key, child in value.items()}
    return value


def _interval_id(bounds: np.ndarray) -> str:
    def stamp(days: float) -> str:
        value = datetime.fromtimestamp(float(days) * 86400.0, tz=UTC)
        return value.strftime("%Y%m%d")

    return f"{stamp(bounds[0])}-{stamp(bounds[1])}"


class _ShardWriter:
    def __init__(
        self,
        root: Path,
        *,
        shard_size: int,
        provenance: dict[str, str],
    ) -> None:
        self.root = root
        self.shard_size = shard_size
        self.provenance = provenance
        self.counts: dict[tuple[str, int], int] = {}
        self.arrays: dict[tuple[str, int, int], zarr.Array] = {}

    def append(
        self,
        embedding: np.ndarray,
        *,
        patch_id: str,
        macro_id: str,
        split: str,
        interval_bounds: np.ndarray,
        epsg: int,
    ) -> dict[str, object]:
        interval_id = _interval_id(interval_bounds)
        partition = (interval_id, epsg)
        count = self.counts.get(partition, 0)
        shard = count // self.shard_size
        offset = count % self.shard_size
        key = (interval_id, epsg, shard)
        relative = Path(f"interval={interval_id}") / f"utm={epsg}" / f"part-{shard:05d}.zarr"
        if key not in self.arrays:
            group = zarr.open_group(str(self.root / relative), mode="w")
            group.attrs.update(
                {
                    "schema_version": "xuannv_v2_embedding_shard_v1",
                    "interval_id": interval_id,
                    "epsg": epsg,
                    **self.provenance,
                }
            )
            self.arrays[key] = group.create_dataset(
                "embedding",
                shape=(0, *embedding.shape),
                chunks=(1, *embedding.shape),
                dtype="float16",
                compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
            )
        array = self.arrays[key]
        array.resize((offset + 1, *embedding.shape))
        array[offset] = embedding.astype(np.float16, copy=False)
        self.counts[partition] = count + 1
        return {
            "patch_id": patch_id,
            "macro_id": macro_id,
            "split": split,
            "interval_id": interval_id,
            "interval_start_days": float(interval_bounds[0]),
            "interval_end_days": float(interval_bounds[1]),
            "epsg": epsg,
            "shard_path": relative.as_posix(),
            "shard_index": offset,
            "embedding_dim": int(embedding.shape[0]),
            "height": int(embedding.shape[1]),
            "width": int(embedding.shape[2]),
            **self.provenance,
        }


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8") + b"\0")
        with child.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def export_v2_sharded(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    output_root: str | Path,
    *,
    device: str | torch.device,
    product_version: str,
    data_manifest_sha256: str,
    config_sha256: str,
    git_sha: str,
    checkpoint_sha256: str,
    model_state_sha256: str,
    run_id: str,
    shard_size: int = 32,
) -> Path:
    """Export finite V2 maps without per-patch NPZ files."""
    if shard_size <= 0:
        raise ValueError("shard_size 必须为正整数")
    product_root = Path(output_root) / product_version
    catalog_path = product_root / "catalog.parquet"
    if catalog_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 V2 catalog: {catalog_path}")
    product_root.mkdir(parents=True, exist_ok=True)
    target_device = torch.device(device)
    model.to(target_device).eval()
    provenance = {
        "product_version": product_version,
        "data_manifest_sha256": data_manifest_sha256,
        "config_sha256": config_sha256,
        "git_sha": git_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": model_state_sha256,
        "run_id": run_id,
    }
    writer = _ShardWriter(product_root, shard_size=shard_size, provenance=provenance)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in batches:
            inputs = _move(batch["model_inputs"], target_device)
            output = model(**inputs)
            embedding = output.embedding_map.detach().float().cpu().numpy()
            intervals = batch["model_inputs"]["output_intervals"].numpy()
            if embedding.ndim != 5 or embedding.shape[:2] != intervals.shape[:2]:
                raise ValueError(
                    f"V2 embedding/interval 形状不匹配: {embedding.shape}/{intervals.shape}"
                )
            if not np.isfinite(embedding).all():
                raise FloatingPointError("V2 embedding 包含 NaN/Inf")
            if not hasattr(output, "observation_selection"):
                raise ValueError("V2 导出要求模型返回逐区间 observation selection")
            selections = {
                product_id: selected.detach().cpu().bool()
                for product_id, selected in output.observation_selection.items()
            }
            for batch_index, patch_id in enumerate(batch["patch_ids"]):
                for interval_index in range(embedding.shape[1]):
                    row = writer.append(
                        embedding[batch_index, interval_index],
                        patch_id=patch_id,
                        macro_id=batch["macro_ids"][batch_index],
                        split=batch["splits"][batch_index],
                        interval_bounds=intervals[batch_index, interval_index],
                        epsg=int(batch["grid_epsgs"][batch_index]),
                    )
                    candidates = batch.get("observation_candidates", [{}])[batch_index]
                    lineage = {}
                    for product_id, selected in selections.items():
                        refs = candidates.get(product_id, [])
                        indices = torch.nonzero(
                            selected[batch_index, interval_index], as_tuple=False
                        ).flatten()
                        lineage[product_id] = [
                            refs[int(index)]
                            for index in indices
                            if int(index) < len(refs) and refs[int(index)] is not None
                        ]
                    row["observation_lineage_json"] = json.dumps(
                        lineage, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    rows.append(row)
    if not rows:
        raise ValueError("V2 导出 batches 为空")
    shard_digests = {
        relative: _directory_sha256(product_root / relative)
        for relative in sorted({str(row["shard_path"]) for row in rows})
    }
    for row in rows:
        row["shard_sha256"] = shard_digests[str(row["shard_path"])]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".catalog.", suffix=".parquet", dir=product_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        os.replace(temporary, catalog_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return catalog_path
