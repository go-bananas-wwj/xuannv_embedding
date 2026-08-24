"""区域无关、逐 patch 原子写入的 embedding 导出。"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def _move_mapping(value: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in value.items()}


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def export_embedding_batches(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    output_root: str | Path,
    *,
    device: str | torch.device,
) -> list[Path]:
    """严格检查形状与有限值后，把每个 patch 写为独立 NPZ。"""
    output_root = Path(output_root)
    target_device = torch.device(device)
    model.to(target_device).eval()
    written: list[Path] = []
    with torch.inference_mode():
        for batch in batches:
            patch_ids = batch.get("patch_ids")
            if not isinstance(patch_ids, list) or not patch_ids:
                raise ValueError("导出 batch 必须包含非空 patch_ids")
            output = model(
                _move_mapping(batch["source_frames"], target_device),
                _move_mapping(batch["source_masks"], target_device),
                batch["timestamps"].to(target_device),
                _move_mapping(batch.get("highres_frames", {}), target_device),
                _move_mapping(batch.get("highres_masks", {}), target_device),
            )
            embedding = output.embedding_map.detach().float().cpu().numpy()
            if embedding.ndim != 5 or embedding.shape[0] != len(patch_ids):
                raise ValueError(f"embedding 输出 batch 形状非法: {embedding.shape}")
            if not np.isfinite(embedding).all():
                raise FloatingPointError("embedding 包含 NaN/Inf")
            timestamps = batch["timestamps"].detach().cpu().numpy()
            for index, patch_id in enumerate(patch_ids):
                path = output_root / f"{patch_id}.npz"
                if path.exists():
                    raise FileExistsError(f"拒绝覆盖已导出的 embedding: {path}")
                _atomic_npz(path, embedding=embedding[index], timestamps=timestamps[index])
                written.append(path)
    return written
