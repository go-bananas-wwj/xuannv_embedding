from __future__ import annotations

import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


class CheckpointError(ValueError):
    """checkpoint 格式、元数据或严格加载失败。"""


FORMAT_VERSION = "1"
V2_FORMAT_VERSION = "2"
_REQUIRED_FIELDS = {
    "format_version",
    "config_sha256",
    "git_sha",
    "source_schema",
    "regions",
    "epoch",
    "model",
    "criterion",
    "optimizer",
    "scheduler",
    "metrics",
}


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "npu": (
            torch.npu.get_rng_state()
            if hasattr(torch, "npu") and torch.npu.is_available()
            else None
        ),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    if state["npu"] is not None and hasattr(torch, "npu") and torch.npu.is_available():
        npu_state = state["npu"].detach().cpu().to(dtype=torch.uint8).contiguous()
        torch.npu.set_rng_state(npu_state)


def _validate_metadata(
    *,
    config_sha256: str,
    git_sha: str,
    source_schema: dict[str, Any],
    regions: list[str],
    epoch: int,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", config_sha256):
        raise CheckpointError("config_sha256 必须是 64 位小写十六进制摘要")
    if not re.fullmatch(r"[0-9a-f]{7,64}", git_sha):
        raise CheckpointError("git_sha 必须是 7-64 位小写十六进制提交摘要")
    if not source_schema or not isinstance(source_schema, dict):
        raise CheckpointError("source_schema 必须是非空 mapping")
    if not regions or not all(isinstance(region, str) and region for region in regions):
        raise CheckpointError("regions 必须是非空字符串列表")
    if len(set(regions)) != len(regions):
        raise CheckpointError("regions 不得重复")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise CheckpointError("epoch 必须是非负整数")


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    criterion: nn.Module | None = None,
    optimizer: Optimizer,
    scheduler: Any | None,
    epoch: int,
    config_sha256: str,
    git_sha: str,
    source_schema: dict[str, Any],
    regions: list[str],
    metrics: dict[str, Any] | None = None,
) -> None:
    """按版本化生产合同原子保存训练状态。"""
    _validate_metadata(
        config_sha256=config_sha256,
        git_sha=git_sha,
        source_schema=source_schema,
        regions=regions,
        epoch=epoch,
    )
    state = {
        "format_version": FORMAT_VERSION,
        "config_sha256": config_sha256,
        "git_sha": git_sha,
        "source_schema": source_schema,
        "regions": list(regions),
        "epoch": epoch,
        "model": model.state_dict(),
        "criterion": criterion.state_dict() if criterion is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metrics": dict(metrics or {}),
    }
    _atomic_torch_save(state, Path(path))


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    criterion: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    device: str | torch.device = "cpu",
    expected_config_sha256: str | None = None,
    expected_source_schema: dict[str, Any] | None = None,
    expected_regions: list[str] | None = None,
) -> dict[str, Any]:
    """严格加载新格式 checkpoint；不接受缺失溯源元数据的文件。"""
    try:
        state = torch.load(Path(path), map_location=device, weights_only=True)
    except (OSError, RuntimeError) as exc:
        raise CheckpointError(f"无法加载 checkpoint: {exc}") from exc
    if not isinstance(state, dict):
        raise CheckpointError("checkpoint 顶层必须是 mapping")
    missing = sorted(_REQUIRED_FIELDS - set(state))
    if missing:
        raise CheckpointError(f"checkpoint 缺少字段: {', '.join(missing)}")
    if state["format_version"] != FORMAT_VERSION:
        raise CheckpointError(f"不支持的 checkpoint format_version: {state['format_version']!r}")
    _validate_metadata(
        config_sha256=state["config_sha256"],
        git_sha=state["git_sha"],
        source_schema=state["source_schema"],
        regions=state["regions"],
        epoch=state["epoch"],
    )
    if expected_config_sha256 is None:
        raise CheckpointError("严格加载必须提供 expected config_sha256")
    if expected_source_schema is None:
        raise CheckpointError("严格加载必须提供 expected source_schema")
    if expected_regions is None:
        raise CheckpointError("严格加载必须提供 expected regions")
    if state["config_sha256"] != expected_config_sha256:
        raise CheckpointError("checkpoint config_sha256 与当前配置不一致")
    if state["source_schema"] != expected_source_schema:
        raise CheckpointError("checkpoint source_schema 与当前配置不一致")
    if state["regions"] != expected_regions:
        raise CheckpointError("checkpoint regions 与当前配置不一致")
    try:
        model.load_state_dict(state["model"], strict=True)
        if criterion is not None:
            if state["criterion"] is None:
                raise CheckpointError("checkpoint 未保存 criterion 状态")
            criterion.load_state_dict(state["criterion"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint 状态严格加载失败: {exc}") from exc
    return state


def save_v2_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    step: int,
    config_sha256: str,
    git_sha: str,
    data_manifest_sha256: str,
    product_schema: dict[str, Any],
    temporal_contract: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    sampler_state: dict[str, int] | None = None,
    rank_rng_states: list[dict[str, Any]] | None = None,
) -> None:
    """Save an explicitly V2-only checkpoint with data and interval provenance."""
    if not re.fullmatch(r"[0-9a-f]{64}", data_manifest_sha256):
        raise CheckpointError("data_manifest_sha256 必须是 64 位小写十六进制摘要")
    _validate_metadata(
        config_sha256=config_sha256,
        git_sha=git_sha,
        source_schema=product_schema,
        regions=["china-national-local-v2"],
        epoch=step,
    )
    if not temporal_contract:
        raise CheckpointError("temporal_contract 必须是非空 mapping")
    rng_state = capture_rng_state()
    state = {
        "format_version": V2_FORMAT_VERSION,
        "config_sha256": config_sha256,
        "git_sha": git_sha,
        "data_manifest_sha256": data_manifest_sha256,
        "product_schema": product_schema,
        "temporal_contract": temporal_contract,
        "step": step,
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metrics": dict(metrics or {}),
        "rng_state": rng_state["torch"],
        "python_rng_state": rng_state["python"],
        "numpy_rng_state": rng_state["numpy"],
        "npu_rng_state": rng_state["npu"],
        "sampler_state": dict(sampler_state or {}),
        "rank_rng_states": list(rank_rng_states or []),
    }
    _atomic_torch_save(state, Path(path))


def load_v2_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    expected_config_sha256: str,
    expected_data_manifest_sha256: str,
    expected_product_schema: dict[str, Any],
    expected_temporal_contract: dict[str, Any],
    device: str | torch.device = "cpu",
    restore_rng: bool = True,
    rng_rank: int | None = None,
) -> dict[str, Any]:
    """Strictly load V2 state; V1 checkpoints and configs are never remapped."""
    try:
        state = torch.load(Path(path), map_location=device, weights_only=True)
    except (OSError, RuntimeError) as exc:
        raise CheckpointError(f"无法加载 V2 checkpoint: {exc}") from exc
    required = {
        "format_version",
        "config_sha256",
        "git_sha",
        "data_manifest_sha256",
        "product_schema",
        "temporal_contract",
        "step",
        "model",
        "criterion",
        "optimizer",
        "scheduler",
        "metrics",
        "rng_state",
        "python_rng_state",
        "numpy_rng_state",
        "npu_rng_state",
        "sampler_state",
        "rank_rng_states",
    }
    if not isinstance(state, dict):
        raise CheckpointError("V2 checkpoint 顶层必须是 mapping")
    if state.get("format_version") != V2_FORMAT_VERSION:
        raise CheckpointError("V2 runtime 明确拒绝 V1 checkpoint，不执行静默 remap")
    missing = sorted(required - set(state))
    if missing:
        raise CheckpointError(f"V2 checkpoint 缺少字段: {', '.join(missing)}")
    expected = {
        "config_sha256": expected_config_sha256,
        "data_manifest_sha256": expected_data_manifest_sha256,
        "product_schema": expected_product_schema,
        "temporal_contract": expected_temporal_contract,
    }
    for name, value in expected.items():
        if state[name] != value:
            raise CheckpointError(f"V2 checkpoint {name} 与当前运行合同不一致")
    try:
        model.load_state_dict(state["model"], strict=True)
        criterion.load_state_dict(state["criterion"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        target_device = torch.device(device)
        for optimizer_state in optimizer.state.values():
            for name, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[name] = value.to(target_device)
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"V2 checkpoint 状态严格加载失败: {exc}") from exc
    if restore_rng:
        if rng_rank is not None and state["rank_rng_states"]:
            if rng_rank >= len(state["rank_rng_states"]):
                raise CheckpointError(f"checkpoint 缺少 rank {rng_rank} RNG 状态")
            restore_rng_state(state["rank_rng_states"][rng_rank])
        else:
            restore_rng_state(
                {
                    "torch": state["rng_state"],
                    "python": state["python_rng_state"],
                    "numpy": state["numpy_rng_state"],
                    "npu": state["npu_rng_state"],
                }
            )
    return state
