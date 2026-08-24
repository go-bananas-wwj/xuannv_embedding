from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


class CheckpointError(ValueError):
    """checkpoint 格式、元数据或严格加载失败。"""


FORMAT_VERSION = "1"
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
