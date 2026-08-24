"""精简、可分布式包装的 P10C 训练运行时。"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any, Callable

import torch
from torch import nn
from torch.optim import Optimizer

from xuannv_embedding.training.losses import TotalLoss


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type != "cpu")
    if isinstance(value, dict):
        return {key: _move(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_move(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(_move(child, device) for child in value)
    return value


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "npu":
        import torch_npu

        return torch_npu.npu.amp.autocast()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def _grad_scaler(device: torch.device, enabled: bool):
    if not enabled:
        return None
    if device.type == "npu":
        import torch_npu

        return torch_npu.npu.amp.GradScaler()
    if device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    return None


class TrainingSystem(nn.Module):
    """把模型与带参数的 semantic probe 置于同一 DDP 边界。"""

    def __init__(self, model: nn.Module, criterion: TotalLoss) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        output = self.model(
            batch["source_frames"],
            batch["source_masks"],
            batch["timestamps"],
            batch.get("highres_frames"),
            batch.get("highres_masks"),
        )
        return self.criterion(
            output,
            batch["targets"],
            batch["target_masks"],
            batch.get("supervised_labels"),
            batch.get("supervised_label_masks"),
        )


def _unwrap(system: nn.Module) -> TrainingSystem:
    module = system.module if hasattr(system, "module") else system
    if not isinstance(module, TrainingSystem):
        raise TypeError("system 必须是 TrainingSystem 或其 DDP 包装")
    return module


def train_steps(
    system: nn.Module,
    batches: Iterable[dict[str, Any]],
    optimizer: Optimizer,
    *,
    device: torch.device,
    epochs: int,
    start_epoch: int = 0,
    gradient_accumulation_steps: int,
    amp: bool,
    scheduler: Any | None = None,
    epoch_end_callback: Callable[[int], None] | None = None,
) -> dict[str, float | int]:
    """训练有限个 epoch，并返回可序列化的发布门禁摘要。"""
    if epochs <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("epochs 与 gradient_accumulation_steps 必须是正整数")
    if start_epoch < 0:
        raise ValueError("start_epoch 必须是非负整数")
    system.to(device)
    system.train()
    scaler = _grad_scaler(device, amp)
    optimizer.zero_grad(set_to_none=True)
    batch_count = 0
    optimizer_steps = 0
    loss_sum = 0.0

    end_epoch = start_epoch + epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        _unwrap(system).criterion.set_epoch(epoch)
        pending = 0
        for raw_batch in batches:
            batch = _move(raw_batch, device)
            with _autocast(device, amp):
                losses = system(batch)
                loss = losses["total"] / gradient_accumulation_steps
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"训练 loss 非有限值: {float(loss.detach().cpu())}")
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            pending += 1
            batch_count += 1
            loss_sum += float(losses["total"].detach().float().cpu())
            if pending == gradient_accumulation_steps:
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending = 0
        if pending:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if scheduler is not None:
            scheduler.step()
        if epoch_end_callback is not None:
            epoch_end_callback(epoch)

    if batch_count == 0:
        raise ValueError("训练 batches 为空")
    return {
        "epochs": epochs,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "batches": batch_count,
        "optimizer_steps": optimizer_steps,
        "loss": loss_sum / batch_count,
    }
