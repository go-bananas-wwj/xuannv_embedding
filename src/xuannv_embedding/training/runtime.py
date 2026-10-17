"""精简、可分布式包装的 P10C 训练运行时。"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from copy import deepcopy
from math import isfinite
from time import perf_counter
from typing import Any, Callable

import torch
from torch import nn
from torch.optim import Optimizer

from xuannv_embedding.training.distillation import freeze_teacher
from xuannv_embedding.training.losses import TotalLoss


def _model_view(model: nn.Module, view: dict[str, Any], *, decode: bool = True):
    optional = {
        key: view[key]
        for key in (
            "highres_months",
            "output_months",
            "highres_observations",
            "reconstruction_requests",
            "source_validity_masks",
        )
        if key in view
    }
    # Annual teachers only need embeddings; avoid observation-specific decoders.
    if hasattr(model, "context_pooling"):
        optional["decode"] = decode
    return model(
        view["source_frames"],
        view["source_masks"],
        view["timestamps"],
        view.get("highres_frames"),
        view.get("highres_masks"),
        source_pixel_masks=view.get("source_pixel_masks"),
        **optional,
    )


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


def _cuda_amp_dtype() -> torch.dtype:
    """bf16 无需 loss scaling 且对 vMF 归一化更稳；仅在硬件不支持时回退 fp16。"""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "npu":
        import torch_npu

        return torch_npu.npu.amp.autocast()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=_cuda_amp_dtype())
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
        if _cuda_amp_dtype() is torch.bfloat16:
            return None
        return torch.amp.GradScaler("cuda")
    return None


def _observe_gradient_norm(system: nn.Module, optimizer: Optimizer, scaler: Any | None) -> float:
    """Observe gradients; reject nonfinite updates when no scaler can skip them."""
    if scaler is not None:
        scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in system.parameters() if parameter.grad is not None]
    norm = (
        torch.linalg.vector_norm(torch.stack([gradient.float().norm() for gradient in gradients]))
        if gradients
        else torch.tensor(0.0)
    )
    if scaler is None and not bool(torch.isfinite(norm)):
        raise FloatingPointError("Nonfinite gradients; optimizer update rejected")
    return float(norm)


class TrainingSystem(nn.Module):
    """把模型与带参数的 semantic probe 置于同一 DDP 边界。"""

    def __init__(
        self,
        model: nn.Module,
        criterion: TotalLoss,
        teacher: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.teacher_model = None if teacher is None else freeze_teacher(teacher)
        self.ema_model = None
        self.ema_decay = 0.999
        self.ema_updates = 0

    def train(self, mode: bool = True) -> TrainingSystem:
        super().train(mode)
        if self.teacher_model is not None:
            self.teacher_model.eval()
        if self.ema_model is not None:
            self.ema_model.eval()
        return self

    def set_ema(self, decay: float) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.ema_decay = decay
        self.ema_model = freeze_teacher(deepcopy(self.model))

    @torch.no_grad()
    def update_ema(self) -> None:
        if self.ema_model is None:
            return
        for target, source in zip(
            self.ema_model.parameters(), self.model.parameters(), strict=True
        ):
            target.lerp_(source.detach(), 1 - self.ema_decay)
        for target, source in zip(self.ema_model.buffers(), self.model.buffers(), strict=True):
            target.copy_(source)
        self.ema_updates += 1

    def auxiliary_state(self) -> dict[str, Any]:
        if self.ema_model is None:
            return {}
        return {
            "ema": self.ema_model.state_dict(),
            "ema_updates": self.ema_updates,
            "ema_decay": self.ema_decay,
        }

    def restore_auxiliary_state(self, state: dict[str, Any]) -> None:
        if self.ema_model is None:
            if state:
                raise ValueError("Checkpoint contains EMA but configuration disables it")
            return
        if "ema" not in state or state.get("ema_decay") != self.ema_decay:
            raise ValueError("EMA checkpoint missing or decay differs")
        self.ema_model.load_state_dict(state["ema"], strict=True)
        self.ema_updates = int(state["ema_updates"])

    def set_teacher(self, teacher: nn.Module, device: torch.device | None = None) -> None:
        if device is not None:
            teacher.to(device)
        self.teacher_model = freeze_teacher(teacher)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        output = _model_view(self.model, batch)
        teacher_output = None
        teacher_view = batch.get("teacher_view")
        if self.teacher_model is not None:
            if teacher_view is None:
                raise ValueError("Frozen teacher requires an explicit teacher_view")
            with torch.no_grad():
                teacher_output = _model_view(self.teacher_model, teacher_view, decode=False)
        loss_optional = (
            {"highres_targets": batch["highres_targets"]} if "highres_targets" in batch else {}
        )
        if "independent_targets" in batch:
            loss_optional["independent_targets"] = batch["independent_targets"]
        if self.ema_model is not None:
            if "consistency_view" not in batch:
                raise ValueError("EMA consistency requires an independent consistency_view")
            with torch.no_grad():
                loss_optional["consistency_output"] = _model_view(
                    self.ema_model, batch["consistency_view"], decode=False
                )
        return self.criterion(
            output,
            batch["targets"],
            batch["target_masks"],
            batch.get("supervised_labels"),
            batch.get("supervised_label_masks"),
            teacher_output=teacher_output,
            **loss_optional,
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
    step_callback: Callable[[dict[str, Any]], None] | None = None,
    update_diagnostics_interval: int = 0,
) -> dict[str, float | int]:
    """训练有限个 epoch，并返回可序列化的发布门禁摘要。"""
    if epochs <= 0 or gradient_accumulation_steps <= 0 or update_diagnostics_interval < 0:
        raise ValueError("epochs 与 gradient_accumulation_steps 必须是正整数")
    if start_epoch < 0:
        raise ValueError("start_epoch 必须是非负整数")
    system.to(device)
    system.train()
    scaler = _grad_scaler(device, amp)
    if update_diagnostics_interval and scaler is not None:
        raise ValueError("Actual-update diagnostics require BF16/FP32 without a GradScaler")
    optimizer.zero_grad(set_to_none=True)
    batch_count = 0
    optimizer_steps = 0
    loss_sum = 0.0
    started = perf_counter()
    previous_end = started
    data_wait_seconds = 0.0
    sample_count = 0
    gradient_norm = 0.0
    skipped_steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    end_epoch = start_epoch + epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        _unwrap(system).criterion.set_epoch(epoch)
        pending = 0
        for raw_batch in batches:
            batch_started = perf_counter()
            data_wait_seconds += batch_started - previous_end
            sample_count += int(raw_batch["timestamps"].shape[0])
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
            update_metrics = {}
            batch_count += 1
            loss_sum += float(losses["total"].detach().float().cpu())
            if pending == gradient_accumulation_steps:
                gradient_norm = _observe_gradient_norm(system, optimizer, scaler)
                if not isfinite(gradient_norm):
                    # fp16/NPU 的 GradScaler 会在溢出时跳过这一步并下调 scale，
                    # 非有限范数属于预期路径，只统计不中断训练。
                    skipped_steps += 1
                if scaler is None:
                    snapshot = None
                    if update_diagnostics_interval and (
                        optimizer_steps == 0
                        or (optimizer_steps + 1) % update_diagnostics_interval == 0
                    ):
                        from xuannv_embedding.training.update_diagnostics import snapshot_parameters

                        snapshot = snapshot_parameters(_unwrap(system))
                    optimizer.step()
                    if snapshot is not None:
                        from xuannv_embedding.training.update_diagnostics import measure_updates

                        update_metrics = measure_updates(snapshot)
                        del snapshot
                else:
                    scaler.step(optimizer)
                    scaler.update()
                if isfinite(gradient_norm):
                    _unwrap(system).update_ema()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending = 0
            # 同步只为让 step_callback 的计时可信；无回调时保留 H2D/计算重叠。
            if step_callback is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            previous_end = perf_counter()
            if step_callback is not None:
                step_callback(
                    {
                        "batch": batch_count,
                        "optimizer_steps": optimizer_steps,
                        "loss": float(losses["total"].detach().float().cpu()),
                        "reconstruction": float(
                            losses.get("recon", losses["total"]).detach().float().cpu()
                        ),
                        "gradient_norm": gradient_norm,
                        "parameter_updates": update_metrics,
                        "elapsed_seconds": previous_end - started,
                        "data_wait_seconds": data_wait_seconds,
                        "step_seconds": previous_end - batch_started,
                        "components": {
                            key: float(value.detach().float().cpu())
                            for key, value in losses.items()
                            if value.numel() == 1
                        },
                    }
                )
        if pending:
            for parameter in system.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_accumulation_steps / pending)
            gradient_norm = _observe_gradient_norm(system, optimizer, scaler)
            if not isfinite(gradient_norm):
                skipped_steps += 1
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            if isfinite(gradient_norm):
                _unwrap(system).update_ema()
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
        "nonfinite_gradient_steps": skipped_steps,
        "loss": loss_sum / batch_count,
        "elapsed_seconds": perf_counter() - started,
        "data_wait_seconds": data_wait_seconds,
        "samples_per_rank": sample_count,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
    }
