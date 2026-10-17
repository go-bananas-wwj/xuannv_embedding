"""P10C 已验证的 AdamW 与 epoch 级 warmup-cosine 调度。"""

from __future__ import annotations

import math
from collections.abc import Iterable

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model: nn.Module | Iterable[nn.Parameter],
    lr: float,
    weight_decay: float,
    *,
    fused: bool = False,
    auxiliary_lr_multiplier: float = 1.0,
    vmf_lr_multiplier: float = 1.0,
) -> Optimizer:
    """构造生产训练使用的 AdamW；``fused`` 仅用于 CUDA，把逐参数更新合并为少量 kernel。"""
    parameters = model.parameters() if isinstance(model, nn.Module) else model
    if not math.isfinite(vmf_lr_multiplier) or vmf_lr_multiplier < 0:
        raise ValueError("vMF learning-rate multiplier must be nonnegative and finite")
    if vmf_lr_multiplier != 1.0:
        annual = getattr(model, "model", None)
        projection = getattr(getattr(annual, "bottleneck", None), "proj", None)
        if projection is None or not hasattr(annual, "temporal_sources"):
            raise ValueError("vMF projection policy requires an annual TrainingSystem")
        projection_ids = {id(p) for p in projection.parameters()}
        if vmf_lr_multiplier == 0:
            projection.requires_grad_(False)
        targets = getattr(getattr(model, "criterion", None), "target_supervision", None)
        head_ids = {id(p) for p in targets.parameters()} if targets is not None else set()
        if auxiliary_lr_multiplier != 1 and not head_ids:
            raise ValueError("Auxiliary learning-rate group requires target supervision heads")
        if not math.isfinite(auxiliary_lr_multiplier) or auxiliary_lr_multiplier <= 0:
            raise ValueError("Auxiliary learning-rate multiplier must be positive and finite")
        groups = {"model": [], "vmf_projection": [], "target_heads": []}
        for parameter in parameters:
            if id(parameter) in projection_ids:
                if vmf_lr_multiplier:
                    groups["vmf_projection"].append(parameter)
            elif id(parameter) in head_ids:
                groups["target_heads"].append(parameter)
            else:
                groups["model"].append(parameter)
        scales = {
            "model": 1.0,
            "vmf_projection": vmf_lr_multiplier,
            "target_heads": auxiliary_lr_multiplier,
        }
        return AdamW(
            [
                {"params": values, "lr": lr * scales[name], "name": name}
                for name, values in groups.items()
                if values
            ],
            lr=lr,
            weight_decay=weight_decay,
            **({"fused": True} if fused else {}),
        )
    if not math.isfinite(auxiliary_lr_multiplier) or auxiliary_lr_multiplier <= 0:
        raise ValueError("Auxiliary learning-rate multiplier must be positive and finite")
    if auxiliary_lr_multiplier != 1:
        targets = getattr(getattr(model, "criterion", None), "target_supervision", None)
        if targets is None:
            raise ValueError("Auxiliary learning-rate group requires target supervision heads")
        head_ids = {id(parameter) for parameter in targets.heads.parameters()}
        parameters = list(parameters)
        parameters = [
            {
                "params": [parameter for parameter in parameters if id(parameter) not in head_ids],
                "lr": lr,
            },
            {
                "params": [parameter for parameter in parameters if id(parameter) in head_ids],
                "lr": lr * auxiliary_lr_multiplier,
            },
        ]
    if fused:
        return AdamW(parameters, lr=lr, weight_decay=weight_decay, fused=True)
    return AdamW(parameters, lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> LambdaLR:
    """按绝对 epoch 进行线性 warmup，随后 cosine 退火至零。"""
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs 必须是非负整数")
    if total_epochs <= 0:
        raise ValueError("total_epochs 必须是正整数")
    if warmup_epochs > total_epochs:
        raise ValueError("warmup_epochs 不得超过 total_epochs")

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
