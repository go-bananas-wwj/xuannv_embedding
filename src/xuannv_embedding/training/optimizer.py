"""P10C 已验证的 AdamW 与 epoch 级 warmup-cosine 调度。"""

from __future__ import annotations

import math
from collections.abc import Iterable

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model: nn.Module | Iterable[nn.Parameter], lr: float, weight_decay: float
) -> Optimizer:
    """构造生产训练使用的 AdamW。"""
    parameters = model.parameters() if isinstance(model, nn.Module) else model
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
