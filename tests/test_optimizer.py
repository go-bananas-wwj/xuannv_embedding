from __future__ import annotations

import pytest
import torch

from xuannv_embedding.training.optimizer import build_optimizer, build_scheduler


def test_optimizer_uses_configured_adamw_hyperparameters() -> None:
    layer = torch.nn.Linear(2, 1)
    optimizer = build_optimizer(layer, lr=3e-4, weight_decay=0.05)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)


def test_scheduler_is_linear_warmup_then_cosine_by_absolute_epoch() -> None:
    layer = torch.nn.Linear(2, 1)
    optimizer = build_optimizer(layer, lr=1.0, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, warmup_epochs=2, total_epochs=6)
    schedule = scheduler.lr_lambdas[0]

    assert schedule(0) == pytest.approx(0.5)
    assert schedule(1) == pytest.approx(1.0)
    assert schedule(2) == pytest.approx(1.0)
    assert schedule(4) == pytest.approx(0.5)
    assert schedule(6) == pytest.approx(0.0)


@pytest.mark.parametrize("warmup,total", [(-1, 2), (0, 0), (3, 2)])
def test_scheduler_rejects_invalid_epoch_ranges(warmup: int, total: int) -> None:
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters())

    with pytest.raises(ValueError):
        build_scheduler(optimizer, warmup_epochs=warmup, total_epochs=total)
