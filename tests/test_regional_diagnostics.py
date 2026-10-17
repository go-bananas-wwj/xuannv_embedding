import json

import pytest
import torch
from torch import nn

from xuannv_embedding.training.regional_diagnostics import (
    append_training_curve,
    initialize_trained_public,
)


def test_curve_preserves_previous_updates_and_group_rates(tmp_path):
    path = tmp_path / "curve.jsonl"
    for step in [1, 20]:
        append_training_curve(
            path,
            step=step,
            losses={"total": 0.5 / step, "recon_s2": 0.2},
            rates={"public": 1e-6, "highres": 1e-4},
            elapsed_seconds=step * 2,
            skipped_updates=0,
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["optimizer_steps"] for r in rows] == [1, 20]
    assert rows[1]["learning_rates"]["public"] == 1e-6
    assert rows[0]["losses"]["total"] == 0.5
    assert rows[1]["losses"]["total"] == 0.025


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_curve_rejects_nonfinite_values_without_modifying_history(tmp_path, value):
    path = tmp_path / "curve.jsonl"
    path.write_text('{"previous": true}\n')
    with pytest.raises(ValueError):
        append_training_curve(
            path,
            step=1,
            losses={"total": value},
            rates={"public": 1e-6},
            elapsed_seconds=1,
            skipped_updates=0,
        )
    assert path.read_text() == '{"previous": true}\n'


def test_public_initialization_preserves_fresh_highres_and_copies_all_public_heads():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(2, 2)
            self.encoders = nn.Linear(2, 3)

    system = nn.Module()
    system.model = Model()
    system.criterion = nn.Linear(2, 1)
    source_model, source_criterion = nn.Linear(2, 2), nn.Linear(2, 1)
    original_highres = system.model.encoders.weight.detach().clone()
    state = {
        "model": {"base." + k: v for k, v in source_model.state_dict().items()},
        "criterion": source_criterion.state_dict(),
    }
    initialize_trained_public(system, state)
    assert torch.equal(system.model.base.weight, source_model.weight)
    assert torch.equal(system.criterion.weight, source_criterion.weight)
    assert torch.equal(system.model.encoders.weight, original_highres)


def test_public_initialization_rejects_highres_parent_and_partial_base():
    system = nn.Module()
    system.model = nn.Module()
    system.model.base = nn.Linear(2, 2)
    system.criterion = nn.Linear(2, 1)
    for model in [{"encoders.x": torch.ones(1)}, {"base.weight": torch.ones(2, 2)}]:
        with pytest.raises((ValueError, RuntimeError)):
            initialize_trained_public(
                system, {"model": model, "criterion": system.criterion.state_dict()}
            )
