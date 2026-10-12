from types import SimpleNamespace

import pytest
import torch

from xuannv_embedding.models.model import AEFOutput
from xuannv_embedding.training.losses import TotalLoss
from xuannv_embedding.training.regional_refinement import apply_reconstruction_weights


def test_reconstruction_override_changes_selected_gradient_and_preserves_other_sources():
    criterion = TotalLoss(
        {
            "optical": {"loss_type": "l1", "weight": 0.9},
            "public": {"loss_type": "l1", "weight": 0.8},
        },
        uniformity_weight=0.0,
    )
    system = SimpleNamespace(criterion=SimpleNamespace(base=criterion))
    x = torch.ones(1, 1, 1, 2, 2, requires_grad=True)
    y = torch.ones_like(x, requires_grad=True)
    embedding = torch.ones(1, 1, 2, 2, 2)
    output = AEFOutput(embedding, embedding.mean((3, 4)), {"optical": x, "public": y})
    targets = {"optical": torch.zeros_like(x), "public": torch.zeros_like(y)}
    masks = {k: torch.ones(1, 1, 2, 2) for k in targets}
    before = criterion(output, targets, masks)["total"]
    grad_before = torch.autograd.grad(before, (x, y), retain_graph=True)
    apply_reconstruction_weights(system, {"optical": 0.225})
    after = criterion(output, targets, masks)["total"]
    grad_after = torch.autograd.grad(after, (x, y))
    torch.testing.assert_close(grad_after[0], grad_before[0] / 4)
    torch.testing.assert_close(grad_after[1], grad_before[1])
    assert criterion.target_cfg["public"]["weight"] == 0.8


@pytest.mark.parametrize(
    "weights",
    [
        {"absent": 1.0},
        {"map": 1.0},
        {"optical": -1},
        {"optical": float("nan")},
        {"optical": True},
        [],
    ],
)
def test_reconstruction_override_rejects_unregistered_or_invalid_objective(weights):
    criterion = SimpleNamespace(
        target_cfg={
            "optical": {"loss_type": "l1", "weight": 0.9},
            "map": {"loss_type": "ce", "weight": 0.45},
        }
    )
    system = SimpleNamespace(criterion=SimpleNamespace(base=criterion))
    with pytest.raises(ValueError):
        apply_reconstruction_weights(system, weights)
