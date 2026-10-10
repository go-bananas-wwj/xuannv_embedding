import json

import pytest
import torch

from xuannv_embedding.training.regional_base import (
    apply_initial_weights,
    require_training_approval,
    set_reference_window,
)


def test_scratch_does_not_read_parent_weights(tmp_path):
    model = torch.nn.Linear(2, 2)
    before = model.weight.detach().clone()
    apply_initial_weights(model, None, mode="scratch", checkpoint=tmp_path / "absent.pt")
    assert torch.equal(before, model.weight)


def test_transfer_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        apply_initial_weights(
            torch.nn.Linear(2, 2), None, mode="transfer", checkpoint=tmp_path / "x"
        )


def test_transfer_loads_every_weight_and_rejects_partial_state(tmp_path):
    parent = torch.nn.Linear(2, 2)
    child = torch.nn.Linear(2, 2)
    p = tmp_path / "parent.pt"
    torch.save({"model": parent.state_dict(), "criterion": None}, p)
    apply_initial_weights(child, None, mode="transfer", checkpoint=p)
    assert torch.equal(parent.weight, child.weight)
    torch.save({"model": {"weight": parent.weight}, "criterion": None}, p)
    with pytest.raises(RuntimeError):
        apply_initial_weights(child, None, mode="transfer", checkpoint=p)


def test_reference_window_updates_month_binning_without_changing_weights():
    class Monthly:
        ref_year = 2025
        ref_month = 12
        num_months = 6

    class Model:
        ref_year = 2025
        ref_month = 12
        num_months = 6
        monthly_embed = Monthly()

    model = Model()
    set_reference_window(model, ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10"])
    assert model.ref_month == model.monthly_embed.ref_month == 5
    with pytest.raises(ValueError):
        set_reference_window(model, ["2025-05", "2025-07"])


def test_training_requires_approval_bound_to_exact_spec(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    with pytest.raises(ValueError):
        require_training_approval(spec, None)
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({"approved": True, "spec_sha256": "wrong"}))
    with pytest.raises(ValueError):
        require_training_approval(spec, approval)


def test_semantic_probe_ignores_pixels_outside_region_mask():
    from xuannv_embedding.training.losses import SemanticProbeLoss

    objective = SemanticProbeLoss(embed_dim=2, tasks=["field"], month_index=0)
    embedding = torch.zeros(1, 1, 2, 2, 2)
    mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    label = torch.zeros(1, 2, 2)
    before, stats = objective(embedding, {"field": label}, {"field": mask})
    label[:, 1, :] = 1
    after, _ = objective(embedding, {"field": label}, {"field": mask})
    assert torch.equal(before, after)
    assert stats["semantic_probe_valid_pixels"].item() == 1


def test_amp_overflow_is_not_counted_as_successful_update():
    from xuannv_embedding.training.regional_base import step_optimizer

    class Scaler:
        scale = 1024

        def get_scale(self):
            return self.scale

        def step(self, optimizer):
            pass

        def update(self):
            self.scale /= 2

    assert step_optimizer(object(), Scaler()) is False
