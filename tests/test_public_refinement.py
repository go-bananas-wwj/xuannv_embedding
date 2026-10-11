from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xuannv_embedding.config import InputSourceConfig
from xuannv_embedding.training.public_refinement import PublicRefinementModel, validate_public_model
from xuannv_embedding.training.regional_refinement import parameter_groups


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.stp_encoder = nn.Sequential(nn.Dropout(0.1))
        self.decoders = nn.ModuleDict({"map": nn.Linear(2, 2)})

    def forward(self, frames, masks, timestamps):
        return self.encoder(frames["x"])


def test_public_control_preserves_base_output_and_gradient_without_highres():
    base = Base()
    model = PublicRefinementModel(base)
    x = torch.randn(3, 2)
    torch.testing.assert_close(model({"x": x}, {}, None, {}, {}), base({"x": x}, {}, None))
    model({"x": x}, {}, None).sum().backward()
    assert base.encoder.weight.grad is not None
    assert set(model.state_dict()) == {"base." + k for k in base.state_dict()}


def test_public_control_rejects_even_fully_masked_highres_data():
    model = PublicRefinementModel(Base())
    with pytest.raises(ValueError, match="highres"):
        model({"x": torch.randn(1, 2)}, {}, None, {"hr": torch.zeros(1)}, {"hr": torch.zeros(1)})


def test_public_control_matches_deterministic_vmf_and_trainable_stp_mode():
    model = PublicRefinementModel(Base()).train()
    assert not model.base.training
    assert model.base.stp_encoder.training
    assert model.base.encoder.weight.requires_grad


def test_public_validator_rejects_highres_input_or_unregistered_continuous_target():
    temporal = InputSourceConfig(channels=2, role="temporal")
    model = SimpleNamespace(
        input_sources={"x": temporal},
        highres_transformer=None,
        target_heads={"x": SimpleNamespace(source="x", loss_type="continuous")},
    )
    validate_public_model(model)
    model.input_sources["hr"] = InputSourceConfig(channels=3, role="highres")
    with pytest.raises(ValueError, match="public"):
        validate_public_model(model)
    del model.input_sources["hr"]
    model.target_heads["teacher"] = SimpleNamespace(source="hr", loss_type="continuous")
    with pytest.raises(ValueError, match="target"):
        validate_public_model(model)


def test_public_parameter_groups_keep_common_encoder_rates_and_frozen_probe():
    from xuannv_embedding.training.regional_refinement import StaticObjective
    from xuannv_embedding.training.runtime import TrainingSystem

    system = TrainingSystem(
        PublicRefinementModel(Base()), StaticObjective(nn.Linear(2, 2), 2, "map", 3)
    )
    groups = parameter_groups(
        system, {"public": 3e-6, "adaptation": 3e-5, "static": 1e-4}, head_only=False
    )
    assert {g["name"]: g["lr"] for g in groups} == {
        "public": 3e-6,
        "adaptation": 3e-5,
        "static": 1e-4,
    }
    assert not any(p.requires_grad for p in system.criterion.base.parameters())
