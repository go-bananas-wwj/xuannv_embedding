import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xuannv_embedding.config import Config
from xuannv_embedding.export.context import sha
from xuannv_embedding.models.model import AEFOutput
from xuannv_embedding.training.regional_base import RegionalObjective, read_spec


class FeatureLoss(nn.Module):
    def forward(self, output, *args):
        return {"total": output.embedding_map.square().mean()}


def test_zero_static_weight_preserves_base_gradient_and_removes_static_gradient():
    objective = RegionalObjective(FeatureLoss(), 2, 3, {}, static_weight=0.0)
    feature = torch.randn(1, 2, 2, 2, 2, requires_grad=True)
    output = AEFOutput(feature, feature.mean(dim=(3, 4)), {})
    targets = {"esa_worldcover": torch.ones(1, 2, 2, dtype=torch.long)}
    masks = {"esa_worldcover": torch.ones(1, 2, 2)}
    result = objective(output, targets, masks)
    result["total"].backward()
    assert torch.allclose(feature.grad, 2 * feature.detach() / feature.numel())
    assert torch.count_nonzero(objective.static_decoder.weight.grad) == 0
    assert result["static_weight"].item() == 0.0


def test_static_weight_changes_only_static_contribution():
    base = RegionalObjective(FeatureLoss(), 2, 3, {})
    stronger = RegionalObjective(FeatureLoss(), 2, 3, {}, static_weight=0.5)
    stronger.load_state_dict(base.state_dict(), strict=True)
    feature = torch.randn(1, 2, 2, 2, 2)
    output = AEFOutput(feature, feature.mean(dim=(3, 4)), {})
    targets = {"esa_worldcover": torch.ones(1, 2, 2, dtype=torch.long)}
    masks = {"esa_worldcover": torch.ones(1, 2, 2)}
    old, new = base(output, targets, masks), stronger(output, targets, masks)
    assert old["static_weight"].item() == 0.25
    assert torch.equal(old["recon_esa_worldcover"], new["recon_esa_worldcover"])
    assert torch.allclose(new["total"] - old["total"], old["recon_esa_worldcover"] * 0.25)


def write_spec(tmp_path, **extra_training):
    config_path = Path(__file__).parents[1] / "configs/examples/china_p10c_pilot.yaml"
    config = Config.from_yaml(config_path)
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {"state": "complete", "months": config.data.months, "records": [{"path": "unused.pt"}]}
        )
    )
    spec = {
        "schema": "regional-base-v1",
        "mode": "scratch",
        "checkpoint": None,
        "config": {"path": str(config_path), "sha256": sha(config_path)},
        "cache": {"path": str(cache_path), "sha256": sha(cache_path)},
        "output": str(tmp_path / "output"),
        "training": {
            "steps": 1,
            "head_steps": 0,
            "warmup_steps": 0,
            "seed": 41,
            "effective_batch": 48,
            "rates": {"public": 0.001, "highres": 0.001, "heads": 0.001},
            "weight_decay": 0.05,
            "save_every": 1,
            "min_free_gib": 1,
            **extra_training,
        },
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_registered_static_weight_is_accepted(tmp_path, weight):
    spec, _, _ = read_spec(write_spec(tmp_path, static_weight=weight))
    assert spec["training"]["static_weight"] == weight


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), True, "0.5", None])
def test_invalid_static_weight_is_rejected_before_build(tmp_path, weight):
    with pytest.raises(ValueError, match="static.*weight"):
        read_spec(write_spec(tmp_path, static_weight=weight))


def test_unknown_training_field_still_rejected(tmp_path):
    with pytest.raises(ValueError, match="training settings"):
        read_spec(write_spec(tmp_path, unregistered_weight=0.5))


def test_legacy_spec_without_static_weight_still_accepted(tmp_path):
    spec, _, _ = read_spec(write_spec(tmp_path))
    assert "static_weight" not in spec["training"]


@pytest.mark.parametrize("settings, expected", [({}, 0.25), ({"static_weight": 0.5}, 0.5)])
def test_cli_registers_effective_static_weight(tmp_path, monkeypatch, settings, expected):
    from xuannv_embedding.training import regional_base

    def build(config, **kwargs):
        return object(), {"static_weight": kwargs["static_weight"]}

    monkeypatch.setattr(regional_base, "build_regional_system", build)
    path = write_spec(tmp_path, **settings)
    assert regional_base.main(["--spec", str(path), "--phase", "validate"]) == 0
    receipt = json.loads((tmp_path / "output/initialization.json").read_text())
    assert receipt["static_weight"] == expected
    assert receipt["optimizer_steps"] == 0
