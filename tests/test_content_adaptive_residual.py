import copy

import pytest
import torch

from xuannv_embedding.models.content_adaptive_residual import (
    ContentAdaptiveResidual,
    adaptive_residual_loss,
)


def inputs(batch=2):
    generator = torch.Generator().manual_seed(41)
    public = torch.randn(batch, 64, 4, 5, generator=generator)
    return {
        "public": public,
        "semantic": public + 0.1 * torch.randn(public.shape, generator=generator),
        "geometry": public + 0.3 * torch.randn(public.shape, generator=generator),
        "spectral": torch.randn(batch, 95, 4, 5, generator=generator),
        "spectral_mask": torch.ones(batch, 95, 4, 5, dtype=torch.bool),
    }


def teacher():
    generator = torch.Generator().manual_seed(19)
    values = {}
    for source, channels in [("public", 64), ("spectral", 95)]:
        values[source + "_mean"] = torch.zeros(1, channels, 1, 1)
        values[source + "_std"] = torch.ones(1, channels, 1, 1)
    for name, classes in [("worldcover", 12), ("area", 7), ("geometry", 4)]:
        values[name + "_head.weight"] = torch.randn(classes, 64, 1, 1, generator=generator)
        values[name + "_head.bias"] = torch.zeros(classes)
    return values


def model():
    result = ContentAdaptiveResidual(width=8, initial_alpha=0.15)
    result.set_teacher(teacher())
    return result


def test_initial_gate_matches_registered_mixture_and_keeps_coordinate_bounds():
    data = inputs()
    net = model()
    out = net(**data)
    torch.testing.assert_close(out.alpha, torch.full_like(data["public"], 0.15))
    expected = 0.85 * data["semantic"] + 0.15 * data["geometry"]
    torch.testing.assert_close(out.embedding, expected)
    with torch.no_grad():
        net.project.weight.normal_(0, 0.3)
    out = net(**data)
    assert (out.alpha >= 0).all() and (out.alpha <= 1).all()
    assert out.alpha.std() > 0
    lower = torch.minimum(data["semantic"], data["geometry"])
    upper = torch.maximum(data["semantic"], data["geometry"])
    assert (out.embedding >= lower - 1e-6).all()
    assert (out.embedding <= upper + 1e-6).all()


def test_frozen_teacher_uses_original_clipped_anchor_and_scaled_correction():
    net = model()
    data = inputs(1)
    data["public"].fill_(20)
    data["semantic"] = data["public"] + 0.2
    data["geometry"] = data["public"] + 0.4
    out = net(**data)
    expected_white = torch.full_like(data["public"], 6) + (out.embedding - data["public"])
    torch.testing.assert_close(out.worldcover_logits, net.worldcover_head(expected_white))
    assert not torch.allclose(out.worldcover_logits, net.worldcover_head(out.embedding))


def test_only_gate_receives_gradients_and_source_values_are_not_mutated():
    net = model()
    data = inputs()
    for name in ["public", "semantic", "geometry", "spectral"]:
        data[name].requires_grad_()
    before = {name: tensor.detach().clone() for name, tensor in data.items()}
    labels = {
        name: torch.ones(2, 4, 5, dtype=torch.long) for name in ["worldcover", "area", "geometry"]
    }
    weights = {
        name: torch.ones(count)
        for name, count in [("worldcover", 12), ("area", 7), ("geometry", 4)]
    }
    out = net(**data)
    losses = adaptive_residual_loss(
        out, labels, torch.ones(2, 4, 5, dtype=torch.bool), weights, preservation_weight=0.05
    )
    losses["loss"].backward()
    assert net.project.weight.grad.abs().sum() > 0
    for head in [net.worldcover_head, net.area_head, net.geometry_head]:
        assert all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in head.parameters()
        )
    for name, tensor in data.items():
        assert torch.equal(tensor.detach(), before[name])
        assert tensor.grad is None


def test_missing_observations_and_empty_weak_domain_remain_finite_and_ignored():
    net = model()
    data = inputs(1)
    for name in ["public", "semantic", "geometry"]:
        data[name].zero_()
    data["spectral"].fill_(float("nan"))
    data["spectral_mask"].zero_()
    out = net(**data)
    assert torch.isfinite(out.embedding).all() and not out.embedding.any()
    labels = {
        name: torch.zeros(1, 4, 5, dtype=torch.long) for name in ["worldcover", "area", "geometry"]
    }
    weights = {
        name: torch.ones(count)
        for name, count in [("worldcover", 12), ("area", 7), ("geometry", 4)]
    }
    loss = adaptive_residual_loss(
        out, labels, torch.zeros(1, 4, 5, dtype=torch.bool), weights, preservation_weight=0.2
    )
    assert all(value.item() == 0 for value in loss.values())
    loss["loss"].backward()


def test_teacher_binding_is_atomic_strict_and_survives_cold_state_loading():
    net = ContentAdaptiveResidual(width=8)
    with pytest.raises(ValueError, match="teacher"):
        net(**inputs())
    invalid = teacher()
    invalid["spectral_std"].zero_()
    before = copy.deepcopy(net.state_dict())
    with pytest.raises(ValueError):
        net.set_teacher(invalid)
    assert all(torch.equal(before[key], value) for key, value in net.state_dict().items())
    net.set_teacher(teacher())
    cold = ContentAdaptiveResidual(width=8)
    cold.load_state_dict(net.state_dict(), strict=True)
    data = inputs()
    torch.testing.assert_close(net(**data).embedding, cold(**data).embedding, rtol=0, atol=0)


@pytest.mark.parametrize(
    "change", [{"width": True}, {"width": 0}, {"initial_alpha": 0}, {"initial_alpha": float("nan")}]
)
def test_invalid_model_controls_are_rejected(change):
    with pytest.raises(ValueError):
        ContentAdaptiveResidual(**change)


def test_invalid_input_shapes_and_observed_nonfinite_values_are_rejected():
    net = model()
    data = inputs()
    data["geometry"] = data["geometry"][:, :63]
    with pytest.raises(ValueError):
        net(**data)
    data = inputs()
    data["spectral_mask"] = None
    with pytest.raises(ValueError):
        net(**data)


def test_unknown_teacher_fields_and_invalid_weak_loss_contracts_are_rejected():
    net = model()
    values = teacher()
    values["unexpected"] = torch.ones(1)
    with pytest.raises(ValueError):
        net.set_teacher(values)
    out = net(**inputs(1))
    labels = {
        name: torch.ones(1, 4, 5, dtype=torch.long) for name in ["worldcover", "area", "geometry"]
    }
    weights = {
        name: torch.ones(count)
        for name, count in [("worldcover", 12), ("area", 7), ("geometry", 4)]
    }
    domain = torch.ones(1, 4, 5, dtype=torch.bool)
    for invalid in [True, float("nan"), -0.1]:
        with pytest.raises(ValueError):
            adaptive_residual_loss(out, labels, domain, weights, preservation_weight=invalid)
    bad = copy.deepcopy(labels)
    bad["worldcover"] = bad["worldcover"].float()
    with pytest.raises(ValueError):
        adaptive_residual_loss(out, bad, domain, weights, preservation_weight=0.05)
    bad = copy.deepcopy(labels)
    bad["geometry"].fill_(4)
    with pytest.raises(ValueError):
        adaptive_residual_loss(out, bad, domain, weights, preservation_weight=0.05)
    bad_weights = copy.deepcopy(weights)
    bad_weights["area"][1] = -1
    with pytest.raises(ValueError):
        adaptive_residual_loss(out, labels, domain, bad_weights, preservation_weight=0.05)
    with pytest.raises(ValueError):
        adaptive_residual_loss(out, labels, domain.float(), weights, preservation_weight=0.05)
    data = inputs()
    data["spectral"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        net(**data)
