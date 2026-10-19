import copy

import pytest
import torch
from test_content_adaptive_residual import inputs, teacher

from xuannv_embedding.models.content_adaptive_residual import adaptive_residual_loss
from xuannv_embedding.models.public_fallback_gate import PublicFallbackGate


def example():
    data = inputs()
    data["foundation"] = data["public"] + 0.05
    net = PublicFallbackGate(width=8, initial_weights=[0.05, 0.8075, 0.1425])
    net.set_teacher(teacher())
    return net, data


def weak_loss(output):
    shape = (len(output.embedding), *output.embedding.shape[-2:])
    labels = {
        key: torch.ones(shape, dtype=torch.long) for key in ["worldcover", "area", "geometry"]
    }
    weights = {key: torch.ones(n) for key, n in [("worldcover", 12), ("area", 7), ("geometry", 4)]}
    return adaptive_residual_loss(
        output, labels, torch.ones(shape, dtype=torch.bool), weights, preservation_weight=0.2
    )["loss"]


def test_initial_mixture_preserves_three_source_contract_and_has_zero_anchor_change():
    net, data = example()
    out = net(**data)
    expected = 0.05 * data["foundation"] + 0.8075 * data["semantic"] + 0.1425 * data["geometry"]
    torch.testing.assert_close(out.embedding, expected)
    torch.testing.assert_close(out.expert_weights.sum(1), torch.ones_like(out.embedding))
    torch.testing.assert_close(
        out.standardized_change, torch.zeros_like(out.embedding), atol=1e-6, rtol=0
    )
    assert all(
        p.requires_grad
        for head in [net.worldcover_head, net.area_head, net.geometry_head]
        for p in head.parameters()
    )


def test_public_expert_can_be_selected_and_all_outputs_stay_in_coordinate_bounds():
    net, data = example()
    with torch.no_grad():
        net.project.bias[:64].fill_(30)
        net.project.bias[64:].fill_(-30)
    out = net(**data)
    torch.testing.assert_close(out.embedding, data["foundation"], atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        net.project.weight.normal_(0, 0.5)
        net.project.bias.zero_()
    out = net(**data)
    sources = torch.stack([data["foundation"], data["semantic"], data["geometry"]])
    assert (out.embedding >= sources.amin(0) - 1e-6).all()
    assert (out.embedding <= sources.amax(0) + 1e-6).all()


def test_head_warmup_does_not_update_gate_then_joint_learning_reaches_it():
    net, data = example()
    for key in ["public", "foundation", "semantic", "geometry", "spectral"]:
        data[key].requires_grad_()
    gate_before = {
        k: v.clone() for k, v in net.state_dict().items() if k.startswith(("body.", "project."))
    }
    head_before = net.worldcover_head.weight.clone()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    weak_loss(net(**data, heads_only=True)).backward()
    assert all(p.grad is None for p in [*net.body.parameters(), *net.project.parameters()])
    optimizer.step()
    assert all(torch.equal(net.state_dict()[k], v) for k, v in gate_before.items())
    assert not torch.equal(net.worldcover_head.weight, head_before)
    optimizer.zero_grad(set_to_none=True)
    weak_loss(net(**data)).backward()
    assert net.project.weight.grad.abs().sum() > 0
    for key in ["public", "foundation", "semantic", "geometry", "spectral"]:
        assert data[key].grad is None


def test_invalid_public_expert_and_cold_state_loading_are_checked():
    net, data = example()
    out = net(**data)
    cold = PublicFallbackGate(width=8, initial_weights=[0.05, 0.8075, 0.1425])
    cold.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(out.embedding, cold(**data).embedding, atol=0, rtol=0)
    bad = copy.deepcopy(data)
    bad["foundation"] = bad["foundation"][:, :63]
    with pytest.raises(ValueError):
        net(**bad)
    bad = copy.deepcopy(data)
    bad["foundation"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        net(**bad)


@pytest.mark.parametrize(
    "weights",
    [[0, 0.85, 0.15], [0.1, 0.8], [0.1, 0.8, 0.2], [True, 0, 0], [float("nan"), 0.8, 0.2]],
)
def test_invalid_initial_expert_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        PublicFallbackGate(initial_weights=weights)
