import torch

from xuannv_embedding.models.semantic_residual import SemanticResidualFusion, semantic_loss


def sample():
    torch.manual_seed(41)
    public = torch.randn(2, 64, 16, 16) * torch.linspace(0.1, 10, 64)[None, :, None, None]
    spectral = torch.randn(2, 95, 16, 16)
    detail = torch.randn(2, 64, 16, 16)
    context = torch.randn(2, 1024, 5, 5)
    mask = torch.ones(2, 1, 5, 5)
    visible = torch.ones_like(spectral, dtype=torch.bool)
    return public, spectral, detail, context, mask, visible


def run(model, values, **kwargs):
    p, s, d, c, m, v = values
    return model(p, s, d, c, m, spectral_mask=v, context_source_shape=(80, 80), **kwargs)


def test_zero_initialization_retains_all_nonunit_public_coordinates_exactly():
    values = sample()
    model = SemanticResidualFusion(width=16)
    out = run(model, values)
    assert torch.equal(out.embedding, values[0])
    assert out.worldcover_logits.shape == (2, 12, 16, 16)
    assert out.area_logits.shape == (2, 7, 16, 16)
    assert out.geometry_logits.shape == (2, 4, 16, 16)
    torch.testing.assert_close(out.gates.sum(1), torch.ones(2, 16, 16))


def test_missing_both_local_sources_is_exact_public_fallback_even_after_learning_biases():
    values = list(sample())
    model = SemanticResidualFusion(width=16)
    with torch.no_grad():
        model.public_delta.bias.fill_(0.3)
        model.detail_delta.bias.fill_(2)
        model.context_delta.bias.fill_(-2)
    values[2] = torch.zeros_like(values[2])
    values[4].zero_()
    # Unobserved native tensors cannot leak their placeholders or bias.
    values[3].fill_(float("nan"))
    out = run(model, values)
    assert torch.equal(out.embedding, out.public_embedding)
    assert torch.count_nonzero(out.gates[:, 1:]) == 0
    assert torch.isfinite(out.embedding).all()


def test_source_scales_apply_relative_to_public_channel_standard_deviation():
    values = list(sample())
    model = SemanticResidualFusion(width=16)
    scale = torch.linspace(0.1, 2, 64)
    model.set_statistics("public", torch.zeros(64), scale)
    with torch.no_grad():
        model.public_delta.bias.fill_(0.5)
    out = run(model, values, burnin=True)
    assert torch.equal(out.embedding, values[0])
    out = run(model, values)
    expected = values[0] + scale[None, :, None, None] * (0.25 * torch.tanh(torch.tensor(0.5)))
    torch.testing.assert_close(out.public_embedding, expected)


def test_final_preservation_penalizes_local_change_not_only_public_branch():
    values = sample()
    model = SemanticResidualFusion(width=16)
    with torch.no_grad():
        model.detail_delta.bias.fill_(1)
    out = run(model, values)
    labels = {
        k: torch.zeros(2, 16, 16, dtype=torch.long) for k in ["worldcover", "area", "geometry"]
    }
    targets = {"spectral": torch.zeros_like(values[1]), "detail": None, "context": None}
    weights = {k: torch.ones(n) for k, n in [("worldcover", 12), ("area", 7), ("geometry", 4)]}
    losses = semantic_loss(out, labels, targets, values[-1], weights)
    assert torch.equal(out.public_embedding, values[0])
    assert losses["preservation"] > 0


def test_semantic_heads_backpropagate_without_updating_frozen_observations():
    values = list(sample())
    for i in [0, 1, 2, 3]:
        values[i].requires_grad_()
    model = SemanticResidualFusion(width=16)
    out = run(model, values)
    labels = {
        k: torch.ones(2, 16, 16, dtype=torch.long) for k in ["worldcover", "area", "geometry"]
    }
    targets = {"spectral": torch.zeros_like(values[1]), "detail": values[2], "context": values[2]}
    weights = {k: torch.ones(n) for k, n in [("worldcover", 12), ("area", 7), ("geometry", 4)]}
    loss = semantic_loss(out, labels, targets, values[-1], weights)["loss"]
    loss.backward()
    assert model.context_delta.weight.grad.abs().sum() > 0
    assert model.detail_delta.weight.grad.abs().sum() > 0
    assert all(values[i].grad is None for i in [0, 1, 2, 3])


def test_no_observation_labels_or_local_targets_produce_finite_zero_loss():
    values = list(sample())
    values[0].zero_()
    values[1].zero_()
    values[2].zero_()
    values[4].zero_()
    values[-1].zero_()
    model = SemanticResidualFusion(width=16)
    out = run(model, values)
    labels = {
        k: torch.ones(2, 16, 16, dtype=torch.long) for k in ["worldcover", "area", "geometry"]
    }
    targets = {"spectral": torch.zeros_like(values[1]), "detail": None, "context": None}
    weights = {k: torch.ones(n) for k, n in [("worldcover", 12), ("area", 7), ("geometry", 4)]}
    losses = semantic_loss(out, labels, targets, values[-1], weights)
    assert torch.isfinite(losses["loss"]) and losses["loss"] == 0


def test_fine_coverage_cannot_expand_weak_map_label_domain():
    values = list(sample())
    values[-1].zero_()
    model = SemanticResidualFusion(width=16)
    out = run(model, values)
    labels = {
        k: torch.ones(2, 16, 16, dtype=torch.long) for k in ["worldcover", "area", "geometry"]
    }
    targets = {"spectral": torch.zeros_like(values[1]), "detail": None, "context": None}
    weights = {k: torch.ones(n) for k, n in [("worldcover", 12), ("area", 7), ("geometry", 4)]}
    losses = semantic_loss(out, labels, targets, values[-1], weights)
    assert all(losses[k] == 0 for k in ["worldcover", "area", "geometry"])
