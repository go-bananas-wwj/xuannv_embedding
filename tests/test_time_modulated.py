"""A real observation axis, time masks and trainable conditioning are contractual."""

import pytest
import torch

from xuannv_embedding.models.time_modulated import TimeModulatedFusion, diversity_loss


def inputs(views=3):
    torch.manual_seed(27)
    b, h = 2, 8
    return {
        "public": torch.randn(b, 64, h, h),
        "spectral": torch.randn(b, 95, h, h),
        "detail": torch.randn(b, 64, h, h),
        "views": torch.randn(b, views, 1024, 2, 2),
        "view_mask": torch.ones(b, views, 1, 2, 2),
        "queries": torch.randn(b, 12, 64, 2, 2),
        "query_mask": torch.ones(b, 12, 1, 2, 2),
        "support": torch.nn.functional.one_hot(torch.arange(views), 12).float(),
        "spectral_mask": torch.ones(b, 95, h, h, dtype=torch.bool),
        "context_source_shape": (32, 32),
    }


def test_zero_initialized_conditioning_retains_public_and_view_permutation():
    x = inputs()
    model = TimeModulatedFusion(width=8)
    y = model(**x)
    assert torch.equal(y.embedding, x["public"])
    with torch.no_grad():
        model.base.context_delta.weight.fill_(0.03)
    y = model(**x)
    perm = torch.tensor([2, 0, 1])
    z = model(
        **{
            **x,
            "views": x["views"][:, perm],
            "view_mask": x["view_mask"][:, perm],
            "support": x["support"][perm],
        }
    )
    torch.testing.assert_close(y.embedding, z.embedding)
    assert y.white_features.shape == x["public"].shape


def test_time_affine_and_view_attention_receive_gradients():
    model = TimeModulatedFusion(width=8)
    with torch.no_grad():
        model.base.context_delta.weight.normal_(0, 0.05)
    model(**inputs()).embedding.square().mean().backward()
    for layer in [model.affine, model.attention]:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0


def test_absent_local_is_exact_learned_public_even_with_nonzero_biases():
    model = TimeModulatedFusion(width=8)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02)
    x = inputs(1)
    x["detail"] = None
    x["views"].fill_(float("nan"))
    x["view_mask"].zero_()
    y = model(**x)
    assert torch.equal(y.embedding, y.public_embedding)
    assert not y.context_mask.any()
    assert torch.isfinite(y.embedding).all()


def test_missing_query_cannot_modulate_a_real_local_observation():
    model = TimeModulatedFusion(width=8)
    x = inputs(1)
    x["query_mask"].zero_()
    x["queries"].fill_(float("nan"))
    with torch.no_grad():
        model.base.context_delta.weight.normal_(0, 0.05)
    y = model(**x)
    with torch.no_grad():
        model.affine.weight.normal_()
        model.affine.bias.fill_(7)
    z = model(**x)
    assert torch.equal(y.embedding, z.embedding)


def test_year_precision_is_one_view_and_invalid_time_support_is_rejected():
    model = TimeModulatedFusion(width=8)
    x = inputs(1)
    x["support"].fill_(1 / 12)
    assert torch.isfinite(model(**x).embedding).all()
    for bad in [torch.zeros(1, 12), torch.ones(1, 12), torch.full((1, 12), float("nan"))]:
        with pytest.raises(ValueError, match="support"):
            model(**{**x, "support": bad})


def test_masked_diversity_matches_direct_moments_and_ignores_unknown_pixels():
    torch.manual_seed(71)
    x = torch.randn(2, 64, 9, 9, requires_grad=True)
    m = torch.ones(2, 1, 9, 9, dtype=torch.bool)
    m[1, :, ::4, ::4] = False
    losses = diversity_loss(x, m, step=0)
    rows = x[:, :, ::4, ::4].permute(0, 2, 3, 1)[m[:, :, ::4, ::4].squeeze(1)]
    variance = (1 - torch.sqrt(rows.var(0, unbiased=True) + 1e-4)).relu().mean()
    covariance = torch.cov(rows.T)
    covariance = covariance - torch.diag_embed(covariance.diag())
    torch.testing.assert_close(losses["variance"], variance)
    torch.testing.assert_close(losses["covariance"], covariance.square().sum() / 64)
    sum(losses.values()).backward()
    assert torch.isfinite(x.grad).all()
    zero = diversity_loss(x, torch.zeros_like(m), step=0)
    assert all(v.item() == 0 for v in zero.values())
