import pytest
import torch

from xuannv_embedding.models.temporal_feature_fusion import (
    TemporalFeatureFusion,
    temporal_feature_loss,
)


def inputs():
    torch.manual_seed(41)
    return (
        torch.randn(2, 12, 19, 8, 8),
        torch.ones(2, 12, 3, 8, 8, dtype=torch.bool),
        torch.arange(202401, 202413),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 5, 8, 8),
        torch.randn(2, 64, 8, 8),
    )


def test_temporal_fusion_retains_calendar_association_and_monthly_output():
    frames, masks, months, public, terrain, local = inputs()
    model = TemporalFeatureFusion(width=16).eval()
    actual = model(frames, masks, months, public, terrain, local)
    assert actual.monthly_embedding.shape == (2, 12, 64, 8, 8)
    assert actual.worldcover_logits.shape == (2, 12, 8, 8)
    torch.testing.assert_close(actual.embedding, actual.monthly_embedding.mean(1))
    permutation = torch.arange(11, -1, -1)
    same = model(
        frames[:, permutation], masks[:, permutation], months[permutation], public, terrain, local
    )
    changed = model(frames[:, permutation], masks[:, permutation], months, public, terrain, local)
    torch.testing.assert_close(same.embedding, actual.embedding, atol=2e-6, rtol=1e-5)
    assert not torch.allclose(changed.embedding, actual.embedding)


def test_temporal_fusion_exact_public_fallback_when_local_imagery_is_missing():
    frames, masks, months, public, terrain, local = inputs()
    model = TemporalFeatureFusion(width=16).eval()
    initial = model(frames, masks, months, public, terrain, local)
    assert torch.equal(initial.monthly_embedding, initial.public_monthly_embedding)
    with torch.no_grad():
        model.local_correction.weight.normal_(std=0.1)
    absent = model(frames, masks, months, public, terrain)
    missing = model(frames, masks, months, public, terrain, torch.zeros_like(local))
    local[:, :, :2] = 0
    present = model(frames, masks, months, public, terrain, local)
    assert torch.equal(absent.embedding, missing.embedding)
    assert torch.equal(
        present.monthly_embedding[:, :, :, :2], present.public_monthly_embedding[:, :, :, :2]
    )
    assert not torch.equal(
        present.monthly_embedding[:, :, :, 2:], present.public_monthly_embedding[:, :, :, 2:]
    )


def test_temporal_fusion_losses_use_public_label_domain_and_frozen_observations():
    frames, masks, months, public, terrain, local = inputs()
    frames.requires_grad_()
    public.requires_grad_()
    local.requires_grad_()
    model = TemporalFeatureFusion(width=16)
    result = model(frames, masks, months, public, terrain, local)
    labels = torch.randint(1, 12, (2, 8, 8))
    loss = temporal_feature_loss(
        result, frames, masks, public, local, labels, torch.ones(12), model
    )
    assert torch.isfinite(loss["loss"])
    loss["loss"].backward()
    assert model.monthly_projection[-1].weight.grad.abs().sum() > 0
    assert model.local_correction.weight.grad.abs().sum() > 0
    assert frames.grad is None and public.grad is None and local.grad is None
    empty = torch.zeros_like(masks)
    for imagery in [None, local]:
        result = model(
            torch.full_like(frames, float("nan")), empty, months, public, terrain, imagery
        )
        loss = temporal_feature_loss(
            result, frames, empty, public, imagery, labels, torch.ones(12), model
        )
        assert torch.isfinite(result.embedding).all() and loss["worldcover"] == 0


@pytest.mark.parametrize("bad", [202400, 202413])
def test_temporal_fusion_rejects_invalid_calendar_month(bad):
    frames, masks, months, public, terrain, _ = inputs()
    months[0] = bad
    with pytest.raises(ValueError, match="calendar"):
        TemporalFeatureFusion(width=16)(frames, masks, months, public, terrain)
