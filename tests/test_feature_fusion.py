import torch

from xuannv_embedding.models.feature_fusion import FrozenFeatureFusion, feature_fusion_loss


def inputs():
    torch.manual_seed(41)
    return [torch.randn(2, c, 8, 8) for c in [64, 95, 64]]


def test_frozen_feature_fusion_zero_initialization_preserves_public_coordinates_exactly():
    public, spectral, semantic = inputs()
    model = FrozenFeatureFusion(width=16)
    result = model(public, spectral, semantic)
    assert torch.equal(result.embedding, public)
    assert result.worldcover_logits.shape == (2, 12, 8, 8)
    assert result.spectral_reconstruction.shape == spectral.shape


def test_frozen_feature_fusion_missing_imagery_equals_the_learned_public_branch():
    public, spectral, semantic = inputs()
    model = FrozenFeatureFusion(width=16)
    with torch.no_grad():
        model.public_adapter[-1].bias.fill_(0.2)
        model.highres.correction.weight.normal_(std=0.1)
    semantic[:, :, :2] = 0
    absent = model(public, spectral, None)
    missing = model(public, spectral, torch.zeros_like(semantic))
    full = model(public, spectral, semantic)
    assert torch.equal(absent.embedding, absent.public_embedding)
    assert torch.equal(missing.embedding, absent.embedding)
    assert torch.equal(full.embedding[:, :, :2], full.public_embedding[:, :, :2])
    assert not torch.equal(full.embedding[:, :, 2:], full.public_embedding[:, :, 2:])


def test_frozen_feature_fusion_learns_from_real_class_labels_but_not_upstream_encoders():
    public, spectral, semantic = [x.requires_grad_() for x in inputs()]
    model = FrozenFeatureFusion(width=16)
    result = model(public, spectral, semantic)
    labels = torch.randint(1, 12, (2, 8, 8))
    losses = feature_fusion_loss(
        result,
        public,
        spectral,
        semantic,
        labels,
        spectral_mask=torch.ones_like(spectral, dtype=torch.bool),
        class_weights=torch.ones(12),
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.public_adapter[-1].weight.grad.abs().sum() > 0
    assert model.highres.correction.weight.grad.abs().sum() > 0
    assert all(x.grad is None for x in [public, spectral, semantic])


def test_frozen_feature_fusion_no_observations_produce_zero_finite_training_loss():
    public, spectral, _ = inputs()
    spectral.zero_()
    model = FrozenFeatureFusion(width=16)
    result = model(public, spectral)
    losses = feature_fusion_loss(
        result,
        public,
        spectral,
        None,
        torch.ones(2, 8, 8, dtype=torch.long),
        spectral_mask=torch.zeros_like(spectral, dtype=torch.bool),
        class_weights=torch.ones(12),
    )
    assert torch.isfinite(losses["loss"]) and losses["loss"] == 0


def test_feature_fusion_weak_loss_normalizes_only_observed_nonignored_labels():
    public, spectral, semantic = inputs()
    model = FrozenFeatureFusion(width=16)
    result = model(public, spectral, semantic)
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    labels[0, 0, 0] = 1
    expected = torch.nn.functional.cross_entropy(
        result.worldcover_logits[0, :, 0, 0][None], torch.ones(1, dtype=torch.long)
    )
    losses = feature_fusion_loss(
        result,
        public,
        spectral,
        semantic,
        labels,
        spectral_mask=torch.ones_like(spectral, dtype=torch.bool),
        class_weights=torch.ones(12),
    )
    torch.testing.assert_close(losses["worldcover"], expected)


def test_feature_fusion_weak_label_exposure_does_not_expand_when_imagery_is_added():
    public, spectral, semantic = inputs()
    spectral.zero_()
    model = FrozenFeatureFusion(width=16)
    labels = torch.ones(2, 8, 8, dtype=torch.long)
    for imagery in [None, semantic]:
        result = model(public, spectral, imagery)
        losses = feature_fusion_loss(
            result,
            public,
            spectral,
            imagery,
            labels,
            spectral_mask=torch.zeros_like(spectral, dtype=torch.bool),
            class_weights=torch.ones(12),
        )
        assert losses["worldcover"] == 0


def test_feature_fusion_can_retain_old_and_annual_spectral_information_together():
    public, spectral, semantic = inputs()
    model = FrozenFeatureFusion(width=16, spectral_channels=190)
    both_periods = torch.cat([spectral, 2 * spectral], 1)
    result = model(public, both_periods, semantic)
    assert result.spectral_reconstruction.shape == both_periods.shape
    assert model.spectral_mean.shape[1] == 190
    assert torch.equal(result.embedding, public)
