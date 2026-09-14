import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift


def test_unsigned_structural_features_ignore_invalid_values_and_intensity_inversion():
    from xuannv_embedding.data_process.v5_cfog_alignment import cfog_features

    values = gaussian_filter(np.random.default_rng(42).normal(size=(80, 80)), 1)
    valid = np.ones(values.shape, bool)
    valid[30:35, 20:40] = False
    features, support = cfog_features(values, valid)
    changed = values.copy()
    changed[~valid] = np.nan
    other, other_support = cfog_features(changed, valid)
    inverted, _ = cfog_features(12 - 3 * values, valid)
    assert features.shape == (9, 80, 80)
    assert np.array_equal(support, other_support)
    np.testing.assert_array_equal(features, other)
    np.testing.assert_allclose(features, inverted, atol=1e-12)
    assert not support[26:39, 16:44].any()
    assert not support[:4].any() and not features[:, ~support].any()
    with pytest.raises(ValueError, match="finite"):
        cfog_features(changed, np.ones_like(valid))


def test_feature_correlation_matches_direct_masked_channel_centering():
    from xuannv_embedding.data_process.v5_cfog_alignment import feature_ncc

    rng = np.random.default_rng(51)
    a, b = rng.normal(size=(9, 5, 6)), rng.normal(size=(9, 8, 9))
    ma, mb = rng.random((5, 6)) > 0.1, rng.random((8, 9)) > 0.1
    score = feature_ncc(a, b, ma, mb)
    for y, x in np.ndindex(score.shape):
        mask = ma & mb[y : y + 5, x : x + 6]
        if mask.sum() < 30 * 0.7:
            assert score[y, x] == -np.inf
            continue
        u, v = a[:, mask], b[:, y : y + 5, x : x + 6][:, mask]
        u, v = u - u.mean(axis=1, keepdims=True), v - v.mean(axis=1, keepdims=True)
        expected = (u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum())
        assert score[y, x] == pytest.approx(expected, abs=1e-12)


def test_structural_alignment_recovers_fractional_shifts_with_separate_cloud_masks():
    from xuannv_embedding.data_process.v5_cfog_alignment import audit_cfog

    a = gaussian_filter(np.random.default_rng(42).normal(size=(192, 192)), 1.5)
    valid = np.ones(a.shape, bool)
    valid[82:108, 70:118] = False
    for delta in [(0, 0), (0, 0.5), (0, 1.5), (2, -1)]:
        b = shift(-3 * a + 8, delta, order=1, mode="constant", cval=0)
        mb = shift(valid.astype(float), delta, order=1, mode="constant", cval=0) >= 1 - 1e-6
        result = audit_cfog(a, b, valid, mb, gsd=5)
        assert result["status"] != "uncertain"
        np.testing.assert_allclose(
            np.array(result["translation_yx_m"]) / 5, -np.array(delta), atol=0.35
        )
        assert not result["pixel_fusion_authorized"]
        assert len(result["windows"]) <= 4
        for w in result["windows"]:
            assert w["valid_pixels"] >= 64 * 64 * 0.7


def test_structural_alignment_rejects_flat_ramps_periodicity_and_unrelated_textures():
    from xuannv_embedding.data_process.v5_cfog_alignment import audit_cfog

    yy, xx = np.indices((192, 192))
    a = gaussian_filter(np.random.default_rng(43).normal(size=xx.shape), 1.5)
    b = gaussian_filter(np.random.default_rng(44).normal(size=xx.shape), 1.5)
    mask = np.ones(xx.shape, bool)
    for left, right in [
        (xx * 0 + 1, xx * 0 + 1),
        (xx.astype(float), xx.astype(float)),
        (
            np.sin(xx * np.pi / 4) + np.cos(yy * np.pi / 4),
            np.sin(xx * np.pi / 4) + np.cos(yy * np.pi / 4),
        ),
        (a, b),
    ]:
        assert audit_cfog(left, right, mask, mask, gsd=5)["status"] == "uncertain"
    with pytest.raises(ValueError):
        audit_cfog(a, b, mask, mask, gsd=float("nan"))


def test_structural_layout_keeps_usable_interior_without_double_eroding_selection():
    from xuannv_embedding.data_process.v5_cfog_alignment import audit_cfog

    a = gaussian_filter(np.random.default_rng(19).normal(size=(256, 256)), 1.5)
    valid = np.zeros(a.shape, bool)
    valid[70:186, 70:186] = True
    result = audit_cfog(a, a, valid, valid, gsd=5)
    assert result["status"] == "passed"
    assert result["valid_windows"] == 4
