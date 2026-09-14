import numpy as np
from scipy.ndimage import gaussian_filter, shift

from xuannv_embedding.data_process.v5_alignment import audit_translation


def test_textureless_alignment_remains_uncertain():
    result = audit_translation(
        np.ones((128, 128)), np.ones((128, 128)), np.ones((128, 128), dtype=bool), gsd=10
    )
    assert result["status"] == "uncertain"


def test_known_translation_is_measured_without_modifying_inputs():
    reference = gaussian_filter(np.random.default_rng(42).normal(size=(128, 128)), 1)
    moving = shift(reference, (2, -1), mode="reflect")
    before = moving.copy()
    result = audit_translation(reference, moving, np.ones((128, 128), dtype=bool), gsd=10)
    assert result["status"] == "over_limit"
    np.testing.assert_allclose(result["translation_yx_m"], [-20, 10], atol=2)
    np.testing.assert_array_equal(moving, before)


def test_aligned_texture_passes_consistent_windows():
    reference = gaussian_filter(np.random.default_rng(24).normal(size=(128, 128)), 1)
    result = audit_translation(reference, reference.copy(), np.ones((128, 128), dtype=bool), gsd=10)
    assert result["status"] == "passed"
    assert result["valid_windows"] >= 3


def test_realistic_smooth_background_does_not_hide_a_seven_meter_shift():
    y, x = np.mgrid[:256, :256]
    texture = gaussian_filter(np.random.default_rng(57).normal(size=x.shape), 3)
    reference = ((x + 3 * y) / 256 + texture).astype("f4")
    moving = shift(reference, (0, 1.5), order=1, mode="constant", cval=0)
    valid = np.ones(x.shape, bool)
    valid[:, :2] = False
    result = audit_translation(reference, moving, valid, gsd=5)
    assert result["status"] == "over_limit"
    np.testing.assert_allclose(result["translation_yx_m"], [0, -7.5], atol=1)


def test_ambiguous_linear_ramp_cannot_prove_zero_translation():
    y, x = np.mgrid[:160, :160]
    reference = (x + 3 * y).astype("f4")
    result = audit_translation(reference, reference + 100, np.ones(x.shape, bool), gsd=8)
    assert result["status"] == "uncertain"
