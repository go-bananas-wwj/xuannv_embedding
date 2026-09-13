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
