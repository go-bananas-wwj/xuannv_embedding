import numpy as np
import pytest

from xuannv_embedding.export.journal_matched_fusion import shared_feature_parts


def test_matched_fusion_parts_reconstruct_the_deployed_feature_exactly():
    rng = np.random.default_rng(41)
    base = rng.normal(size=(64, 4, 4)).astype("float16")
    final = (base.astype("float32") + rng.normal(0, 0.1, base.shape)).astype("float16")
    parts = shared_feature_parts(base, final)
    assert parts.shape == (128, 4, 4)
    np.testing.assert_array_equal(parts[:64] + parts[64:], final.astype("float32"))
    np.testing.assert_array_equal(parts[:64], base.astype("float32"))


def test_matched_fusion_requires_equal_native_embedding_grids():
    with pytest.raises(ValueError):
        shared_feature_parts(np.zeros((64, 2, 2), "float16"), np.zeros((64, 3, 3), "float16"))
