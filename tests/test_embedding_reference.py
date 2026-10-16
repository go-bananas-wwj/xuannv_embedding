import numpy as np
import pytest

from xuannv_embedding.data_process.embedding_reference import dequantize, normalize_vectors


def test_dequantization_is_signed_square_and_masks_nodata():
    raw = np.tile(np.array([-128, -127, -64, 0, 64, 127], dtype=np.int8), (64, 1, 1))
    result = dequantize(raw)
    assert np.isnan(result[:, :, 0]).all()
    np.testing.assert_allclose(
        result[0, 0, 1:],
        np.array([-(127**2), -(64**2), 0, 64**2, 127**2]) / 127.5**2,
        rtol=2e-7,
    )
    with pytest.raises(ValueError, match="int8"):
        dequantize(raw.astype(np.float32))
    raw[0, 0, 1] = -128
    with pytest.raises(ValueError, match="mask"):
        dequantize(raw)


def test_normalization_preserves_missing_vectors_instead_of_imputing():
    values = np.array([[[3, np.nan, 0]], [[4, np.nan, 0]]], dtype=np.float32)
    normalized, valid = normalize_vectors(values)
    np.testing.assert_allclose(normalized[:, 0, 0], [0.6, 0.8])
    assert valid.tolist() == [[True, False, False]]
    assert np.isnan(normalized[:, 0, 1:]).all()
