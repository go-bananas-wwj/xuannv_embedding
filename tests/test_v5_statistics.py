import numpy as np
import pytest

from xuannv_embedding.data_process.v5_statistics import StreamingBandStatistics


def test_statistics_match_direct_masked_values_across_chunks():
    stats = StreamingBandStatistics(2, reservoir_size=32)
    stats.update(
        np.array([[1.0, 3.0, 99.0], [2.0, 4.0, 99.0]]),
        np.array([[True, True, False], [True, True, False]]),
    )
    stats.update(np.array([[5.0, 7.0], [6.0, 8.0]]), np.ones((2, 2), dtype=bool))
    result = stats.finish()
    np.testing.assert_allclose(result["mean"], [4, 5])
    np.testing.assert_allclose(result["std"], np.sqrt([5, 5]))
    assert result["count"] == [4, 4]


def test_empty_or_constant_channel_cannot_create_fake_statistics():
    stats = StreamingBandStatistics(1)
    with pytest.raises(ValueError):
        stats.finish()
    stats.update(np.ones((1, 4)), np.ones((1, 4), dtype=bool))
    with pytest.raises(ValueError):
        stats.finish()
