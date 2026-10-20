import numpy as np
import pytest

from xuannv_embedding.downstream.manuscript_evidence import paired_statistics


def test_spatial_bootstrap_reuses_fixed_predictions_and_thresholds():
    y = np.tile([0, 0, 1, 1] * 4, 20).astype("int8")
    candidate = {
        "truth": y,
        "scores": y.astype("float32"),
        "tiles": np.repeat(np.arange(20), 16),
        "valid_indices": np.arange(320),
    }
    baseline = {**candidate, "scores": 1 - candidate["scores"]}
    records = [
        {"bounds": [i % 5 * 40, i // 5 * 40, (i % 5 + 1) * 40, (i // 5 + 1) * 40]}
        for i in range(20)
    ]
    result = paired_statistics(candidate, baseline, 0.5, 0.5, list(range(20)), records, 4)
    assert result["f1_difference"] == 1
    assert result["interval95"] == [1.0, 1.0]
    bad = {**baseline, "truth": 1 - y}
    with pytest.raises(ValueError, match="positions"):
        paired_statistics(candidate, bad, 0.5, 0.5, list(range(20)), records, 4)
