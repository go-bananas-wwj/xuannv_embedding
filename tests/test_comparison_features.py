import numpy as np
import pytest
import torch

from xuannv_embedding.downstream.comparison_features import raw_features, validate_grid


def test_raw_features_preserve_channels_and_zero_invalid_observations():
    sample = {
        "source_frames": {"s2": torch.ones(2, 3, 4, 4)},
        "source_masks": {"s2": torch.ones(2, 1, 4, 4)},
    }
    sample["source_frames"]["s2"][-1, :, 0, 0] = float("nan")
    sample["source_masks"]["s2"][-1, :, 0, 0] = 0
    result = raw_features(sample, ["s2"])
    assert result.shape == (4, 4, 4)
    assert np.isfinite(result).all()
    assert (result[:, 0, 0] == 0).all()


def test_external_grid_rejects_geographic_offset():
    reference = {"bounds": [0, 0, 1280, 1280], "shape": [128, 128]}
    validate_grid(reference, [0, 0, 1280, 1280])
    with pytest.raises(ValueError):
        validate_grid(reference, [10, 0, 1290, 1280])
