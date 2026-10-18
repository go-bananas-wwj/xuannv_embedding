import numpy as np
import pytest
import torch

from xuannv_embedding.export.semantic_features import (
    align_patch_features,
    restore_rgb,
)


def test_rgb_normalization_uses_physical_values_and_satellite_statistics():
    image = torch.zeros(3, 16, 16)
    mean = [51.0, 102.0, 153.0]
    std = [10.0, 10.0, 10.0]
    output = restore_rgb(image, torch.ones(1, 16, 16), mean, std, [0.2, 0.4, 0.6], [0.1, 0.2, 0.3])
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
    image[:, 0, 0] = 100
    output = restore_rgb(image, torch.zeros(1, 16, 16), mean, std, [0.2, 0.4, 0.6], [0.1, 0.2, 0.3])
    assert torch.count_nonzero(output) == 0


def test_patch_alignment_accounts_for_padding_without_stretching_geography():
    # Three width-wise tokens centered at 8, 24 and 40 source pixels.
    values = torch.tensor([8.0, 24.0, 40.0]).expand(1, 1, 2, 3).clone()
    result = align_patch_features(values, (30, 46), (15, 23), patch=16)
    assert result.shape == (1, 1, 15, 23)
    # Output cell 8 is centered at source x=17, not padded-width x=17.739.
    assert result[0, 0, 7, 8].item() == pytest.approx(17.0, abs=1e-5)
    with pytest.raises(ValueError):
        align_patch_features(values, (32, 49), (15, 23), patch=16)


def test_patch_alignment_is_finite_for_tiny_output():
    values = torch.ones(1, 3, 2, 2)
    assert np.isfinite(align_patch_features(values, (31, 31), (1, 1), patch=16).numpy()).all()
