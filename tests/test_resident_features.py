import numpy as np
import pytest
import torch
from test_semantic_refinement import fixture_spec

from xuannv_embedding.training.resident_features import ResidentFeatures
from xuannv_embedding.training.semantic_refinement import TrainingData, specification


def compare(a, b):
    for left, right in zip(a, b):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            assert all(torch.equal(left[k], right[k]) for k in left)
        elif left is None:
            assert right is None
        else:
            assert torch.equal(left, right)


def test_resident_gathers_exact_streaming_values_masks_targets_and_duplicate_positions(tmp_path):
    path, _ = fixture_spec(tmp_path)
    spec, index = specification(path)
    data = TrainingData(spec, index)
    resident = ResidentFeatures(data, torch.device("cpu"), chunk_size=1)
    compare(data.batch([1, 0, 1]), resident.batch([1, 0, 1]))
    assert all(not x.requires_grad for x in resident.values if isinstance(x, torch.Tensor))
    assert resident.resident_bytes > 0
    # Subsequent steps must never reread or recast disk arrays.
    data.batch = lambda *_: pytest.fail("streaming batch was called after residency")
    assert resident.batch([0])[0].shape == (1, 64, 4, 4)
    for ids in [[], [-1], [2], [True], [0.1]]:
        with pytest.raises(ValueError, match="positions"):
            resident.batch(ids)


def test_resident_masks_unobserved_native_nan_and_preserves_missing_sources(tmp_path):
    path, _ = fixture_spec(tmp_path)
    spec, index = specification(path)
    data = TrainingData(spec, index)
    data.extra["context_mask"] = np.zeros((2, 1, 1, 1), np.float16)
    data.extra["context"] = np.full((2, 1024, 1, 1), np.nan, np.float16)
    data.detail = None
    resident = ResidentFeatures(data, torch.device("cpu"), chunk_size=1)
    compare(data.batch([0, 1]), resident.batch([0, 1]))
    data.extra = {k: None for k in data.extra}
    absent = ResidentFeatures(data, torch.device("cpu"))
    compare(data.batch([1]), absent.batch([1]))


def test_resident_float16_values_are_exactly_promoted_before_training(tmp_path):
    path, _ = fixture_spec(tmp_path)
    spec, index = specification(path)
    data = TrainingData(spec, index)
    # Include signed zero and the smallest half subnormal, with no new precision loss.
    values = np.arange(2048, dtype=np.uint16).view(np.float16)
    data.public = np.resize(values, (2, 64, 4, 4))
    resident = ResidentFeatures(data, torch.device("cpu"))
    compare(data.batch([0, 1]), resident.batch([0, 1]))
    assert resident.batch([0, 1])[0].dtype == torch.float32
