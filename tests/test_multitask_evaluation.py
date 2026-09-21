"""Regression cases for leakage boundaries and nonclassification evaluation."""

import numpy as np
import pytest

from xuannv_embedding.downstream.multitask import (
    block_regression_data,
    check_partition,
    normalized_score,
    regression_metrics,
    retrieval_prototypes,
)


def test_partition_rejects_validation_test_overlap():
    with pytest.raises(ValueError, match="overlap"):
        check_partition({"train": [0], "validation": [1], "test": [1], "buffer": []}, 2)


def test_partition_rejects_missing_and_out_of_range_indices():
    with pytest.raises(ValueError):
        check_partition({"train": [0], "validation": [1], "test": [3], "buffer": []}, 3)


def test_block_regression_ignores_invalid_reference_and_masks_features():
    x = np.ones((1, 4, 4, 2), dtype=np.float32)
    y = np.zeros((1, 4, 4), dtype=np.int8)
    y[0, :2, :2] = 1
    y[0, 0, 0] = -1
    x[0, 0, 0] = 999
    features, fractions, tile_ids = block_regression_data(x, y, [0], block=2, minimum=0.75)
    np.testing.assert_array_equal(features, np.ones((4, 2)))
    np.testing.assert_array_equal(fractions, [1, 0, 0, 0])
    np.testing.assert_array_equal(tile_ids, [0, 0, 0, 0])
    assert len(block_regression_data(x, y, [0], block=2, minimum=0.8)[0]) == 3


def test_regression_reports_constant_targets_without_fabricated_r_squared():
    metrics = regression_metrics(np.zeros(3), np.ones(3))
    assert metrics == {"rmse": 1.0, "mae": 1.0, "bias": 1.0, "r2": None}


def test_retrieval_uses_only_training_components_and_nested_seeds():
    y = np.ones((3, 4, 4), dtype=np.int8)
    x = np.zeros((3, 4, 4, 2), dtype=np.float32)
    x[0, ..., 0] = 1
    x[1, ..., 1] = 1
    x[2] = 999
    p1, ids1 = retrieval_prototypes(x, y, [0, 1], ["a", "b", "forbidden"], 1, 7)
    p2, ids2 = retrieval_prototypes(x, y, [0, 1], ["a", "b", "forbidden"], 2, 7)
    assert ids1 == ids2[:1]
    np.testing.assert_array_equal(p1, p2[:1])
    assert {r["tile"] for r in ids2} == {0, 1}
    np.testing.assert_allclose(np.linalg.norm(p2, axis=1), 1)


def test_multitask_score_balances_families_and_rejects_unpaired_results():
    base = [
        {"key": "c", "family": "C", "error": 0.5},
        {"key": "r", "family": "R", "error": 0.2},
        {"key": "q", "family": "Q", "error": 0.8},
    ]
    candidate = [{**r, "error": r["error"] * 0.9} for r in base]
    assert normalized_score(base, candidate)["score"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="paired"):
        normalized_score(base, candidate[:-1])
    with pytest.raises(ValueError, match="degenerate"):
        normalized_score([{**r, "error": 0} for r in base], candidate)


def test_multitask_classification_records_paired_support_without_test_features(tmp_path):
    from xuannv_embedding.downstream.multitask import _classification

    rng = np.random.default_rng(8)
    y = np.tile(np.indices((8, 8))[1] >= 4, (3, 1, 1)).astype(np.int8)
    x = np.stack([y, rng.normal(0, 0.01, y.shape)], axis=-1).astype(np.float32)
    result = _classification(x, y, [0, 1], [2], ["a", "b", "c"], 1, 5, tmp_path / "a.npz")
    assert result["ap"] == 1
    assert result["ba"] == 1
    assert result["validation_pixels"] == 64
    assert set(result["support_tiles"]) <= {0, 1}
