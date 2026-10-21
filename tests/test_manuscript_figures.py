import numpy as np
import pytest

from xuannv_embedding.downstream.manuscript_figures import score_matrix


def test_score_matrix_preserves_task_order_and_both_heads():
    rows = [
        {"head": h, "task": t, "all_region": {"f1": value}}
        for h in ["conv3x3", "mlp"]
        for t, value in [("water", 0.8), ("crop", 0.6)]
    ]
    result = score_matrix({"rows": rows}, ["crop", "water"], "all_region")
    np.testing.assert_array_equal(result, [[0.6, 0.8], [0.6, 0.8]])


def test_score_matrix_rejects_duplicate_or_missing_results():
    row = {"head": "mlp", "task": "crop", "all_region": {"f1": 0.8}}
    with pytest.raises(ValueError):
        score_matrix({"rows": [row, row]}, ["crop"], "all_region")
    with pytest.raises(ValueError):
        score_matrix({"rows": [row]}, ["crop"], "all_region")
