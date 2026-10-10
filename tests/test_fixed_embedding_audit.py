import numpy as np
import pytest
import torch

from xuannv_embedding.downstream.fixed_audit import (
    balanced_positions,
    nested_support,
    paired_bootstrap,
)
from xuannv_embedding.export.context import context_window


def test_context_reads_real_neighbors_and_preserves_center():
    tiles = {(0, 0): torch.ones(1, 4, 4), (1, 0): torch.full((1, 4, 4), 2.0)}
    out = context_window(tiles, (0, 0), 1)
    assert out.shape == (1, 6, 6)
    assert torch.equal(out[:, 1:5, 1:5], tiles[(0, 0)])
    assert (out[:, 1:5, 5] == 2).all()
    assert (out[:, :, 0] == 0).all()


def test_support_is_nested_and_ignores_missing_classes():
    y = np.array([[[0, 1]], [[0, 0]], [[0, 1]], [[0, 1]]])
    a = nested_support(y, ["a", "b", "c", "d"], [0, 1, 2, 3], 1, 42)
    b = nested_support(y, ["a", "b", "c", "d"], [0, 1, 2, 3], 3, 42)
    assert b[:1] == a
    assert 1 not in b
    with pytest.raises(ValueError):
        nested_support(y, ["a", "b", "c", "d"], [0, 1, 2, 3], 4, 42)


def test_balanced_positions_excludes_invalid_labels():
    y = np.array([-1, 0, 0, 1, 1, 1])
    p = balanced_positions(y, 2, 42)
    assert len(p) == 4
    assert 0 not in p
    assert (y[p] == 0).sum() == 2


def test_paired_bootstrap_identical_scores_has_zero_difference():
    counts = np.array([[4, 2, 1], [8, 3, 2]])
    assert paired_bootstrap(counts, counts, repeats=200) == [0.0, 0.0]


def test_accelerated_cosine_knn_matches_reference_on_distinct_points():
    from sklearn.neighbors import KNeighborsClassifier

    from xuannv_embedding.downstream.fixed_audit import accelerated_knn

    rng = np.random.default_rng(9)
    x = rng.normal(size=(30, 8)).astype("float32")
    q = rng.normal(size=(12, 8)).astype("float32")
    y = np.arange(30) % 2
    ref = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(x, y)
    np.testing.assert_allclose(
        accelerated_knn(x, y, q, "cpu"), ref.predict_proba(q)[:, 1], atol=1e-6
    )


def test_perfect_boundaries_and_small_objects_have_full_recall():
    from xuannv_embedding.downstream.fixed_audit import score_tiles

    y = np.zeros((1, 16, 16), dtype=np.int8)
    y[:, 6:9, 6:9] = 1
    metrics = score_tiles(y, y.astype(float), 0.5)
    assert metrics["f1"] == 1
    assert metrics["boundary_f1_10m"] == 1
    assert metrics["boundary_f1_20m"] == 1
    assert metrics["small_object_recall"] == 1
    assert metrics["small_objects"] == 1
