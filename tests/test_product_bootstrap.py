import numpy as np
from sklearn.metrics import average_precision_score


def test_block_weighted_ap_matches_explicit_duplicate_tiles_and_ties():
    from xuannv_embedding.downstream.product_bootstrap import bootstrap_ap

    labels = np.array([[1, 0, -1, 1], [0, 0, 1, 0], [0, 0, 0, 0]])
    scores = np.array([[0.5, 0.5, np.nan, 0.2], [0.8, 0.2, 0.2, 0.1], [1, 0.1, 0, 0]])
    weights = np.array([[1, 1, 1], [2, 1, 0], [0, 0, 3], [0, 2, 1]])
    actual = bootstrap_ap(labels, scores, weights)
    for i, counts in enumerate(weights):
        blocks = np.repeat(np.arange(3), counts)
        y, s = labels[blocks].ravel(), scores[blocks].ravel()
        valid = y >= 0
        expected = average_precision_score(y[valid], s[valid]) if y[valid].sum() else 0.0
        np.testing.assert_allclose(actual[i], expected, atol=1e-14)


def test_bootstrap_weights_match_paired_count_schedule():
    from xuannv_embedding.downstream.product_bootstrap import block_weights

    got = block_weights(57)
    indices = np.random.default_rng(20260921).integers(57, size=(2000, 57))
    np.testing.assert_array_equal(got.sum(1), 57)
    np.testing.assert_array_equal(got[0], np.bincount(indices[0], minlength=57))
