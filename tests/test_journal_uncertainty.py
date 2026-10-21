import numpy as np
import pytest

from xuannv_embedding.downstream.journal_uncertainty import paired_mae_interval


def test_paired_interval_preserves_pixel_weighting_with_unequal_tile_sizes():
    ours = np.array([[1, 0.2], [9, 1.8]])
    baseline = np.array([[1, 0.3], [9, 2.7]])
    result = paired_mae_interval(ours, baseline, repeats=100, seed=41)
    assert result["baseline_minus_fusion_mae"] == pytest.approx(0.1)
    assert result["percentile95"] == pytest.approx([0.1, 0.1])


def test_paired_interval_requires_the_same_query_cells():
    with pytest.raises(ValueError):
        paired_mae_interval(np.array([[2, 0.1]]), np.array([[3, 0.3]]))


def test_query_statistics_exclude_global_ids_with_gaps_in_the_inventory():
    from xuannv_embedding.downstream.journal_uncertainty import query_statistics

    rows = [{"mixed_cells": [2, 0, 0.4, 0]}, {"mixed_cells": [8, 0, 7, 0]}]
    selected = query_statistics(rows, [11, 90], {90}, "mixed_cells")
    assert selected == [[2, 0.4]]
