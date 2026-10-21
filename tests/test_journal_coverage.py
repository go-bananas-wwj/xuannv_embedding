import numpy as np
import pytest

from xuannv_embedding.downstream.journal_coverage import coverage_domains


def test_coverage_domains_detect_internal_edge_without_inventing_tile_edges():
    coverage = np.zeros((2, 6, 6), np.float32)
    coverage[0, :, :3] = 1
    coverage[1] = 1
    domains = coverage_domains(coverage)
    assert domains["covered"].sum() == 54
    assert domains["uncovered"].sum() == 18
    assert domains["boundary"][0, :, 2:4].all()
    assert domains["boundary"].sum() == 12
    assert not domains["boundary"][1].any()


def test_coverage_domains_reject_nonfinite_or_out_of_range_masks():
    with pytest.raises(ValueError):
        coverage_domains(np.full((1, 3, 3), np.nan))
