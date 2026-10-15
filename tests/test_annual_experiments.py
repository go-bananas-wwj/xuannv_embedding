from collections import Counter
from types import SimpleNamespace

import pytest

from xuannv_embedding.training.annual_experiments import coverage_stratum, stratified_indices


def records():
    return [
        SimpleNamespace(
            patch_id=f"p{index}",
            grid={"parent_key": "32650:0:0"},
            provenance={
                "observations": {
                    "s2": [1],
                    "GF6_PAN_c1": [1] if index < 20 else [],
                    "optical": [1] if index < 10 else [],
                }
            },
        )
        for index in range(100)
    ]


def test_hash_sampling_preserves_coverage_proportions_and_excludes_development_patches():
    source = records()
    selected = stratified_indices(source, 40, "seed")
    assert len(set(selected)) == 40
    assert Counter(coverage_stratum(source[index]) for index in selected) == {
        "32650:ms1:pan1": 4,
        "32650:ms0:pan1": 4,
        "32650:ms0:pan0": 32,
    }
    assert selected == stratified_indices(source, 40, "seed")
    assert selected != stratified_indices(source, 40, "different-seed")
    excluded = {source[index].patch_id for index in selected}
    alternative = stratified_indices(source, 40, "seed", excluded)
    assert not set(alternative).intersection(selected)
    with pytest.raises(ValueError, match="eligible"):
        stratified_indices(source, 61, "seed", excluded)
