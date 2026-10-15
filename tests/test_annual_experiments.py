from collections import Counter
from types import SimpleNamespace

import pytest

from xuannv_embedding.training.annual_experiments import coverage_stratum, stratified_indices


def test_cluster_bootstrap_preserves_pairing_and_groups_adjacent_patches():
    import numpy as np

    from xuannv_embedding.downstream.annual_ablation import (
        folds_for,
        paired_group_bootstrap,
        spatial_group,
    )

    assert spatial_group("parent_32650:32:64") == spatial_group("parent_32650:33:65")
    assert spatial_group("parent_32650:32:64") != spatial_group("parent_32650:48:64")
    assert (
        folds_for(["a", "b", "a", "c", "d", "e"])[0] == folds_for(["a", "b", "a", "c", "d", "e"])[2]
    )
    a = np.arange(10, dtype=float).reshape(5, 2) + 1
    result = paired_group_bootstrap(
        a, a, ["a", "b", "c", "d", "e"], lambda x: x[0] / x.sum(), draws=100
    )
    assert result["difference"] == 0 and result["CI95"] == [0, 0]
    assert result["groups"] == 5


def test_five_meter_counts_measure_regions_and_one_pixel_boundary_tolerance():
    import torch

    from xuannv_embedding.downstream.annual_ablation import city_counts, city_metrics

    truth = torch.zeros(1, 16, 16, dtype=torch.uint8)
    truth[:, 4:12, 7:9] = 1
    exact = city_metrics(city_counts(truth.bool(), truth).sum(0))
    assert exact == {"F1": 1, "IoU": 1, "boundaryF1": 1}
    shifted = city_metrics(city_counts(truth.roll(1, -1).bool(), truth).sum(0))
    assert shifted["F1"] < 1 and shifted["boundaryF1"] == 1


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
