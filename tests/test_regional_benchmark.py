import json

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from xuannv_embedding.downstream.regional_benchmark import (
    _confusion,
    classification_metrics,
    sample_support,
    select_classes,
)


def test_followup_registers_all_comparisons_without_per_task_month_selection():
    from xuannv_embedding.training.regional_followup import jobs

    registered = jobs()
    assert len(registered) == 215
    assert len({tuple(sorted(job.items())) for job in registered}) == 215
    for feature in ["aef", "public", "highres"]:
        assert len([j for j in registered if j["feature"] == feature]) == 45
    assert {j["budget"] for j in registered if j["feature"].endswith("april")} == {200}


def test_figure_scenes_are_determined_by_reference_labels():
    from xuannv_embedding.downstream.regional_figures import choose_scenes

    labels = np.array([[[1, 1], [1, 40]], [[1, 40], [40, 40]]])
    assert choose_scenes(labels, np.ones_like(labels, bool), [1, 40]) == [(1, 0), (40, 1)]


def test_historical_diagnostics_join_by_patch_id_and_keep_actual_reference_years(tmp_path):
    from xuannv_embedding.downstream.regional_benchmark import historical_buildings

    (tmp_path / "evaluation").mkdir()
    (tmp_path / "cache/highres").mkdir(parents=True)
    (tmp_path / "evaluation/building_legacy_patch_targets.csv").write_text(
        "patch_id,coverage_fraction,density_per_km2,reference_years\n"
        + "".join(f"p{i},{i / 10},{i * 10},2018;2021\n" for i in reversed(range(9)))
    )
    (tmp_path / "cache/highres/cache.json").write_text(
        json.dumps({"records": [{"patch_id": f"p{i}"} for i in range(9)]})
    )
    folds = [{"fold": 0, "train": [0, 1, 2], "validation": [3, 4, 5], "test": [6, 7, 8]}]
    result = historical_buildings(
        {"data": str(tmp_path), "trial": str(tmp_path)},
        np.arange(9 * 2 * 2 * 2).reshape(9, 2, 2, 2),
        folds,
    )
    assert result["reference_years"] == [2018, 2021]
    assert result["eligible_for_current_year_building_accuracy"] is False
    assert result["folds"][0]["metrics"]["coverage_fraction"]["mae"] < 0.1


def test_dense_readout_standardization_keeps_float32_inputs():
    torch.set_num_threads(1)
    features = np.ones((1, 3, 8, 8), np.float32)
    scaler = StandardScaler().fit(np.array([[0, 0, 0], [2, 2, 2]], np.float32))
    result = _confusion(
        torch.nn.Conv2d(3, 2, 1),
        features,
        np.zeros((1, 8, 8), np.int64),
        np.ones((1, 8, 8), bool),
        [0],
        [1, 2],
        scaler,
        torch.device("cpu"),
    )
    assert result.sum() == 64


def test_dense_readout_skips_tiles_without_eligible_labels():
    features = np.ones((1, 3, 8, 8), np.float32)
    scaler = StandardScaler().fit(np.array([[0, 0, 0], [2, 2, 2]], np.float32))
    result = _confusion(
        torch.nn.Conv2d(3, 2, 1),
        features,
        np.zeros((1, 8, 8), np.int64),
        np.zeros((1, 8, 8), bool),
        [0],
        [1, 2],
        scaler,
        torch.device("cpu"),
    )
    assert result.sum() == 0


def test_multiclass_metrics_retain_declining_and_empty_prediction_classes():
    confusion = np.array([[8, 2, 0], [1, 9, 0], [0, 10, 0]])
    report = classification_metrics(confusion, [1, 2, 3])
    assert report["per_class"]["3"]["f1"] == 0
    assert report["per_class"]["3"]["recall"] == 0
    assert report["accuracy"] == pytest.approx(17 / 30)
    assert report["macro_f1"] < report["accuracy"]


def test_support_is_nested_distributed_and_uses_only_training_labels():
    labels = np.tile(np.arange(2)[None, None, :], (6, 10, 5))
    allowed = np.ones_like(labels, dtype=bool)
    small = sample_support(labels, allowed, [0, 1, 2], [0, 1], 3, seed=41)
    large = sample_support(labels, allowed, [0, 1, 2], [0, 1], 10, seed=41)
    assert set(small) <= set(large)
    assert set(np.unravel_index(small, labels.shape)[0]) <= {0, 1, 2}
    assert len(set(np.unravel_index(small, labels.shape)[0])) == 3
    assert np.bincount(labels.ravel()[small]).tolist() == [3, 3]
    assert large == sample_support(labels, allowed, [0, 1, 2], [0, 1], 10, seed=41)


def test_class_selection_depends_on_spatial_support_and_counts():
    labels = np.zeros((15, 4, 4), dtype=np.int16)
    labels[:, :, 2:] = 2
    labels[0, 0, 0] = 7
    folds = [{"train": list(range(9)), "validation": [9, 10, 11], "test": [12, 13, 14]}]
    bounds = [(i * 1280, 0, (i + 1) * 1280, 1280) for i in range(15)]
    selected, reasons = select_classes(
        labels, np.ones_like(labels, bool), folds, bounds, min_train=10, min_query=2
    )
    assert selected == [2]
    assert "7" in reasons
