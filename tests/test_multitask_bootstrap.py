import numpy as np
import pytest
from sklearn.metrics import average_precision_score, mean_squared_error

from xuannv_embedding.downstream.multitask_bootstrap import (
    paired_seed_summary,
    seed_metric_draws,
    tile_weights,
)


def bundle(truth, predictions, tiles, weights, metric="ap", **kwargs):
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim == 1:
        predictions = predictions[None, None]
    return seed_metric_draws(
        truth,
        predictions,
        tiles,
        weights,
        tile_ids=kwargs.pop("tile_ids", ("west", "east")),
        training_seeds=kwargs.pop("training_seeds", (41,)),
        support_seeds=kwargs.pop("support_seeds", (17,)),
        metric=metric,
        **kwargs,
    )


@pytest.mark.parametrize("metric", ["ap", "rmse"])
def test_pooled_metrics_match_literal_tile_duplication_with_ties(metric):
    truth = np.array([1, 0, 1, 0, 0])
    prediction = np.array([0.5, 0.5, 0.9, 0.9, 0.1])
    tiles = np.array([0, 0, 1, 1, 1])
    weights = np.array([[2, 0], [1, 1], [0, 2]])
    result = bundle(truth, prediction, tiles, weights, metric)
    expected = []
    for counts in weights:
        indices = np.concatenate(
            [np.tile(np.flatnonzero(tiles == i), count) for i, count in enumerate(counts)]
        )
        if metric == "ap":
            value = average_precision_score(truth[indices], prediction[indices])
        else:
            value = np.sqrt(mean_squared_error(truth[indices], prediction[indices]))
        expected.append(value)
    np.testing.assert_allclose(result.draws[0, 0], expected, atol=1e-14)
    assert result.observed[0, 0] == pytest.approx(expected[1])


def test_mean_metrics_across_training_and_support_seeds_precedes_difference():
    weights = np.array([[1, 1], [2, 0], [0, 2]])
    truth, tiles = [0, 1], [0, 1]
    baseline = bundle(truth, [0.5, 0.5], tiles, weights, "rmse")
    candidate = bundle(
        truth,
        [[[0, 0]], [[1, 1]]],
        tiles,
        weights,
        "rmse",
        training_seeds=(41, 42),
    )
    result = paired_seed_summary(baseline, candidate)
    # Averaging predictions first would instead produce zero observed difference.
    assert result["observed_difference"] == pytest.approx(np.sqrt(0.5) - 0.5)
    np.testing.assert_allclose(result["differences"], [np.sqrt(0.5) - 0.5, 0, 0])
    assert result["difference_direction"] == "candidate_minus_baseline"


def test_identical_predictions_give_exact_zero_paired_interval():
    weights = tile_weights(("west", "east"), repeats=40)
    result = bundle([0, 1, 0, 1], [0.1, 0.8, 0.3, 0.7], [0, 0, 1, 1], weights)
    paired = paired_seed_summary(result, result)
    assert paired["interval95"] == [0.0, 0.0]
    assert paired["defined_draws"] == 40


def test_support_seeds_are_averaged_as_metrics_not_ensembled_predictions():
    weights = np.array([[1, 1], [2, 0], [0, 2]])
    # Each tile contains both classes, so every AP replicate is defined.
    truth, tiles = [0, 1, 0, 1], [0, 0, 1, 1]
    baseline = bundle(
        truth,
        [[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]],
        tiles,
        weights,
        support_seeds=(17, 18),
    )
    candidate = bundle(
        truth,
        [[[0, 1, 0, 1], [1, 0, 1, 0]]],
        tiles,
        weights,
        support_seeds=(17, 18),
    )
    result = paired_seed_summary(baseline, candidate)
    assert result["observed_difference"] == pytest.approx(0.25)
    np.testing.assert_allclose(result["interval95"], [0.25, 0.25])


def test_missing_positive_draw_is_explicit_and_cannot_produce_unconditional_interval():
    weights = np.array([[2, 0], [1, 1], [0, 2]])
    result = bundle([0, 1], [0.1, 0.9], [0, 1], weights)
    assert np.isnan(result.draws[0, 0, 0])
    paired = paired_seed_summary(result, result)
    assert paired["observed_difference"] == 0
    assert paired["defined_draws"] == 2
    assert paired["interval95"] is None


def test_empty_selected_domain_is_undefined_for_regression():
    result = bundle([0.2], [0.4], [0], [[2, 0], [0, 2]], "rmse")
    assert result.draws[0, 0, 0] == pytest.approx(0.2)
    assert np.isnan(result.draws[0, 0, 1])


@pytest.mark.parametrize(
    "change",
    [
        {"tile_ids": ("east", "west")},
        {"support_seeds": (18,)},
        {"truth": [1, 0]},
        {"tiles": [1, 0]},
        {"weights": [[1, 1], [0, 2]]},
        {"metric": "rmse"},
    ],
)
def test_pairing_rejects_different_domains_draws_or_support_conditions(change):
    args = dict(truth=[0, 1], predictions=[0.1, 0.9], tiles=[0, 1], weights=[[1, 1], [2, 0]])
    baseline = bundle(**args)
    args.update(change)
    with pytest.raises(ValueError, match="paired"):
        paired_seed_summary(baseline, bundle(**args))


@pytest.mark.parametrize(
    "change",
    [
        {"weights": [[0, 0]]},
        {"weights": [[-1, 3]]},
        {"weights": [[0.5, 1.5]]},
        {"weights": [[1, float("nan")]]},
        {"tiles": [0, 2]},
        {"tiles": [0, 0.5]},
        {"predictions": [0.1, float("inf")]},
        {"truth": [0, 2]},
        {"tile_ids": ("west", "west")},
        {"training_seeds": (41, 41)},
        {"support_seeds": (17, 18)},
    ],
)
def test_invalid_scientific_inputs_are_rejected(change):
    args = dict(truth=[0, 1], predictions=[0.1, 0.9], tiles=[0, 1], weights=[[1, 1]])
    args.update(change)
    with pytest.raises(ValueError):
        bundle(**args)


def test_default_resampling_matches_registered_2000_draw_schedule():
    expected = np.random.default_rng(20260921).integers(2, size=(2000, 2))
    counts = np.array([np.bincount(row, minlength=2) for row in expected])
    np.testing.assert_array_equal(tile_weights(("west", "east")), counts)
