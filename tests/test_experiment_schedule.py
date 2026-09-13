import pytest

from xuannv_embedding.training.experiment_schedule import select_learning_rate


def test_selection_pairs_all_seeds_and_uses_mean_validation_score():
    records = [
        {"lr": lr, "seed": seed, "score": score}
        for lr, scores in [(1e-4, [1.0, 2.0, 3.0]), (3e-4, [0.0, 3.0, 6.0])]
        for seed, score in zip([41, 42, 43], scores)
    ]
    assert select_learning_rate(records) == (1e-4, {1e-4: 2.0, 3e-4: 3.0})


def test_selection_rejects_missing_seed_or_nonfinite_score():
    with pytest.raises(ValueError):
        select_learning_rate([{"lr": 1e-4, "seed": 41, "score": 1.0}])
    with pytest.raises(ValueError):
        select_learning_rate(
            [
                {"lr": lr, "seed": seed, "score": float("nan")}
                for lr in [1e-4, 3e-4]
                for seed in [41, 42, 43]
            ]
        )
