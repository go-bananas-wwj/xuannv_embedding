import pytest

from xuannv_embedding.training.regional_tuning import select_validation_winner


def test_selection_uses_validation_only_and_keeps_parent_if_no_improvement():
    candidates = {
        "parent": {"validation_macro_f1": 0.5, "test_macro_f1": 0.1},
        "candidate": {"validation_macro_f1": 0.49, "test_macro_f1": 0.99},
    }
    assert select_validation_winner(candidates) == "parent"
    candidates["candidate"]["validation_macro_f1"] = 0.51
    assert select_validation_winner(candidates) == "candidate"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_validation_score_is_rejected(value):
    with pytest.raises(ValueError):
        select_validation_winner({"candidate": {"validation_macro_f1": value}})


def test_empty_validation_candidates_are_rejected():
    with pytest.raises(ValueError):
        select_validation_winner({})
