import pytest

from xuannv_embedding.training.regional_tuning import (
    select_validation_winner,
    stage_schedule,
    stop_after_validation,
)


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


def test_stage_schedule_keeps_legacy_budget_and_validates_longer_budgets():
    assert stage_schedule({}, 800) == [800]
    assert stage_schedule({"stages": [800, 1600, 2400]}, 2400) == [800, 1600, 2400]
    for stages in [[], [800, 800], [1600, 800], [True], [800, 1200]]:
        with pytest.raises(ValueError):
            stage_schedule({"stages": stages}, 2400)


def test_early_stop_counts_validation_stagnation_without_reading_test_scores():
    assert not stop_after_validation([0.5, 0.49], patience=2)
    assert stop_after_validation([0.5, 0.49, 0.48], patience=2)
    assert not stop_after_validation([0.5, 0.49, 0.51], patience=2)


@pytest.mark.parametrize(
    "stages,scores,budget,selected,completed",
    [([2, 4, 6], [0.5, 0.6, 0.55], 6, 4, 6), ([2, 4, 6, 8], [0.6, 0.59, 0.58, 0.7], 8, 2, 6)],
)
def test_staged_worker_selects_checkpoint_and_stops_without_test_scores(
    tmp_path, monkeypatch, stages, scores, budget, selected, completed
):
    import json

    from xuannv_embedding.export import regional_trial
    from xuannv_embedding.training import regional_tuning as tuning

    directory = tmp_path / "candidate"
    run = directory / "run"
    run.mkdir(parents=True)
    state = {"step": 0}
    spec = {"output": str(run), "training": {"steps": budget}}
    monkeypatch.setattr(tuning, "read_spec", lambda path: (spec, None, None))
    monkeypatch.setattr(
        tuning,
        "select_checkpoint",
        lambda *args: (
            run / f"step{state['step']:06d}.pt" if state["step"] else None,
            state["step"],
            [],
        ),
    )
    trained, exported = [], []

    def train(command, **kwargs):
        step = int(command[command.index("--stop-after-updates") + 1])
        trained.append(step)
        state["step"] = step
        (run / ("final.pt" if step == budget else f"step{step:06d}.pt")).touch()

    monkeypatch.setattr(tuning.subprocess, "run", train)

    def benchmark(plan, candidate, *, step=None):
        root = directory / "stages" / f"step{step:06d}" / "benchmark"
        root.mkdir(parents=True, exist_ok=True)
        return {"output": str(root)}

    monkeypatch.setattr(tuning, "candidate_benchmark", benchmark)

    def export(spec_path, checkpoint, output, **kwargs):
        assert checkpoint.exists()
        exported.append(state["step"])

    monkeypatch.setattr(regional_trial, "export", export)
    monkeypatch.setattr(
        tuning,
        "validate_candidate",
        lambda *args, step, **kwargs: {"validation_macro_f1": scores[stages.index(step)]},
    )
    candidate = {
        "name": "candidate",
        "spec": "unused",
        "approval": "unused",
        "devices": [1, 2],
        "micro_batch": 4,
        "stages": stages,
    }
    tuning.run_candidate({"output": str(tmp_path)}, candidate)
    status = json.loads((directory / "status.json").read_text())
    assert status["selected_step"] == selected
    assert status["completed_steps"] == completed
    assert trained == [1, *[s for s in stages if s <= completed]]
    assert exported == [s for s in stages if s <= completed]
    assert (run / "selected.pt").resolve() == run / f"step{selected:06d}.pt"
    assert (
        directory / "benchmark"
    ).resolve() == directory / "stages" / f"step{selected:06d}" / "benchmark"
