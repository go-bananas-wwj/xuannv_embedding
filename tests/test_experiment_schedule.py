import pytest

from xuannv_embedding.training.experiment_schedule import select_learning_rate


def test_follow_isolates_compiler_artifacts_from_code_and_other_runs(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from xuannv_embedding.training import experiment_schedule as schedule

    code = tmp_path / "code"
    code.mkdir()
    cache = tmp_path / "cache" / "cache.json"
    cache.parent.mkdir()
    cache.write_text("{}")
    jobs = []
    for lr in [1e-4, 3e-4]:
        for seed in [41, 42, 43]:
            output = tmp_path / f"pilot_{lr}_{seed}"
            output.mkdir()
            config = output / "config.yaml"
            config.write_text("{}")
            (output / "status.json").write_text(
                json.dumps({"state": "complete", "epoch": 20, "best_validation": 1 / lr})
            )
            jobs.append(
                dict(
                    name=output.name,
                    output=str(output),
                    config=str(config),
                    config_sha256=schedule._sha(config),
                    seed=seed,
                    lr=lr,
                    epochs=20,
                    pid=123,
                )
            )
    (tmp_path / "experiment_registry.json").write_text(
        json.dumps(
            dict(
                runs=jobs,
                code_snapshot=str(code),
                git_sha="fixed",
                cache_sha256=schedule._sha(cache),
            )
        )
    )
    monkeypatch.setattr(schedule, "_alive", lambda pid: False)

    def check_output(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return "fixed\n"
        if command[0] == "git":
            return ""
        return "\n".join(f"No running processes found in NPU {i} " for i in [1, 2, 3])

    directories = []

    def launch(command, *, cwd, **kwargs):
        from pathlib import Path

        directories.append(Path(cwd))
        (Path(cwd) / "fusion_result.json").write_text("compiler artifact")
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "status.json").write_text('{"state":"complete","epoch":800}')
        return SimpleNamespace(pid=456)

    monkeypatch.setattr(schedule.subprocess, "check_output", check_output)
    monkeypatch.setattr(schedule.subprocess, "Popen", launch)
    schedule.follow(tmp_path)
    assert not (code / "fusion_result.json").exists()
    assert len(set(directories)) == 3
    assert all((directory / "fusion_result.json").exists() for directory in directories)
    assert json.loads((tmp_path / "controller_status.json").read_text())["state"] == (
        "baselines_complete"
    )


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
