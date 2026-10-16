import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_annual_model import annual_config
from test_target_supervision import target_spec

from xuannv_embedding.config import Config, ConfigError
from xuannv_embedding.training.target_supervision import validate_research_budget


def test_bounded_research_requires_explicit_fresh_initialization_and_total_budget():
    target = target_spec(engineering_only=False, research_max_updates=1024)
    valid = dict(steps=512, epochs=2, initialize_from=Path("original.pt"), resume=None)
    validate_research_budget(target, **valid)
    for changes in (
        {"steps": 0},
        {"epochs": 3},
        {"epochs": 0},
        {"initialize_from": None},
        {"resume": Path("partial.pt")},
    ):
        with pytest.raises(ValueError):
            validate_research_budget(target, **{**valid, **changes})
    # Default formal/engineering modes retain their existing separate contracts.
    validate_research_budget(target_spec(), steps=0, epochs=200, initialize_from=None, resume=None)


def test_research_config_is_strict_and_cannot_also_claim_engineering_mode(tmp_path):
    config = annual_config()
    raw = replace(
        config, training=replace(config.training, independent_targets=target_spec())
    ).to_dict()
    target = raw["training"]["independent_targets"]
    for dataset in raw["data"]["datasets"]:
        dataset["source_map"] = {name: name for name in raw["model"]["input_sources"]}
    target["research_max_updates"] = 1024
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="mutually exclusive"):
        Config.from_yaml(path)
    target["engineering_only"] = False
    for value in (-1, True, 1.5):
        target["research_max_updates"] = value
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError):
            Config.from_yaml(path)
    target["research_max_updates"] = 1024
    path.write_text(yaml.safe_dump(raw))
    assert Config.from_yaml(path).training.independent_targets.research_max_updates == 1024


def test_reconstruction_never_requests_training_supervision(tmp_path, monkeypatch):
    from xuannv_embedding.training import annual, annual_experiments

    config = annual_config()
    config = replace(config, training=replace(config.training, independent_targets=target_spec()))
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"arms": {"joint": {"config": "config.yaml", "checkpoint": "model.pt"}}})
    )
    monkeypatch.setattr(annual_experiments, "load_model", lambda *args: (config, None, {}))

    class Captured(Exception):
        pass

    def loss(c):
        assert c.training.independent_targets is None
        assert c.training.highres_visible_weight == 0
        assert c.training.uniformity_weight == 0
        raise Captured

    monkeypatch.setattr(annual, "AnnualLoss", loss)
    with pytest.raises(Captured):
        annual_experiments.reconstruction(SimpleNamespace(plan=plan, arm="joint", device="cpu"))


@pytest.mark.parametrize("research", [0, 1024])
def test_bounded_research_does_not_release_formal_training(tmp_path, monkeypatch, research):
    from xuannv_embedding.data import targets
    from xuannv_embedding.training import annual_data

    target = target_spec(engineering_only=False, research_max_updates=research)
    config = annual_config()
    config = replace(config, training=replace(config.training, independent_targets=target))
    reader = SimpleNamespace(
        preparation={"formal_training_released": False}, dem_mean=100, dem_std=20
    )
    monkeypatch.setattr(targets, "AnnualTargetReader", lambda *args, **kwargs: reader)

    class DatasetReached(Exception):
        pass

    def dataset(*args, **kwargs):
        raise DatasetReached

    monkeypatch.setattr(annual_data, "annual_dataset", dataset)
    expected = DatasetReached if research else ValueError
    with pytest.raises(expected):
        annual_data.build_annual_stream(
            config,
            distributed=False,
            max_records=None,
            max_steps=512,
            start_epoch=0,
            teacher_enabled=False,
        )
    assert reader.preparation["formal_training_released"] is False


def test_comparison_freezes_only_paired_recipes_and_rejects_evaluation_overlap(
    tmp_path, monkeypatch
):
    import copy

    from xuannv_embedding.training import target_comparison
    from xuannv_embedding.training.annual_experiments import digest, write_json

    prepared = tmp_path / "ready"
    (prepared / "local_targets").mkdir(parents=True)
    positions = [f"parent_32643:{i}:0" for i in range(8192)]
    write_json(prepared / "experiment_splits.json", {"screen_patch_ids": positions})
    write_json(
        prepared / "preparation.json",
        {
            "splits_sha256": digest(prepared / "experiment_splits.json"),
            "formal_training_released": False,
        },
    )
    write_json(prepared / "local_targets/manifest.json", {})
    raw = annual_config().to_dict()
    target = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(target_spec()).items()}
    target.update(
        prepared_root=str(prepared), preparation_sha256=digest(prepared / "preparation.json")
    )
    raw["training"]["independent_targets"] = target
    dataset = raw["data"]["datasets"][0]
    dataset["source_map"] = {name: name for name in raw["model"]["input_sources"]}
    raw["data"]["datasets"] = [copy.deepcopy(dataset), copy.deepcopy(dataset)]
    raw["data"]["datasets"][1]["region"] = "national2021"
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump(raw))
    manifests = []
    for year in (2020, 2021):
        path = tmp_path / f"{year}.train.manifest.jsonl"
        path.write_text("Frozen fixture indexed through the manifest reader")
        manifests.append({"path": str(path), "sha256": digest(path), "records": 8192})
    monkeypatch.setattr(
        target_comparison,
        "IndexedManifest",
        lambda p: [SimpleNamespace(patch_id=i) for i in positions],
    )
    parent = tmp_path / "parent.json"
    write_json(parent, {"manifests": {"screen": manifests}})
    local = tmp_path / "local.json"
    write_json(
        local,
        {
            "parent_plan": {"path": str(parent), "sha256": digest(parent)},
            "configs": {"multisource": {"path": str(template), "sha256": digest(template)}},
        },
    )
    labels = tmp_path / "labels"
    labels.mkdir()
    write_json(labels / "labels.json", {"records": []})
    cities = tmp_path / "cities.json"
    write_json(cities, [])
    evaluation = tmp_path / "evaluation.json"
    old = {
        "labels": str(labels),
        "cities": str(cities),
        "osm": str(tmp_path / "osm"),
        "assets": {},
        "initialize_config": str(template),
        "initialize_checkpoint": str(tmp_path / "original.pt"),
        "datasets": [],
        "probe_steps": 600,
        "baselines": {"original_exact": {}},
    }
    write_json(evaluation, old)
    result = target_comparison.prepare_comparison(local, evaluation, tmp_path / "comparison")
    assert len(result["arms"]) == 4 and not result["formal_target_training_released"]
    for seed in (42, 17):
        a = Config.from_yaml(result["arms"][f"control_s{seed}"]["config"])
        b = Config.from_yaml(result["arms"][f"multisource_s{seed}"]["config"])
        assert a.model == b.model and a.data == b.data
        assert a.experiment.seed == b.experiment.seed == seed
        assert a.training.independent_targets is None
        assert b.training.independent_targets.research_max_updates == 1024
        assert b.training.independent_targets.ramp_batches == 128
        assert a.training == replace(b.training, independent_targets=None)
        assert result["arms"][f"multisource_s{seed}"]["steps_per_epoch"] * b.training.epochs == 1024
    with pytest.raises(FileExistsError):
        target_comparison.prepare_comparison(local, evaluation, tmp_path / "comparison")
    write_json(labels / "labels.json", {"records": [{"patch_id": positions[0]}]})
    with pytest.raises(ValueError, match="overlaps"):
        target_comparison.prepare_comparison(local, evaluation, tmp_path / "overlap")
    assert not (tmp_path / "overlap").exists()
