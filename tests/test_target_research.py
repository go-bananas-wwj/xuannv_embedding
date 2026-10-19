import json
from dataclasses import replace
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
