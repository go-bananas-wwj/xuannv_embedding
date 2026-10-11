"""Public-observation control with no high-resolution inputs, targets or ancestors."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from xuannv_embedding.config import Config
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.runtime import TrainingSystem

PROTOCOL = "public-static-refinement-v1"


class PublicRefinementModel(nn.Module):
    """Keep public optimizer names and deterministic vMF identical to the HR route."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def train(self, mode: bool = True) -> PublicRefinementModel:
        super().train(mode)
        self.base.eval()
        self.base.stp_encoder.train(mode)
        return self

    def forward(
        self, source_frames, source_masks, timestamps, highres_frames=None, highres_masks=None
    ):
        if highres_frames or highres_masks:
            raise ValueError("public control cannot consume highres observations")
        return self.base(source_frames, source_masks, timestamps)


def validate_public_model(model: Any) -> None:
    if not model.input_sources or any(s.role != "temporal" for s in model.input_sources.values()):
        raise ValueError("public control requires exclusively temporal public inputs")
    if model.highres_transformer is not None:
        raise ValueError("public control cannot register a highres transformer")
    for target in model.target_heads.values():
        if target.source not in model.input_sources and target.loss_type != "categorical":
            raise ValueError("public control cannot reconstruct an unregistered target source")


def initialize(spec, cache, labels):
    from xuannv_embedding.training.regional_refinement import (
        StaticObjective,
        registered,
        rename_targets,
    )

    config = Config.from_yaml(spec["parent_config"]["path"])
    validate_public_model(config.model)
    parent = json.loads(Path(spec["parent_registration"]["path"]).read_text())
    expected = {
        "protocol",
        "initial_config",
        "initial_registration",
        "checkpoint",
        "config_sha256",
        "cache_sha256",
    }
    if (
        not expected.issubset(parent)
        or set(parent) - expected - {"initial_static_decoder"}
        or parent["protocol"] != "public-base-ancestor-v1"
    ):
        raise ValueError("public control requires an exact public ancestor registration")
    if (
        parent["config_sha256"] != spec["parent_config"]["sha256"]
        or parent["cache_sha256"] != spec["cache"]["sha256"]
        or parent["checkpoint"] != spec["parent_checkpoint"]
    ):
        raise ValueError("public control ancestor identities differ")
    initial_config = Config.from_yaml(registered(parent["initial_config"]))
    original = json.loads(registered(parent["initial_registration"]).read_text())
    validate_public_model(initial_config.model)
    if (
        original.get("adaptation")
        or original["config_sha256"] != parent["initial_config"]["sha256"]
    ):
        raise ValueError("public ancestor cannot inherit a highres adaptation")
    if asdict(config.model) != asdict(initial_config.model):
        raise ValueError("public ancestor model or reconstruction schema changed")
    if config.data.months != initial_config.data.months or [
        d.region for d in config.data.datasets
    ] != [d.region for d in initial_config.data.datasets]:
        raise ValueError("public ancestor geography or observation window changed")
    schema = {k: asdict(v) for k, v in config.model.input_sources.items()}
    if cache["model_inputs"] != schema:
        raise ValueError("public cache schema differs from the model")
    system = build_training_system(config)
    state = load_training_checkpoint(
        spec["parent_checkpoint"]["path"],
        model=system.model,
        criterion=system.criterion,
        expected_config_sha256=parent["initial_config"]["sha256"],
        expected_source_schema=schema,
        expected_regions=[d.region for d in config.data.datasets],
    )
    if state["git_sha"] != original["git_sha"]:
        raise ValueError("public checkpoint code identity changed")
    system.model = PublicRefinementModel(system.model)
    rename_targets(system.model, system.criterion, spec["target_aliases"])
    system.criterion.set_epoch(config.training.epochs)
    objective = StaticObjective(
        system.criterion, config.model.embed_dim, labels["target"], labels["classes"]
    )
    if "initial_static_decoder" in parent:
        state = torch.load(
            registered(parent["initial_static_decoder"]), map_location="cpu", weights_only=True
        )
        if state.get("optimizer_steps") != 0:
            raise ValueError("shared static initialization must be untrained")
        objective.static_decoder.load_state_dict(state["state_dict"], strict=True)
    return TrainingSystem(system.model, objective), config
