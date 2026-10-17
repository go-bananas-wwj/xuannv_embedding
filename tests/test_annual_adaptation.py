import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_annual_model import annual_config, raw_batch
from test_target_supervision import target_spec

from xuannv_embedding.config import Config, ConfigError
from xuannv_embedding.training.annual import prepare_annual_batch
from xuannv_embedding.training.annual_experiments import model_state_digest
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.optimizer import build_optimizer
from xuannv_embedding.training.runtime import train_steps
from xuannv_embedding.training.target_supervision import (
    research_budget_record,
    validate_research_resume,
)
from xuannv_embedding.training.target_warmup import (
    FORMAT,
    head_contract,
    initialize_readouts,
)
from xuannv_embedding.training.update_diagnostics import measure_updates, snapshot_parameters


def supervised_config():
    cfg = annual_config()
    return replace(cfg, training=replace(cfg.training, independent_targets=target_spec()))


def test_default_optimizer_parameter_order_retains_historical_resume_contract():
    system = build_training_system(annual_config())
    optimizer = build_optimizer(system, lr=1e-5, weight_decay=0.01)
    assert optimizer.param_groups[0]["params"] == list(system.parameters())


def test_freezing_projection_keeps_upstream_gradient_and_excludes_only_projection():
    system = build_training_system(supervised_config())
    optimizer = build_optimizer(
        system,
        lr=1e-4,
        weight_decay=0.01,
        auxiliary_lr_multiplier=30,
        vmf_lr_multiplier=0,
    )
    projection = system.model.bottleneck.proj.weight
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert not projection.requires_grad and id(projection) not in ids
    upstream = torch.randn(2, 8, 4, 4, requires_grad=True)
    system.model.bottleneck.eval()
    output = system.model.bottleneck(upstream)
    (output[:, 0] ** 2).sum().backward()
    assert upstream.grad.norm() > 0 and projection.grad is None
    before = projection.detach().clone()
    optimizer.step()
    assert torch.equal(before, projection)
    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx([1e-4, 3e-3])


def test_projection_multiplier_controls_actual_adam_step_without_scaling_other_groups():
    original = build_training_system(supervised_config())
    a, b = copy.deepcopy(original), copy.deepcopy(original)
    opts = [
        build_optimizer(s, lr=1e-3, weight_decay=0, auxiliary_lr_multiplier=30, vmf_lr_multiplier=m)
        for s, m in [(a, 1), (b, 0.1)]
    ]
    metrics = []
    for system, optimizer in zip([a, b], opts, strict=True):
        for p in system.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        snapshot = snapshot_parameters(system)
        optimizer.step()
        metrics.append(measure_updates(snapshot))
    assert metrics[1]["vmf_projection"]["update_l2"] / metrics[0]["vmf_projection"][
        "update_l2"
    ] == pytest.approx(0.1, rel=3e-4)
    assert metrics[0]["highres"]["update_l2"] == metrics[1]["highres"]["update_l2"]
    assert metrics[0]["target_heads"]["update_l2"] == metrics[1]["target_heads"]["update_l2"]


def test_gradient_scale_is_not_the_fresh_adam_update_scale():
    systems = [build_training_system(annual_config()) for _ in range(2)]
    systems[1].load_state_dict(systems[0].state_dict())
    metrics = []
    for system, scale in zip(systems, [1.0, 100.0], strict=True):
        optimizer = build_optimizer(system, lr=1e-4, weight_decay=0)
        system.model.bottleneck.proj.weight.grad = torch.full_like(
            system.model.bottleneck.proj.weight, scale
        )
        snapshot = snapshot_parameters(system)
        optimizer.step()
        metrics.append(measure_updates(snapshot)["vmf_projection"])
    assert metrics[1]["raw_gradient_l2"] / metrics[0]["raw_gradient_l2"] == pytest.approx(100)
    assert metrics[1]["update_l2"] / metrics[0]["update_l2"] == pytest.approx(1, rel=1e-5)


def test_actual_update_observation_does_not_change_training_and_reports_declared_steps():
    cfg = annual_config()
    first = build_training_system(cfg)
    systems = [first, copy.deepcopy(first)]
    batch = prepare_annual_batch(raw_batch(), cfg, training=True)
    observed = []
    for system, interval in zip(systems, [0, 2], strict=True):
        optimizer = build_optimizer(system, lr=1e-4, weight_decay=0.01)
        torch.manual_seed(2026)
        train_steps(
            system,
            [batch] * 3,
            optimizer,
            device=torch.device("cpu"),
            epochs=1,
            gradient_accumulation_steps=1,
            amp=False,
            update_diagnostics_interval=interval,
            step_callback=observed.append if interval else None,
        )
    assert all(torch.equal(v, systems[1].state_dict()[k]) for k, v in first.state_dict().items())
    assert [row["optimizer_steps"] for row in observed if row["parameter_updates"]] == [1, 2]
    assert observed[0]["parameter_updates"]["vmf_projection"]["update_l2"] > 0


def test_warmed_joint_initialization_checks_file_and_original_encoder(tmp_path):
    cfg = supervised_config()
    system = build_training_system(cfg)
    module = system.criterion.target_supervision
    initial = copy.deepcopy(module.state_dict())
    path = tmp_path / "heads.pt"
    torch.save(
        {
            "format": FORMAT,
            "contract": head_contract(cfg),
            "model_state_sha256": model_state_digest(system.model),
            "model_optimizer_updates": 0,
            "encoder_unchanged": True,
            "head_optimizer_updates": 4096,
            "heads": {"42": initial},
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps({"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    )
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
    identity = initialize_readouts(path, system, cfg, seed=42)
    assert not identity["optimizer_state_restored"]
    assert all(torch.equal(v, module.state_dict()[k]) for k, v in initial.items())
    with torch.no_grad():
        system.model.bottleneck.proj.weight.add_(1)
    with pytest.raises(ValueError, match="identity"):
        initialize_readouts(path, system, cfg, seed=42)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="digest"):
        initialize_readouts(path, system, cfg, seed=42)


def test_bounded_resume_checks_cumulative_budget_and_original_initialization():
    target = target_spec(engineering_only=False, research_max_updates=4096)
    state = {
        "epoch": 1,
        "metrics": {
            "initialization_identity": {"checkpoint_sha256": "a" * 64},
            "research_budget": research_budget_record(target, steps=512, completed_epochs=2),
        },
    }
    validate_research_resume(target, state, steps=512, epochs=6)
    for steps, epochs in [(512, 7), (256, 6), (0, 1), (512, 0)]:
        with pytest.raises(ValueError):
            validate_research_resume(target, state, steps=steps, epochs=epochs)
    for key in ["research_budget", "initialization_identity"]:
        broken = copy.deepcopy(state)
        del broken["metrics"][key]
        with pytest.raises(ValueError):
            validate_research_resume(target, broken, steps=512, epochs=1)


def test_adaptation_schema_rejects_invalid_values_and_monthly_projection_policy(tmp_path):
    cfg = annual_config()
    raw = cfg.to_dict()
    for row in raw["data"]["datasets"]:
        row["source_map"] = {name: name for name in raw["model"]["input_sources"]}
    path = tmp_path / "config.yaml"
    for field, value in [
        ("annual_vmf_lr_multiplier", -1),
        ("annual_vmf_lr_multiplier", True),
        ("update_diagnostics_interval", 1.5),
    ]:
        changed = copy.deepcopy(raw)
        changed["training"][field] = value
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ConfigError):
            Config.from_yaml(path)
    monthly = Config.from_yaml(Path("configs/production/haidian_p10c_v1.yaml")).to_dict()
    monthly["training"]["annual_vmf_lr_multiplier"] = 0
    path.write_text(yaml.safe_dump(monthly))
    with pytest.raises(ConfigError):
        Config.from_yaml(path)


@pytest.mark.parametrize("resume", [False, True])
def test_matrix_runner_limits_epochs_and_initializes_heads_only_on_fresh_run(
    tmp_path, monkeypatch, resume
):
    from xuannv_embedding.training import annual_experiments as experiment

    folder = tmp_path / "arm"
    folder.mkdir()
    config = folder / "config.yaml"
    config.write_text("frozen config")
    heads = tmp_path / "heads.pt"
    heads.write_bytes(b"frozen readouts")
    checkpoint = folder / "model.pt"
    if resume:
        (folder / "model.epoch-0001.pt").write_bytes(b"periodic")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "epochs": 2,
                "assets": {},
                "initialize_checkpoint": "original.pt",
                "initialize_config": "original.yaml",
                "arms": {
                    "joint": {
                        "config": str(config),
                        "config_sha256": experiment.digest(config),
                        "checkpoint": str(checkpoint),
                        "steps_per_epoch": 512,
                        "changes": {},
                        "warmed_heads": {
                            "path": str(heads),
                            "sha256": experiment.digest(heads),
                            "seed": 42,
                        },
                    }
                },
            }
        )
    )
    monkeypatch.setattr(experiment, "load_model", lambda *_args: (None, None, {"epoch": 0}))
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment.subprocess, "run", run)
    experiment.run(SimpleNamespace(plan=plan, arms=None, gpus="0,1"))
    command = commands[0]
    assert command[command.index("--epochs") + 1] == ("1" if resume else "2")
    assert ("--initialize-target-heads" in command) is not resume
    assert ("--resume" in command) is resume
