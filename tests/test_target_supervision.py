from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from test_annual_model import annual_config, raw_batch
from torch import nn

from xuannv_embedding.config import Config, ConfigError, IndependentTargetsConfig
from xuannv_embedding.training.annual import prepare_annual_batch
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.optimizer import build_optimizer
from xuannv_embedding.training.runtime import _model_view
from xuannv_embedding.training.target_supervision import TargetSupervision


def target_spec(**changes):
    return replace(
        IndependentTargetsConfig(
            schema="annual-dem40m-worldcover-v1",
            target_root=Path("/absent/targets"),
            prepared_root=Path("/absent/prepared"),
            preparation_sha256="0" * 64,
            scope="screen",
            weights={"dem_elevation": 0.1, "worldcover": 0.1},
            dem_mean=100,
            dem_std=20,
            ramp_batches=0,
        ),
        **changes,
    )


def test_head_initialization_is_paired_and_does_not_touch_cpu_or_accelerator_rng(monkeypatch):
    def fail_seed(*args, **kwargs):
        raise AssertionError("Accelerator RNG must not be seeded by target head construction")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", fail_seed)
    before = torch.random.get_rng_state().clone()
    dem = TargetSupervision(4, {"dem_elevation": 0.1}, dem_mean=0, dem_std=1)
    wc = TargetSupervision(4, {"worldcover": 0.1}, dem_mean=0, dem_std=1)
    both = TargetSupervision(4, {"worldcover": 0.1, "dem_elevation": 0.1}, dem_mean=0, dem_std=1)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(dem.heads["dem_elevation"].weight, both.heads["dem_elevation"].weight)
    assert torch.equal(wc.heads["worldcover"].weight, both.heads["worldcover"].weight)


def test_missing_labels_produce_finite_zero_and_keep_every_head_in_gradient_graph():
    module = TargetSupervision(
        4, {"dem_elevation": 0.1, "worldcover": 0.1}, dem_mean=100, dem_std=20
    )
    embedding = torch.randn(2, 4, 16, 16, requires_grad=True)
    targets = {
        "dem_elevation": {
            "values": torch.full((2, 2, 2), float("nan")),
            "valid": torch.zeros(2, 2, 2, dtype=torch.bool),
        },
        "worldcover": {
            "values": torch.full((2, 8, 8), -100),
            "valid": torch.zeros(2, 8, 8, dtype=torch.bool),
        },
    }
    result = module.loss(module(embedding), targets)
    assert result["total"].item() == 0
    result["total"].backward()
    assert embedding.grad is not None and not embedding.grad.any()
    for parameter in module.parameters():
        assert parameter.grad is not None and not parameter.grad.any()


def test_height_loss_uses_valid_count_and_reports_metres():
    module = TargetSupervision(4, {"dem_elevation": 1.0}, dem_mean=10, dem_std=2)
    nn.init.zeros_(module.heads["dem_elevation"].weight)
    nn.init.constant_(module.heads["dem_elevation"].bias, 2)
    embedding = torch.zeros(2, 4, 8, 8)
    target = {
        "dem_elevation": {
            "values": torch.tensor([[[12.0]], [[9999.0]]]),
            "valid": torch.tensor([[[True]], [[False]]]),
        }
    }
    result = module.loss(module(embedding), target)
    assert result["total"].item() == 1
    assert result["target_dem_elevation_MAE_m"].item() == 2
    assert result["target_dem_elevation_valid_pixels"].item() == 1
    result["total"].backward()
    assert module.heads["dem_elevation"].bias.grad.item() == 1


def _ddp_target_worker(rank, rendezvous, directory):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        module = TargetSupervision(4, {"dem_elevation": 1.0}, dem_mean=0, dem_std=1)
        nn.init.zeros_(module.heads["dem_elevation"].weight)
        nn.init.constant_(module.heads["dem_elevation"].bias, 2)
        wrapped = nn.parallel.DistributedDataParallel(module, broadcast_buffers=False)
        prediction = wrapped(torch.zeros(1, 4, 8, 8))
        target = {
            "dem_elevation": {
                "values": torch.full((1, 1, 1), 10.0),
                "valid": torch.full((1, 1, 1), bool(rank)),
            }
        }
        result = module.loss(prediction, target)
        result["total"].backward()
        Path(directory, f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "grad_bias": module.heads["dem_elevation"].bias.grad.item(),
                    "global_mae": result["target_dem_elevation_MAE_m"].item(),
                    "valid_pixels": result["target_dem_elevation_valid_pixels"].item(),
                }
            )
        )
    finally:
        dist.destroy_process_group()


def test_ddp_empty_rank_matches_global_valid_pixel_gradient(tmp_path):
    torch.multiprocessing.spawn(
        _ddp_target_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2
    )
    for rank in (0, 1):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result == {"grad_bias": -1, "global_mae": 8, "valid_pixels": 1}


def test_target_values_do_not_enter_encoder_and_target_gradients_reach_stp():
    config = annual_config()
    config = replace(
        config,
        model=replace(config.model, annual_context_pooling=True, annual_exact_mask_coverage=True),
        training=replace(config.training, independent_targets=target_spec()),
    )
    system = build_training_system(config)
    raw = raw_batch()
    raw["independent_targets"] = {
        "dem_elevation": {
            "values": torch.full((1, 4, 4), 120.0),
            "valid": torch.ones(1, 4, 4, dtype=torch.bool),
        },
        "worldcover": {
            "values": torch.zeros(1, 16, 16, dtype=torch.long),
            "valid": torch.ones(1, 16, 16, dtype=torch.bool),
        },
    }
    batch = prepare_annual_batch(
        raw, config, training=True, generator=torch.Generator().manual_seed(3)
    )
    assert batch["independent_targets"]["worldcover"]["valid"].all()
    system.model.eval()
    first = _model_view(system.model, batch)
    batch["independent_targets"]["worldcover"]["values"].fill_(10)
    second = _model_view(system.model, batch)
    assert torch.equal(first.embedding_map, second.embedding_map)
    module = system.criterion.target_supervision
    auxiliary = module.loss(module(first.embedding_map[:, 0]), batch["independent_targets"])
    parameter = next(
        parameter
        for name, parameter in system.model.named_parameters()
        if name.startswith("temporal_encoder.") and parameter.requires_grad
    )
    (gradient,) = torch.autograd.grad(auxiliary["total"], parameter)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    optimizer = build_optimizer(system, lr=3e-5, weight_decay=0.01, auxiliary_lr_multiplier=10)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([3e-5, 3e-4])
    losses = system(batch)
    assert "target_worldcover" in losses
    losses["total"].backward()
    assert system.criterion.target_batches.item() == 1


@pytest.mark.parametrize(
    "change",
    [
        {"scope": "confirmation"},
        {"weights": {"unknown": 0.1}},
        {"dem_std": -1},
        {"encoder_input": True},
    ],
)
def test_target_config_rejects_scope_and_contract_bypasses(tmp_path, change):
    from dataclasses import asdict

    import yaml

    raw = annual_config().to_dict()
    spec = asdict(target_spec())
    spec.update(target_root="/absent/targets", prepared_root="/absent/prepared")
    spec.update(change)
    raw["training"]["independent_targets"] = spec
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        Config.from_yaml(path)
