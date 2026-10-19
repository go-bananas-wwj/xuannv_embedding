from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_annual_model import annual_config
from test_target_supervision import target_spec
from torch import nn

from xuannv_embedding.training.target_warmup import (
    FORMAT,
    HeadReadouts,
    continue_readouts,
    freeze_encoder,
    head_contract,
    head_step,
    restore_warmed_heads,
)


def config():
    value = annual_config()
    return replace(value, training=replace(value.training, independent_targets=target_spec()))


class FixedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.register_buffer("constant", torch.tensor(3.0))

    def forward(self, source_frames, *_args, **_kwargs):
        return SimpleNamespace(embedding_map=source_frames["features"][:, None] * self.weight)


def batch():
    return {
        "source_frames": {"features": torch.randn(2, 64, 16, 16)},
        "source_masks": {},
        "timestamps": torch.zeros(2, 1),
        "independent_targets": {
            "dem_elevation": {
                "values": torch.ones(2, 2, 2) * 90,
                "valid": torch.ones(2, 2, 2).bool(),
            },
            "worldcover": {
                "values": torch.ones(2, 8, 8).long(),
                "valid": torch.ones(2, 8, 8).bool(),
            },
        },
    }


def test_warmup_updates_only_heads_and_replicas_match_separate_optimizers():
    cfg = config()
    model = FixedEncoder()
    freeze_encoder(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    heads = HeadReadouts(cfg, [42, 17])
    data = batch()
    optimizer = torch.optim.AdamW(heads.parameters(), lr=3e-3)
    separate = HeadReadouts(cfg, [42, 17])
    with torch.no_grad():
        features = model(data["source_frames"]).embedding_map[:, 0]
    for module in separate.readouts.values():
        opt = torch.optim.AdamW(module.parameters(), lr=3e-3)
        module.loss(module(features), data["independent_targets"])["total"].backward()
        opt.step()
    head_step(model, heads, data, optimizer, device=torch.device("cpu"))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in heads.state_dict().items():
        assert torch.equal(value, separate.state_dict()[key]), key
    assert all(int(row["step"]) == 1 for row in optimizer.state.values())


def test_warmup_refuses_unfrozen_model_or_optimizer_with_encoder_parameters():
    heads = HeadReadouts(config(), [42])
    model = FixedEncoder()
    optimizer = torch.optim.AdamW(heads.parameters())
    with pytest.raises(ValueError, match="frozen"):
        head_step(model, heads, batch(), optimizer, device=torch.device("cpu"))
    freeze_encoder(model)
    optimizer = torch.optim.AdamW([*heads.parameters(), *model.parameters()])
    with pytest.raises(ValueError, match="only"):
        head_step(model, heads, batch(), optimizer, device=torch.device("cpu"))


def test_readout_artifact_strictly_checks_original_encoder_and_target_contract(tmp_path):
    cfg = config()
    heads = HeadReadouts(cfg, [42])
    path = tmp_path / "heads.pt"
    torch.save(
        {
            "format": FORMAT,
            "contract": head_contract(cfg),
            "model_state_sha256": "a" * 64,
            "encoder_unchanged": True,
            "model_optimizer_updates": 0,
            "heads": {"42": heads.readouts["42"].state_dict()},
        },
        path,
    )
    module = HeadReadouts(cfg, [17]).readouts["17"]
    args = dict(seed=42, expected_contract=head_contract(cfg), expected_model_sha256="a" * 64)
    restore_warmed_heads(path, module, **args)
    assert all(
        torch.equal(value, module.state_dict()[key])
        for key, value in heads.readouts["42"].state_dict().items()
    )
    with pytest.raises(ValueError, match="identity"):
        restore_warmed_heads(path, module, **{**args, "expected_model_sha256": "b" * 64})
    with pytest.raises(ValueError, match="identity"):
        restore_warmed_heads(path, module, **{**args, "expected_contract": {}})


def test_readout_continuation_preserves_optimizer_and_checks_traversal_identity(tmp_path):
    import copy
    import hashlib
    import json

    cfg = config()
    cfg = replace(cfg, data=replace(cfg.data, batch_size=2))
    model = FixedEncoder()
    freeze_encoder(model)
    data = batch()
    heads = HeadReadouts(cfg, [42, 17])
    initial = {seed: copy.deepcopy(m.state_dict()) for seed, m in heads.readouts.items()}
    lr = cfg.training.lr * cfg.training.independent_targets.head_lr_multiplier
    optimizer = torch.optim.AdamW(heads.parameters(), lr=lr)
    head_step(model, heads, data, optimizer, device=torch.device("cpu"))
    path = tmp_path / "heads.pt"
    torch.save(
        {
            "format": FORMAT,
            "contract": head_contract(cfg),
            "model_state_sha256": "a" * 64,
            "encoder_unchanged": True,
            "model_optimizer_updates": 0,
            "head_optimizer_updates": 1,
            "feature_seed": cfg.experiment.seed,
            "git_sha": "abcd123",
            "source_code_sha256": "b" * 64,
            "heads": {s: m.state_dict() for s, m in heads.readouts.items()},
            "head_initialization": initial,
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "world_size": 1,
                "global_annual_exposures": 2,
                "head_optimizer_updates_per_seed": 1,
            }
        )
    )
    resumed = HeadReadouts(cfg, [42, 17])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=lr)
    args = dict(config=cfg, model_sha256="a" * 64, world_size=1, epoch_batches=1)
    step, _, parent = continue_readouts(path, resumed, resumed_optimizer, **args)
    assert step == 1 and parent["noise_rng_restarted"] is True
    head_step(model, heads, data, optimizer, device=torch.device("cpu"))
    head_step(model, resumed, data, resumed_optimizer, device=torch.device("cpu"))
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in heads.state_dict().items())
    with pytest.raises(ValueError, match="traversal"):
        continue_readouts(path, resumed, resumed_optimizer, **{**args, "epoch_batches": 2})
    with pytest.raises(ValueError, match="identity"):
        continue_readouts(path, resumed, resumed_optimizer, **{**args, "world_size": 2})


def _distributed_warmup_worker(rank, rendezvous, directory):
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
        torch.manual_seed(2026)
        data = batch()
        for item in data["independent_targets"].values():
            item["valid"][1] = False

        def partition(value):
            if isinstance(value, torch.Tensor):
                return value[rank : rank + 1]
            return {key: partition(child) for key, child in value.items()}

        data = partition(data)
        model = FixedEncoder()
        freeze_encoder(model)
        heads = HeadReadouts(config(), [42, 17])
        wrapped = nn.parallel.DistributedDataParallel(heads, broadcast_buffers=False)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=3e-3)
        head_step(model, wrapped, data, optimizer, device=torch.device("cpu"))
        assert model.weight.item() == 1 and model.weight.grad is None
        torch.save(heads.state_dict(), Path(directory) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_distributed_frozen_heads_match_global_objective_with_empty_rank(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_warmup_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    torch.manual_seed(2026)
    data = batch()
    for item in data["independent_targets"].values():
        item["valid"][1] = False
    model = FixedEncoder()
    freeze_encoder(model)
    heads = HeadReadouts(config(), [42, 17])
    optimizer = torch.optim.AdamW(heads.parameters(), lr=3e-3)
    head_step(model, heads, data, optimizer, device=torch.device("cpu"))
    for rank in (0, 1):
        saved = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        for key, value in heads.state_dict().items():
            assert torch.allclose(value, saved[key], atol=1e-6), (rank, key)
