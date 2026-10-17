from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_annual_model import annual_config
from test_target_supervision import target_spec
from torch import nn

from xuannv_embedding.training.target_gradients import (
    gradient_statistics,
    single_task_objectives,
)
from xuannv_embedding.training.target_warmup import (
    FORMAT,
    HeadReadouts,
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


def test_single_task_gradients_sum_to_the_actual_weighted_production_objective():
    cfg = config()
    heads = HeadReadouts(cfg, [42]).readouts["42"]
    data = batch()
    features = data["source_frames"]["features"].requires_grad_()
    terms = single_task_objectives(cfg, heads, features, data["independent_targets"])
    full = heads.loss(heads(features), data["independent_targets"])["total"]
    sum_gradient = torch.autograd.grad(sum(terms.values()), features, retain_graph=True)[0]
    full_gradient = torch.autograd.grad(full, features)[0]
    assert torch.allclose(sum(terms.values()), full)
    assert torch.allclose(sum_gradient, full_gradient, atol=1e-8)


def test_gradient_cosine_distinguishes_opposition_from_missing_support():
    vectors = {
        "image": {"embedding": torch.tensor([2.0, 0])},
        "auxiliary_total": {"embedding": torch.tensor([-1.0, 0])},
        "absent": {"embedding": torch.zeros(2)},
    }
    result = gradient_statistics(vectors)["embedding"]
    assert result["cosines"]["image"]["auxiliary_total"] == -1
    assert result["cosines"]["image"]["absent"] is None
    assert result["auxiliary_to_image_norm_ratio"] == 0.5


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
