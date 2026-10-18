import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from test_annual_feature_anchor import anchor_batch, anchor_config
from test_annual_highpass_anchor import _block_means

from xuannv_embedding.config import Config, ConfigError
from xuannv_embedding.training.annual import (
    native_feature_anchor_loss,
    weighted_feature_anchor_loss,
)
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.optimizer import build_optimizer
from xuannv_embedding.training.target_warmup import head_contract


def _independent_loss(student, reference, support):
    """Explicit projection, not the production convex combination."""
    low_student = _block_means(student.float(), support)
    low_reference = _block_means(reference.detach().float(), support)
    coarse = torch.where(support, low_student - low_reference, 0.0)
    fine = torch.where(
        support, student.float() - low_student - (reference.detach().float() - low_reference), 0.0
    )
    return (0.5 * (fine.square() + 0.1 * coarse.square()) * support).sum() / support.sum().clamp(
        min=1
    )


def test_weighted_value_and_gradient_match_independent_projection_with_missing_edges():
    torch.manual_seed(29)
    student = torch.randn(2, 3, 5, 7, requires_grad=True)
    reference = torch.randn_like(student).requires_grad_()
    support = torch.rand(2, 1, 5, 7) > 0.35
    support[..., :2, :2] = False
    actual = weighted_feature_anchor_loss(student, reference, support)
    expected = _independent_loss(student, reference, support)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    a = torch.autograd.grad(actual, student, retain_graph=True)[0]
    b = torch.autograd.grad(expected, student)[0]
    assert torch.allclose(a, b, atol=1e-7, rtol=1e-6)
    actual.backward()
    assert reference.grad is None
    assert student.grad[~support.expand_as(student)].count_nonzero() == 0


def test_weighted_keeps_fine_penalty_and_restores_tenth_block_mean_penalty():
    reference = torch.zeros(1, 2, 4, 6)
    support = torch.ones(1, 1, 4, 6, dtype=torch.bool)
    block = torch.tensor([[1.0, 2.0, -1.0], [3.0, -2.0, 4.0]])
    constant = block.repeat_interleave(2, -2).repeat_interleave(2, -1)[None, None]
    constant = constant.expand_as(reference).requires_grad_()
    loss = weighted_feature_anchor_loss(constant, reference, support)
    assert torch.allclose(loss, 0.1 * native_feature_anchor_loss(constant, reference, support))
    loss.backward()
    assert constant.grad.abs().sum() > 0
    checker = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(2, 3)
    fine = checker[None, None].expand_as(reference)
    assert weighted_feature_anchor_loss(fine, reference, support) == pytest.approx(1.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_weighted_masks_nans_empty_support_and_single_pixel_edge(dtype):
    reference = torch.zeros(1, 2, 3, 3, dtype=dtype, requires_grad=True)
    values = torch.tensor(
        [[1.0, -1.0, float("nan")], [float("nan")] * 3, [float("nan"), float("nan"), 19.0]],
        dtype=dtype,
    )
    student = values[None, None].repeat(1, 2, 1, 1).requires_grad_()
    support = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    support[..., 0, :2] = True
    support[..., 2, 2] = True
    loss = weighted_feature_anchor_loss(student, reference, support)
    assert float(loss.detach()) == pytest.approx(12.7, abs=1e-5)
    loss.backward()
    assert reference.grad is None and torch.isfinite(student.grad).all()
    assert student.grad[~support.expand_as(student)].count_nonzero() == 0
    assert student.grad[..., 2, 2].abs().sum() > 0
    student.grad.zero_()
    zero = weighted_feature_anchor_loss(student, reference, support & False)
    assert zero == 0
    zero.backward()
    assert student.grad.count_nonzero() == 0 and torch.isfinite(student.grad).all()
    assert weighted_feature_anchor_loss(
        torch.ones(1, 2, 1, 1), torch.zeros(1, 2, 1, 1), torch.ones(1, 1, 1, 1)
    ) == pytest.approx(0.1)


def test_weighted_config_is_annual_only_and_preserves_head_identity(tmp_path):
    cfg = anchor_config()
    weighted = replace(
        cfg, training=replace(cfg.training, annual_feature_anchor_mode="weighted_10m")
    )
    raw = weighted.to_dict()
    for dc in raw["data"]["datasets"]:
        dc["source_map"] = {name: name for name in raw["model"]["input_sources"]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert Config.from_yaml(path).training.annual_feature_anchor_mode == "weighted_10m"
    monthly = Config.from_yaml(Path("configs/production/haidian_p10c_v1.yaml")).to_dict()
    monthly["training"]["annual_feature_anchor_mode"] = "weighted_10m"
    path.write_text(yaml.safe_dump(monthly))
    with pytest.raises(ConfigError, match="annual5m"):
        Config.from_yaml(path)
    from test_target_supervision import target_spec

    supervised = replace(cfg, training=replace(cfg.training, independent_targets=target_spec()))
    altered = replace(
        supervised, training=replace(supervised.training, annual_feature_anchor_mode="weighted_10m")
    )
    assert head_contract(supervised) == head_contract(altered)


def test_weighted_uses_same_clean_forward_frozen_reference_and_no_extra_rng():
    torch.manual_seed(83)
    cfg = anchor_config()
    cfg = replace(cfg, training=replace(cfg.training, annual_feature_anchor_mode="weighted_10m"))
    system = build_training_system(cfg)
    teacher = copy.deepcopy(system.model)
    system.set_teacher(teacher)
    system.train()
    build_optimizer(system, lr=1e-3, weight_decay=0, vmf_lr_multiplier=0)
    batch = anchor_batch(cfg)
    before = {key: value.clone() for key, value in teacher.state_dict().items()}
    assert system(batch)["feature_anchor"] == 0
    with torch.no_grad():
        for parameter in system.model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.02 * torch.randn_like(parameter))
    calls = []
    hook = system.model.register_forward_hook(
        lambda module, args, kwargs, output: calls.append((kwargs, torch.get_rng_state())),
        with_kwargs=True,
    )
    result = system(batch)
    hook.remove()
    assert len(calls) == 2 and calls[1][0]["sample_vmf"] is False
    assert torch.equal(calls[0][1], calls[1][1])
    assert result["feature_anchor"] > 0
    assert result["feature_anchor_weighted"] == 10 * result["feature_anchor"]
    result["feature_anchor"].backward()
    assert sum(p.grad.abs().sum() for p in system.model.parameters() if p.grad is not None) > 0
    assert system.model.bottleneck.proj.weight.grad is None
    assert not teacher.training and all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(before[key], value) for key, value in teacher.state_dict().items())


def _weighted_global_worker(rank, rendezvous, output):
    import test_annual_highpass_anchor as fixture

    # Reuse only the actual DDP fixture; expected values below use a separate projection.
    fixture.highpass_feature_anchor_loss = weighted_feature_anchor_loss
    fixture._highpass_global_mean_worker(rank, rendezvous, output)


def test_weighted_global_mean_gradient_matches_centralized_projection(tmp_path):
    import json
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_weighted_global_worker,
            args=(rank, (tmp_path / "rendezvous").as_uri(), str(tmp_path)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=60)
        assert all(process.exitcode == 0 for process in processes)
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 3.0, -2.0]]).reshape(2, 1, 2, 2)
        reference = torch.tensor([[0.0, 1.0, 2.0, 1.0], [1.0, 0.0, 0.0, 2.0]]).reshape(2, 1, 2, 2)
        expected = []
        for pattern in ["unequal", "zero_local", "zero_global"]:
            support = torch.ones(2, 1, 2, 2, dtype=torch.bool)
            support[0, ..., 1, :] = False
            if pattern == "zero_local":
                support[0].zero_()
            if pattern == "zero_global":
                support.zero_()
            coefficient = torch.tensor(0.3, requires_grad=True)
            loss = _independent_loss(coefficient * values, reference, support)
            gradient = torch.autograd.grad(loss, coefficient)[0]
            expected.append((float(loss.detach()), float(gradient)))
        for rank in range(2):
            rows = json.loads((tmp_path / f"rank{rank}.json").read_text())
            for actual, (loss, gradient) in zip(rows, expected, strict=True):
                assert actual["loss"] == pytest.approx(loss, abs=1e-6)
                assert actual["gradient"] == pytest.approx(gradient, abs=1e-6)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
