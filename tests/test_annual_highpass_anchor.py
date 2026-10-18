import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from test_annual_feature_anchor import anchor_batch, anchor_config

from xuannv_embedding.config import Config, ConfigError
from xuannv_embedding.training.annual import (
    highpass_feature_anchor_loss,
    native_feature_anchor_loss,
)
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.optimizer import build_optimizer
from xuannv_embedding.training.target_warmup import head_contract


def _block_means(values, valid):
    """Explicit independent projection for checking values and gradients."""
    rows = []
    for top in range(0, values.shape[-2], 2):
        columns = []
        for left in range(0, values.shape[-1], 2):
            tile = values[..., top : top + 2, left : left + 2]
            mask = valid[..., top : top + 2, left : left + 2].float()
            clean = torch.where(mask > 0, tile, 0.0)
            mean = (clean * mask).sum((-2, -1), keepdim=True) / mask.sum(
                (-2, -1), keepdim=True
            ).clamp(min=1)
            columns.append(mean.expand_as(tile))
        rows.append(torch.cat(columns, -1))
    return torch.cat(rows, -2)


def test_highpass_is_the_fine_component_of_whole_feature_error_with_missing_edge_pixels():
    torch.manual_seed(19)
    student = torch.randn(2, 3, 5, 7, requires_grad=True)
    reference = torch.randn_like(student).requires_grad_()
    valid = torch.rand(2, 1, 5, 7) > 0.4
    valid[..., :2, :2] = False
    whole = native_feature_anchor_loss(student, reference, valid)
    student_low = _block_means(student, valid)
    reference_low = _block_means(reference.detach(), valid)
    coarse = native_feature_anchor_loss(student_low, reference_low, valid)
    fine = highpass_feature_anchor_loss(student, reference, valid)
    assert float(whole.detach()) == pytest.approx(float((coarse + fine).detach()), rel=1e-6)
    expected = torch.autograd.grad(whole, student, retain_graph=True)[0]
    actual = torch.autograd.grad(coarse + fine, student, retain_graph=True)[0]
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    fine.backward()
    assert reference.grad is None and torch.isfinite(student.grad).all()
    assert student.grad[~valid.expand_as(student)].count_nonzero() == 0


def test_highpass_leaves_block_means_free_and_detects_changes_inside_the_blocks():
    torch.manual_seed(4)
    reference = torch.randn(1, 3, 4, 6)
    offset = torch.randn(1, 3, 2, 3).repeat_interleave(2, -2).repeat_interleave(2, -1)
    student = (reference + offset).requires_grad_()
    support = torch.ones(1, 1, 4, 6, dtype=torch.bool)
    loss = highpass_feature_anchor_loss(student, reference, support)
    assert loss.abs() < 1e-12
    loss.backward()
    assert student.grad.abs().max() < 1e-7
    assert native_feature_anchor_loss(student, reference, support) > 0.1
    checkerboard = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(2, 3)
    changed = (reference + checkerboard).requires_grad_()
    assert torch.allclose(_block_means(changed, support), _block_means(reference, support))
    fine = highpass_feature_anchor_loss(changed, reference, support)
    assert float(fine.detach()) == pytest.approx(1.5)
    fine.backward()
    assert changed.grad.abs().sum() > 0


def test_highpass_masks_invalid_nans_empty_blocks_and_single_pixel_edges():
    reference = torch.zeros(1, 2, 3, 3, requires_grad=True)
    values = torch.tensor(
        [
            [1.0, -1.0, float("nan")],
            [float("nan"), float("nan"), float("nan")],
            [float("nan"), float("nan"), 19.0],
        ]
    )
    student = values[None, None].repeat(1, 2, 1, 1).requires_grad_()
    support = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    support[..., 0, :2] = True
    support[..., 2, 2] = True
    loss = highpass_feature_anchor_loss(student, reference, support)
    assert float(loss.detach()) == pytest.approx(2 / 3)
    loss.backward()
    assert reference.grad is None
    assert torch.isfinite(student.grad).all()
    assert student.grad[~support.expand_as(student)].count_nonzero() == 0
    assert student.grad[..., 2, 2].count_nonzero() == 0
    student.grad.zero_()
    zero = highpass_feature_anchor_loss(student, reference, support * False)
    assert zero == 0
    zero.backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad.count_nonzero() == 0
    single = highpass_feature_anchor_loss(
        torch.ones(1, 2, 1, 1), torch.zeros(1, 2, 1, 1), torch.ones(1, 1, 1, 1)
    )
    assert single == 0
    with pytest.raises(ValueError, match="same BCHW"):
        highpass_feature_anchor_loss(student, reference[..., :2], support)
    with pytest.raises(ValueError, match="B1HW"):
        highpass_feature_anchor_loss(student, reference, support[:, 0])


def test_highpass_configuration_is_strict_annual_only_and_keeps_the_head_contract(tmp_path):
    cfg = anchor_config()
    hp = replace(cfg, training=replace(cfg.training, annual_feature_anchor_mode="highpass_10m"))
    raw = hp.to_dict()
    for dc in raw["data"]["datasets"]:
        dc["source_map"] = {name: name for name in raw["model"]["input_sources"]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert Config.from_yaml(path).training.annual_feature_anchor_mode == "highpass_10m"
    for value in [True, False, 2, None, "", "highpass", [], {}]:
        current = copy.deepcopy(raw)
        current["training"]["annual_feature_anchor_mode"] = value
        path.write_text(yaml.safe_dump(current))
        with pytest.raises(ConfigError, match="annual_feature_anchor_mode"):
            Config.from_yaml(path)
    raw["training"].pop("annual_feature_anchor_mode")
    path.write_text(yaml.safe_dump(raw))
    assert Config.from_yaml(path).training.annual_feature_anchor_mode == "native"
    monthly = Config.from_yaml(Path("configs/production/haidian_p10c_v1.yaml")).to_dict()
    monthly["training"]["annual_feature_anchor_mode"] = "highpass_10m"
    monthly["training"]["annual_feature_anchor_weight"] = 0
    path.write_text(yaml.safe_dump(monthly))
    with pytest.raises(ConfigError, match="annual5m"):
        Config.from_yaml(path)
    with pytest.raises(ValueError, match="anchor mode"):
        build_training_system(
            replace(cfg, training=replace(cfg.training, annual_feature_anchor_mode="bad"))
        )
    from test_target_supervision import target_spec

    supervised = replace(cfg, training=replace(cfg.training, independent_targets=target_spec()))
    altered = replace(
        supervised, training=replace(supervised.training, annual_feature_anchor_mode="highpass_10m")
    )
    assert head_contract(supervised) == head_contract(altered)


def test_highpass_runs_on_clean_means_and_backpropagates_without_updating_reference():
    torch.manual_seed(81)
    cfg = anchor_config()
    cfg = replace(cfg, training=replace(cfg.training, annual_feature_anchor_mode="highpass_10m"))
    system = build_training_system(cfg)
    teacher = copy.deepcopy(system.model)
    system.set_teacher(teacher)
    system.train()
    build_optimizer(system, lr=1e-3, weight_decay=0, vmf_lr_multiplier=0)
    batch = anchor_batch(cfg)
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
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
    assert result["feature_anchor_weighted"] == result["feature_anchor"] * 10
    result["feature_anchor"].backward()
    assert sum(p.grad.abs().sum() for p in system.model.parameters() if p.grad is not None) > 0
    assert system.model.bottleneck.proj.weight.grad is None
    assert not teacher.training and all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items())


def _highpass_global_mean_worker(rank, rendezvous, output):
    import json

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.3)
        wrapped = DistributedDataParallel(model)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0] if rank == 0 else [2.0, -1.0, 3.0, -2.0])[
            :, None
        ]
        reference = torch.tensor(
            [0.0, 1.0, 2.0, 1.0] if rank == 0 else [1.0, 0.0, 0.0, 2.0]
        ).reshape(1, 1, 2, 2)
        rows = []
        for pattern in ["unequal", "zero_local", "zero_global"]:
            support = torch.ones(1, 1, 2, 2, dtype=torch.bool)
            if rank == 0:
                support[..., 1, :] = False
            if pattern == "zero_global" or (pattern == "zero_local" and rank == 0):
                support.zero_()
            wrapped.zero_grad(set_to_none=True)
            student = wrapped(values).reshape(1, 1, 2, 2)
            loss = highpass_feature_anchor_loss(student, reference, support)
            loss.backward()
            rows.append({"loss": float(loss.detach()), "gradient": float(model.weight.grad)})
        Path(output, f"rank{rank}.json").write_text(json.dumps(rows))
    finally:
        dist.destroy_process_group()


def test_highpass_global_pixel_mean_and_spatial_gradient_match_centralized_values(tmp_path):
    import json
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_highpass_global_mean_worker,
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
        # Rank0: squared-error numerator .1225, derivative -.35, 2 valid pixels.
        # Rank1: numerator3.19, derivative8.6, 4 valid pixels. DDP must use 6 total.
        for rank in range(2):
            rows = json.loads((tmp_path / f"rank{rank}.json").read_text())
            for row, loss, gradient in zip(
                rows, [3.3125 / 6, 0.7975, 0.0], [1.375, 2.15, 0.0], strict=True
            ):
                assert row["loss"] == pytest.approx(loss, abs=1e-6)
                assert row["gradient"] == pytest.approx(gradient, abs=1e-6)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
