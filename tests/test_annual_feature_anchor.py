import copy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import yaml
from test_annual_model import annual_config, raw_batch

from xuannv_embedding.config import Config, ConfigError
from xuannv_embedding.models.bottleneck import VMFBottleneck
from xuannv_embedding.training.annual import native_feature_anchor_loss, prepare_annual_batch
from xuannv_embedding.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from xuannv_embedding.training.cli import build_training_system, teacher_contract_matches
from xuannv_embedding.training.optimizer import build_optimizer, build_scheduler


def anchor_config():
    cfg = annual_config()
    return replace(
        cfg,
        model=replace(cfg.model, annual_context_pooling=True, annual_exact_mask_coverage=True),
        training=replace(cfg.training, annual_feature_anchor_weight=10, annual_vmf_lr_multiplier=0),
    )


def anchor_batch(cfg):
    raw = raw_batch()
    batch = prepare_annual_batch(
        raw, cfg, training=True, generator=torch.Generator().manual_seed(9)
    )
    batch["teacher_view"] = prepare_annual_batch(raw, cfg, training=False)
    return batch


def test_deterministic_bottleneck_keeps_gradients_without_drawing_noise():
    torch.manual_seed(7)
    model = VMFBottleneck(4, 64).train()
    values = torch.randn(2, 4, 3, 3, requires_grad=True)
    before = torch.get_rng_state()
    mean = model(values, sample=False)
    assert torch.equal(before, torch.get_rng_state())
    sampled = model(values)
    assert not torch.equal(before, torch.get_rng_state())
    assert not torch.equal(mean, sampled)
    model.eval()
    assert torch.equal(mean, model(values))
    mean[:, 0].sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0


def test_native_anchor_preserves_details_masks_nodata_and_has_no_hidden_normalization():
    reference = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], requires_grad=True)
    student = reference.flip(-1).detach().requires_grad_()
    valid = torch.ones(1, 1, 1, 2)
    assert torch.equal(reference.mean(-1), student.mean(-1))
    loss = native_feature_anchor_loss(student, reference, valid)
    assert loss == 1
    loss.backward()
    assert reference.grad is None and student.grad.abs().sum() > 0
    student = torch.tensor([[[[2.0, float("nan")]], [[0.0, float("nan")]]]], requires_grad=True)
    assert native_feature_anchor_loss(student, reference, valid * torch.tensor([1, 0])) == 0.5
    zero = native_feature_anchor_loss(student, reference, valid * 0)
    assert zero == 0
    zero.backward()
    assert torch.isfinite(student.grad).all() and student.grad.count_nonzero() == 0
    with pytest.raises(ValueError, match="same BCHW"):
        native_feature_anchor_loss(student, reference[:, :, :, :1], valid)
    with pytest.raises(ValueError, match="B1HW"):
        native_feature_anchor_loss(student, reference, valid[:, 0])


def test_anchor_uses_clean_deterministic_student_and_frozen_reference_without_a_predictor():
    torch.manual_seed(8)
    cfg = anchor_config()
    system = build_training_system(cfg)
    teacher = copy.deepcopy(system.model)
    system.set_teacher(teacher)
    system.train()
    build_optimizer(system, lr=1e-3, weight_decay=0, vmf_lr_multiplier=0)
    batch = anchor_batch(cfg)
    calls = []

    def capture(module, args, kwargs, output):
        calls.append((kwargs, output, torch.get_rng_state()))

    hook = system.model.register_forward_hook(capture, with_kwargs=True)
    initial = system(batch)
    hook.remove()
    assert initial["feature_anchor"] == 0
    assert system.criterion.latent_predictor is None and system.criterion.view_predictor is None
    assert len(calls) == 2
    assert calls[0][0].get("sample_vmf") is None
    assert calls[1][0]["sample_vmf"] is False and calls[1][0]["decode"] is False
    assert torch.equal(calls[0][2], calls[1][2])
    assert calls[1][1].embedding_map.requires_grad
    assert not calls[1][1].reconstructions
    assert not torch.equal(
        calls[0][0]["source_pixel_masks"]["s2"], calls[1][0]["source_pixel_masks"]["s2"]
    )
    assert all(
        torch.equal(calls[1][0]["source_pixel_masks"][k], v)
        for k, v in batch["teacher_view"]["source_pixel_masks"].items()
    )
    before = {name: value.clone() for name, value in teacher.state_dict().items()}
    with torch.no_grad():
        next(system.model.temporal_encoder.parameters()).add_(0.05)
    losses = system(batch)
    assert losses["feature_anchor"] > 0
    assert torch.equal(losses["feature_anchor_weighted"], losses["feature_anchor"] * 10)
    losses["feature_anchor"].backward()
    assert system.model.bottleneck.proj.weight.grad is None
    assert not system.model.bottleneck.proj.weight.requires_grad
    assert (
        sum(
            p.grad.abs().sum()
            for p in system.model.temporal_encoder.parameters()
            if p.grad is not None
        )
        > 0
    )
    assert not teacher.training and all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(v, before[k]) for k, v in teacher.state_dict().items())


def test_disabled_anchor_does_not_change_parameters_or_run_an_extra_student_view():
    cfg = anchor_config()
    disabled = replace(cfg, training=replace(cfg.training, annual_feature_anchor_weight=0))
    torch.manual_seed(11)
    system = build_training_system(disabled)
    enabled = build_training_system(cfg)
    assert list(enabled.model.state_dict()) == list(system.model.state_dict())
    assert list(enabled.criterion.state_dict()) == list(system.criterion.state_dict())
    calls = []
    hook = system.model.register_forward_hook(lambda *args: calls.append(True))
    result = system(anchor_batch(disabled))
    hook.remove()
    assert len(calls) == 1 and "feature_anchor" not in result
    with pytest.raises(ValueError, match="requires clean student and frozen reference"):
        enabled(anchor_batch(cfg))


def test_anchor_configuration_and_reference_contract_are_strict(tmp_path):
    cfg = anchor_config()
    raw = cfg.to_dict()
    for dc in raw["data"]["datasets"]:
        dc["source_map"] = {name: name for name in raw["model"]["input_sources"]}
    path = tmp_path / "config.yaml"
    for value in [True, -1, float("nan"), "1"]:
        changed = copy.deepcopy(raw)
        changed["training"]["annual_feature_anchor_weight"] = value
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ConfigError):
            Config.from_yaml(path)
    path.write_text(yaml.safe_dump(raw))
    assert Config.from_yaml(path).training.annual_feature_anchor_weight == 10
    monthly = Config.from_yaml(Path("configs/production/haidian_p10c_v1.yaml")).to_dict()
    monthly["training"]["annual_feature_anchor_weight"] = 1
    path.write_text(yaml.safe_dump(monthly))
    with pytest.raises(ConfigError, match="annual5m"):
        Config.from_yaml(path)
    reference = replace(cfg, model=replace(cfg.model, annual_context_pooling=False))
    assert teacher_contract_matches(cfg, reference)
    small = replace(reference, model=replace(reference.model, annual_feature_dim=16))
    assert not teacher_contract_matches(cfg, small)
    reordered = replace(
        reference,
        model=replace(
            reference.model, input_sources=dict(reversed(reference.model.input_sources.items()))
        ),
    )
    assert not teacher_contract_matches(cfg, reordered)
    assert not teacher_contract_matches(
        cfg, replace(reference, data=replace(reference.data, highres_max_observations=1))
    )


def test_anchor_named_optimizer_and_scheduler_resume_reproduce_the_next_update(tmp_path):
    torch.manual_seed(33)
    cfg = anchor_config()
    system = build_training_system(cfg)
    reference = copy.deepcopy(system.model)
    system.set_teacher(copy.deepcopy(reference))
    optimizer = build_optimizer(system, lr=1e-4, weight_decay=0.01, vmf_lr_multiplier=0)
    scheduler = build_scheduler(optimizer, warmup_epochs=0, total_epochs=4)
    batch = anchor_batch(cfg)

    def step(module, opt):
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(101)
        losses = module(batch)
        losses["total"].backward()
        opt.step()
        return float(losses["total"])

    step(system, optimizer)
    scheduler.step()
    checkpoint = tmp_path / "anchor.pt"
    schema = {name: asdict(source) for name, source in cfg.model.input_sources.items()}
    regions = [dc.region for dc in cfg.data.datasets]
    save_training_checkpoint(
        checkpoint,
        model=system.model,
        criterion=system.criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        config_sha256="c" * 64,
        git_sha="f" * 40,
        source_schema=schema,
        regions=regions,
        optimizer_named_parameters=system.named_parameters(),
    )
    restored = build_training_system(cfg)
    restored.set_teacher(copy.deepcopy(reference))
    restored_opt = build_optimizer(restored, lr=1e-4, weight_decay=0.01, vmf_lr_multiplier=0)
    restored_scheduler = build_scheduler(restored_opt, warmup_epochs=0, total_epochs=4)
    state = load_training_checkpoint(
        checkpoint,
        model=restored.model,
        criterion=restored.criterion,
        optimizer=restored_opt,
        scheduler=restored_scheduler,
        optimizer_named_parameters=restored.named_parameters(),
        expected_config_sha256="c" * 64,
        expected_source_schema=schema,
        expected_regions=regions,
    )
    # Historical optimizer layouts retain frozen entries. Adam must not create
    # moments or apply decay for the reference, whose gradients stay absent.
    teacher_slots = [
        index
        for group, names in zip(
            state["optimizer"]["param_groups"], state["optimizer_parameter_names"], strict=True
        )
        for index, name in zip(group["params"], names, strict=True)
        if name.startswith("teacher_model.")
    ]
    assert teacher_slots and all(
        index not in state["optimizer"]["state"] for index in teacher_slots
    )
    assert step(system, optimizer) == step(restored, restored_opt)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            system.model.state_dict().values(), restored.model.state_dict().values(), strict=True
        )
    )
    assert all(p.grad is None for p in restored.teacher_model.parameters())
