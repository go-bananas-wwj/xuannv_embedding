from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_annual_model import annual_config, model_forward, raw_batch

from xuannv_embedding.training.annual import (
    AnnualLoss,
    multiscale_latent_loss,
    prepare_annual_batch,
)
from xuannv_embedding.training.cli import build_training_system, teacher_contract_matches
from xuannv_embedding.training.runtime import train_steps


@pytest.mark.parametrize("weight", [0.0, 0.25, 1.0])
def test_visible_weight_is_a_normalized_pixel_weight_and_preserves_heldout_gradients(weight):
    config = annual_config()
    config = replace(config, training=replace(config.training, highres_visible_weight=weight))
    prediction = torch.zeros(1, 2, 2, requires_grad=True)
    hidden = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    visible = 1 - hidden
    target = {
        "source": "sensor",
        "quality_weight": 0.5,
        "partitions": {
            "heldout": {"values": torch.ones(1, 2, 2), "mask": hidden},
            "visible": {"values": torch.full((1, 2, 2), 3.0), "mask": visible},
        },
    }
    output = SimpleNamespace(
        embedding_map=torch.ones(1, 1, 64, 2, 2),
        embedding=torch.ones(1, 1, 64),
        validity_mask=torch.ones(1, 1, 1, 2, 2, dtype=torch.bool),
        reconstructions={
            name: torch.zeros(1, 1, head.channels, 2, 2)
            for name, head in config.model.target_heads.items()
        },
        highres_reconstructions=[[prediction]],
    )
    targets = {name: torch.zeros_like(values) for name, values in output.reconstructions.items()}
    masks = {name: torch.zeros(1, 1, 2, 2) for name in targets}
    losses = AnnualLoss(config)(output, targets, masks, highres_targets=[[target]])
    expected = 0.5 * (1 + 9 * weight) / (1 + 3 * weight)
    assert losses["highres_recon"].item() == pytest.approx(expected)
    assert losses["highres_heldout_recon"].item() == pytest.approx(0.5)
    assert losses["highres_visible_recon"].item() == pytest.approx(1.5)
    losses["total"].backward()
    assert prediction.grad[0, 0, 0] != 0
    assert bool((prediction.grad[0, 1] != 0).all()) is (weight > 0)


@pytest.mark.parametrize("weight,legacy_visible", [(0.0, False), (1.0, True)])
def test_weight_endpoints_match_legacy_highres_objectives(weight, legacy_visible):
    config = annual_config()
    config = replace(
        config,
        training=replace(
            config.training,
            highres_visible_targets=legacy_visible,
            input_masking=replace(config.training.input_masking, spatial_block_size=4),
        ),
    )
    split = replace(config, training=replace(config.training, highres_visible_weight=weight))
    torch.manual_seed(0)
    raw = raw_batch()
    # At native 5 m resolution the geographic blocks do not mix partitions
    # within a target pixel; 2 m PAN is tested separately for support conservation.
    raw["highres_observations"][0] = [
        observation
        for observation in raw["highres_observations"][0]
        if observation["metadata"]["source"] not in config.model.annual_pan_sources
    ]
    model = build_training_system(config).model.eval()
    losses = []
    for variant in [config, split]:
        batch = prepare_annual_batch(
            raw, variant, training=True, generator=torch.Generator().manual_seed(11)
        )
        output = model_forward(model, batch)
        losses.append(
            AnnualLoss(variant)(
                output,
                batch["targets"],
                batch["target_masks"],
                highres_targets=batch["highres_targets"],
            )
        )
    assert torch.allclose(losses[0]["highres_recon"], losses[1]["highres_recon"], atol=1e-6)


def test_native_pan_supervision_partitions_conserve_support():
    config = annual_config()
    config = replace(config, training=replace(config.training, highres_visible_weight=0.1))
    batch = prepare_annual_batch(
        raw_batch(32), config, training=True, generator=torch.Generator().manual_seed(11)
    )
    for target in batch["highres_targets"][0]:
        parts = target["partitions"]
        assert torch.allclose(
            parts["heldout"]["mask"] + parts["visible"]["mask"],
            torch.ones_like(parts["heldout"]["mask"]),
            atol=1e-6,
        )


def test_coarse_distillation_ignores_unsupported_pixels_and_stops_target_gradients():
    torch.manual_seed(3)
    student = torch.randn(1, 4, 8, 8, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    support = torch.ones(1, 1, 8, 8)
    support[:, :, :4] = 0
    loss = multiscale_latent_loss(student, teacher, support, [2, 4, 8])
    loss.backward()
    assert teacher.grad is None
    assert student.grad[:, :, :4].count_nonzero() == 0
    assert student.grad[:, :, 4:].abs().sum() > 0
    assert multiscale_latent_loss(student, student, support, [2, 4, 8]).abs() < 1e-6
    assert multiscale_latent_loss(student, teacher, support * 0, [2, 4, 8]) == 0


def test_annual_teacher_can_have_a_different_backbone_capacity():
    student = annual_config()
    teacher = replace(
        student,
        model=replace(student.model, stp=replace(student.model.stp, space_dim=16, time_dim=16)),
    )
    assert teacher_contract_matches(student, teacher)
    assert not teacher_contract_matches(
        student, replace(teacher, model=replace(teacher.model, embed_dim=32))
    )


@pytest.mark.parametrize("objective", ["teacher", "view"])
def test_semantic_objectives_alone_train_the_annual_backbone(objective):
    torch.manual_seed(8)
    base = annual_config()
    config = replace(
        base,
        model=replace(base.model, annual_context_pooling=True),
        training=replace(
            base.training,
            latent_prediction_weight=0.1 if objective == "teacher" else 0.0,
            latent_prediction_pool_factors=[2, 4, 8],
            view_consistency_weight=0.1 if objective == "view" else 0.0,
        ),
    )
    system = build_training_system(config)
    raw = raw_batch(32)
    batch = prepare_annual_batch(
        raw, config, training=True, generator=torch.Generator().manual_seed(1)
    )
    if objective == "teacher":
        system.set_teacher(build_training_system(base).model)
        batch["teacher_view"] = prepare_annual_batch(raw, base, training=False)
        key = "latent_prediction"
    else:
        system.set_ema(0.99)
        batch["consistency_view"] = prepare_annual_batch(
            raw, config, training=True, generator=torch.Generator().manual_seed(2)
        )
        key = "view_consistency"
    system(batch)[key].backward()
    assert (
        sum(
            parameter.grad.abs().sum()
            for parameter in system.model.temporal_encoder.parameters()
            if parameter.grad is not None
        )
        > 0
    )
    assert system.model.bottleneck.proj.weight.grad.abs().sum() > 0
    for target in [system.teacher_model, system.ema_model]:
        if target is not None:
            assert all(parameter.grad is None for parameter in target.parameters())


def test_ema_updates_only_on_optimizer_steps_and_roundtrips_auxiliary_state():
    torch.manual_seed(8)
    base = annual_config()
    config = replace(
        base,
        model=replace(base.model, annual_context_pooling=True),
        training=replace(
            base.training,
            view_consistency_weight=0.1,
            latent_prediction_weight=0.1,
            latent_prediction_pool_factors=[2, 4, 8],
        ),
    )
    system = build_training_system(config)
    teacher = build_training_system(base).model
    system.set_teacher(teacher)
    system.set_ema(0.5)
    raw = raw_batch()
    batch = prepare_annual_batch(
        raw, config, training=True, generator=torch.Generator().manual_seed(1)
    )
    batch["teacher_view"] = prepare_annual_batch(raw, config, training=False)
    batch["consistency_view"] = prepare_annual_batch(
        raw, config, training=True, generator=torch.Generator().manual_seed(2)
    )
    assert not torch.equal(
        batch["source_pixel_masks"]["s2"], batch["consistency_view"]["source_pixel_masks"]["s2"]
    )
    original = system.model.bottleneck.proj.weight.detach().clone()
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-3)
    summary = train_steps(
        system,
        [batch],
        optimizer,
        device=torch.device("cpu"),
        epochs=1,
        gradient_accumulation_steps=2,
        amp=False,
    )
    assert summary["optimizer_steps"] == system.ema_updates == 1
    expected = (original + system.model.bottleneck.proj.weight.detach()) / 2
    assert torch.allclose(system.ema_model.bottleneck.proj.weight, expected)
    assert not system.ema_model.training and not system.teacher_model.training
    assert all(parameter.grad is None for parameter in system.ema_model.parameters())
    assert all(
        torch.equal(value, teacher_before[key]) for key, value in teacher.state_dict().items()
    )
    restored = build_training_system(config)
    restored.set_ema(0.5)
    restored.restore_auxiliary_state(system.auxiliary_state())
    assert restored.ema_updates == 1
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            system.ema_model.parameters(), restored.ema_model.parameters(), strict=True
        )
    )
    with pytest.raises(ValueError, match="missing"):
        restored.restore_auxiliary_state({})
