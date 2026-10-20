import copy
from dataclasses import replace

import pytest
import torch

from xuannv_embedding.config import TransformerAdapterSettings
from xuannv_embedding.models.highres_transformer import HighResTransformerModel
from xuannv_embedding.models.model import AEFModel


def example(**settings_overrides):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    base = AEFModel(
        embed_dim=8,
        stem_dim=8,
        num_months=2,
        ref_year=2025,
        ref_month=12,
        sensor_channels={"s2": 3},
        source_roles={"s2": "temporal"},
        target_heads={"s2_recon": ("continuous", 3)},
        stp={
            "space_dim": 16,
            "time_dim": 16,
            "precision_dim": 16,
            "num_blocks": 3,
            "num_heads": 2,
            "precision_scale": 1,
            "temporal_fusion": "gated_sum",
            "highres_fusion_to_embedding": False,
        },
    )
    settings = TransformerAdapterSettings(
        dim=16,
        heads=2,
        layers=2,
        injection_blocks=(1, 2),
        patch_pixels=2,
        window_cells=4,
        reference_gsd_m=10.0,
        window_chunk=32,
    )
    settings = replace(settings, **settings_overrides)
    model = HighResTransformerModel(
        base,
        {"optical": 3, "sar": 1},
        {"optical_recon": 3, "sar_recon": 1},
        settings=settings,
        freeze_base=True,
    )
    frames = {"s2": torch.randn(1, 2, 3, 16, 16)}
    masks = {"s2": torch.ones(1, 2)}
    dates = torch.tensor([[202512, 202601]])
    highres = {"optical": torch.randn(1, 2, 3, 53, 53), "sar": torch.randn(1, 2, 1, 32, 32)}
    valid = {k: torch.ones(1, 2, 1, *v.shape[-2:]) for k, v in highres.items()}
    return model, frames, masks, dates, highres, valid


def test_zero_injection_and_all_missing_restore_exact_base():
    model, x, m, t, hi, valid = example()
    model.eval()
    expected = model.base(x, m, t).embedding_map
    actual = model(x, m, t, hi, valid).embedding_map
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for injector in model.injectors.values():
        torch.nn.init.normal_(injector.output.weight, std=0.02)
        torch.nn.init.constant_(injector.output.bias, 0.2)
    absent = {k: torch.zeros_like(v) for k, v in valid.items()}
    hi = {k: torch.full_like(v, float("nan")) for k, v in hi.items()}
    actual = model(x, m, t, hi, absent).embedding_map
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_both_sources_learn_through_frozen_stp_and_checkpointing():
    model, x, m, t, hi, valid = example()
    old = copy.deepcopy(model.base.state_dict())
    model.base.stp_encoder.gradient_checkpointing = True
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.002)
    for _ in range(3):
        optimizer.zero_grad()
        out = model(x, m, t, hi, valid)
        loss = sum(value.square().mean() for value in out.reconstructions.values())
        loss.backward()
        optimizer.step()
    for key, val in old.items():
        torch.testing.assert_close(model.base.state_dict()[key], val, rtol=0, atol=0)
    for source in hi:
        grad = model.encoders[source].projection.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())


def test_window_encoding_preserves_months_and_ignores_invalid_values():
    model, x, m, t, hi, valid = example()
    encoder = model.encoders["optical"]
    valid["optical"][..., :9, :13] = 0
    altered = hi["optical"].clone()
    altered[..., :9, :13] = float("nan")
    a = encoder(hi["optical"], valid["optical"], (16, 16), t)
    b = encoder(altered, valid["optical"], (16, 16), t)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    altered = hi["optical"].clone()
    altered[:, 1] += 10
    b = encoder(altered, valid["optical"], (16, 16), t)
    torch.testing.assert_close(a[0][:, 0], b[0][:, 0], rtol=0, atol=0)
    assert not torch.equal(a[0][:, 1], b[0][:, 1])


def test_invalid_spatial_grid_and_static_input_rejected():
    model, x, m, t, hi, valid = example()
    with pytest.raises(ValueError, match="monthly"):
        model(x, m, t, {k: v[:, 0] for k, v in hi.items()}, valid)
    with pytest.raises(ValueError, match="divisible"):
        model.encoders["optical"](hi["optical"], valid["optical"], (17, 16), t)


def test_static_highres_is_encoded_once_without_fabricated_months():
    model, x, m, t, hi, valid = example()
    model.eval()
    observed = []
    hook = model.encoders["optical"].register_forward_pre_hook(
        lambda module, args: observed.append((args[0].shape, args[3].clone()))
    )
    result = model(x, m, t, {"optical": hi["optical"][:, 0]}, {"optical": valid["optical"][:, 0]})
    hook.remove()
    assert observed[0][0] == (1, 1, 3, 53, 53)
    assert observed[0][1].tolist() == [[0]]
    assert result.embedding_map.shape == (1, 2, 8, 16, 16)
    assert torch.isfinite(result.embedding_map).all()


def test_static_highres_missing_pixels_restore_base_and_receive_gradients():
    model, x, m, t, hi, valid = example()
    image, mask = hi["optical"][:, 0], valid["optical"][:, 0]
    for injector in model.injectors.values():
        torch.nn.init.normal_(injector.output.weight, std=0.02)
    model.eval()
    absent = model(x, m, t, {"optical": image * float("nan")}, {"optical": mask * 0})
    torch.testing.assert_close(absent.embedding_map, model.base(x, m, t).embedding_map)
    model.train()
    result = model(x, m, t, {"optical": image}, {"optical": mask})
    result.embedding_map.square().mean().backward()
    grad = model.encoders["optical"].projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_joint_mode_updates_public_backbone_without_changing_checkpoint_keys():
    model, x, m, t, hi, valid = example()
    keys = set(model.state_dict())
    model.set_freeze_base(False)
    model.train()
    result = model(x, m, t, hi, valid)
    result.reconstructions["s2_recon"].square().mean().backward()
    gradients = [p.grad for p in model.base.parameters() if p.grad is not None]
    assert gradients and any(g.abs().sum() > 0 for g in gradients)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert set(model.state_dict()) == keys
    model.set_freeze_base(True)
    assert not any(p.requires_grad for p in model.base.parameters())


def test_joint_constructor_and_missing_static_observation_match_public_base():
    model, x, m, t, hi, valid = example(allow_base_only=True)
    joint = HighResTransformerModel(
        copy.deepcopy(model.base),
        {"optical": 3},
        {"optical_recon": 3},
        settings=model.settings,
        freeze_base=False,
    ).eval()
    missing = {"optical": torch.zeros_like(valid["optical"][:, 0])}
    images = {"optical": torch.full_like(hi["optical"][:, 0], float("nan"))}
    actual = joint(x, m, t, images, missing).embedding_map
    torch.testing.assert_close(actual, joint.base(x, m, t).embedding_map, rtol=0, atol=0)
