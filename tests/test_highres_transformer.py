import copy

import pytest
import torch

from xuannv_embedding.config import TransformerAdapterSettings
from xuannv_embedding.models.highres_transformer import HighResTransformerModel
from xuannv_embedding.models.model import AEFModel


def example():
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
