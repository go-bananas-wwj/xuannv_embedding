import copy
from dataclasses import replace

import pytest
import torch
from test_highres_transformer import example

from xuannv_embedding.config import (
    ConfigError,
    STPConfig,
    TransformerAdapterSettings,
    _parse_transformer,
)
from xuannv_embedding.models.highres_transformer import (
    CrossResolutionInjector,
    HighResWindowEncoder,
)


def injector_input():
    precision = torch.zeros(1, 1, 4, 4, 8)
    tokens = torch.zeros(1, 1, 1, 4, 8)
    valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    positions = torch.zeros(1, 4, 2)
    return precision, {"optical": (tokens, valid, positions)}


def test_null_option_scales_projection_bias_and_can_retain_base():
    settings = TransformerAdapterSettings(dim=8, heads=2, allow_base_only=True)
    model = CrossResolutionInjector(8, ["optical"], settings)
    with torch.no_grad():
        model.gates["optical"].weight.zero_()
        model.gates["optical"].bias.zero_()
        model.output.weight.zero_()
        model.output.bias.fill_(2)
        model.base_gate.weight.zero_()
        model.base_gate.bias.zero_()
    x, encoded = injector_input()
    torch.testing.assert_close(model(x, encoded), torch.ones_like(x))
    model.base_gate.bias.data.fill_(80)
    torch.testing.assert_close(model(x, encoded), x, atol=1e-30, rtol=0)
    model.base_gate.bias.data.fill_(-80)
    torch.testing.assert_close(model(x, encoded), torch.full_like(x, 2))


@pytest.mark.parametrize("coverage", [False, True])
def test_full_missing_after_training_preserves_base_and_checkpoint_reload(coverage):
    model, x, mask, dates, highres, valid = example(allow_base_only=True, coverage_gating=coverage)
    frozen = copy.deepcopy(model.base.state_dict())
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        output = model(x, mask, dates, highres, valid)
        sum(v.square().mean() for v in output.reconstructions.values()).backward()
        optimizer.step()
    assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in frozen.items())
    assert all(p.grad is None for p in model.base.parameters())
    grad = next(iter(model.injectors.values())).base_gate.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    model.eval()
    missing = {k: torch.zeros_like(v) for k, v in valid.items()}
    invalid = {k: torch.full_like(v, float("nan")) for k, v in highres.items()}
    expected = model.base(x, mask, dates).embedding_map
    actual = model(x, mask, dates, invalid, missing).embedding_map
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    clone, *_ = example(allow_base_only=True, coverage_gating=coverage)
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.eval()
    torch.testing.assert_close(
        clone(x, mask, dates, highres, valid).embedding_map,
        model(x, mask, dates, highres, valid).embedding_map,
        rtol=0,
        atol=0,
    )


def test_optional_modules_do_not_change_shared_initialization_or_rng():
    baseline, *_ = example()
    rng = torch.random.get_rng_state()
    modified, *_ = example(allow_base_only=True, coverage_gating=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, modified.state_dict()[key]), key


def test_coverage_counts_actual_pixels_in_partial_boundary_patches():
    settings = TransformerAdapterSettings(
        dim=8, heads=2, patch_pixels=2, window_cells=4, coverage_gating=True
    )
    encoder = HighResWindowEncoder(1, settings).eval()
    image = torch.randn(1, 1, 1, 5, 5)
    mask = torch.ones_like(image)
    dates = torch.tensor([[202601]])
    *_, coverage = encoder(image, mask, (4, 4), dates)
    torch.testing.assert_close(coverage, torch.ones_like(coverage), rtol=0, atol=0)
    mask[..., -1, :] = 0
    *_, coverage = encoder(image, mask, (4, 4), dates)
    torch.testing.assert_close(coverage, torch.full_like(coverage, 0.8))


def test_coverage_changes_fusion_without_reading_labels():
    settings = TransformerAdapterSettings(dim=8, heads=2, coverage_gating=True)
    injector = CrossResolutionInjector(8, ["optical", "sar"], settings)
    with torch.no_grad():
        injector.output.weight.copy_(torch.eye(8))
        injector.output.bias.zero_()
        for index, name in enumerate(["optical", "sar"]):
            injector.gates[name].weight.zero_()
            injector.gates[name].bias.zero_()
            injector.attention[name].proj.weight.zero_()
            injector.attention[name].proj.bias.fill_(index + 1)
            injector.coverage_gates[name].weight.fill_(3)
    x, encoded = injector_input()
    source = encoded["optical"]
    full = torch.ones(1, 1, 1, 1)
    a = {"optical": (*source, full), "sar": (*source, full)}
    b = {"optical": (*source, full), "sar": (*source, full * 0.1)}
    assert (injector(x, b) < injector(x, a)).all()


def test_pooling_ignores_token_order_and_missing_token_values():
    settings = TransformerAdapterSettings(dim=8, heads=2, spatial_readout="mean")
    injector = CrossResolutionInjector(8, ["optical"], settings)
    torch.nn.init.normal_(injector.output.weight)
    x, encoded = injector_input()
    tokens, valid, positions = encoded["optical"]
    tokens.normal_()
    valid[..., -1] = False
    original = injector(x, {"optical": (tokens, valid, positions)})
    tokens[..., -1, :] = float("nan")
    changed = injector(x, {"optical": (tokens.flip(-2), valid.flip(-1), positions.flip(-2))})
    torch.testing.assert_close(changed, original)
    model, xx, masks, dates, hi, vv = example(spatial_readout="mean")
    model.train()
    sum(
        t.square().mean() for t in model(xx, masks, dates, hi, vv).reconstructions.values()
    ).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize(
    "field,value",
    [("allow_base_only", 1), ("coverage_gating", "yes"), ("spatial_readout", "invalid")],
)
def test_selective_settings_reject_invalid_values(field, value):
    with pytest.raises(ConfigError):
        _parse_transformer({field: value}, STPConfig())


def test_selective_settings_are_independent_and_defaults_are_legacy():
    defaults = TransformerAdapterSettings()
    assert not defaults.allow_base_only and not defaults.coverage_gating
    assert defaults.spatial_readout == "attention"
    actual = _parse_transformer(
        {"allow_base_only": True, "coverage_gating": False, "spatial_readout": "mean"},
        STPConfig(),
    )
    assert actual == replace(defaults, allow_base_only=True, spatial_readout="mean")
