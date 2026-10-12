from dataclasses import replace
from pathlib import Path

import pytest
import torch

from xuannv_embedding.config import Config, STPConfig, TransformerAdapterSettings
from xuannv_embedding.training.regional_base import build_regional_system


def paired_configs():
    config = Config.from_yaml(Path(__file__).parents[1] / "configs/examples/china_p10c_pilot.yaml")
    model = replace(
        config.model,
        embed_dim=8,
        stem_dim=8,
        num_months=2,
        stp=STPConfig(
            space_dim=16,
            time_dim=16,
            precision_dim=16,
            precision_scale=1,
            num_blocks=3,
            num_heads=2,
            temporal_fusion="gated_sum",
            highres_fusion_to_embedding=False,
        ),
        highres_transformer=TransformerAdapterSettings(
            dim=16, heads=2, layers=1, injection_blocks=(1,), window_chunk=32
        ),
    )
    full = replace(config, model=model, data=replace(config.data, months=["2025-01", "2025-02"]))
    sources = {k: v for k, v in model.input_sources.items() if v.role == "temporal"}
    targets = {
        k: v
        for k, v in model.target_heads.items()
        if v.source in sources or v.loss_type == "categorical"
    }
    public = replace(
        full,
        model=replace(model, input_sources=sources, target_heads=targets, highres_transformer=None),
    )
    return full, public


def test_public_region_starts_with_identical_untrained_common_weights():
    torch.set_num_threads(1)
    full, public = paired_configs()
    reference, _ = build_regional_system(full, mode="scratch", checkpoint=None, seed=41)
    control, receipt = build_regional_system(
        public, mode="scratch", checkpoint=None, seed=41, initialization_config=full
    )
    for key, value in reference.model.base.state_dict().items():
        assert torch.equal(value, control.model.base.state_dict()[key]), key
    for key, value in reference.criterion.state_dict().items():
        assert torch.equal(value, control.criterion.state_dict()[key]), key
    assert receipt["parent_weights_read"] is False
    assert receipt["shared_untrained_initialization"] is True
    assert not hasattr(control.model, "encoders")
    assert set(control.criterion.base.target_cfg) == set(public.model.target_heads) - {
        "worldcover"
    } | {"osm_landcover"}


def test_public_region_has_finite_gradients_and_no_highres_reconstructions():
    full, public = paired_configs()
    system, _ = build_regional_system(
        public, mode="scratch", checkpoint=None, seed=41, initialization_config=full
    )
    frames = {
        k: torch.randn(1, 2, v.channels, 16, 16) for k, v in public.model.input_sources.items()
    }
    masks = {k: torch.ones(1, 2) for k in frames}
    system.train()
    result = system.model(frames, masks, torch.tensor([[202501, 202502]]), {}, {})
    assert not any("highres" in k for k in result.reconstructions)
    result.embedding_map.square().mean().backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in system.model.parameters()
    )
    with pytest.raises(ValueError, match="highres"):
        system.model(frames, masks, None, {"image": torch.zeros(1)}, {"image": torch.zeros(1)})


def test_public_region_rejects_changed_common_architecture():
    full, public = paired_configs()
    altered = replace(public, model=replace(public.model, stem_dim=16))
    with pytest.raises(ValueError, match="common.*architecture"):
        build_regional_system(
            altered, mode="scratch", checkpoint=None, seed=41, initialization_config=full
        )


def test_public_region_can_continue_its_own_checkpoint(tmp_path):
    full, public = paired_configs()
    parent, _ = build_regional_system(
        public, mode="scratch", checkpoint=None, seed=41, initialization_config=full
    )
    checkpoint = tmp_path / "public.pt"
    torch.save(
        {"model": parent.model.state_dict(), "criterion": parent.criterion.state_dict()}, checkpoint
    )
    child, receipt = build_regional_system(
        public, mode="transfer", checkpoint=checkpoint, seed=2, initialization_config=full
    )
    assert receipt["parent_weights_read"] is True
    for key, value in parent.model.state_dict().items():
        assert torch.equal(value, child.model.state_dict()[key])
