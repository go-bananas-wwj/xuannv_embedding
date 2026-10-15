from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_annual_model import annual_config, model_forward, raw_batch

from xuannv_embedding.training.annual import AnnualLoss, prepare_annual_batch
from xuannv_embedding.training.cli import build_training_system


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
