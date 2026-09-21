from dataclasses import replace
from pathlib import Path

import pytest
import torch

from xuannv_embedding.config import Config
from xuannv_embedding.models.annual import (
    area_resample,
    masked_resample,
    overlap_weights,
)
from xuannv_embedding.training.annual import prepare_annual_batch
from xuannv_embedding.training.cli import build_training_system


def annual_config():
    config = Config.from_yaml(Path("configs/production/annual5m_v1.yaml"))
    sources = {
        name: value
        for name, value in config.model.input_sources.items()
        if name in {"s2", "s1", "GF6_PAN_c1", "JL1GP01_PMS1_5m_c6_d00302d8f981"}
    }
    model = replace(
        config.model,
        input_sources=sources,
        annual_feature_dim=8,
        annual_pan_sources=["GF6_PAN_c1"],
        stem_dim=8,
        stp=replace(config.model.stp, space_dim=8, time_dim=8, num_blocks=1, num_heads=2),
    )
    return replace(
        config, model=model, training=replace(config.training, gradient_checkpointing=False)
    )


def raw_batch(size=16):
    sources = {"s2": torch.randn(1, 12, 10, size, size), "s1": torch.randn(1, 12, 2, size, size)}
    highres = []
    for source, channels, resolution in [
        ("GF6_PAN_c1", 1, size * 5),
        ("JL1GP01_PMS1_5m_c6_d00302d8f981", 6, size * 2),
    ]:
        for month in (1, 6):
            highres.append(
                {
                    "values": torch.randn(channels, resolution, resolution),
                    "mask": torch.ones(resolution, resolution, dtype=torch.bool),
                    "metadata": {
                        "source": source,
                        "date": f"2020-{month:02d}-15",
                        "relative_registration": {"status": "measured", "offset_m": 0},
                    },
                }
            )
    return {
        "source_frames": sources,
        "source_masks": {name: torch.ones(1, 12, dtype=torch.bool) for name in sources},
        "source_pixel_masks": {
            name: torch.ones(1, 12, size, size, dtype=torch.bool) for name in sources
        },
        "timestamps": torch.arange(202001, 202013)[None],
        "highres_observations": [highres],
        "metadata": [{"patch_id": "parent", "year": 2020}],
    }


def model_forward(model, batch):
    return model(
        **{
            key: batch[key]
            for key in (
                "source_frames",
                "source_masks",
                "timestamps",
                "source_pixel_masks",
                "highres_observations",
                "reconstruction_requests",
            )
        }
    )


def test_fractional_area_weights_and_conservation():
    weights = overlap_weights(5, 2, torch.tensor(0.0))
    assert torch.allclose(weights, torch.tensor([[0.4, 0.4, 0.2, 0, 0], [0, 0, 0.2, 0.4, 0.4]]))
    values = torch.randn(2, 3, 15, 20, requires_grad=True)
    result = area_resample(values, (6, 8))
    assert torch.allclose(result.mean((-2, -1)), values.mean((-2, -1)), atol=1e-6)
    result.sum().backward()
    assert torch.isfinite(values.grad).all()
    mask = torch.ones(1, 1, 5, 5)
    mask[:, :, :, 2] = 0
    values = torch.ones(1, 1, 5, 5)
    values[:, :, :, 2] = float("nan")
    pooled, support = masked_resample(values, mask, (2, 2))
    assert torch.allclose(pooled, torch.ones_like(pooled))
    assert torch.allclose(support, torch.full_like(support, 0.8))


def test_annual_forward_backward_and_frozen_noise():
    config = annual_config()
    system = build_training_system(config)
    batch = prepare_annual_batch(
        raw_batch(), config, training=True, generator=torch.Generator().manual_seed(9)
    )
    losses = system(batch)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert system.model.bottleneck.log_kappa.grad is None
    assert system.model.gate.weight.grad is not None
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in system.parameters()
        if parameter.grad is not None
    )
    assert system.model.highres_encoders["GF6_PAN_c1"].stem.weight.grad.abs().sum() > 0


def test_annual_permutation_missing_fallback_and_pixel_masks():
    config = annual_config()
    model = build_training_system(config).model.eval()
    batch = prepare_annual_batch(raw_batch(), config, training=False)
    with torch.no_grad():
        output = model_forward(model, batch)
        batch["highres_observations"][0].reverse()
        reordered = model_forward(model, batch)
        assert torch.allclose(output.embedding_map, reordered.embedding_map, atol=1e-6)
        assert output.embedding_map.shape == (1, 1, 64, 32, 32)
        assert torch.allclose(output.embedding_map.norm(dim=2), torch.ones(1, 1, 32, 32), atol=1e-5)
        for observation in batch["highres_observations"][0]:
            observation["mask"].zero_()
            observation["values"].fill_(float("nan"))
        missing = model_forward(model, batch)
        batch["highres_observations"] = [[]]
        absent = model_forward(model, batch)
        assert torch.equal(missing.embedding_map, absent.embedding_map)
        for mask in batch["source_pixel_masks"].values():
            mask.zero_()
        empty = model_forward(model, batch)
        assert not empty.validity_mask.any()
        assert torch.isfinite(empty.embedding_map).all()


def test_annual_shared_geographic_mask_and_hidden_targets():
    config = annual_config()
    config = replace(
        config,
        training=replace(
            config.training,
            input_masking=replace(config.training.input_masking, spatial_block_size=4),
        ),
    )
    raw = raw_batch()
    batch = prepare_annual_batch(
        raw, config, training=True, generator=torch.Generator().manual_seed(11)
    )
    for source, values in batch["source_frames"].items():
        mask = batch["source_pixel_masks"][source]
        assert torch.equal(mask, batch["source_pixel_masks"]["s2"])
        assert (values[~mask[:, :, None].expand_as(values)] == 0).all()
        assert torch.equal(batch["targets"][source + "_recon"], raw["source_frames"][source])
    for observation in batch["highres_observations"][0]:
        mask = observation["mask"]
        assert (observation["values"][:, ~mask] == 0).all()
        _, coverage = masked_resample(mask[None, None].float(), mask[None, None].float(), (16, 16))
        assert (coverage[0, 0] <= batch["source_pixel_masks"]["s2"][0].any(dim=0)).all()


def test_annual_rejects_unmeasured_registration():
    raw = raw_batch()
    raw["highres_observations"][0][0]["metadata"]["relative_registration"]["status"] = "unresolved"
    with pytest.raises(ValueError, match="Unmeasured"):
        prepare_annual_batch(raw, annual_config(), training=False)
