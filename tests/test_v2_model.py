from __future__ import annotations

from pathlib import Path

import torch

from xuannv_embedding.config import V2Config
from xuannv_embedding.data.contracts import ProductSpec
from xuannv_embedding.models import build_v2_model
from xuannv_embedding.models.multires_adapters import MultiResolutionProductAdapter
from xuannv_embedding.models.v2_model import XuannvV2Model


def _product(
    product_id: str,
    role: str,
    channels: int,
    gsd: float,
    *,
    time_precision: str = "month",
) -> ProductSpec:
    return ProductSpec(
        product_id=product_id,
        role=role,
        bands=tuple(f"b{index}" for index in range(channels)),
        native_gsd_m=tuple(gsd for _ in range(channels)),
        stored_gsd_m=gsd,
        dtype="float32",
        time_precision=time_precision,
        already_resampled=gsd == 10,
        qa_available=role == "highres",
    )


def _products() -> dict[str, ProductSpec]:
    return {
        "s2": _product("s2", "dense", 10, 10),
        "s1": _product("s1", "dense", 2, 10),
        "landsat": _product("landsat", "dense", 6, 30),
        "optical_2m": _product("optical_2m", "highres", 4, 2, time_precision="exact"),
        "optical_5m": _product("optical_5m", "highres", 4, 5, time_precision="exact"),
    }


def _model(*, mode: str = "causal_window") -> XuannvV2Model:
    return XuannvV2Model(
        _products(),
        embedding_dim=64,
        stem_dim=16,
        spatial_dim=64,
        temporal_dim=32,
        precision_dim=32,
        num_blocks=1,
        num_heads=4,
        temporal_mode=mode,
        dense_lookback_days=365,
        highres_structure_days=730,
        highres_appearance_days=90,
        highres_structure_max_observations=8,
        highres_appearance_max_observations=4,
    )


def _batch() -> dict[str, object]:
    batch_size = 2
    source_frames = {
        "s2": torch.randn(batch_size, 2, 10, 32, 32),
        "s1": torch.randn(batch_size, 3, 2, 32, 32),
        "landsat": torch.randn(batch_size, 1, 6, 11, 11),
    }
    source_pixel_masks = {
        name: torch.ones(value.shape[0], value.shape[1], 1, value.shape[3], value.shape[4])
        for name, value in source_frames.items()
    }
    return {
        "source_frames": source_frames,
        "source_pixel_masks": source_pixel_masks,
        "source_observation_masks": {
            "s2": torch.ones(batch_size, 2, dtype=torch.bool),
            "s1": torch.ones(batch_size, 3, dtype=torch.bool),
            "landsat": torch.ones(batch_size, 1, dtype=torch.bool),
        },
        "source_time_bounds": {
            "s2": torch.tensor([[[10.0, 40.0], [100.0, 130.0]]]).repeat(batch_size, 1, 1),
            "s1": torch.tensor([[[0.0, 30.0], [60.0, 90.0], [130.0, 160.0]]]).repeat(
                batch_size, 1, 1
            ),
            "landsat": torch.tensor([[[20.0, 50.0]]]).repeat(batch_size, 1, 1),
        },
        "source_available_at": {
            "s2": torch.tensor([[40.0, 130.0]]).repeat(batch_size, 1),
            "s1": torch.tensor([[30.0, 90.0, 160.0]]).repeat(batch_size, 1),
            "landsat": torch.tensor([[50.0]]).repeat(batch_size, 1),
        },
        "output_intervals": torch.tensor([[[80.0, 100.0], [140.0, 170.0]]]).repeat(
            batch_size, 1, 1
        ),
        "highres_frames": {
            "optical_2m": torch.randn(batch_size, 2, 4, 160, 160),
            "optical_5m": torch.randn(batch_size, 1, 4, 64, 64),
        },
        "highres_masks": {
            "optical_2m": torch.ones(batch_size, 2, 1, 160, 160),
            "optical_5m": torch.ones(batch_size, 1, 1, 64, 64),
        },
        "highres_acquired_at": {
            "optical_2m": torch.tensor([[70.0, 150.0]]).repeat(batch_size, 1),
            "optical_5m": torch.tensor([[95.0]]).repeat(batch_size, 1),
        },
        "highres_available_at": {
            "optical_2m": torch.tensor([[72.0, 152.0]]).repeat(batch_size, 1),
            "optical_5m": torch.tensor([[96.0]]).repeat(batch_size, 1),
        },
        "highres_geotransforms": {
            "optical_2m": torch.tensor([2.0, 0.0, 0.0, 0.0, -2.0, 320.0])
            .view(1, 1, 6)
            .repeat(batch_size, 2, 1),
            "optical_5m": torch.tensor([5.0, 0.0, 0.0, 0.0, -5.0, 320.0])
            .view(1, 1, 6)
            .repeat(batch_size, 1, 1),
        },
        "output_geotransforms": torch.tensor([10.0, 0.0, 0.0, 0.0, -10.0, 320.0])
        .view(1, 6)
        .repeat(batch_size, 1),
        "output_size": (32, 32),
    }


def test_v2_model_accepts_independent_timelines_and_multires_scenes() -> None:
    model = _model().eval()

    with torch.no_grad():
        output = model(**_batch())

    assert output.embedding_map.shape == (2, 2, 64, 32, 32)
    assert output.reconstructions["s2"].shape == (2, 2, 10, 32, 32)
    assert output.highres_detail_stats["optical_2m"].shape == (2, 2, 3, 32, 32)
    assert not torch.isnan(output.embedding_map).any()
    norms = torch.linalg.vector_norm(output.embedding_map, dim=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_causal_output_is_invariant_to_future_dense_and_highres_observations() -> None:
    torch.manual_seed(1)
    model = _model(mode="causal_window").eval()
    batch = _batch()
    with torch.no_grad():
        baseline = model(**batch).embedding_map[:, 0]
        batch["source_frames"]["s2"][:, 1].add_(10_000)
        batch["highres_frames"]["optical_2m"][:, 1].add_(10_000)
        changed = model(**batch).embedding_map[:, 0]

    assert torch.allclose(baseline, changed, atol=1e-6, rtol=0)


def test_within_period_excludes_observations_outside_output_interval() -> None:
    torch.manual_seed(2)
    model = _model(mode="within_period").eval()
    batch = _batch()
    with torch.no_grad():
        baseline = model(**batch).embedding_map[:, 0]
        batch["source_frames"]["s1"][:, 0].mul_(1000)
        changed = model(**batch).embedding_map[:, 0]

    assert torch.allclose(baseline, changed, atol=1e-6, rtol=0)


def test_centered_window_can_use_future_observation_but_not_out_of_window() -> None:
    torch.manual_seed(3)
    model = _model(mode="centered_window").eval()
    batch = _batch()
    with torch.no_grad():
        baseline = model(**batch).embedding_map[:, 0]
        batch["source_frames"]["s2"][:, 1].add_(100)
        future_changed = model(**batch).embedding_map[:, 0]

    assert not torch.allclose(baseline, future_changed)

    outside = _batch()
    outside["source_time_bounds"]["s2"][:, 1] = torch.tensor([1000.0, 1001.0])
    outside["source_available_at"]["s2"][:, 1] = 1001.0
    with torch.no_grad():
        baseline = model(**outside).embedding_map.clone()
        outside["source_frames"]["s2"][:, 1].add_(10_000)
        changed = model(**outside).embedding_map
    assert torch.allclose(baseline, changed, atol=1e-6, rtol=0)


def test_highres_native_convolution_receives_gradient_before_alignment() -> None:
    model = _model().train()
    model.gradient_checkpointing = True
    output = model(**_batch())

    output.embedding_map[:, :, 0].mean().backward()

    gradient = model.highres_adapters["optical_2m"].native_stem[0].weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0


def test_highres_products_have_independent_adapters_and_scene_axis() -> None:
    model = _model()

    assert set(model.highres_adapters) == {"optical_2m", "optical_5m"}
    assert (
        model.highres_adapters["optical_2m"].native_stem[0].weight
        is not model.highres_adapters["optical_5m"].native_stem[0].weight
    )


def test_native_downsampling_depth_follows_declared_gsd() -> None:
    expected = {0.5: 3, 2.0: 1, 3.0: 0, 5.0: 0}

    for gsd, blocks in expected.items():
        spec = _product(f"highres_{gsd}", "highres", 4, gsd, time_precision="exact")
        adapter = MultiResolutionProductAdapter(spec, dim=8)
        assert len(adapter.downsample_blocks) == blocks
        assert adapter.feature_gsd_m <= 5.0


def test_affine_alignment_masks_highres_scene_outside_output_footprint() -> None:
    spec = _product("optical_5m", "highres", 4, 5, time_precision="exact")
    adapter = MultiResolutionProductAdapter(spec, dim=8).eval()
    frames = torch.ones(1, 1, 4, 8, 8)
    masks = torch.ones(1, 1, 1, 8, 8)
    source_transform = torch.tensor([[[5.0, 0.0, 0.0, 0.0, -5.0, 40.0]]])
    far_target = torch.tensor([[10.0, 0.0, 1000.0, 0.0, -10.0, 1040.0]])

    with torch.no_grad():
        features, aligned_mask, quality = adapter(
            frames, masks, source_transform, (4, 4), far_target
        )

    assert torch.count_nonzero(aligned_mask).item() == 0
    assert torch.count_nonzero(features).item() == 0
    assert quality.item() == 0


def test_missing_dense_product_is_masked_without_nan_or_value_leakage() -> None:
    model = _model().eval()
    batch = _batch()
    batch["source_observation_masks"]["s2"].zero_()
    batch["source_pixel_masks"]["s2"].zero_()
    with torch.no_grad():
        baseline = model(**batch).embedding_map
        batch["source_frames"]["s2"].add_(1_000_000)
        changed = model(**batch).embedding_map

    assert not torch.isnan(changed).any()
    assert torch.allclose(baseline, changed, atol=1e-6, rtol=0)


def test_production_profile_has_about_100m_parameters() -> None:
    model = XuannvV2Model(
        _products(),
        embedding_dim=64,
        stem_dim=64,
        spatial_dim=768,
        temporal_dim=384,
        precision_dim=192,
        num_blocks=8,
        num_heads=12,
        temporal_mode="causal_window",
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 90_000_000 <= parameter_count <= 120_000_000


def test_shipped_production_config_builds_v2_model() -> None:
    root = Path(__file__).parents[1]
    config = V2Config.from_yaml(root / "configs/production/china_v2_highres_adapt.yaml")

    model = build_v2_model(config)

    assert isinstance(model, XuannvV2Model)
    assert model.temporal_mode == "causal_window"
    assert model.gradient_checkpointing is True


def test_v2_model_source_does_not_use_einops_or_l2_outside_vmf() -> None:
    root = Path(__file__).parents[1] / "src/xuannv_embedding/models"
    v2_sources = [root / "v2_model.py", root / "multires_adapters.py", root / "interval_fusion.py"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in v2_sources)

    assert "einops" not in text
    assert "F.normalize" not in text
