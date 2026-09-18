from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import yaml
from rasterio.transform import from_origin

from xuannv_embedding.config import Config, ConfigError
from xuannv_embedding.data.raster_dataset import RegionRasterDataset, _month, collate_region_batch
from xuannv_embedding.data_process.observation_raster import sha256_file
from xuannv_embedding.data_process.p0 import sanitize_observation
from xuannv_embedding.data_process.pilot_cache import MONTHS, _smoke_config
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.masking import _drop_highres_source, apply_input_masking
from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest


def small_model() -> AEFModel:
    return AEFModel(
        sensor_channels={"s2": 2, "detail": 1},
        embed_dim=8,
        stem_dim=8,
        target_heads={"s2": ("continuous", 2)},
        source_roles={"s2": "temporal", "detail": "highres"},
        num_months=24,
        ref_year=2020,
        ref_month=1,
        stp={
            "space_dim": 8,
            "time_dim": 8,
            "precision_dim": 8,
            "num_blocks": 1,
            "num_heads": 2,
            "time_attention_mode": "none",
        },
    ).eval()


def model_inputs() -> dict:
    return {
        "source_frames": {"s2": torch.randn(1, 2, 2, 16, 16)},
        "source_masks": {"s2": torch.ones(1, 2)},
        "timestamps": torch.tensor([[202001, 202107]]),
        "output_months": torch.tensor([[202001, 202107]]),
        "highres_frames": {"detail": torch.randn(1, 3, 1, 32, 32)},
        "highres_masks": {"detail": torch.ones(1, 3, 1, 32, 32)},
        "highres_months": {"detail": torch.tensor([[202001, 202001, 202107]])},
    }


def test_observation_permutation_and_masked_fallback() -> None:
    torch.manual_seed(31)
    model = small_model()
    batch = model_inputs()
    with torch.no_grad():
        original = model(**batch).embedding_map
        permutation = torch.tensor([2, 0, 1])
        reordered = dict(batch)
        for name in ("highres_frames", "highres_masks", "highres_months"):
            reordered[name] = {"detail": batch[name]["detail"][:, permutation]}
        shuffled = model(**reordered).embedding_map
        assert torch.allclose(original, shuffled, atol=1e-6, rtol=1e-5)
        batch["highres_masks"]["detail"].zero_()
        masked = model(**batch).embedding_map
        baseline = dict(batch)
        baseline["highres_frames"] = {}
        baseline["highres_masks"] = {}
        baseline["highres_months"] = {}
        absent = model(**baseline).embedding_map
        assert torch.equal(masked, absent)
        assert masked.shape == (1, 2, 8, 16, 16)


def test_month_binding_native_encoding_and_gradients() -> None:
    model = small_model()
    batch = model_inputs()
    batch["highres_masks"]["detail"][:, 2] = 0
    seen = []
    hook = model.highres_encoders["detail"].conv1.register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[0].shape[-2:])
    )
    output = model(**batch)
    hook.remove()
    assert seen == [(32, 32)]
    baseline = dict(batch)
    baseline["highres_frames"] = {}
    expected = model(**baseline)
    assert torch.equal(output.embedding_map[:, 1], expected.embedding_map[:, 1])
    assert not torch.allclose(output.embedding_map[:, 0], expected.embedding_map[:, 0])
    output.reconstructions["s2"].square().mean().backward()
    gradient = model.highres_encoders["detail"].conv1.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_reject_unbound_observation() -> None:
    model = small_model()
    batch = model_inputs()
    batch["highres_months"]["detail"][0, 0] = 202002
    with pytest.raises(ValueError, match="selected output month"):
        model(**batch)


def test_modality_dropout_preserves_observation_batch_shape() -> None:
    frames = torch.ones(2, 3, 1, 16, 16)
    masks = torch.ones(2, 3, 1, 16, 16)
    dropped, availability, ratio = _drop_highres_source(frames, masks, 1.0)
    assert dropped.shape == frames.shape and availability.shape == masks.shape
    assert dropped.count_nonzero() == availability.count_nonzero() == 0
    assert float(ratio) == 1.0


def test_spatial_dropout_broadcasts_across_observations() -> None:
    batch = model_inputs()
    batch["source_frames"]["s2"] = batch["source_frames"]["s2"].repeat(2, 1, 1, 1, 1)
    batch["source_masks"]["s2"] = batch["source_masks"]["s2"].repeat(2, 1)
    batch["highres_frames"]["detail"] = batch["highres_frames"]["detail"].repeat(2, 1, 1, 1, 1)
    batch["highres_masks"]["detail"] = batch["highres_masks"]["detail"].repeat(2, 1, 1, 1, 1)
    masked = apply_input_masking(
        batch,
        {
            "enabled": True,
            "spatial_block_prob": 1.0,
            "spatial_block_size": 4,
            "spatial_block_ratio": 0.5,
        },
    )
    assert masked["highres_frames"]["detail"].shape == (2, 3, 1, 32, 32)
    assert masked["highres_masks"]["detail"].min() == 0
    assert torch.equal(
        masked["highres_masks"]["detail"][:, 0], masked["highres_masks"]["detail"][:, 1]
    )


def write_tiff(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[1],
        width=values.shape[2],
        count=values.shape[0],
        dtype=values.dtype,
        crs="EPSG:32650",
        transform=from_origin(512000, 3585280, 80, 80),
    ) as target:
        target.write(values)


def test_p0_masks_exclude_extremes_without_altering_source(tmp_path: Path) -> None:
    pilot, output = tmp_path / "pilot", tmp_path / "output"
    values = np.full((2, 16, 16), 17, dtype=np.uint16)
    values[:, 0, 0] = 0
    values[1, 1, 1] = 65535
    source = pilot / "s2.tif"
    write_tiff(source, values)
    original_hash = sha256_file(source)
    original = {"materialized_path": "s2.tif", "sha256": original_hash}
    result = sanitize_observation((pilot, output, original, "s2.tif"))
    assert sha256_file(source) == original_hash == sha256_file(output / "s2.tif")
    assert result["band_counts"] == [254, 254]
    assert result["band_mean"] == [17, 17]
    with rasterio.open(output / result["mask_path"]) as raster:
        mask = raster.read(1)
        assert mask.sum() == 254 and mask[0, 0] == mask[1, 1] == 0


def test_target_month_dataset_and_observation_collation(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, ["s2"], "train")
    config["model"]["input_sources"]["detail"] = {"channels": 1, "role": "highres"}
    config["data"].update(
        target_months=["2020-01", "2021-07"],
        highres_mode="observations",
        highres_max_observations=2,
        patch_size=16,
    )
    config["data"]["datasets"][0]["source_map"]["detail"] = "detail"
    paths = {
        "s2": ["s2/2020/01/frame.tif", "s2/2021/07/frame.tif"],
        "detail": ["detail/2021/07/frame.tif"],
    }
    for source, source_paths in paths.items():
        channels = 10 if source == "s2" else 1
        for relative in source_paths:
            write_tiff(tmp_path / relative, np.ones((channels, 16, 16), dtype=np.uint16))
        (tmp_path / "statistics").mkdir(exist_ok=True)
        (tmp_path / "statistics" / f"{source}_stats.json").write_text(
            json.dumps({"mean": [0] * channels, "std": [1] * channels})
        )
    write_manifest(
        tmp_path / "train.manifest.jsonl",
        [ManifestRecord(patch_id="sample", region="national_pilot", sources=paths)],
        months=MONTHS,
        generator_version="test",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    parsed = Config.from_yaml(config_path)
    dataset = RegionRasterDataset(parsed, parsed.data.datasets[0])
    batch = collate_region_batch([dataset[0]])
    assert batch["source_frames"]["s2"].shape == (1, 2, 10, 16, 16)
    assert batch["highres_frames"]["detail"].shape == (1, 2, 1, 16, 16)
    assert batch["highres_months"]["detail"].tolist() == [[202107, 0]]
    assert batch["output_months"].tolist() == [[202001, 202107]]
    config["data"]["target_months"] = ["2022-01"]
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ConfigError, match="target_months"):
        Config.from_yaml(config_path)


def test_month_prefers_date_directories_over_hashed_filename() -> None:
    assert _month("highres/GF6_PAN_c1/2020/10/parent/frame-64509491.tif") == 202010
    assert _month("highres/GF6_PAN_c1/2020/10/parent/frame-20210401.tif") == 202010
    assert _month("s2/frame-20210401.tif") == 202104
    assert _month("s2/frame-20219999.tif") == 0
    assert _month("s2/frame-without-date.tif") == 0
    with pytest.raises(ValueError, match="冲突的月份目录"):
        _month("highres/2020/10/2021/04/frame.tif")
