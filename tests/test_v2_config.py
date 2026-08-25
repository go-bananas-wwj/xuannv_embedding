from __future__ import annotations

from pathlib import Path

import pytest

from xuannv_embedding.config import ConfigError, V2Config


def _valid_v2() -> str:
    return """
schema_version: "2"
paths:
  data_root: /data2/xuannv_embedding/v2
  source_root: /data2/china_xuannv_embedding/data
  grid_package: /data/xuannv_embedding/outputs/china_full_1280m_grid_package_v1_20260805
network_policy:
  allow_remote_metadata: false
  allow_remote_pixels: false
  missing_local_observation: mask
products:
  s2_local:
    role: dense
    bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
    native_gsd_m: [10, 10, 10, 20, 20, 20, 10, 20, 20, 20]
    stored_gsd_m: 10
    dtype: uint16
    time_precision: month
    already_resampled: true
    qa_available: false
  planet_3m:
    role: highres
    bands: [blue, green, red, nir]
    native_gsd_m: [3, 3, 3, 3]
    stored_gsd_m: 3
    dtype: uint16
    time_precision: exact
    already_resampled: false
    qa_available: true
temporal:
  mode: causal_window
  dense_lookback_days: 365
  highres_structure_days: 730
  highres_appearance_days: 90
  highres_structure_max_observations: 8
  highres_appearance_max_observations: 4
model:
  embedding_dim: 64
  stem_dim: 64
  spatial_dim: 768
  temporal_dim: 384
  precision_dim: 192
  num_blocks: 8
  num_heads: 12
  gradient_checkpointing: true
training:
  epochs: 1
  lr: 0.0001
  weight_decay: 0.05
  batch_size: 2
  gradient_accumulation_steps: 8
  amp: true
validation_profiles:
  mini-real:
    records: 16
    steps: 2
    batch_size: 2
  smoke:
    records: 652
    steps: 50
    batch_size: 1
"""


def _write(tmp_path: Path, value: str) -> Path:
    path = tmp_path / "v2.yaml"
    path.write_text(value, encoding="utf-8")
    return path


def test_loads_strict_offline_v2_contract(tmp_path: Path) -> None:
    config = V2Config.from_yaml(_write(tmp_path, _valid_v2()))

    assert config.schema_version == "2"
    assert config.paths.source_root == Path("/data2/china_xuannv_embedding/data")
    assert config.products["s2_local"].bands == (
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    )
    assert config.network_policy.allow_remote_pixels is False
    assert config.validation_profiles["smoke"].records == 652


def test_rejects_v1_for_v2_runtime(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="schema_version.*2"):
        V2Config.from_yaml(_write(tmp_path, _valid_v2().replace('"2"', '"1"', 1)))


def test_rejects_unknown_v2_field(tmp_path: Path) -> None:
    value = _valid_v2().replace(
        "  allow_remote_pixels: false", "  allow_remote_pixels: false\n  retry: true"
    )
    with pytest.raises(ConfigError, match="未知字段"):
        V2Config.from_yaml(_write(tmp_path, value))


def test_rejects_remote_pixels_in_offline_national_config(tmp_path: Path) -> None:
    value = _valid_v2().replace("allow_remote_pixels: false", "allow_remote_pixels: true")
    with pytest.raises(ConfigError, match="远程像元"):
        V2Config.from_yaml(_write(tmp_path, value))


def test_rejects_band_and_gsd_count_mismatch(tmp_path: Path) -> None:
    value = _valid_v2().replace(
        "native_gsd_m: [10, 10, 10, 20, 20, 20, 10, 20, 20, 20]",
        "native_gsd_m: [10]",
    )
    with pytest.raises(ConfigError, match="bands.*native_gsd_m"):
        V2Config.from_yaml(_write(tmp_path, value))


def test_rejects_invalid_temporal_window(tmp_path: Path) -> None:
    value = _valid_v2().replace("highres_appearance_days: 90", "highres_appearance_days: 800")
    with pytest.raises(ConfigError, match="appearance.*structure"):
        V2Config.from_yaml(_write(tmp_path, value))


def test_shipped_v2_configs_are_offline_and_use_real_s2_bands() -> None:
    root = Path(__file__).parents[1]
    paths = [
        root / "configs/production/china_v2_dense_2020_2021.yaml",
        root / "configs/production/china_v2_highres_adapt.yaml",
    ]

    configs = [V2Config.from_yaml(path) for path in paths]

    assert all(not config.network_policy.allow_remote_pixels for config in configs)
    assert all(config.products["s2_local"].bands[-2:] == ("B11", "B12") for config in configs)
    assert all("B09" not in config.products["s2_local"].bands for config in configs)
    assert configs[1].paths.product_roots["planetscope_3m_sr"] == Path(
        "/data/xuannv_embedding/raw/haidian/highres_optical/_unzipped"
    )
    assert "harbin_0_5m" in configs[1].paths.legacy_unverified_roots
