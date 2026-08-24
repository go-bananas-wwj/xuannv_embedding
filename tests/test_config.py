from __future__ import annotations

from pathlib import Path

import pytest

from xuannv_embedding.config import Config, ConfigError
from xuannv_embedding.models import build_model


def _valid_config() -> str:
    return """
schema_version: "1"
paths:
  data_root: /data/xuannv_embedding
  output_root: /data/xuannv_embedding/outputs
  artifact_root: /data/xuannv_embedding/artifacts
experiment:
  name: contract_test
  seed: 42
  output_dir: /data/xuannv_embedding/outputs/contract_test
model:
  embed_dim: 64
  stem_dim: 32
  num_months: 2
  ref_year: 2025
  ref_month: 12
  input_sources:
    s2: {channels: 12, role: temporal}
    highres_optical: {channels: 3, role: highres}
    highres_sar: {channels: 1, role: highres}
  target_heads:
    s2_recon: {source: s2, loss_type: continuous, channels: 12, weight: 0.8}
    osm_building: {source: osm_building, loss_type: categorical, channels: 2, weight: 0.2}
  stp:
    space_dim: 512
    time_dim: 256
    precision_dim: 128
    precision_scale: 1
    num_blocks: 6
    num_heads: 8
    temporal_fusion: gated_sum
    time_attention_mode: full
    highres_fusion_to_embedding: true
training:
  epochs: 800
  lr: 2.0e-6
  weight_decay: 0.05
  warmup_epochs: 30
  gradient_accumulation_steps: 2
  save_every: 200
  amp: true
  gradient_checkpointing: true
  uniformity_weight: 0.06
  uniformity_warmup_epochs: 60
  uniformity_temperature: 2.0
  semantic_probe_weight: 0.14
  semantic_probe_warmup_epochs: 80
  semantic_probe_tasks: [osm_building]
  semantic_probe_task_weights: {osm_building: 1.3}
  semantic_probe_pos_weight: 2.0
  semantic_probe_pos_weights: {osm_building: 3.0}
  semantic_probe_hidden_dim: 0
  semantic_probe_hard_negative_ratio: 0.02
  semantic_probe_hard_negative_weight: 0.35
  semantic_probe_hard_negative_warmup_epochs: 120
  input_masking:
    enabled: true
    drop_availability_masks: false
    modality_dropout_probs: {s2: 0.18}
    month_dropout_prob: 0.65
    max_months_per_sample: 2
    spatial_block_prob: 0.65
    spatial_block_size: 16
    spatial_block_ratio: 0.32
data:
  months: [2025-12, 2026-01]
  batch_size: 3
  num_workers: 8
  patch_size: 128
  datasets:
    - region: haidian
      manifest_path: processed/haidian/manifest.jsonl
      statistics_dir: statistics/haidian
      patch_grid_path: grids/haidian.parquet
      source_map:
        s2_haidian: s2
        optical_haidian: highres_optical
        sar_haidian: highres_sar
      supervised_label_roots:
        osm_building: processed/haidian/labels/osm_building
      sampling_weight: 1.0
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_strict_region_agnostic_config(tmp_path: Path) -> None:
    cfg = Config.from_yaml(_write(tmp_path, _valid_config()))

    assert cfg.schema_version == "1"
    assert cfg.model.input_sources["highres_optical"].role == "highres"
    assert cfg.data.datasets[0].source_map["optical_haidian"] == "highres_optical"
    assert cfg.data.dataset_for_region("haidian").statistics_dir == Path("statistics/haidian")
    assert cfg.to_dict()["model"]["input_sources"]["s2"]["channels"] == 12

    model = build_model(cfg.model, gradient_checkpointing=False)
    assert set(model.temporal_stem_bank.encoders) == {"s2"}
    assert set(model.highres_encoders) == {"highres_optical", "highres_sar"}


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        ("  seed: 42", "  seed: 42\n  surprise: true", "未知字段"),
        ("  seed: 42", "  seed: 42\n  use_wandb: true", "未知字段"),
        (
            "    s2: {channels: 12, role: temporal}",
            "    s2: {channels: 12, role: temporal, city: haidian}",
            "未知字段",
        ),
        ('schema_version: "1"', 'schema_version: "1"\n_base_: base.yaml', "_base_"),
    ],
)
def test_rejects_unknown_fields_and_base(
    tmp_path: Path, needle: str, replacement: str, message: str
) -> None:
    path = _write(tmp_path, _valid_config().replace(needle, replacement))

    with pytest.raises(ConfigError, match=message):
        Config.from_yaml(path)


def test_rejects_duplicate_yaml_mapping_key(tmp_path: Path) -> None:
    text = _valid_config().replace(
        "  seed: 42",
        "  seed: 42\n  seed: 7",
    )

    with pytest.raises(ConfigError, match="重复字段"):
        Config.from_yaml(_write(tmp_path, text))


def test_rejects_duplicate_source_mapping(tmp_path: Path) -> None:
    text = _valid_config().replace(
        "        sar_haidian: highres_sar",
        "        sar_haidian: highres_sar\n        sar_copy: highres_sar",
    )

    with pytest.raises(ConfigError, match="重复映射"):
        Config.from_yaml(_write(tmp_path, text))


def test_rejects_target_channel_conflict(tmp_path: Path) -> None:
    text = _valid_config().replace(
        "s2_recon: {source: s2, loss_type: continuous, channels: 12, weight: 0.8}",
        "s2_recon: {source: s2, loss_type: continuous, channels: 11, weight: 0.8}",
    )

    with pytest.raises(ConfigError, match="通道冲突"):
        Config.from_yaml(_write(tmp_path, text))


def test_rejects_month_count_and_reference_conflicts(tmp_path: Path) -> None:
    count_conflict = _valid_config().replace("  num_months: 2", "  num_months: 3")
    with pytest.raises(ConfigError, match="月份冲突"):
        Config.from_yaml(_write(tmp_path, count_conflict))

    ref_conflict = _valid_config().replace("  ref_month: 12", "  ref_month: 11")
    with pytest.raises(ConfigError, match="月份冲突"):
        Config.from_yaml(_write(tmp_path, ref_conflict))


def test_rejects_unknown_or_missing_source_slot(tmp_path: Path) -> None:
    text = _valid_config().replace("optical_haidian: highres_optical", "optical_haidian: x")

    with pytest.raises(ConfigError, match="未知规范 source"):
        Config.from_yaml(_write(tmp_path, text))


def test_mixed_regions_select_statistics_by_region(tmp_path: Path) -> None:
    second = """
    - region: harbin
      manifest_path: processed/harbin/manifest.jsonl
      statistics_dir: statistics/harbin
      patch_grid_path: grids/harbin.parquet
      source_map:
        s2_harbin: s2
        optical_harbin: highres_optical
      supervised_label_roots:
        osm_building: processed/harbin/labels/osm_building
      sampling_weight: 1.5
"""
    cfg = Config.from_yaml(_write(tmp_path, _valid_config() + second))

    assert cfg.data.dataset_for_region("haidian").statistics_dir == Path("statistics/haidian")
    assert cfg.data.dataset_for_region("harbin").statistics_dir == Path("statistics/harbin")
    with pytest.raises(ConfigError, match="未配置区域"):
        cfg.data.dataset_for_region("beijing")


def test_shipped_configs_are_self_contained_and_share_canonical_sources() -> None:
    root = Path(__file__).parents[1]
    paths = [
        root / "configs/production/haidian_p10c_v1.yaml",
        root / "configs/production/harbin_p10c.yaml",
        root / "configs/production/mixed_haidian_harbin_p10c.yaml",
        root / "configs/examples/china_p10c_pilot.yaml",
    ]

    configs = [Config.from_yaml(path) for path in paths]

    expected_sources = {"s2", "s1", "landsat", "highres_optical", "highres_sar"}
    assert all(set(config.model.input_sources) == expected_sources for config in configs)
    assert all("_base_" not in path.read_text(encoding="utf-8") for path in paths)
    assert [dataset.region for dataset in configs[2].data.datasets] == ["haidian", "harbin"]
    assert all(
        not task.startswith(("haidian_", "harbin_"))
        for config in configs
        for task in config.training.semantic_probe_tasks
    )
