"""P10C 生产模型组件。"""

from __future__ import annotations

from dataclasses import asdict

from xuannv_embedding.config import ModelConfig, V2Config
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.models.v2_model import XuannvV2Model


def build_model(
    config: ModelConfig,
    *,
    gradient_checkpointing: bool = False,
) -> AEFModel:
    """从严格公共配置构造模型，不读取 region 或物理 source 名称。"""
    return AEFModel(
        sensor_channels=config.sensor_channels,
        embed_dim=config.embed_dim,
        target_heads=config.decoder_specs,
        stem_dim=config.stem_dim,
        stp=asdict(config.stp),
        num_months=config.num_months,
        ref_year=config.ref_year,
        ref_month=config.ref_month,
        gradient_checkpointing=gradient_checkpointing,
        source_roles=config.source_roles,
    )


def build_v2_model(config: V2Config) -> XuannvV2Model:
    """Build the explicitly incompatible V2 interval model."""
    return XuannvV2Model.from_config(config)


__all__ = ["AEFModel", "XuannvV2Model", "build_model", "build_v2_model"]
