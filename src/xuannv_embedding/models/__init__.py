"""P10C 生产模型组件。"""

from __future__ import annotations

from dataclasses import asdict

from xuannv_embedding.config import ModelConfig
from xuannv_embedding.models.model import AEFModel


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


__all__ = ["AEFModel", "build_model"]
