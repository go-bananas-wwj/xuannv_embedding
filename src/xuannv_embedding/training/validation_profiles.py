"""V2 validation-profile builders and provenance hashes."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from xuannv_embedding.config import V2Config, ValidationProfileConfig
from xuannv_embedding.models.v2_model import XuannvV2Model
from xuannv_embedding.training.losses import V2TotalLoss


def data_manifest_sha256(root: Path) -> str:
    paths = [
        root / "locks" / "local_archive_sha256.jsonl",
        root / "registry" / "local_archive_inventory.parquet",
        root / "registry" / "split_80_10_10.parquet",
        root / "observations" / "index" / "availability.parquet",
    ]
    paths.extend(sorted((root / "statistics").glob("*.json")))
    paths.extend(sorted((root / "observations" / "highres").glob("*/scenes.parquet")))
    paths.extend(sorted((root / "observations" / "highres").glob("*/patch_observations.parquet")))
    paths.extend(sorted((root / "labels").glob("*/patch_observations.parquet")))
    if not any(path.parent.name == "statistics" for path in paths):
        raise FileNotFoundError(f"数据 manifest 缺少训练波段统计量: {root / 'statistics'}")
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"数据 manifest 组成文件不存在: {path}")
        digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def build_profile_model(config: V2Config, profile: ValidationProfileConfig) -> XuannvV2Model:
    if profile.model_profile == "production":
        return XuannvV2Model.from_config(config)
    products = {
        product_id: product.to_product_spec(product_id)
        for product_id, product in config.products.items()
        if product.role in {"dense", "highres"}
    }
    return XuannvV2Model(
        products,
        embedding_dim=config.model.embedding_dim,
        stem_dim=16,
        spatial_dim=64,
        temporal_dim=32,
        precision_dim=32,
        num_blocks=1,
        num_heads=4,
        temporal_mode=config.temporal.mode,
        dense_lookback_days=config.temporal.dense_lookback_days,
        highres_structure_days=config.temporal.highres_structure_days,
        highres_appearance_days=config.temporal.highres_appearance_days,
        highres_structure_max_observations=(config.temporal.highres_structure_max_observations),
        highres_appearance_max_observations=(config.temporal.highres_appearance_max_observations),
        gradient_checkpointing=config.model.gradient_checkpointing,
    )


def build_v2_criterion(config: V2Config) -> V2TotalLoss:
    training = config.training
    return V2TotalLoss(
        embed_dim=config.model.embedding_dim,
        reconstruction_weights=training.reconstruction_weights,
        uniformity_weight=training.uniformity_weight,
        uniformity_warmup_epochs=training.uniformity_warmup_epochs,
        uniformity_temperature=training.uniformity_temperature,
        semantic_probe_weight=training.semantic_probe_weight,
        semantic_probe_warmup_epochs=training.semantic_probe_warmup_epochs,
        semantic_probe_tasks=training.semantic_probe_tasks,
        semantic_probe_hidden_dim=training.semantic_probe_hidden_dim,
        highres_detail_weight=training.highres_detail_weight,
    )


def v2_product_schema(config: V2Config) -> dict[str, object]:
    return {
        product_id: asdict(product.to_product_spec(product_id))
        for product_id, product in config.products.items()
    }


def v2_temporal_contract(config: V2Config) -> dict[str, object]:
    return asdict(config.temporal)


def assert_macro_disjoint(registry_path: Path) -> dict[str, int]:
    table = pq.read_table(registry_path, columns=["macro_id", "split"])
    counts: dict[str, int] = {}
    seen: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        selected = table.filter(pc.equal(table["split"], split))
        values = {str(value) for value in selected["macro_id"].to_pylist()}
        seen[split] = values
        counts[split] = selected.num_rows
    overlaps = {
        f"{left}/{right}": sorted(seen[left] & seen[right])
        for index, left in enumerate(seen)
        for right in list(seen)[index + 1 :]
        if seen[left] & seen[right]
    }
    if overlaps:
        raise ValueError(f"smoke registry 存在跨 split macro_id: {overlaps}")
    return counts
