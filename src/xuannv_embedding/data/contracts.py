from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xuannv_embedding.config import InputSourceConfig, RegionDatasetConfig
from xuannv_embedding.utils.manifest import (
    ManifestRecord,
    SourceValue,
    load_legacy_manifest,
    load_manifest,
    manifest_meta_path,
)


class RegionContractError(ValueError):
    """区域记录与公共 source 合同不一致。"""


@dataclass(frozen=True)
class CanonicalRecord:
    patch_id: str
    region: str
    sources: dict[str, SourceValue]
    availability: dict[str, bool]
    statistics_dir: Path
    source_patch_id: str | None = None


def _is_available(value: SourceValue) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return bool(value)
    return True


def canonicalize_record(
    record: ManifestRecord,
    dataset: RegionDatasetConfig,
    input_sources: dict[str, InputSourceConfig],
) -> CanonicalRecord:
    """把区域物理 source 映射到模型规范槽位，并显式保留缺失状态。"""
    if record.region != dataset.region:
        raise RegionContractError(f"区域冲突: record={record.region!r}, dataset={dataset.region!r}")

    canonical_sources: dict[str, SourceValue] = {source: None for source in input_sources}
    availability = {source: False for source in input_sources}
    for physical_source, canonical_source in dataset.source_map.items():
        if canonical_source not in input_sources:
            # target-only source（例如 worldcover）由 target pipeline 处理。
            continue
        value = record.sources.get(physical_source)
        canonical_sources[canonical_source] = value
        availability[canonical_source] = _is_available(value)

    return CanonicalRecord(
        patch_id=record.patch_id,
        region=record.region,
        sources=canonical_sources,
        availability=availability,
        statistics_dir=dataset.statistics_dir,
        source_patch_id=record.source_patch_id,
    )


def load_region_records(
    dataset: RegionDatasetConfig,
    *,
    expected_months: list[str],
) -> list[ManifestRecord]:
    """读取 v1 或显式 legacy manifest，并只返回当前区域记录。"""
    if manifest_meta_path(dataset.manifest_path).is_file():
        document = load_manifest(dataset.manifest_path)
        if document.meta.months != expected_months:
            raise RegionContractError(
                f"月份冲突: manifest={document.meta.months}, config={expected_months}"
            )
        records = document.records
    else:
        records = load_legacy_manifest(dataset.manifest_path, region=dataset.region)

    return [record for record in records if record.region == dataset.region]
