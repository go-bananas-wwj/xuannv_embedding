from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

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
class ProductSpec:
    """V2 product-level spectral, spatial, and temporal contract."""

    product_id: str
    role: Literal["dense", "highres", "target"]
    bands: tuple[str, ...]
    native_gsd_m: tuple[float, ...]
    stored_gsd_m: float
    dtype: str
    time_precision: Literal["exact", "day", "month", "static"]
    already_resampled: bool
    qa_available: bool

    def __post_init__(self) -> None:
        if not self.product_id:
            raise ValueError("product_id 不能为空")
        if not self.bands or len(self.bands) != len(self.native_gsd_m):
            raise ValueError("波段 bands 与原生 GSD native_gsd_m 必须一一对应")
        if len(set(self.bands)) != len(self.bands):
            raise ValueError("波段名称不得重复")
        if self.stored_gsd_m <= 0 or any(gsd <= 0 for gsd in self.native_gsd_m):
            raise ValueError("GSD 必须大于 0")


@dataclass(frozen=True)
class ObservationRef:
    """Reference to one immutable V2 observation, including honest time precision."""

    patch_id: str
    product_id: str
    interval_start: datetime
    interval_end: datetime
    acquired_at: datetime | None
    available_at: datetime
    archive_path: Path
    member_name: str
    present: bool
    quality_status: str
    time_precision: Literal["exact", "day", "month", "static"]

    def __post_init__(self) -> None:
        if self.interval_end <= self.interval_start:
            raise ValueError("观测 interval_end 必须晚于 interval_start")
        if self.available_at < self.interval_end:
            raise ValueError("available_at 不得早于 interval_end")
        if self.time_precision == "month" and self.acquired_at is not None:
            raise ValueError("month 时间精度不得虚构 acquired_at")
        if self.time_precision in {"exact", "day"} and self.acquired_at is None:
            raise ValueError(f"{self.time_precision} 时间精度必须提供 acquired_at")
        if self.present and (not self.member_name or not self.quality_status):
            raise ValueError("存在的观测必须包含 member_name 和 quality_status")


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
