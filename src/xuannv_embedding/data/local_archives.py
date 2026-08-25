"""Read immutable local monthly ZIP archives without network access."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
from rasterio.io import MemoryFile

from xuannv_embedding.data.contracts import ObservationRef, ProductSpec


class ArchiveContractError(ValueError):
    """A local archive violates the frozen V2 product or grid contract."""


@dataclass(frozen=True)
class LocalArchive:
    product_id: str
    year: int
    month: int
    path: Path

    def __post_init__(self) -> None:
        raw_path = str(self.path)
        if raw_path.startswith(("http:/", "https:/", "s3:/", "gs:/", "az:/")):
            raise ArchiveContractError("archive_path 必须是本地路径")
        if not self.product_id or self.year < 1970 or not 1 <= self.month <= 12:
            raise ArchiveContractError("本地 archive 描述非法")


@dataclass(frozen=True)
class ArchiveIndex:
    archive: LocalArchive
    size_bytes: int
    member_count: int
    member_list_sha256: str
    observations: tuple[ObservationRef, ...]
    missing_patch_ids: tuple[str, ...]
    extra_patch_ids: tuple[str, ...]


@dataclass(frozen=True)
class RasterAudit:
    archive_path: Path
    member_name: str
    processing_state: str
    channels: int
    height: int
    width: int
    dtype: str
    crs: str | None
    transform: tuple[float, ...]
    bounds: tuple[float, float, float, float]
    finite_fraction: float
    zero_fraction: float
    constant: bool
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    if not 1 <= month <= 12:
        raise ArchiveContractError(f"非法月份: {month}")
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = (
        datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        if month == 12
        else datetime(year, month + 1, 1, tzinfo=timezone.utc)
    )
    return start, end


def patch_id_from_member(member_name: str) -> str:
    path = Path(member_name)
    if path.suffix.lower() not in {".tif", ".tiff"} or not path.stem:
        raise ArchiveContractError(f"ZIP 成员不是 patch GeoTIFF: {member_name}")
    return path.stem


def _member_list_sha256(infos: list) -> str:
    digest = hashlib.sha256()
    for info in sorted(infos, key=lambda value: value.filename):
        digest.update(f"{info.filename}\0{info.file_size}\0{info.CRC:08x}\n".encode("utf-8"))
    return digest.hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive_sidecar(archive: LocalArchive, product: ProductSpec) -> str:
    """Lock the only local evidence for band order; never infer absent QA or dates."""
    sidecar = archive.path.with_suffix(".zip.txt")
    if not sidecar.is_file():
        raise ArchiveContractError(f"本地 ZIP 缺少波段合同 sidecar: {sidecar}")
    payload = sidecar.read_bytes()
    text = payload.decode("utf-8")
    match = re.search(r"^波段/通道:\s*(.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ArchiveContractError(f"sidecar 缺少波段/通道: {sidecar}")
    declaration = match.group(1).split("(", maxsplit=1)[0]
    bands = tuple(value.strip() for value in declaration.split(",") if value.strip())
    if bands != product.bands:
        raise ArchiveContractError(
            f"sidecar 波段顺序不符合合同: declared={bands}, expected={product.bands}"
        )
    return hashlib.sha256(payload).hexdigest()


def index_local_archive(
    archive: LocalArchive,
    product: ProductSpec,
    expected_patch_ids: set[str],
) -> ArchiveIndex:
    if product.product_id != archive.product_id:
        raise ArchiveContractError("archive 与 ProductSpec 的 product_id 不一致")
    if not archive.path.is_file():
        raise ArchiveContractError(f"本地 ZIP 不存在: {archive.path}")
    start, end = month_bounds(archive.year, archive.month)
    try:
        with ZipFile(archive.path) as handle:
            infos = [info for info in handle.infolist() if not info.is_dir()]
    except BadZipFile as exc:
        raise ArchiveContractError(f"损坏 ZIP: {archive.path}") from exc
    by_patch: dict[str, str] = {}
    for info in infos:
        patch_id = patch_id_from_member(info.filename)
        if patch_id in by_patch:
            raise ArchiveContractError(f"ZIP 中 patch ID 重复: {patch_id}")
        by_patch[patch_id] = info.filename
    actual = set(by_patch)
    extra = tuple(sorted(actual - expected_patch_ids))
    if extra:
        preview = ", ".join(extra[:5])
        raise ArchiveContractError(f"ZIP 包含正式网格之外的多余 patch: {preview}")
    missing = tuple(sorted(expected_patch_ids - actual))
    observations = tuple(
        ObservationRef(
            patch_id=patch_id,
            product_id=product.product_id,
            interval_start=start,
            interval_end=end,
            acquired_at=None,
            available_at=end,
            archive_path=archive.path,
            member_name=by_patch.get(patch_id, ""),
            present=patch_id in actual,
            quality_status="unchecked" if patch_id in actual else "missing_local_observation",
            time_precision="month",
        )
        for patch_id in sorted(expected_patch_ids)
    )
    return ArchiveIndex(
        archive=archive,
        size_bytes=archive.path.stat().st_size,
        member_count=len(infos),
        member_list_sha256=_member_list_sha256(infos),
        observations=observations,
        missing_patch_ids=missing,
        extra_patch_ids=extra,
    )


def _processing_state(product: ProductSpec) -> str:
    if product.role == "highres" and product.time_precision in {"exact", "day"}:
        return "raw_scene"
    if product.time_precision == "month":
        suffix = "already_resampled" if product.already_resampled else "native_grid"
        return f"monthly_patch_{suffix}"
    return "static_product"


def audit_raster_member(
    archive_path: Path,
    member_name: str,
    product: ProductSpec,
    *,
    expected_shape: tuple[int, int],
    expected_crs: str | None = None,
    expected_bounds: tuple[float, float, float, float] | None = None,
) -> RasterAudit:
    failures: list[str] = []
    try:
        with ZipFile(archive_path) as archive:
            payload = archive.read(member_name)
        with MemoryFile(payload) as memory:
            with memory.open() as dataset:
                values = dataset.read()
                channels, height, width = values.shape
                dtype = dataset.dtypes[0]
                crs = dataset.crs.to_string() if dataset.crs is not None else None
                transform = tuple(float(value) for value in dataset.transform)[:6]
                bounds = tuple(float(value) for value in dataset.bounds)
    except (BadZipFile, KeyError, OSError, ValueError) as exc:
        raise ArchiveContractError(f"无法读取 {archive_path}!{member_name}: {exc}") from exc
    if channels != len(product.bands):
        failures.append("channels")
    if (height, width) != expected_shape:
        failures.append("shape")
    if dtype != product.dtype:
        failures.append("dtype")
    if crs is None:
        failures.append("crs")
    elif expected_crs is not None and crs != expected_crs:
        failures.append("crs_alignment")
    if not np.isfinite(np.asarray(transform)).all() or transform[0] <= 0 or transform[4] >= 0:
        failures.append("transform")
    if expected_bounds is not None:
        tolerance = max(abs(transform[0]), abs(transform[4])) + 1e-6
        if any(
            abs(actual - expected) > tolerance for actual, expected in zip(bounds, expected_bounds)
        ):
            failures.append("footprint_alignment")
    finite = np.isfinite(values)
    finite_fraction = float(finite.mean())
    zero_fraction = float(np.equal(values, 0).mean())
    constant = bool(values.size and np.nanmin(values) == np.nanmax(values))
    if finite_fraction < 1.0:
        failures.append("non_finite")
    if zero_fraction == 1.0:
        failures.append("all_zero")
    if constant:
        failures.append("all_constant")
    return RasterAudit(
        archive_path=archive_path,
        member_name=member_name,
        processing_state=_processing_state(product),
        channels=channels,
        height=height,
        width=width,
        dtype=dtype,
        crs=crs,
        transform=transform,
        bounds=bounds,
        finite_fraction=finite_fraction,
        zero_fraction=zero_fraction,
        constant=constant,
        failures=tuple(failures),
    )
