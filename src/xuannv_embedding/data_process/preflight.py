"""V2 preflight classification and local-only validation helpers."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import rasterio

from xuannv_embedding.config import V2Config
from xuannv_embedding.data.local_archives import audit_raster_member


def classify_source(
    role: str,
    time_precision: str,
    already_resampled: bool,
    provenance: str = "verified",
) -> str:
    if provenance == "legacy_unverified":
        return "legacy_unverified"
    if role == "highres" and time_precision in {"exact", "day"}:
        return "raw_scene"
    if time_precision == "month":
        suffix = "already_resampled" if already_resampled else "native_grid"
        return f"monthly_patch_{suffix}"
    return "static_product"


class PreflightError(ValueError):
    """Local V2 data are absent or violate their declared contract."""


def _parse_planet_metadata(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    properties = document.get("properties", {})
    return {
        "scene_id": str(document.get("id", path.stem.removesuffix("_metadata"))),
        "acquired_at": properties.get("acquired"),
        "available_at": properties.get("published"),
        "cloud_percent": properties.get("cloud_percent"),
        "clear_percent": properties.get("clear_percent"),
        "quality_category": properties.get("quality_category"),
        "reported_gsd_m": properties.get("gsd"),
    }


def index_local_highres_and_auxiliary(config: V2Config) -> dict[str, int]:
    root = config.paths.data_root
    highres_count = 0
    for product_id, product_root in config.paths.product_roots.items():
        if product_id not in config.products:
            raise PreflightError(f"product_roots 引用了未知产品: {product_id}")
        product = config.products[product_id]
        rows: list[dict[str, Any]] = []
        for image_path in sorted(product_root.rglob("*_AnalyticMS_SR_clip.tif")):
            metadata_path = image_path.with_name(
                image_path.name.replace("_3B_AnalyticMS_SR_clip.tif", "_metadata.json")
            )
            if not metadata_path.is_file():
                raise PreflightError(f"PlanetScope 影像缺少 metadata: {image_path}")
            metadata = _parse_planet_metadata(metadata_path)
            if not metadata["acquired_at"] or not metadata["available_at"]:
                raise PreflightError(f"高分场景缺少 acquired/published 时间: {metadata_path}")
            udm2_path = image_path.with_name(
                image_path.name.replace("_3B_AnalyticMS_SR_clip.tif", "_3B_udm2_clip.tif")
            )
            with rasterio.open(image_path) as dataset:
                bounds = dataset.bounds
                row = {
                    "product_id": product_id,
                    **metadata,
                    "image_path": str(image_path.resolve()),
                    "qa_path": str(udm2_path.resolve()) if udm2_path.is_file() else None,
                    "qa_present": udm2_path.is_file(),
                    "channels": dataset.count,
                    "height": dataset.height,
                    "width": dataset.width,
                    "dtype": dataset.dtypes[0],
                    "crs": dataset.crs.to_string() if dataset.crs else None,
                    "transform": list(dataset.transform)[:6],
                    "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                    "processing_state": classify_source(
                        product.role,
                        product.time_precision,
                        product.already_resampled,
                        "verified",
                    ),
                    "training_eligible": True,
                }
            if row["channels"] != len(product.bands) or row["dtype"] != product.dtype:
                raise PreflightError(f"高分场景通道或 dtype 不符合合同: {image_path}")
            rows.append(row)
        if not rows:
            raise PreflightError(f"高分 product_root 没有可索引场景: {product_root}")
        output = root / "observations" / "highres" / product_id / "scenes.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd")
        highres_count += len(rows)

    auxiliary_rows: list[dict[str, Any]] = []
    for product_id, source_root in config.paths.auxiliary_roots.items():
        if not source_root.exists():
            raise PreflightError(f"辅助数据目录不存在: {source_root}")
        for path in sorted(source_root.rglob("*.zip")):
            auxiliary_rows.append(
                {
                    "product_id": product_id,
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "processing_state": "local_archive_uninspected",
                    "training_eligible": True,
                }
            )
    if auxiliary_rows:
        output = root / "registry" / "auxiliary_inventory.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(auxiliary_rows), output, compression="zstd")

    legacy_rows: list[dict[str, Any]] = []
    for product_id, source_root in config.paths.legacy_unverified_roots.items():
        if not source_root.exists():
            raise PreflightError(f"legacy 数据目录不存在: {source_root}")
        for path in sorted(source_root.rglob("*.tif")):
            legacy_rows.append(
                {
                    "product_id": product_id,
                    "path": str(path.resolve()),
                    "processing_state": "legacy_unverified",
                    "training_eligible": False,
                }
            )
    if legacy_rows:
        output = root / "registry" / "legacy_unverified_inventory.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(legacy_rows), output, compression="zstd")
    return {
        "highres_scene_count": highres_count,
        "auxiliary_archive_count": len(auxiliary_rows),
        "legacy_file_count": len(legacy_rows),
    }


def _expected_shape(product_id: str) -> tuple[int, int]:
    if product_id in {"s2_local", "s1_local"}:
        return (128, 128)
    if product_id == "landsat_local":
        return (43, 43)
    raise PreflightError(f"未知 dense product shape: {product_id}")


def run_preflight(config: V2Config, *, max_pixel_audits: int = 96) -> dict[str, Any]:
    if config.network_policy.allow_remote_pixels or config.network_policy.allow_remote_metadata:
        raise PreflightError("preflight 要求 metadata 和 pixels 均为本地离线模式")
    root = config.paths.data_root
    archive_path = root / "registry" / "local_archive_inventory.parquet"
    member_path = root / "observations" / "index" / "local_zip_members.parquet"
    availability_path = root / "observations" / "index" / "availability.parquet"
    for path in (archive_path, member_path, availability_path):
        if not path.is_file():
            raise PreflightError(f"缺少 local-index 产物: {path}")
    archives = pq.read_table(archive_path)
    if archives.num_rows != 72:
        raise PreflightError(f"archive 清单应为 72，实际 {archives.num_rows}")
    members = pq.read_table(
        member_path,
        columns=["product_id", "year", "month", "archive_path", "member_name"],
    )
    sample_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for row in members.to_pylist():
        key = (str(row["product_id"]), int(row["year"]), int(row["month"]))
        if key in seen:
            continue
        seen.add(key)
        sample_rows.append(row)
        if len(sample_rows) >= max_pixel_audits:
            break
    audits = []
    for row in sample_rows:
        product_id = str(row["product_id"])
        product = config.products[product_id].to_product_spec(product_id)
        audit = audit_raster_member(
            Path(row["archive_path"]),
            str(row["member_name"]),
            product,
            expected_shape=_expected_shape(product_id),
        )
        audits.append(
            {
                "product_id": product_id,
                "year": int(row["year"]),
                "month": int(row["month"]),
                "member_name": row["member_name"],
                "passed": audit.passed,
                "processing_state": audit.processing_state,
                "failures": list(audit.failures),
                "channels": audit.channels,
                "height": audit.height,
                "width": audit.width,
                "dtype": audit.dtype,
                "crs": audit.crs,
                "finite_fraction": audit.finite_fraction,
                "zero_fraction": audit.zero_fraction,
            }
        )
    availability = pq.read_metadata(availability_path).num_rows
    data_mini = run_data_mini(config)
    passed = len(audits) == 72 and all(row["passed"] for row in audits) and data_mini["passed"]
    report = {
        "schema_version": "xuannv_v2_preflight_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "network_requests": 0,
        "archive_count": archives.num_rows,
        "availability_count": availability,
        "pixel_audit_count": len(audits),
        "pixel_audits": audits,
        "data_mini": data_mini,
        "passed": passed,
    }
    report_root = root / "reports" / "preflight"
    report_root.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".preflight.", dir=report_root)
    temporary = Path(name)
    output = report_root / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {**report, "report_path": str(output)}


def run_data_mini(config: V2Config) -> dict[str, Any]:
    """Read the 16-patch January 2020/2021 mini set directly from local ZIPs."""
    root = config.paths.data_root
    mini_path = root / "registry" / "mini_16.parquet"
    availability_path = root / "observations" / "index" / "availability.parquet"
    if not mini_path.is_file() or not availability_path.is_file():
        raise PreflightError("data mini 需要先运行 local-index")
    mini = pq.read_table(mini_path)
    if mini.num_rows != 16:
        raise PreflightError(f"data mini registry 必须为 16，实际 {mini.num_rows}")
    mini_rows = {str(row["patch_id"]): row for row in mini.to_pylist()}
    observations = pq.read_table(
        availability_path,
        filters=[
            ("patch_id", "in", list(mini_rows)),
            ("month", "=", 1),
            ("year", "in", [2020, 2021]),
        ],
    )
    expected_count = 16 * 3 * 2
    if observations.num_rows != expected_count:
        raise PreflightError(
            f"data mini 应有 {expected_count} 条产品观测，实际 {observations.num_rows}"
        )
    audit_rows: list[dict[str, Any]] = []
    missing_count = 0
    for row in observations.to_pylist():
        if not row["present"]:
            missing_count += 1
            continue
        product_id = str(row["product_id"])
        grid = mini_rows[str(row["patch_id"])]
        product = config.products[product_id].to_product_spec(product_id)
        audit = audit_raster_member(
            Path(row["archive_path"]),
            str(row["member_name"]),
            product,
            expected_shape=_expected_shape(product_id),
            expected_crs=f"EPSG:{int(grid['grid_epsg'])}",
            expected_bounds=tuple(float(value) for value in grid["utm_bounds"]),
        )
        audit_rows.append(
            {
                "patch_id": row["patch_id"],
                "product_id": product_id,
                "year": row["year"],
                "passed": audit.passed,
                "failures": list(audit.failures),
            }
        )
    failed = [row for row in audit_rows if not row["passed"]]
    return {
        "schema_version": "xuannv_v2_data_mini_v1",
        "patch_count": mini.num_rows,
        "observation_count": observations.num_rows,
        "present_count": len(audit_rows),
        "missing_count": missing_count,
        "corrupt_or_misaligned_count": len(failed),
        "network_requests": 0,
        "passed": not failed and missing_count > 0,
        "failures": failed,
    }
