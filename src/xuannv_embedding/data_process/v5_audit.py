"""Source evidence audits and fail-closed data acceptance snapshots (no training)."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from rasterio.io import MemoryFile

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

REQUIRED_GATES = (
    "download",
    "catalog",
    "radiometry",
    "quality",
    "alignment",
    "targets",
    "statistics",
    "sample_index",
    "loader",
    "reproducibility",
    "visual_review",
)


def acceptance_status(checks: dict) -> dict:
    pending = [key for key in REQUIRED_GATES if checks.get(key) is not True]
    return {
        "status": "incomplete" if pending else "ready_for_review",
        "user_accepted": False,
        "training_authorized": False,
        "pending_gates": pending,
        "checks": checks,
        "generated_at": now(),
    }


def audit_radiometry(dense_root: Path, report_root: Path) -> dict:
    """Inspect all monthly archives, without inferring undocumented physical conversions."""
    records, inventories = [], []
    for product, directory, count in [
        ("s2_local", "pc-s2", 10),
        ("s1_local", "pc-s1", 2),
        ("landsat_local", "pc-ls", 6),
    ]:
        for year in (2020, 2021):
            for month in range(1, 13):
                path = (
                    dense_root
                    / directory
                    / str(year)
                    / f"{month:02d}"
                    / f"{directory}_{year}_{month:02d}.zip"
                )
                if not path.is_file():
                    inventories.append(
                        {
                            "product_id": product,
                            "year": year,
                            "month": month,
                            "status": "missing",
                            "path": str(path),
                        }
                    )
                    continue
                sidecar = path.with_name(path.name + ".txt")
                text = sidecar.read_text() if sidecar.exists() else ""
                with ZipFile(path) as archive:
                    names = sorted(n for n in archive.namelist() if n.lower().endswith(".tif"))
                    for index in sorted({0, len(names) // 2, len(names) - 1}):
                        if not names:
                            break
                        name = names[index]
                        with MemoryFile(archive.read(name)) as memory:
                            with memory.open() as source:
                                # Read all bands, including masks, to expose decoder failures.
                                source.read()
                                source.read_masks()
                                conflict = (
                                    product == "landsat_local"
                                    and "red, green, blue" in text
                                    and "B2/B3/B4" in text
                                )
                                record = {
                                    "product_id": product,
                                    "year": year,
                                    "month": month,
                                    "archive_path": str(path),
                                    "member_name": name,
                                    "count": source.count,
                                    "expected_count": count,
                                    "descriptions": list(source.descriptions),
                                    "scales": list(source.scales),
                                    "offsets": list(source.offsets),
                                    "nodata": source.nodata,
                                    "crs": str(source.crs),
                                    "tags_json": json.dumps(source.tags(), ensure_ascii=False),
                                    "sidecar_sha256": sha256(sidecar) if sidecar.exists() else None,
                                    "status": "unverified",
                                    "reason": (
                                        "conflicting Landsat RGB versus B2/B3/B4 mapping"
                                        if conflict
                                        else "export band/scaling/fill provenance missing"
                                    ),
                                }
                                records.append(record)
                    inventories.append(
                        {
                            "product_id": product,
                            "year": year,
                            "month": month,
                            "status": "metadata_sampled",
                            "path": str(path),
                            "tiff_count": len(names),
                            "bytes": path.stat().st_size,
                        }
                    )
    frame = pd.DataFrame(records)
    atomic_parquet(frame, report_root / "radiometry_audit.parquet")
    atomic_parquet(
        frame.loc[frame.status != "verified"], report_root / "band_contract_failures.parquet"
    )
    atomic_parquet(pd.DataFrame(inventories), report_root / "dense_archive_inventory.parquet")
    result = {
        "status": "blocked_on_source_provenance",
        "sampled_rasters": len(frame),
        "archives": len(inventories),
        "sampling": "first_middle_last_per_archive",
        "full_pixel_integrity": "not_performed_by_this_stage",
        "products": sorted(frame.product_id.unique()) if not frame.empty else [],
        "finished_at": now(),
    }
    write_json(report_root / "radiometry_summary.json", result)
    return result


def audit_targets(base_root: Path, dataset_root: Path, report_root: Path) -> dict:
    import zarr

    registry = pd.read_parquet(dataset_root / "registry/national_62000.parquet")
    paths = [
        ("static", "targets/static_10m/full62000_v1.zarr"),
        ("osm", "targets/osm30_10m/full62000_base_v1.zarr"),
        ("reliable_negative", "targets/osm30_10m/reliable_negative_full62000_v1.zarr"),
    ]
    rows = []
    for family, relative in paths:
        path = base_root / relative
        if not path.is_dir():
            rows.append({"family": family, "path": str(path), "status": "missing"})
            continue
        root = zarr.open_group(str(path), mode="r")
        ordered_ids = list(root.attrs.get("patch_ids", []))
        order_ok = ordered_ids == registry.patch_id.tolist()

        def visit(name, obj):
            if not isinstance(obj, zarr.Array):
                return
            # Shape/order audit is distinct from a full label value/provenance audit.
            year = next((year for year in (2020, 2021) if str(year) in name), None)
            temporal = (
                "static"
                if any(term in name for term in ("dem_elevation", "dem_slope"))
                else "annual" if year else "unknown"
            )
            rows.append(
                {
                    "family": family,
                    "path": str(path),
                    "array": name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "year": year,
                    "temporal_mode": temporal,
                    "registry_order_verified": order_ok,
                    "status": "indexed_needs_value_and_provenance_audit",
                    "source_metadata_sha256": sha256(path / ".zattrs"),
                }
            )

        root.visititems(visit)
    atomic_parquet(pd.DataFrame(rows), dataset_root / "targets/manifest.parquet")
    result = {
        "status": "indexed_not_accepted",
        "arrays": len(rows),
        "unverified_order": sum(not r.get("registry_order_verified", False) for r in rows),
        "full_label_value_audit": False,
        "finished_at": now(),
    }
    write_json(report_root / "target_audit.json", result)
    return result


def report_progress(source_root: Path, dataset_root: Path, report_root: Path) -> dict:
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}

    source = read(source_root / "manifests/source.lock.json")
    specs = source.get("archives", [])
    total_bytes = sum(row["bytes"] for row in specs)
    completed = 0
    partial_bytes = 0
    verified_bytes = 0
    for row in specs:
        path = source_root / "packages" / row["archive"]
        marker = source_root / "manifests/extracted" / (row["archive"] + ".json")
        if path.exists():
            verified_bytes += path.stat().st_size
        partial = path.with_name(path.name + ".partial")
        parts = path.with_name(path.name + ".parts")
        part_bytes = sum(p.stat().st_size for p in parts.glob("*.part"))
        part_bytes += sum(p.stat().st_size for p in parts.glob("*.partial"))
        partial_bytes += min(
            row["bytes"], max(partial.stat().st_size if partial.exists() else 0, part_bytes)
        )
        if marker.exists():
            completed += 1
    catalog = read(report_root / "grid_match_report.json")
    checks = {key: False for key in REQUIRED_GATES}
    checks["download"] = bool(specs) and completed == len(specs)
    checks["catalog"] = bool(specs) and catalog.get("processed_archives") == len(specs)
    # Later stages must publish their own verifications; file existence alone never passes them.
    for key in REQUIRED_GATES[2:]:
        verification = read(report_root / "gates" / (key + ".json"))
        input_lock = dataset_root / "locks/input.lock.json"
        checks[key] = (
            verification.get("passed") is True
            and input_lock.exists()
            and verification.get("input_lock_sha256") == sha256(input_lock)
        )
    result = acceptance_status(checks)
    result.update(
        total_archives=len(specs),
        extracted_archives=completed,
        expected_bytes=total_bytes,
        completed_archive_bytes=verified_bytes,
        partial_bytes=partial_bytes,
        catalog=catalog,
    )
    previous = read(dataset_root / "locks/acceptance.json")
    if previous.get("user_accepted"):
        raise ValueError("cannot replace user acceptance with an automatic snapshot")
    write_json(dataset_root / "locks/acceptance.json", result)
    write_json(report_root / "progress.json", result)
    lines = [
        "# V5 数据处理实际状态",
        "",
        f"生成时间：{now()}",
        "",
        "**尚未完成数据验收；未授权训练。**",
        "",
        f"- 完成解压的分包：{completed}/{len(specs)}",
        f"- 已完成下载文件：{verified_bytes / 1e9:.3f} GB；续传文件：{partial_bytes / 1e9:.3f} GB",
        f"- 发布压缩总量：{total_bytes / 1e9:.3f} GB",
        "",
        "| 检查 | 状态 |",
        "|---|---|",
    ]
    lines.extend(
        f'| {key} | {"通过" if value else "未完成或未通过"} |' for key, value in checks.items()
    )
    radiometry = read(report_root / "radiometry_summary.json")
    if radiometry:
        lines += [
            "",
            "## 已核实的基础数据问题",
            "",
            f'已抽查{radiometry.get("sampled_rasters")}张影像，覆盖{radiometry.get("archives")}个月度包。',
            "Landsat波段对应关系存在冲突；基础TIFF未保留足以独立确认缩放、填充值和处理基线的元数据。",
            "待原导出脚本或来源合同补证；不能用猜测生成合格统计量。",
        ]
    lines += [
        "",
        "## 验收材料",
        "",
        "当前为进度快照，不是验收通过报告。",
        "完整视觉样例、配准、统计、样本与加载验证完成后，才允许进入ready_for_review。",
        "具体文件清单与错误位于同目录parquet/JSON报告。",
    ]
    (report_root / "acceptance_report.md").write_text("\n".join(lines) + "\n")
    return result
