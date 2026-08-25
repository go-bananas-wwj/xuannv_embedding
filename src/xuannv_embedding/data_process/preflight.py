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
from xuannv_embedding.data.local_archives import audit_raster_member, verify_archive_lock


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


def build_highres_patch_index(
    candidates_path: Path,
    scenes_path: Path,
    output_path: Path,
) -> int:
    """Spatially join local high-resolution scenes to frozen national patches."""
    candidates = pq.read_table(
        candidates_path, columns=["patch_id", "grid_epsg", "utm_bounds"]
    ).to_pylist()
    scenes = pq.read_table(scenes_path).to_pylist()
    by_epsg: dict[int, list[dict[str, Any]]] = {}
    for patch in candidates:
        by_epsg.setdefault(int(patch["grid_epsg"]), []).append(patch)
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        crs = str(scene.get("crs") or "")
        if not crs.startswith("EPSG:") or not scene.get("training_eligible"):
            continue
        epsg = int(crs.split(":", 1)[1])
        scene_left, scene_bottom, scene_right, scene_top = map(float, scene["bounds"])
        for patch in by_epsg.get(epsg, []):
            left, bottom, right, top = map(float, patch["utm_bounds"])
            width = max(0.0, min(right, scene_right) - max(left, scene_left))
            height = max(0.0, min(top, scene_top) - max(bottom, scene_bottom))
            intersection = width * height
            if intersection <= 0:
                continue
            rows.append(
                {
                    "patch_id": str(patch["patch_id"]),
                    "grid_epsg": epsg,
                    "patch_bounds": [left, bottom, right, top],
                    "product_id": str(scene["product_id"]),
                    "scene_id": str(scene["scene_id"]),
                    "acquired_at": str(scene["acquired_at"]),
                    "available_at": str(scene["available_at"]),
                    "clear_percent": int(scene.get("clear_percent") or 0),
                    "image_path": str(scene["image_path"]),
                    "qa_path": str(scene.get("qa_path") or ""),
                    "qa_present": bool(scene.get("qa_present")),
                    "scene_transform": [float(value) for value in scene["transform"]],
                    "intersection_fraction": intersection / ((right - left) * (top - bottom)),
                    "quality_status": str(scene.get("quality_category") or "unknown"),
                }
            )
    rows.sort(key=lambda row: (row["patch_id"], row["acquired_at"], row["scene_id"]))
    if not rows:
        raise PreflightError(f"高分场景与全国 patch 无空间交集: {scenes_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path, compression="zstd")
    return len(rows)


def build_raster_label_patch_index(
    candidates_path: Path,
    task: str,
    label_root: Path,
    output_path: Path,
) -> int:
    """Index verified local raster labels by their real CRS and footprint."""
    candidates = pq.read_table(
        candidates_path, columns=["patch_id", "grid_epsg", "utm_bounds"]
    ).to_pylist()
    by_epsg: dict[int, list[dict[str, Any]]] = {}
    for patch in candidates:
        by_epsg.setdefault(int(patch["grid_epsg"]), []).append(patch)
    rows: list[dict[str, Any]] = []
    for label_path in sorted(label_root.rglob("*.tif")):
        with rasterio.open(label_path) as dataset:
            if dataset.count != 1 or dataset.crs is None:
                raise PreflightError(f"监督标签必须是单波段且带 CRS: {label_path}")
            epsg = dataset.crs.to_epsg()
            if epsg is None:
                raise PreflightError(f"监督标签 CRS 无 EPSG: {label_path}")
            source_bounds = dataset.bounds
            source_transform = list(dataset.transform)[:6]
            dtype = dataset.dtypes[0]
        for patch in by_epsg.get(epsg, []):
            left, bottom, right, top = map(float, patch["utm_bounds"])
            width = max(0.0, min(right, source_bounds.right) - max(left, source_bounds.left))
            height = max(0.0, min(top, source_bounds.top) - max(bottom, source_bounds.bottom))
            if width * height <= 0:
                continue
            rows.append(
                {
                    "patch_id": str(patch["patch_id"]),
                    "task": task,
                    "label_path": str(label_path.resolve()),
                    "crs": f"EPSG:{epsg}",
                    "transform": [float(value) for value in source_transform],
                    "dtype": dtype,
                    "intersection_fraction": width * height / ((right - left) * (top - bottom)),
                    "provenance": "local_preprocessed_osm_weak_label",
                }
            )
    rows.sort(key=lambda row: (row["patch_id"], row["label_path"]))
    if not rows:
        raise PreflightError(f"监督标签与全国 patch 无空间交集: {task}/{label_root}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path, compression="zstd")
    return len(rows)


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
    highres_patch_ids: set[str] = set()
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
        patch_count = build_highres_patch_index(
            root / "registry" / "candidate_62000.parquet",
            output,
            output.with_name("patch_observations.parquet"),
        )
        if patch_count <= 0:
            raise PreflightError(f"高分 product 没有覆盖全国 1% patch: {product_id}")
        highres_patch_ids.update(
            str(value)
            for value in pq.read_table(
                output.with_name("patch_observations.parquet"), columns=["patch_id"]
            )["patch_id"].to_pylist()
        )
        highres_count += len(rows)

    if highres_patch_ids:
        split = pq.read_table(root / "registry" / "split_80_10_10.parquet")
        selected = [row for row in split.to_pylist() if str(row["patch_id"]) in highres_patch_ids]
        selected.sort(key=lambda row: str(row["patch_id"]))
        pq.write_table(
            pa.Table.from_pylist(selected),
            root / "registry" / "highres_real.parquet",
            compression="zstd",
        )

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
    label_count = 0
    for task, label_root in config.paths.supervised_label_roots.items():
        if not label_root.is_dir():
            raise PreflightError(f"监督标签目录不存在: {task}/{label_root}")
        label_count += build_raster_label_patch_index(
            root / "registry" / "candidate_62000.parquet",
            task,
            label_root,
            root / "labels" / task / "patch_observations.parquet",
        )
    return {
        "highres_scene_count": highres_count,
        "auxiliary_archive_count": len(auxiliary_rows),
        "legacy_file_count": len(legacy_rows),
        "supervised_label_observation_count": label_count,
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
    try:
        archive_lock = verify_archive_lock(
            root / "locks" / "local_archive_sha256.jsonl", expected_count=72
        )
    except ValueError as exc:
        raise PreflightError(f"archive lock 校验失败: {exc}") from exc
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
        "archive_lock": archive_lock,
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


def run_highres_mini(config: V2Config) -> dict[str, Any]:
    """Run one real local high-resolution sample through data, losses and gradients."""
    import torch
    from torch.utils.data import DataLoader

    from xuannv_embedding.data.v2_dataset import V2LocalZipDataset, collate_v2
    from xuannv_embedding.training.losses import V2TotalLoss
    from xuannv_embedding.training.runtime import V2TrainingSystem
    from xuannv_embedding.training.validation_profiles import build_profile_model

    registry = config.paths.data_root / "registry" / "highres_real.parquet"
    if not registry.is_file():
        raise PreflightError("highres mini 需要先生成 highres_real.parquet")
    records = pq.read_table(registry).to_pylist()
    if not records:
        raise PreflightError("highres mini registry 为空")
    selected = None
    selected_rows = None
    for record in records:
        patch_id = str(record["patch_id"])
        rows = []
        for product_id in config.paths.product_roots:
            path = (
                config.paths.data_root
                / "observations"
                / "highres"
                / product_id
                / "patch_observations.parquet"
            )
            rows.extend(pq.read_table(path, filters=[("patch_id", "=", patch_id)]).to_pylist())
        label_available = all(
            pq.read_table(
                config.paths.data_root / "labels" / task / "patch_observations.parquet",
                filters=[("patch_id", "=", patch_id)],
            ).num_rows
            > 0
            for task in config.training.semantic_probe_tasks
        )
        if rows and label_available:
            selected = record
            selected_rows = rows
            break
    if selected is None or selected_rows is None:
        raise PreflightError("highres mini 没有同时具备高分和监督标签的真实 patch")
    temporary_registry = config.paths.data_root / "registry" / "highres_mini_1.parquet"
    pq.write_table(pa.Table.from_pylist([selected]), temporary_registry, compression="zstd")
    output_end = (
        max(
            datetime.fromisoformat(str(row["available_at"]).replace("Z", "+00:00"))
            for row in selected_rows
        ).timestamp()
        / 86400.0
        + 1.0
    )
    dataset = V2LocalZipDataset(
        config,
        temporary_registry,
        spatial_size=32,
        max_records=1,
        output_selection="random_single",
        context_days=config.temporal.dense_lookback_days,
        output_intervals_override=((output_end - 1.0, output_end),),
        allow_incomplete_statistics=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate_v2)))
    profile = config.validation_profiles["mini-real"]
    model = build_profile_model(config, profile)
    criterion = V2TotalLoss(
        embed_dim=config.model.embedding_dim,
        reconstruction_weights={},
        semantic_probe_weight=1.0 if config.training.semantic_probe_tasks else 0.0,
        semantic_probe_tasks=config.training.semantic_probe_tasks,
        semantic_probe_hidden_dim=config.training.semantic_probe_hidden_dim,
        highres_detail_weight=1.0,
    )
    system = V2TrainingSystem(model, criterion)
    result = system(batch)
    result["total"].backward()
    product_id = next(iter(config.paths.product_roots))
    native_gradient = model.highres_adapters[product_id].native_stem[0].weight.grad
    mask_pixels = int(batch["model_inputs"]["highres_masks"][product_id].sum().item())
    detail_pixels = int(batch["detail_masks"][product_id].gt(0).sum().item())
    semantic_pixels = sum(
        int(mask.gt(0).sum().item()) for mask in batch["supervised_label_masks"].values()
    )
    passed = (
        mask_pixels > 0
        and detail_pixels > 0
        and semantic_pixels > 0
        and native_gradient is not None
        and bool(torch.isfinite(native_gradient).all())
        and float(native_gradient.abs().sum()) > 0
        and bool(torch.isfinite(result["total"]))
    )
    return {
        "schema_version": "xuannv_v2_highres_mini_v1",
        "patch_id": str(selected["patch_id"]),
        "product_id": product_id,
        "native_shape": list(batch["model_inputs"]["highres_frames"][product_id].shape),
        "highres_valid_pixels": mask_pixels,
        "detail_valid_pixels": detail_pixels,
        "semantic_valid_pixels": semantic_pixels,
        "detail_loss": float(result["highres_detail"].detach()),
        "semantic_loss": float(result["semantic_probe"].detach()),
        "native_stem_gradient_l1": (
            float(native_gradient.abs().sum()) if native_gradient is not None else 0.0
        ),
        "network_requests": 0,
        "passed": passed,
    }
