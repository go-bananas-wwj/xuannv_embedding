"""Build deterministic V2 registries from the frozen local national grid."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from xuannv_embedding.config import V2Config
from xuannv_embedding.data.local_archives import (
    LocalArchive,
    index_local_archive,
    sha256_file,
    validate_archive_sidecar,
)


class InventoryError(ValueError):
    """The local candidate inventory cannot satisfy the V2 registry contract."""


REQUIRED_GRID_COLUMNS = ("patch_id", "macro_id", "grid_id", "grid_epsg")


def read_sampled_grid(grid_package: Path) -> pa.Table:
    paths = sorted((grid_package / "sampled").glob("utm*/part-*.parquet"))
    if not paths:
        raise InventoryError(f"未找到 sampled parquet: {grid_package}")
    tables = []
    for path in paths:
        schema_names = pq.read_schema(path).names
        missing = sorted(set(REQUIRED_GRID_COLUMNS) - set(schema_names))
        if missing:
            raise InventoryError(f"{path} 缺少字段: {', '.join(missing)}")
        columns = list(REQUIRED_GRID_COLUMNS)
        columns.extend(
            name
            for name in ("longitude", "latitude", "utm_bounds", "wgs84_bounds")
            if name in schema_names
        )
        if "sampled" in schema_names:
            columns.append("sampled")
        tables.append(pq.read_table(path, columns=columns))
    table = pa.concat_tables(tables, promote_options="default")
    if "sampled" in table.column_names:
        table = table.filter(pc.equal(table["sampled"], True)).drop(["sampled"])
    patch_ids = table["patch_id"].to_pylist()
    if len(set(patch_ids)) != len(patch_ids):
        raise InventoryError("sampled grid 包含重复 patch_id")
    return table.sort_by([("patch_id", "ascending")])


def _ordered_groups(table: pa.Table, seed: int) -> list[tuple[str, list[int]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, macro_id in enumerate(table["macro_id"].to_pylist()):
        groups[str(macro_id)].append(index)
    return sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).digest(),
    )


def _take_exact_groups(
    groups: list[tuple[str, list[int]]], target: int
) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    selected: list[tuple[str, list[int]]] = []
    deferred: list[tuple[str, list[int]]] = []
    remaining = target
    for group in groups:
        size = len(group[1])
        if size <= remaining:
            selected.append(group)
            remaining -= size
        else:
            deferred.append(group)
    if remaining:
        raise InventoryError(f"macro_id 分组后无法得到精确的 {target} 条记录")
    return selected, deferred


def build_grouped_split(
    candidates: pa.Table,
    *,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> pa.Table:
    expected = train_count + val_count + test_count
    if candidates.num_rows != expected:
        raise InventoryError(f"候选数量 {candidates.num_rows} 与 split 总数 {expected} 不一致")
    groups = _ordered_groups(candidates, seed)
    train, groups = _take_exact_groups(groups, train_count)
    val, groups = _take_exact_groups(groups, val_count)
    if sum(len(indices) for _, indices in groups) != test_count:
        raise InventoryError(f"macro_id 分组后无法得到精确的 {test_count} 条 test 记录")
    assignment: dict[int, str] = {}
    for split, selected in (("train", train), ("val", val), ("test", groups)):
        for _, indices in selected:
            assignment.update({index: split for index in indices})
    split_array = pa.array([assignment[index] for index in range(candidates.num_rows)])
    return candidates.append_column("split", split_array)


def write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


ARCHIVE_DIRECTORIES = {
    "s2_local": "pc-s2",
    "s1_local": "pc-s1",
    "landsat_local": "pc-ls",
}


def discover_dense_archives(config: V2Config) -> list[LocalArchive]:
    archives: list[LocalArchive] = []
    for product_id, product in config.products.items():
        if product.role != "dense":
            continue
        directory = ARCHIVE_DIRECTORIES.get(product_id)
        if directory is None:
            raise InventoryError(f"没有为 dense product 配置本地 ZIP 目录: {product_id}")
        root = config.paths.source_root / directory
        paths = sorted(root.glob("20??/??/*.zip"))
        if not paths:
            raise InventoryError(f"本地 dense ZIP 不存在: {root}")
        for path in paths:
            try:
                year = int(path.parent.parent.name)
                month = int(path.parent.name)
            except ValueError as exc:
                raise InventoryError(f"本地 ZIP 年月目录非法: {path}") from exc
            archives.append(LocalArchive(product_id, year, month, path.resolve()))
    archives.sort(key=lambda item: (item.product_id, item.year, item.month, str(item.path)))
    return archives


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _replace_parquet(path: Path, tables: list[pa.Table]) -> None:
    if not tables:
        raise InventoryError(f"拒绝写空 parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with pq.ParquetWriter(temporary, tables[0].schema, compression="zstd") as writer:
            for table in tables:
                writer.write_table(table)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _observation_rows(index) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    members: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    for observation in index.observations:
        common = {
            "patch_id": observation.patch_id,
            "product_id": observation.product_id,
            "year": index.archive.year,
            "month": index.archive.month,
            "interval_start": observation.interval_start,
            "interval_end": observation.interval_end,
            "acquired_at": observation.acquired_at,
            "available_at": observation.available_at,
            "archive_path": str(observation.archive_path),
            "member_name": observation.member_name,
            "present": observation.present,
            "quality_status": observation.quality_status,
            "time_precision": observation.time_precision,
        }
        availability.append(common)
        if observation.present:
            members.append(common)
    return members, availability


def build_local_archive_inventory(
    config: V2Config,
    *,
    calculate_sha256: bool = True,
) -> dict[str, Any]:
    """Index all local dense ZIPs and produce immutable V2 Parquet registries."""
    if config.network_policy.allow_remote_pixels:
        raise InventoryError("local-index 禁止远程像元")
    candidates = read_sampled_grid(config.paths.grid_package)
    if candidates.num_rows != 62_000:
        raise InventoryError(f"正式全国 sampled grid 必须为 62000，实际 {candidates.num_rows}")
    root = config.paths.data_root
    registry_root = root / "registry"
    index_root = root / "observations" / "index"
    write_table(candidates, registry_root / "candidate_62000.parquet")
    split = build_grouped_split(
        candidates,
        train_count=49_600,
        val_count=6_200,
        test_count=6_200,
        seed=42,
    )
    write_table(split, registry_root / "split_80_10_10.parquet")
    expected_patch_ids = set(candidates["patch_id"].to_pylist())
    archive_rows: list[dict[str, Any]] = []
    lock_rows: list[dict[str, Any]] = []
    member_path = index_root / "local_zip_members.parquet"
    availability_path = index_root / "availability.parquet"
    index_root.mkdir(parents=True, exist_ok=True)
    member_temporary = member_path.with_name(f".{member_path.name}.tmp")
    availability_temporary = availability_path.with_name(f".{availability_path.name}.tmp")
    member_writer: pq.ParquetWriter | None = None
    availability_writer: pq.ParquetWriter | None = None
    observation_count = 0
    present_count = 0
    archives = discover_dense_archives(config)
    try:
        for archive in archives:
            product = config.products[archive.product_id].to_product_spec(archive.product_id)
            index = index_local_archive(archive, product, expected_patch_ids)
            sidecar_sha256 = validate_archive_sidecar(archive, product)
            checksum = sha256_file(archive.path) if calculate_sha256 else None
            row = {
                "product_id": archive.product_id,
                "year": archive.year,
                "month": archive.month,
                "archive_path": str(archive.path),
                "size_bytes": index.size_bytes,
                "member_count": index.member_count,
                "missing_count": len(index.missing_patch_ids),
                "extra_count": len(index.extra_patch_ids),
                "member_list_sha256": index.member_list_sha256,
                "sidecar_sha256": sidecar_sha256,
                "sha256": checksum,
            }
            archive_rows.append(row)
            lock_rows.append({**row, "locked_at": datetime.now(timezone.utc).isoformat()})
            members, availability = _observation_rows(index)
            member_table = pa.Table.from_pylist(members)
            availability_table = pa.Table.from_pylist(availability)
            if member_writer is None:
                member_writer = pq.ParquetWriter(
                    member_temporary, member_table.schema, compression="zstd"
                )
            if availability_writer is None:
                availability_writer = pq.ParquetWriter(
                    availability_temporary,
                    availability_table.schema,
                    compression="zstd",
                )
            member_writer.write_table(member_table)
            availability_writer.write_table(availability_table)
            present_count += member_table.num_rows
            observation_count += availability_table.num_rows
    except BaseException:
        if member_writer is not None:
            member_writer.close()
        if availability_writer is not None:
            availability_writer.close()
        member_temporary.unlink(missing_ok=True)
        availability_temporary.unlink(missing_ok=True)
        raise
    if member_writer is None or availability_writer is None:
        raise InventoryError("未索引到本地 dense observations")
    member_writer.close()
    availability_writer.close()
    os.replace(member_temporary, member_path)
    os.replace(availability_temporary, availability_path)
    if len(archives) != 72:
        raise InventoryError(f"全国 dense archive 数量必须为 72，实际 {len(archives)}")
    write_table(
        pa.Table.from_pylist(archive_rows),
        registry_root / "local_archive_inventory.parquet",
    )
    _atomic_jsonl(root / "locks" / "local_archive_sha256.jsonl", lock_rows)
    build_validation_registries(root)
    return {
        "schema_version": "xuannv_v2_local_archive_inventory_v1",
        "archive_count": len(archive_rows),
        "candidate_count": candidates.num_rows,
        "observation_count": observation_count,
        "present_count": present_count,
        "sha256_complete": calculate_sha256,
        "output_root": str(root),
    }


def _row_hash(row: dict[str, Any], salt: str) -> bytes:
    return hashlib.sha256(f"{salt}:{row['patch_id']}".encode()).digest()


def _append_registry_row(
    selected: list[dict[str, Any]],
    used: set[str],
    candidates: list[dict[str, Any]],
    *,
    sentinel_type: str,
) -> None:
    for row in candidates:
        if row["patch_id"] not in used:
            selected.append({**row, "sentinel_type": sentinel_type})
            used.add(row["patch_id"])
            return
    raise InventoryError(f"无法选择不重复的 mini sentinel: {sentinel_type}")


def build_validation_registries(root: Path) -> dict[str, int]:
    registry_root = root / "registry"
    split_table = pq.read_table(registry_root / "split_80_10_10.parquet")
    rows = split_table.to_pylist()
    by_patch = {str(row["patch_id"]): row for row in rows}
    availability_path = root / "observations" / "index" / "availability.parquet"
    jan = pq.read_table(
        availability_path,
        columns=["patch_id", "product_id", "year", "month", "present"],
        filters=[("month", "=", 1)],
    )
    products = sorted(set(jan["product_id"].to_pylist()))
    required_slots = {(product, year) for product in products for year in (2020, 2021)}
    present_slots: dict[str, set[tuple[str, int]]] = defaultdict(set)
    missing_by_product: dict[str, set[str]] = defaultdict(set)
    for observation in jan.to_pylist():
        patch_id = str(observation["patch_id"])
        product_id = str(observation["product_id"])
        if observation["present"]:
            present_slots[patch_id].add((product_id, int(observation["year"])))
        else:
            missing_by_product[product_id].add(patch_id)
    complete = {
        patch_id for patch_id, slots in present_slots.items() if required_slots.issubset(slots)
    }
    mini: list[dict[str, Any]] = []
    used: set[str] = set()
    for grid_id in [f"utm{zone}n" for zone in range(43, 54)]:
        eligible = sorted(
            (row for row in rows if row["grid_id"] == grid_id and row["patch_id"] in complete),
            key=lambda row: _row_hash(row, "mini-complete"),
        )
        _append_registry_row(mini, used, eligible, sentinel_type="three_source_complete")
    for product_id in products:
        eligible = sorted(
            (by_patch[patch_id] for patch_id in missing_by_product[product_id]),
            key=lambda row: _row_hash(row, f"missing-{product_id}"),
        )
        _append_registry_row(mini, used, eligible, sentinel_type=f"missing_{product_id}")
    seam_candidates = sorted(
        rows,
        key=lambda row: (
            min(abs(((float(row.get("longitude", 0.0)) + 180.0) % 6.0) - edge) for edge in (0, 6)),
            _row_hash(row, "utm-seam"),
        ),
    )
    _append_registry_row(mini, used, seam_candidates, sentinel_type="utm_seam")
    highres_candidates = sorted(
        rows,
        key=lambda row: (
            (float(row.get("longitude", 0.0)) - 116.3) ** 2
            + (float(row.get("latitude", 0.0)) - 39.9) ** 2,
            _row_hash(row, "highres"),
        ),
    )
    _append_registry_row(mini, used, highres_candidates, sentinel_type="highres_spatial_candidate")
    if len(mini) != 16:
        raise InventoryError(f"mini registry 必须为 16，实际 {len(mini)}")
    write_table(pa.Table.from_pylist(mini), registry_root / "mini_16.parquet")

    core: list[dict[str, Any]] = []
    for split_name, count in (("train", 496), ("val", 62), ("test", 62)):
        eligible = sorted(
            (row for row in rows if row["split"] == split_name),
            key=lambda row: _row_hash(row, f"smoke-{split_name}"),
        )[:count]
        core.extend({**row, "is_sentinel": False, "sentinel_type": ""} for row in eligible)
    smoke_used = {row["patch_id"] for row in core}
    sentinel_pool = sorted(
        (
            by_patch[patch_id]
            for missing in missing_by_product.values()
            for patch_id in missing
            if patch_id not in smoke_used
        ),
        key=lambda row: _row_hash(row, "smoke-sentinel"),
    )
    sentinels: list[dict[str, Any]] = []
    for row in sentinel_pool:
        if row["patch_id"] in smoke_used:
            continue
        sentinels.append({**row, "is_sentinel": True, "sentinel_type": "missing_or_boundary"})
        smoke_used.add(row["patch_id"])
        if len(sentinels) == 32:
            break
    if len(sentinels) != 32:
        raise InventoryError("无法选择 32 个 smoke sentinels")
    smoke = core + sentinels
    write_table(pa.Table.from_pylist(smoke), registry_root / "smoke_620_plus_32.parquet")
    return {"mini_count": len(mini), "smoke_count": len(smoke)}
