"""Enumerate the canonical nationwide 1,280 m parent grid."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from pyproj import Transformer
from shapely import from_wkb, set_precision
from shapely import normalize as normalize_geometry
from shapely.geometry import Point, box
from shapely.ops import transform as transform_geometry

PARENT_SIDE_METERS = 1280
CHINA_OWNER_EPSGS = frozenset(range(32643, 32654))
SHAPEFILE_MAX_BYTES = 1_800_000_000
SHAPEFILE_SAFE_FRACTION = 0.95
SHAPEFILE_ROW_BLOCK_ROWS = 1_000
SHAPEFILE_COMPONENT_SUFFIXES = (".shp", ".shx", ".dbf", ".prj", ".cpg")
AUDIT_BATCH_SIZE = 100_000
AUDIT_DIMENSION_TOLERANCE_M = 0.001
AUDIT_AREA_TOLERANCE_M2 = 1.0
CROSS_ZONE_OVERLAP_FRACTION = 0.01
UTM_SEAM_GLOBAL_DUPLICATE_FRACTION = 0.001
UTM_SEAM_POLICY_VERSION = "adjacent-owner-zone-seam-v1"
GRID_PACKAGE_MANIFEST = "china_full_1280m_grid_package_manifest.json"
GRID_PACKAGE_CHECKSUMS = "SHA256SUMS.json"
AUDIT_EXAMPLE_LIMIT = 100
AUDIT_COORDINATE_TOLERANCE = 1e-9
CANONICAL_METADATA_FIELDS = (
    "schema_version",
    "atlas_version",
    "boundary_version",
    "patch_id",
    "parent_key",
    "macro_id",
    "macro_local_col",
    "macro_local_row",
    "grid_id",
    "grid_epsg",
    "grid_col",
    "grid_row",
    "utm_bounds",
    "wgs84_bounds",
    "longitude",
    "latitude",
    "sampled",
    "identity_hash",
    "footprint_hash",
)


@dataclass(frozen=True)
class GridSpec:
    side_m: int = PARENT_SIDE_METERS
    macro_side_patches: int = 10
    boundary_version: str = "geoBoundaries-CHN-ADM0-frozen-20260726"
    atlas_version: str = "china-full-1280m-v1-20260805"

    def __post_init__(self) -> None:
        if self.side_m != PARENT_SIDE_METERS:
            raise ValueError(f"side_m must be exactly {PARENT_SIDE_METERS}")


@dataclass
class ZoneWriteSummary:
    """Counts and GeoParquet parts emitted for one UTM grid zone."""

    grid_id: str
    all_count: int = 0
    sampled_count: int = 0
    unsampled_count: int = 0
    parquet_parts: list[str] = field(default_factory=list)
    shapefile_parts: list[str] = field(default_factory=list)
    batch_count: int = 0


def parent_key(grid_epsg: int, grid_col: int, grid_row: int) -> str:
    """Return the stable identity for one parent grid cell."""
    return f"{grid_epsg}:{grid_col}:{grid_row}"


def utm_owner_epsg(longitude: float, latitude: float) -> int:
    """Return the EPSG code for the unique half-open UTM zone at a point."""
    zone = max(1, min(60, math.floor((float(longitude) + 180.0) / 6.0) + 1))
    return (32600 if float(latitude) >= 0 else 32700) + zone


def _validate_china_owner_epsg(grid_epsg: int) -> None:
    if grid_epsg not in CHINA_OWNER_EPSGS:
        raise ValueError("grid_epsg must be within China owner EPSGs 32643 through 32653")


def build_patch_record(
    macro: Mapping[str, Any],
    grid_col: int,
    grid_row: int,
    longitude: float,
    latitude: float,
    spec: GridSpec,
) -> dict[str, Any]:
    """Build one canonical parent-grid record from its integer coordinates."""
    grid_epsg = int(macro["grid_epsg"])
    macro_col = int(macro["macro_col"])
    macro_row = int(macro["macro_row"])
    key = parent_key(grid_epsg, grid_col, grid_row)
    return {
        "schema_version": "china_full_1280m_parent_grid_v1",
        "atlas_version": spec.atlas_version,
        "boundary_version": spec.boundary_version,
        "patch_id": f"parent_{key}",
        "parent_key": key,
        "macro_id": str(macro.get("macro_id", f"{macro['grid_id']}_c{macro_col}_r{macro_row}")),
        "macro_local_col": grid_col - macro_col * spec.macro_side_patches,
        "macro_local_row": grid_row - macro_row * spec.macro_side_patches,
        "grid_id": str(macro["grid_id"]),
        "grid_epsg": grid_epsg,
        "grid_col": grid_col,
        "grid_row": grid_row,
        "utm_bounds": [
            grid_col * spec.side_m,
            grid_row * spec.side_m,
            (grid_col + 1) * spec.side_m,
            (grid_row + 1) * spec.side_m,
        ],
        "longitude": longitude,
        "latitude": latitude,
    }


def enumerate_macro_patch_records(
    macro: Mapping[str, Any], boundary_wgs84: Any, spec: GridSpec
) -> Iterator[dict[str, Any]]:
    """Yield owner-zone-valid parent cells whose centers are covered by ADM0."""
    grid_epsg = int(macro["grid_epsg"])
    _validate_china_owner_epsg(grid_epsg)
    macro_col = int(macro["macro_col"])
    macro_row = int(macro["macro_row"])
    to_wgs84 = Transformer.from_crs(grid_epsg, 4326, always_xy=True)
    half_side = spec.side_m / 2

    for local_col in range(spec.macro_side_patches):
        for local_row in range(spec.macro_side_patches):
            grid_col = macro_col * spec.macro_side_patches + local_col
            grid_row = macro_row * spec.macro_side_patches + local_row
            minx = grid_col * spec.side_m
            miny = grid_row * spec.side_m
            longitude, latitude = to_wgs84.transform(minx + half_side, miny + half_side)
            if utm_owner_epsg(longitude, latitude) != grid_epsg:
                continue
            if not boundary_wgs84.covers(Point(longitude, latitude)):
                continue
            yield build_patch_record(macro, grid_col, grid_row, longitude, latitude, spec)


def validate_patch_record(record: Mapping[str, Any], spec: GridSpec) -> None:
    """Raise ValueError when a record violates the canonical parent-grid contract."""
    required = (
        "patch_id",
        "parent_key",
        "macro_id",
        "macro_local_col",
        "macro_local_row",
        "grid_id",
        "grid_epsg",
        "grid_col",
        "grid_row",
        "utm_bounds",
        "longitude",
        "latitude",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"patch record missing required fields: {missing}")

    grid_epsg = record["grid_epsg"]
    grid_col = record["grid_col"]
    grid_row = record["grid_row"]
    if not all(isinstance(value, int) for value in (grid_epsg, grid_col, grid_row)):
        raise ValueError("grid_epsg, grid_col, and grid_row must be integers")
    _validate_china_owner_epsg(grid_epsg)
    expected_key = parent_key(grid_epsg, grid_col, grid_row)
    if record["parent_key"] != expected_key or record["patch_id"] != f"parent_{expected_key}":
        raise ValueError("parent_key and patch_id must match integer grid coordinates")

    expected_bounds = [
        grid_col * spec.side_m,
        grid_row * spec.side_m,
        (grid_col + 1) * spec.side_m,
        (grid_row + 1) * spec.side_m,
    ]
    if record["utm_bounds"] != expected_bounds:
        raise ValueError("utm_bounds must be exact integer-aligned parent-cell bounds")

    for local_field in ("macro_local_col", "macro_local_row"):
        local_value = record[local_field]
        if not isinstance(local_value, int) or not 0 <= local_value < spec.macro_side_patches:
            raise ValueError(f"{local_field} must be within the macrocell")

    longitude = record["longitude"]
    latitude = record["latitude"]
    if not isinstance(longitude, (int, float)) or not isinstance(latitude, (int, float)):
        raise ValueError("longitude and latitude must be numeric")
    if utm_owner_epsg(longitude, latitude) != grid_epsg:
        raise ValueError("grid_epsg must own the WGS84 center point")


@lru_cache(maxsize=len(CHINA_OWNER_EPSGS))
def _transformer_to_wgs84(grid_epsg: int) -> Transformer:
    """Reuse immutable PROJ pipelines across all cells in one owner zone."""
    _validate_china_owner_epsg(grid_epsg)
    return Transformer.from_crs(grid_epsg, 4326, always_xy=True)


@lru_cache(maxsize=len(CHINA_OWNER_EPSGS))
def _transformer_from_wgs84(grid_epsg: int) -> Transformer:
    """Reuse the inverse PROJ pipeline used by exact geometry validation."""
    _validate_china_owner_epsg(grid_epsg)
    return Transformer.from_crs(4326, grid_epsg, always_xy=True)


def _wgs84_geometry(record: Mapping[str, Any]):
    minx, miny, maxx, maxy = record["utm_bounds"]
    to_wgs84 = _transformer_to_wgs84(int(record["grid_epsg"]))
    return transform_geometry(to_wgs84.transform, box(minx, miny, maxx, maxy))


def _geoparquet_records(records: Iterable[Mapping[str, Any]], sampled_keys: set[str]):
    import geopandas as gpd

    rows = []
    for record in records:
        row = dict(record)
        geometry = _wgs84_geometry(record)
        identity = (
            f"{record['atlas_version']}:{record['grid_epsg']}:"
            f"{record['grid_col']}:{record['grid_row']}"
        )
        row["sampled"] = str(record["parent_key"]) in sampled_keys
        row["wgs84_bounds"] = list(geometry.bounds)
        row["identity_hash"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        row["footprint_hash"] = hashlib.sha256(
            normalize_geometry(set_precision(geometry, 1e-9)).wkb
        ).hexdigest()
        row["geometry"] = geometry
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _shapefile_records(records: Iterable[Mapping[str, Any]], sampled_keys: set[str]):
    import geopandas as gpd

    rows = []
    for record in records:
        rows.append(
            {
                "PATCH_ID": record["patch_id"],
                "UTM_EPSG": record["grid_epsg"],
                "GRID_COL": record["grid_col"],
                "GRID_ROW": record["grid_row"],
                "MACRO_ID": record["macro_id"],
                "SAMPLED": int(str(record["parent_key"]) in sampled_keys),
                "geometry": _wgs84_geometry(record),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _shapefile_component_sizes(path: Path) -> dict[str, int]:
    return {
        suffix: path.with_suffix(suffix).stat().st_size
        for suffix in SHAPEFILE_COMPONENT_SUFFIXES
        if path.with_suffix(suffix).exists()
    }


def _remove_shapefile(path: Path) -> None:
    for suffix in SHAPEFILE_COMPONENT_SUFFIXES:
        path.with_suffix(suffix).unlink(missing_ok=True)


def _shapefile_component_limit() -> int:
    return max(1, math.floor(SHAPEFILE_MAX_BYTES * SHAPEFILE_SAFE_FRACTION))


def _fits_shapefile_component_cap(
    records: list[Mapping[str, Any]],
    sampled_keys: set[str],
    probe_path: Path,
) -> bool:
    frame = _shapefile_records(records, sampled_keys)
    try:
        frame.to_file(probe_path, index=False)
        sizes = _shapefile_component_sizes(probe_path).values()
        return all(size < _shapefile_component_limit() for size in sizes)
    finally:
        _remove_shapefile(probe_path)


def _split_shapefile_records(
    records: list[Mapping[str, Any]], sampled_keys: set[str], staging_dir: Path
) -> list[list[Mapping[str, Any]]]:
    probe_path = staging_dir / f".cap-probe-{uuid.uuid4().hex}.shp"
    if _fits_shapefile_component_cap(records, sampled_keys, probe_path):
        return [records]
    if len(records) == 1:
        raise ValueError(
            f"one Shapefile feature exceeds the {SHAPEFILE_MAX_BYTES} byte component cap"
        )
    midpoint = len(records) // 2
    first_half = _split_shapefile_records(records[:midpoint], sampled_keys, staging_dir)
    second_half = _split_shapefile_records(records[midpoint:], sampled_keys, staging_dir)
    return first_half + second_half


def _stage_shapefile_partition(
    records: list[Mapping[str, Any]],
    sampled_keys: set[str],
    staging_root: Path,
    grid_id: str,
    partition: str,
    batch_index: int,
) -> list[Path]:
    records_by_row_block: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in sorted(
        records,
        key=lambda item: (int(item["grid_row"]), int(item["grid_col"]), str(item["parent_key"])),
    ):
        records_by_row_block[int(record["grid_row"]) // SHAPEFILE_ROW_BLOCK_ROWS].append(record)

    staging_dir = staging_root / partition / grid_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for row_block, block_records in sorted(records_by_row_block.items()):
        for part_index, part_records in enumerate(
            _split_shapefile_records(block_records, sampled_keys, staging_dir)
        ):
            path = staging_dir / (
                f"{grid_id}_{partition}_rowblock-{row_block:06d}_part-{batch_index:05d}-{part_index:03d}.shp"
            )
            _shapefile_records(part_records, sampled_keys).to_file(path, index=False)
            sizes = _shapefile_component_sizes(path).values()
            if not all(size < _shapefile_component_limit() for size in sizes):
                raise ValueError(f"Shapefile cap check failed for {path}")
            paths.append(path)
    return paths


def _publish_staged_files(staging_root: Path, output_root: Path) -> list[Path]:
    staged_files = sorted(path for path in staging_root.rglob("*") if path.is_file())
    published = []
    try:
        for staged_path in staged_files:
            destination = output_root / staged_path.relative_to(staging_root)
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite existing output: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(destination)
            published.append(destination)
    except Exception:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    return published


def write_zone_batch(
    batch: list[Mapping[str, Any]],
    sampled_keys: set[str],
    output_root: str | Path,
    summary: ZoneWriteSummary,
) -> None:
    """Write one bounded batch into all, sampled, and unsampled partitions."""
    if not batch:
        return
    output_root = Path(output_root)
    for record in batch:
        if str(record["grid_id"]) != summary.grid_id:
            raise ValueError("each write_zone_records call must contain one grid_id")
        validate_patch_record(record, GridSpec())

    partitions = {
        "all": batch,
        "sampled": [record for record in batch if str(record["parent_key"]) in sampled_keys],
        "unsampled": [record for record in batch if str(record["parent_key"]) not in sampled_keys],
    }
    staging_root = output_root.parent / f".{output_root.name}.batch-{uuid.uuid4().hex}"
    staged_parquet_paths: list[Path] = []
    staged_shapefile_paths: list[Path] = []
    try:
        for partition, partition_records in partitions.items():
            if not partition_records:
                continue
            partition_dir = staging_root / partition / summary.grid_id
            partition_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = partition_dir / f"part-{summary.batch_count:05d}.parquet"
            _geoparquet_records(partition_records, sampled_keys).to_parquet(
                parquet_path, index=False, compression="zstd"
            )
            staged_parquet_paths.append(parquet_path)
            staged_shapefile_paths.extend(
                _stage_shapefile_partition(
                    partition_records,
                    sampled_keys,
                    staging_root,
                    summary.grid_id,
                    partition,
                    summary.batch_count,
                )
            )
        _publish_staged_files(staging_root, output_root)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    summary.all_count += len(batch)
    summary.sampled_count += len(partitions["sampled"])
    summary.unsampled_count += len(partitions["unsampled"])
    summary.parquet_parts.extend(
        str(output_root / path.relative_to(staging_root)) for path in staged_parquet_paths
    )
    summary.shapefile_parts.extend(
        str(output_root / path.relative_to(staging_root)) for path in staged_shapefile_paths
    )
    summary.batch_count += 1


def write_zone_records(
    records: Iterable[Mapping[str, Any]],
    sampled_keys: set[str],
    output_root: str | Path,
    batch_size: int = 100_000,
) -> ZoneWriteSummary:
    """Stream one UTM zone's parent records to disjoint output partitions."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    iterator = iter(records)
    first = next(iterator)
    summary = ZoneWriteSummary(grid_id=str(first["grid_id"]))
    batch = [first]
    for record in iterator:
        batch.append(record)
        if len(batch) == batch_size:
            write_zone_batch(batch, sampled_keys, output_root, summary)
            batch.clear()
    if batch:
        write_zone_batch(batch, sampled_keys, output_root, summary)
    if summary.all_count != summary.sampled_count + summary.unsampled_count:
        raise ValueError(f"partition mismatch for {summary.grid_id}: {summary}")
    return summary


def audit_sample_membership(
    atlas_keys: Iterable[str], sampled_keys: Iterable[str]
) -> dict[str, Any]:
    """Audit whether every sampled parent key appears in the parent atlas exactly once."""
    atlas_counts = Counter(str(key) for key in atlas_keys)
    sampled_counts = Counter(str(key) for key in sampled_keys)
    sampled_key_set = set(sampled_counts)
    duplicate_atlas_keys = sorted(key for key, count in atlas_counts.items() if count > 1)
    duplicate_sampled_keys = sorted(key for key, count in sampled_counts.items() if count > 1)
    missing = sorted(key for key in sampled_counts if atlas_counts[key] == 0)
    matched = sum(
        1 for key, count in sampled_counts.items() if count == 1 and atlas_counts.get(key, 0) == 1
    )
    return {
        "all_count": sum(atlas_counts.values()),
        "sampled_count": sum(sampled_counts.values()),
        "unsampled_count": sum(
            count for key, count in atlas_counts.items() if key not in sampled_key_set
        ),
        "matched": matched,
        "missing": missing,
        "duplicate_atlas_keys": duplicate_atlas_keys,
        "duplicate_sampled_keys": duplicate_sampled_keys,
    }


def sampled_registry_key(record: Mapping[str, Any]) -> str:
    """Return a sampled-registry parent key from either explicit or coordinate fields."""
    if "parent_key" in record:
        return str(record["parent_key"])
    try:
        return parent_key(
            int(record["grid_epsg"]), int(record["grid_col"]), int(record["grid_row"])
        )
    except KeyError as error:
        raise ValueError(
            "sampled registry record needs parent_key or grid_epsg/grid_col/grid_row"
        ) from error


def read_sampled_registry_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read the bounded sampled registry without loading the nationwide parent atlas."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid sampled registry JSONL at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"sampled registry JSONL line {line_number} must be an object")
            sampled_registry_key(record)
            records.append(record)
    return records


def _parquet_paths(output_root: Path, partition: str) -> list[Path]:
    return sorted((output_root / partition).rglob("*.parquet"))


def _iter_parquet_rows(
    paths: Iterable[Path], columns: list[str], batch_size: int
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in paths:
        parquet_file = pq.ParquetFile(path)
        missing_columns = set(columns) - set(parquet_file.schema_arrow.names)
        if missing_columns:
            raise ValueError(f"{path} is missing audit columns: {sorted(missing_columns)}")
        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            yield from batch.to_pylist()


def _iter_audited_parquet_rows(
    paths: Iterable[Path], columns: list[str], batch_size: int
) -> Iterator[dict[str, Any]]:
    """Vectorize geometry normalization while retaining exact per-row audit evidence."""
    import numpy as np
    import pyarrow.parquet as pq
    import shapely

    for path in paths:
        parquet_file = pq.ParquetFile(path)
        missing_columns = set(columns) - set(parquet_file.schema_arrow.names)
        if missing_columns:
            raise ValueError(f"{path} is missing audit columns: {sorted(missing_columns)}")
        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            rows = batch.to_pylist()

            def values(name: str):
                return batch.column(batch.schema.get_field_index(name)).to_numpy(
                    zero_copy_only=False
                )

            grid_epsgs = np.asarray(values("grid_epsg"), dtype=np.int32)
            grid_cols = np.asarray(values("grid_col"), dtype=np.int64)
            grid_rows = np.asarray(values("grid_row"), dtype=np.int64)
            canonical_geometries = np.empty(len(rows), dtype=object)
            for grid_epsg in np.unique(grid_epsgs):
                mask = grid_epsgs == grid_epsg
                utm_geometries = shapely.box(
                    grid_cols[mask] * PARENT_SIDE_METERS,
                    grid_rows[mask] * PARENT_SIDE_METERS,
                    (grid_cols[mask] + 1) * PARENT_SIDE_METERS,
                    (grid_rows[mask] + 1) * PARENT_SIDE_METERS,
                )
                canonical_geometries[mask] = shapely.transform(
                    utm_geometries,
                    _transformer_to_wgs84(int(grid_epsg)).transform,
                    interleaved=False,
                )
            geometries = shapely.from_wkb(values("geometry"))
            normalized_canonical = shapely.normalize(
                shapely.set_precision(canonical_geometries, 1e-9)
            )
            normalized_actual = shapely.normalize(shapely.set_precision(geometries, 1e-9))
            canonical_wkb = shapely.to_wkb(normalized_canonical)
            actual_wkb = shapely.to_wkb(normalized_actual)
            for index, row in enumerate(rows):
                canonical_hash = hashlib.sha256(canonical_wkb[index]).hexdigest()
                actual_hash = hashlib.sha256(actual_wkb[index]).hexdigest()
                canonical_geometry = canonical_geometries[index]
                geometry = geometries[index]
                coordinate_difference = (
                    0.0
                    if canonical_hash == actual_hash
                    else _maximum_footprint_coordinate_difference(canonical_geometry, geometry)
                )
                yield {
                    "row": row,
                    "geometry": geometry,
                    "canonical_geometry": canonical_geometry,
                    "actual_footprint_hash": actual_hash,
                    "canonical_footprint_hash": canonical_hash,
                    "coordinate_difference": coordinate_difference,
                }


def _iter_hashed_parquet_rows(
    paths: Iterable[Path], columns: list[str], batch_size: int
) -> Iterator[dict[str, Any]]:
    """Yield child-partition rows with vectorized normalized geometry hashes."""
    import pyarrow.parquet as pq
    import shapely

    for path in paths:
        parquet_file = pq.ParquetFile(path)
        missing_columns = set(columns) - set(parquet_file.schema_arrow.names)
        if missing_columns:
            raise ValueError(f"{path} is missing audit columns: {sorted(missing_columns)}")
        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            rows = batch.to_pylist()
            geometry_values = batch.column(batch.schema.get_field_index("geometry")).to_numpy(
                zero_copy_only=False
            )
            normalized = shapely.normalize(
                shapely.set_precision(shapely.from_wkb(geometry_values), 1e-9)
            )
            normalized_wkb = shapely.to_wkb(normalized)
            for index, row in enumerate(rows):
                yield {
                    "row": row,
                    "geometry_hash": hashlib.sha256(normalized_wkb[index]).hexdigest(),
                }


def _normalized_footprint_hash(geometry: Any) -> str:
    return hashlib.sha256(normalize_geometry(set_precision(geometry, 1e-9)).wkb).hexdigest()


def _canonical_utm_bounds(grid_col: int, grid_row: int) -> list[int]:
    return [
        grid_col * PARENT_SIDE_METERS,
        grid_row * PARENT_SIDE_METERS,
        (grid_col + 1) * PARENT_SIDE_METERS,
        (grid_row + 1) * PARENT_SIDE_METERS,
    ]


def _canonical_wgs84_geometry(row: Mapping[str, Any]):
    return _wgs84_geometry(
        {
            "grid_epsg": int(row["grid_epsg"]),
            "utm_bounds": _canonical_utm_bounds(int(row["grid_col"]), int(row["grid_row"])),
        }
    )


def _metadata_hash(row: Mapping[str, Any]) -> str:
    metadata = {field: row[field] for field in CANONICAL_METADATA_FIELDS}
    return hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _bounds_match(actual: Iterable[float], expected: Iterable[float]) -> bool:
    return all(
        abs(float(actual_value) - float(expected_value)) <= AUDIT_COORDINATE_TOLERANCE
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )


def _maximum_footprint_coordinate_difference(expected: Any, actual: Any) -> float:
    """Return the largest normalized exterior-coordinate difference in WGS84 degrees."""
    expected = normalize_geometry(set_precision(expected, 1e-9))
    actual = normalize_geometry(set_precision(actual, 1e-9))
    if expected.geom_type != "Polygon" or actual.geom_type != "Polygon":
        return float(max(expected.hausdorff_distance(actual), actual.hausdorff_distance(expected)))
    expected_coordinates = list(expected.exterior.coords)
    actual_coordinates = list(actual.exterior.coords)
    if len(expected_coordinates) != len(actual_coordinates):
        return float(max(expected.hausdorff_distance(actual), actual.hausdorff_distance(expected)))
    return max(
        max(abs(expected_x - actual_x), abs(expected_y - actual_y))
        for (expected_x, expected_y), (actual_x, actual_y) in zip(
            expected_coordinates, actual_coordinates, strict=True
        )
    )


def _audit_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE partition_keys (
            parent_key TEXT PRIMARY KEY,
            all_count INTEGER NOT NULL DEFAULT 0,
            sampled_partition_count INTEGER NOT NULL DEFAULT 0,
            unsampled_partition_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE geometries (
            geometry_id INTEGER PRIMARY KEY,
            grid_epsg INTEGER NOT NULL,
            parent_key TEXT NOT NULL,
            geometry_wkb BLOB NOT NULL
        );
        CREATE TABLE canonical_rows (
            parent_key TEXT PRIMARY KEY,
            metadata_hash TEXT NOT NULL,
            geometry_hash TEXT NOT NULL
        );
        CREATE TABLE sampled_registry_keys (
            parent_key TEXT PRIMARY KEY
        );
        CREATE VIRTUAL TABLE geometry_bounds USING rtree(
            geometry_id, minx, maxx, miny, maxy
        );
        """)
    return connection


def _record_partition_key(
    connection: sqlite3.Connection, parent_key_value: str, partition: str
) -> None:
    field = {
        "all": "all_count",
        "sampled": "sampled_partition_count",
        "unsampled": "unsampled_partition_count",
    }[partition]
    connection.execute(
        f"""
        INSERT INTO partition_keys(parent_key, {field}) VALUES (?, 1)
        ON CONFLICT(parent_key) DO UPDATE SET {field} = {field} + 1
        """,
        (parent_key_value,),
    )


def _audit_overlap(
    connection: sqlite3.Connection,
    geometry: Any,
    grid_epsg: int,
    parent_key_value: str,
    to_equal_area: Transformer,
    *,
    check_overlap: bool = True,
) -> tuple[int, int, float]:
    """Compare one footprint with prior streamed footprints retained on disk."""
    minx, miny, maxx, maxy = geometry.bounds
    same_zone_positive_overlap_count = 0
    cross_zone_overlap_violation_count = 0
    max_cross_zone_overlap_fraction = 0.0
    if check_overlap:
        candidates = list(
            connection.execute(
                """
            SELECT geometries.grid_epsg, geometries.geometry_wkb
            FROM geometry_bounds
            JOIN geometries USING (geometry_id)
            WHERE minx < ? AND maxx > ? AND miny < ? AND maxy > ?
            """,
                (maxx, minx, maxy, miny),
            )
        )
        if candidates:
            projected_geometry = transform_geometry(to_equal_area.transform, geometry)
            for candidate_epsg, candidate_wkb in candidates:
                candidate_geometry = from_wkb(candidate_wkb)
                candidate_projected = transform_geometry(
                    to_equal_area.transform, candidate_geometry
                )
                overlap_area = projected_geometry.intersection(candidate_projected).area
                if overlap_area <= 0:
                    continue
                if int(candidate_epsg) == grid_epsg:
                    same_zone_positive_overlap_count += 1
                    continue
                overlap_fraction = overlap_area / min(
                    projected_geometry.area, candidate_projected.area
                )
                max_cross_zone_overlap_fraction = max(
                    max_cross_zone_overlap_fraction, overlap_fraction
                )
                if overlap_fraction > CROSS_ZONE_OVERLAP_FRACTION:
                    cross_zone_overlap_violation_count += 1
    cursor = connection.execute(
        "INSERT INTO geometries(grid_epsg, parent_key, geometry_wkb) VALUES (?, ?, ?)",
        (grid_epsg, parent_key_value, geometry.wkb),
    )
    geometry_id = int(cursor.lastrowid)
    connection.execute(
        "INSERT INTO geometry_bounds VALUES (?, ?, ?, ?, ?)",
        (geometry_id, minx, maxx, miny, maxy),
    )
    return (
        same_zone_positive_overlap_count,
        cross_zone_overlap_violation_count,
        max_cross_zone_overlap_fraction,
    )


def assess_utm_seam_overlap_policy(
    *,
    overlap_pair_count: int,
    overlap_area_m2: float,
    total_parent_count: int,
    non_adjacent_pair_count: int,
    owner_order_mismatch_count: int,
    off_seam_pair_count: int,
    max_pair_overlap_fraction: float,
    maximum_global_duplicate_fraction: float = UTM_SEAM_GLOBAL_DUPLICATE_FRACTION,
) -> dict[str, Any]:
    """Assess expected overlap where independent UTM grids meet at owner-zone seams."""
    integer_values = (
        overlap_pair_count,
        total_parent_count,
        non_adjacent_pair_count,
        owner_order_mismatch_count,
        off_seam_pair_count,
    )
    if any(value < 0 for value in integer_values) or total_parent_count == 0:
        raise ValueError("UTM seam counts must be non-negative and total_parent_count positive")
    if overlap_area_m2 < 0 or not 0 <= max_pair_overlap_fraction <= 1:
        raise ValueError("UTM seam area and pair-overlap fraction are invalid")
    if not 0 < maximum_global_duplicate_fraction < 1:
        raise ValueError("maximum_global_duplicate_fraction must be between zero and one")

    global_fraction = overlap_area_m2 / (total_parent_count * PARENT_SIDE_METERS**2)
    passed = (
        non_adjacent_pair_count == 0
        and owner_order_mismatch_count == 0
        and off_seam_pair_count == 0
        and global_fraction <= maximum_global_duplicate_fraction
    )
    return {
        "schema_version": "china_full_1280m_utm_seam_audit_v1",
        "policy_version": UTM_SEAM_POLICY_VERSION,
        "overlap_pair_count": overlap_pair_count,
        "overlap_area_m2": overlap_area_m2,
        "total_parent_count": total_parent_count,
        "global_duplicate_area_fraction": global_fraction,
        "maximum_global_duplicate_area_fraction": maximum_global_duplicate_fraction,
        "non_adjacent_pair_count": non_adjacent_pair_count,
        "owner_order_mismatch_count": owner_order_mismatch_count,
        "off_seam_pair_count": off_seam_pair_count,
        "max_pair_overlap_fraction": max_pair_overlap_fraction,
        "passed": passed,
    }


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} 必须是 JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_utm_seam_audit_to_package(
    output_root: str | Path,
    seam_path: str | Path,
    base_audit: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify that the seam decision is an immutable member of this exact grid package."""
    output_root = Path(output_root).resolve()
    manifest_path = output_root / GRID_PACKAGE_MANIFEST
    checksum_path = output_root / GRID_PACKAGE_CHECKSUMS
    if not manifest_path.is_file():
        raise ValueError(f"grid package manifest 不存在: {manifest_path}")
    if not checksum_path.is_file():
        raise ValueError(f"grid package checksum manifest 不存在: {checksum_path}")
    if (
        not isinstance(expected_manifest_sha256, str)
        or not all(
            character in "0123456789abcdef" for character in expected_manifest_sha256.lower()
        )
        or len(expected_manifest_sha256) != 64
    ):
        raise ValueError("package manifest SHA-256 必须是 64 位十六进制字符串")
    actual_manifest_sha256 = _file_sha256(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256.lower():
        raise ValueError("grid package manifest SHA-256 与外部信任锚不一致")

    manifest = _read_json_object(manifest_path, "grid package manifest")
    if manifest.get("schema_version") != "china_full_1280m_grid_package_v1":
        raise ValueError("grid package manifest schema_version 不受支持")
    audits = manifest.get("audits")
    if not isinstance(audits, Mapping):
        raise ValueError("grid package manifest 缺少 audits object")

    def resolve_member(field: str) -> tuple[str, Path]:
        relative = audits.get(field)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"grid package manifest 缺少 audits.{field}")
        candidate = (output_root / relative).resolve()
        try:
            candidate.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(f"audits.{field} 逃逸 grid package 根目录") from exc
        if not candidate.is_file():
            raise ValueError(f"grid package audit 文件不存在: {candidate}")
        return relative, candidate

    seam_relative, expected_seam_path = resolve_member("utm_seam")
    membership_relative, membership_path = resolve_member("final_membership")
    legacy_relative, legacy_path = resolve_member("legacy_pair_threshold")
    if Path(seam_path).resolve() != expected_seam_path:
        raise ValueError("UTM seam audit 未绑定到 grid package manifest 指定路径")
    seam_audit = _read_json_object(expected_seam_path, "UTM seam audit")
    membership_audit = _read_json_object(membership_path, "membership audit")
    legacy_audit = _read_json_object(legacy_path, "legacy pair-threshold audit")

    if seam_audit.get("schema_version") != "china_full_1280m_utm_seam_audit_v1":
        raise ValueError("UTM seam audit schema_version 不受支持")
    if seam_audit.get("policy_version") != UTM_SEAM_POLICY_VERSION:
        raise ValueError("UTM seam audit policy_version 不受支持")
    if seam_audit.get("passed") is not True:
        raise ValueError("UTM seam audit did not pass")
    if manifest.get("membership_audit") != membership_audit:
        raise ValueError("membership audit 与 grid package manifest 内嵌副本不一致")
    if membership_audit.get("cross_zone_seam_audit") != seam_audit:
        raise ValueError("UTM seam audit 与 membership audit 内嵌副本不一致")
    if legacy_audit.get("schema_version") != "china_full_1280m_membership_audit_v1":
        raise ValueError("legacy pair-threshold audit schema_version 不受支持")
    frozen_legacy_count = legacy_audit.get("cross_zone_overlap_violation_count")
    if (
        isinstance(frozen_legacy_count, bool)
        or not isinstance(frozen_legacy_count, int)
        or frozen_legacy_count < 0
    ):
        raise ValueError("legacy pair-threshold count 无效")
    if membership_audit.get("legacy_cross_zone_pair_over_1pct_count") != frozen_legacy_count:
        raise ValueError("冻结 membership 与 legacy pair-threshold audit 不一致")
    for field_name, value in legacy_audit.items():
        if field_name in {"passed", "cross_zone_overlap_violation_count"}:
            continue
        if membership_audit.get(field_name) != value:
            raise ValueError(f"冻结 membership 与 legacy audit 不一致: {field_name}")

    checksum_manifest = _read_json_object(checksum_path, "grid package checksum manifest")
    if checksum_manifest.get("schema_version") != "xuannv_package_sha256_v1":
        raise ValueError("grid package checksum schema_version 不受支持")
    entries = checksum_manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("grid package checksum files 必须是 list")
    by_path: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError("grid package checksum entry 无效")
        relative = str(entry["path"])
        if relative in by_path:
            raise ValueError(f"grid package checksum path 重复: {relative}")
        by_path[relative] = entry
    member_paths = (
        GRID_PACKAGE_MANIFEST,
        membership_relative,
        seam_relative,
        legacy_relative,
    )
    verified: dict[str, str] = {}
    for relative in member_paths:
        path = output_root / relative
        entry = by_path.get(relative)
        if entry is None:
            raise ValueError(f"grid package checksum 缺少: {relative}")
        digest = _file_sha256(path)
        if entry.get("sha256") != digest or entry.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"grid package checksum/size 不匹配: {relative}")
        verified[relative] = digest

    if base_audit.get("schema_version") != "china_full_1280m_membership_audit_v1":
        raise ValueError("live membership audit schema_version 不受支持")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("grid package manifest 缺少 counts object")
    for manifest_key, audit_key in (
        ("all", "all_count"),
        ("sampled", "sampled_count"),
        ("unsampled", "unsampled_count"),
    ):
        expected = counts.get(manifest_key)
        if expected != base_audit.get(audit_key) or expected != membership_audit.get(audit_key):
            raise ValueError(f"grid package {manifest_key} count 与 live/stored audit 不一致")
    if seam_audit.get("total_parent_count") != base_audit.get("all_count"):
        raise ValueError("UTM seam total_parent_count 与 live grid audit 不一致")

    ignored_live_fields = {"passed", "cross_zone_overlap_violation_count"}
    for field_name, value in base_audit.items():
        if field_name in ignored_live_fields:
            continue
        if membership_audit.get(field_name) != value:
            raise ValueError(f"live membership audit 与冻结 audit 不一致: {field_name}")
    live_cross_zone_count = int(base_audit.get("cross_zone_overlap_violation_count", 0))

    binding = {
        "schema_version": "china_full_1280m_grid_package_binding_v1",
        "manifest_sha256": actual_manifest_sha256,
        "verified_member_sha256": verified,
        "frozen_legacy_pair_threshold_count": frozen_legacy_count,
        "live_pair_threshold_count": live_cross_zone_count,
        "passed": True,
    }
    return seam_audit, binding


def reconcile_utm_seam_audit(
    base_audit: Mapping[str, Any], seam_audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace the obsolete per-pair 1% gate only when every other strict gate passed."""
    if seam_audit.get("passed") is not True:
        raise ValueError("UTM seam audit did not pass")

    blocking_count_fields = (
        "missing_sampled_count",
        "duplicate_parent_key_count",
        "sampled_unsampled_intersection_count",
        "sampled_flag_mismatch_count",
        "partition_mismatch_count",
        "exact_partition_membership_mismatch_count",
        "child_partition_unknown_parent_key_count",
        "child_partition_metadata_mismatch_count",
        "child_partition_geometry_mismatch_count",
        "footprint_coordinate_mismatch_count",
        "stored_utm_bounds_mismatch_count",
        "stored_wgs84_bounds_mismatch_count",
        "invalid_geometry_count",
        "owner_zone_mismatch_count",
        "same_zone_positive_overlap_count",
    )
    blocking = [field for field in blocking_count_fields if int(base_audit.get(field, 0)) != 0]
    if base_audit.get("missing") or base_audit.get("duplicate_atlas_keys"):
        blocking.append("membership_examples")
    if base_audit.get("duplicate_sampled_keys"):
        blocking.append("duplicate_sampled_keys")
    hash_mismatches = base_audit.get("hash_mismatches", {})
    if not isinstance(hash_mismatches, Mapping) or any(
        int(value) != 0 for value in hash_mismatches.values()
    ):
        blocking.append("hash_mismatches")
    if blocking:
        raise ValueError(f"base audit has other blocking failures: {sorted(set(blocking))}")

    reconciled = dict(base_audit)
    reconciled["live_cross_zone_pair_over_1pct_count"] = int(
        reconciled.get("cross_zone_overlap_violation_count", 0)
    )
    reconciled["cross_zone_overlap_violation_count"] = 0
    reconciled["cross_zone_policy_version"] = seam_audit["policy_version"]
    reconciled["cross_zone_seam_audit"] = dict(seam_audit)
    reconciled["passed"] = True
    return reconciled


def audit_grid_package(
    output_root: str | Path,
    sampled_registry: Iterable[Mapping[str, Any]],
    *,
    batch_size: int = AUDIT_BATCH_SIZE,
) -> dict[str, Any]:
    """Stream a partitioned parent atlas and return reproducible membership/geometry audits."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_root = Path(output_root)
    all_paths = _parquet_paths(output_root, "all")
    if not all_paths:
        raise ValueError(f"no all-partition GeoParquet files found below {output_root}")

    registry_records = list(sampled_registry)
    registry_keys = [sampled_registry_key(record) for record in registry_records]
    registry_audit = audit_sample_membership([], registry_keys)
    registry_key_counts = Counter(registry_keys)
    registry_by_key = {
        sampled_registry_key(record): record
        for record in registry_records
        if registry_key_counts[sampled_registry_key(record)] == 1
    }
    sampled_key_set = set(registry_by_key)

    counters = Counter()
    max_footprint_coordinate_difference = 0.0
    to_equal_area = Transformer.from_crs(4326, 6933, always_xy=True)
    temporary_parent = output_root.parent if output_root.parent.exists() else None
    with tempfile.TemporaryDirectory(
        prefix=".china-full-grid-audit-", dir=temporary_parent
    ) as temporary_dir:
        connection = _audit_database(Path(temporary_dir) / "audit.sqlite")
        try:
            connection.executemany(
                "INSERT INTO sampled_registry_keys(parent_key) VALUES (?)",
                ((key,) for key in sampled_key_set),
            )
            all_columns = [*CANONICAL_METADATA_FIELDS, "geometry"]
            for audited in _iter_audited_parquet_rows(all_paths, all_columns, batch_size):
                row = audited["row"]
                parent_key_value = str(row["parent_key"])
                grid_epsg = int(row["grid_epsg"])
                geometry = audited["geometry"]
                canonical_bounds = _canonical_utm_bounds(int(row["grid_col"]), int(row["grid_row"]))
                canonical_geometry = audited["canonical_geometry"]
                canonical_footprint_hash = audited["canonical_footprint_hash"]
                actual_footprint_hash = audited["actual_footprint_hash"]
                _record_partition_key(connection, parent_key_value, "all")
                counters["all_count"] += 1
                if bool(row["sampled"]):
                    counters["sampled_count"] += 1
                else:
                    counters["unsampled_count"] += 1
                if (parent_key_value in sampled_key_set) != bool(row["sampled"]):
                    counters["sampled_flag_mismatch_count"] += 1
                if utm_owner_epsg(float(row["longitude"]), float(row["latitude"])) != grid_epsg:
                    counters["owner_zone_mismatch_count"] += 1

                expected_identity = (
                    f"{row['atlas_version']}:{grid_epsg}:"
                    f"{int(row['grid_col'])}:{int(row['grid_row'])}"
                )
                expected_identity_hash = hashlib.sha256(
                    expected_identity.encode("utf-8")
                ).hexdigest()
                if row["identity_hash"] != expected_identity_hash:
                    counters["identity_hash_mismatch_count"] += 1
                if list(row["utm_bounds"]) != canonical_bounds:
                    counters["stored_utm_bounds_mismatch_count"] += 1
                if not _bounds_match(row["wgs84_bounds"], canonical_geometry.bounds):
                    counters["stored_wgs84_bounds_mismatch_count"] += 1
                if row["footprint_hash"] != canonical_footprint_hash:
                    counters["footprint_hash_mismatch_count"] += 1
                registry_record = registry_by_key.get(parent_key_value)
                if registry_record is not None:
                    registry_footprint_hash = registry_record.get(
                        "canonical_wgs84_footprint_hash", registry_record.get("footprint_hash")
                    )
                    if (
                        registry_footprint_hash is not None
                        and str(registry_footprint_hash) != actual_footprint_hash
                    ):
                        counters["sampled_registry_footprint_hash_mismatch_count"] += 1

                coordinate_difference = audited["coordinate_difference"]
                max_footprint_coordinate_difference = max(
                    max_footprint_coordinate_difference, coordinate_difference
                )
                if coordinate_difference > AUDIT_COORDINATE_TOLERANCE:
                    counters["footprint_coordinate_mismatch_count"] += 1
                if coordinate_difference != 0.0:
                    projected_geometry = transform_geometry(
                        _transformer_from_wgs84(grid_epsg).transform, geometry
                    )
                    minx, miny, maxx, maxy = projected_geometry.bounds
                    if (
                        abs((maxx - minx) - PARENT_SIDE_METERS) > AUDIT_DIMENSION_TOLERANCE_M
                        or abs((maxy - miny) - PARENT_SIDE_METERS) > AUDIT_DIMENSION_TOLERANCE_M
                        or abs(projected_geometry.area - PARENT_SIDE_METERS**2)
                        > AUDIT_AREA_TOLERANCE_M2
                    ):
                        counters["invalid_geometry_count"] += 1
                same_zone_count, cross_zone_count, overlap_fraction = _audit_overlap(
                    connection,
                    geometry,
                    grid_epsg,
                    parent_key_value,
                    to_equal_area,
                    check_overlap=True,
                )
                counters["same_zone_positive_overlap_count"] += same_zone_count
                counters["cross_zone_overlap_violation_count"] += cross_zone_count
                counters["max_cross_zone_overlap_fraction"] = max(
                    counters["max_cross_zone_overlap_fraction"], overlap_fraction
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO canonical_rows(parent_key, metadata_hash, geometry_hash)
                    VALUES (?, ?, ?)
                    """,
                    (parent_key_value, _metadata_hash(row), actual_footprint_hash),
                )
            connection.commit()

            for partition in ("sampled", "unsampled"):
                for hashed in _iter_hashed_parquet_rows(
                    _parquet_paths(output_root, partition), all_columns, batch_size
                ):
                    row = hashed["row"]
                    parent_key_value = str(row["parent_key"])
                    _record_partition_key(connection, parent_key_value, partition)
                    canonical_row = connection.execute(
                        """
                        SELECT metadata_hash, geometry_hash FROM canonical_rows
                        WHERE parent_key = ?
                        """,
                        (parent_key_value,),
                    ).fetchone()
                    if canonical_row is None:
                        counters["child_partition_unknown_parent_key_count"] += 1
                        continue
                    if _metadata_hash(row) != canonical_row[0]:
                        counters["child_partition_metadata_mismatch_count"] += 1
                    if hashed["geometry_hash"] != canonical_row[1]:
                        counters["child_partition_geometry_mismatch_count"] += 1
            connection.commit()

            duplicate_parent_keys = [
                row[0]
                for row in connection.execute(
                    "SELECT parent_key FROM partition_keys WHERE all_count > 1 ORDER BY parent_key "
                    "LIMIT ?",
                    (AUDIT_EXAMPLE_LIMIT,),
                )
            ]
            counters["duplicate_parent_key_count"] = connection.execute(
                "SELECT COUNT(*) FROM partition_keys WHERE all_count > 1"
            ).fetchone()[0]
            counters["partition_mismatch_count"] = connection.execute("""
                SELECT COUNT(*) FROM partition_keys
                WHERE all_count != 1
                   OR sampled_partition_count + unsampled_partition_count != all_count
                """).fetchone()[0]
            counters["sampled_unsampled_intersection_count"] = connection.execute("""
                SELECT COUNT(*) FROM partition_keys
                WHERE sampled_partition_count > 0 AND unsampled_partition_count > 0
                """).fetchone()[0]
            counters["exact_partition_membership_mismatch_count"] = connection.execute("""
                SELECT COUNT(*)
                FROM partition_keys
                LEFT JOIN sampled_registry_keys USING (parent_key)
                WHERE all_count != 1
                   OR (
                        sampled_registry_keys.parent_key IS NOT NULL
                        AND (sampled_partition_count != 1 OR unsampled_partition_count != 0)
                   )
                   OR (
                        sampled_registry_keys.parent_key IS NULL
                        AND (sampled_partition_count != 0 OR unsampled_partition_count != 1)
                   )
                """).fetchone()[0]
            missing = []
            matched = 0
            for key in sampled_key_set:
                all_count = connection.execute(
                    "SELECT all_count FROM partition_keys WHERE parent_key = ?", (key,)
                ).fetchone()
                if all_count is None or all_count[0] == 0:
                    missing.append(key)
                elif all_count[0] == 1:
                    matched += 1
            missing.sort()
        finally:
            connection.close()

    hash_mismatches = {
        "identity_hash": counters["identity_hash_mismatch_count"],
        "footprint_hash": counters["footprint_hash_mismatch_count"],
        "sampled_registry_footprint_hash": counters[
            "sampled_registry_footprint_hash_mismatch_count"
        ],
    }
    passed = not any(
        (
            missing,
            registry_audit["duplicate_sampled_keys"],
            counters["duplicate_parent_key_count"],
            counters["sampled_flag_mismatch_count"],
            counters["partition_mismatch_count"],
            counters["sampled_unsampled_intersection_count"],
            counters["exact_partition_membership_mismatch_count"],
            counters["child_partition_unknown_parent_key_count"],
            counters["child_partition_metadata_mismatch_count"],
            counters["child_partition_geometry_mismatch_count"],
            counters["identity_hash_mismatch_count"],
            counters["footprint_hash_mismatch_count"],
            counters["sampled_registry_footprint_hash_mismatch_count"],
            counters["stored_utm_bounds_mismatch_count"],
            counters["stored_wgs84_bounds_mismatch_count"],
            counters["footprint_coordinate_mismatch_count"],
            counters["invalid_geometry_count"],
            counters["owner_zone_mismatch_count"],
            counters["same_zone_positive_overlap_count"],
            counters["cross_zone_overlap_violation_count"],
        )
    )
    return {
        "schema_version": "china_full_1280m_membership_audit_v1",
        "all_count": counters["all_count"],
        "sampled_count": counters["sampled_count"],
        "unsampled_count": counters["unsampled_count"],
        "matched": matched,
        "missing": missing,
        "missing_sampled_count": len(missing),
        "duplicate_atlas_keys": duplicate_parent_keys,
        "duplicate_parent_key_count": counters["duplicate_parent_key_count"],
        "duplicate_sampled_keys": registry_audit["duplicate_sampled_keys"],
        "sampled_unsampled_intersection_count": counters["sampled_unsampled_intersection_count"],
        "sampled_flag_mismatch_count": counters["sampled_flag_mismatch_count"],
        "partition_mismatch_count": counters["partition_mismatch_count"],
        "exact_partition_membership_mismatch_count": counters[
            "exact_partition_membership_mismatch_count"
        ],
        "child_partition_unknown_parent_key_count": counters[
            "child_partition_unknown_parent_key_count"
        ],
        "child_partition_metadata_mismatch_count": counters[
            "child_partition_metadata_mismatch_count"
        ],
        "child_partition_geometry_mismatch_count": counters[
            "child_partition_geometry_mismatch_count"
        ],
        "hash_mismatches": hash_mismatches,
        "max_footprint_coordinate_difference": max_footprint_coordinate_difference,
        "footprint_coordinate_mismatch_count": counters["footprint_coordinate_mismatch_count"],
        "stored_utm_bounds_mismatch_count": counters["stored_utm_bounds_mismatch_count"],
        "stored_wgs84_bounds_mismatch_count": counters["stored_wgs84_bounds_mismatch_count"],
        "invalid_geometry_count": counters["invalid_geometry_count"],
        "owner_zone_mismatch_count": counters["owner_zone_mismatch_count"],
        "same_zone_positive_overlap_count": counters["same_zone_positive_overlap_count"],
        "cross_zone_overlap_violation_count": counters["cross_zone_overlap_violation_count"],
        "max_cross_zone_overlap_fraction": counters["max_cross_zone_overlap_fraction"],
        "passed": passed,
    }


def write_grid_package_audit(audit: Mapping[str, Any], output_path: str | Path) -> Path:
    """Persist an audit result as stable, human-readable JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
