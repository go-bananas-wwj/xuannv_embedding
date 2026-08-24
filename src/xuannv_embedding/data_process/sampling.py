"""Materialize the reviewed China sampling design as auditable GIS files.

The current national atlas has not yet been enriched with nationwide semantic
rasters or quarterly source-quality summaries.  This tool therefore emits a
fixed 62,000-location *design candidate* registry.  The final 1,500 locations
are an explicitly named semantic-audit reserve and must be re-ranked after the
WorldCover/OSM/DEM/source-quality atlas is available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from shapely import normalize as normalize_geometry
from shapely import set_precision
from shapely.geometry import Point, box
from shapely.ops import transform as transform_geometry
from shapely.prepared import prep

PATCH_SIDE_METERS = 1280
MACRO_SIDE_PATCHES = 10
COAST_MIN_FRACTION = 0.001
SCHEMA_VERSION = "xuannv_china_quarterly_sampling_candidate_v1"
TIER_TO_LAYER = {
    "base_spatial": "base_expected_1pct",
    "spatial_stratified_supplement": "utm_spatial_balance_supplement",
    "spatial_stratified_supplement_fallback": "utm_spatial_balance_supplement",
    "coastal_ocean_adjacent_supplement": "coastal_supplement",
    "semantic_audit_reserve": "semantic_and_difficult_supplement",
}


@lru_cache(maxsize=None)
def _transformer(source_epsg: int, target_epsg: int) -> Transformer:
    return Transformer.from_crs(source_epsg, target_epsg, always_xy=True)


def stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def utm_owner_zone(longitude: float) -> int:
    """Return the unique half-open six-degree UTM zone for a longitude."""
    return max(1, min(60, int(math.floor((float(longitude) + 180.0) / 6.0)) + 1))


def utm_owner_epsg(longitude: float, latitude: float) -> int:
    return (32600 if float(latitude) >= 0 else 32700) + utm_owner_zone(longitude)


def patch_geometry_utm(record: dict[str, Any]) -> Any:
    left = int(record["grid_col"]) * PATCH_SIDE_METERS
    bottom = int(record["grid_row"]) * PATCH_SIDE_METERS
    return box(left, bottom, left + PATCH_SIDE_METERS, bottom + PATCH_SIDE_METERS)


def patch_geometry_wgs84(record: dict[str, Any]) -> Any:
    transformer = _transformer(int(record["grid_epsg"]), 4326)
    return transform_geometry(transformer.transform, patch_geometry_utm(record))


def _candidate_order(seed: int, macro_id: str) -> list[tuple[int, int]]:
    cells = [
        (patch_col, patch_row)
        for patch_row in range(MACRO_SIDE_PATCHES)
        for patch_col in range(MACRO_SIDE_PATCHES)
    ]
    return sorted(
        cells,
        key=lambda cell: stable_hash(seed, f"candidate-cell:{macro_id}:{cell[0]}:{cell[1]}"),
    )


def _candidate_from_macro(
    macro: dict[str, Any],
    country_geometry: Any,
    seed: int,
    used_patch_ids: set[str],
    used_identities: set[tuple[int, int, int]],
    candidate_predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any] | None:
    epsg = int(macro["grid_epsg"])
    transformer = _transformer(epsg, 4326)
    left, bottom, _, _ = [float(value) for value in macro["utm_bounds"]]
    macro_id = str(macro["macro_id"])
    for patch_col, patch_row in _candidate_order(seed, macro_id):
        grid_col = int(macro["macro_col"]) * MACRO_SIDE_PATCHES + patch_col
        grid_row = int(macro["macro_row"]) * MACRO_SIDE_PATCHES + patch_row
        patch_id = f"candidate_{macro_id}_c{patch_col}_r{patch_row}"
        identity = (epsg, grid_col, grid_row)
        if patch_id in used_patch_ids or identity in used_identities:
            continue
        longitude, latitude = transformer.transform(
            left + (patch_col + 0.5) * PATCH_SIDE_METERS,
            bottom + (patch_row + 0.5) * PATCH_SIDE_METERS,
        )
        if epsg != utm_owner_epsg(longitude, latitude):
            continue
        if not country_geometry.covers(Point(longitude, latitude)):
            continue
        candidate = {
            "schema_version": SCHEMA_VERSION,
            "patch_id": patch_id,
            "macro_id": macro_id,
            "grid_id": str(macro["grid_id"]),
            "grid_epsg": epsg,
            "grid_col": grid_col,
            "grid_row": grid_row,
            "longitude": round(longitude, 7),
            "latitude": round(latitude, 7),
            "admin1": str(macro.get("admin1", "unknown")),
            "sampling_seed": seed,
        }
        if candidate_predicate is not None and not candidate_predicate(candidate):
            continue
        return candidate
    return None


def choose_replacements(
    inventory: list[dict[str, Any]],
    country_geometry: Any,
    existing: list[dict[str, Any]],
    count: int,
    seed: int,
    sampling_layer: str,
    primary_reasons: list[str],
    forbid_supplement_macros: set[str],
) -> list[dict[str, Any]]:
    """Choose deterministic, owner-zone-valid reserve candidates."""
    if count != len(primary_reasons):
        raise ValueError("count and primary_reasons length must match")
    used_patch_ids = {str(item["patch_id"]) for item in existing}
    used_identities = {
        (
            int(item["grid_epsg"]),
            int(item["grid_col"]),
            int(item["grid_row"]),
        )
        for item in existing
    }
    ordered = sorted(
        inventory,
        key=lambda item: stable_hash(seed, f"{sampling_layer}:macro:{item['macro_id']}"),
    )
    selected: list[dict[str, Any]] = []
    used_macros = set(forbid_supplement_macros)
    for macro in ordered:
        if len(selected) == count:
            break
        macro_id = str(macro["macro_id"])
        if macro_id in used_macros:
            continue
        candidate = _candidate_from_macro(
            macro,
            country_geometry,
            seed + len(selected),
            used_patch_ids,
            used_identities,
        )
        if candidate is None:
            continue
        reason = primary_reasons[len(selected)]
        is_semantic = sampling_layer == "semantic_and_difficult_supplement"
        candidate.update(
            {
                "status": "design_candidate_not_quality_eligible",
                "sampling_tier": (
                    "semantic_audit_reserve" if is_semantic else "deterministic_replenishment"
                ),
                "sampling_layer": sampling_layer,
                "sampling_primary_reason": reason,
                "sampling_reasons": (
                    [
                        f"provisional_quota:{reason}",
                        "requires_semantic_atlas_reranking",
                    ]
                    if is_semantic
                    else [reason]
                ),
                "semantic_verification_status": ("pending" if is_semantic else "not_applicable"),
            }
        )
        selected.append(candidate)
        used_patch_ids.add(str(candidate["patch_id"]))
        used_identities.add(
            (
                int(candidate["grid_epsg"]),
                int(candidate["grid_col"]),
                int(candidate["grid_row"]),
            )
        )
        used_macros.add(macro_id)
    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} reserve points, expected {count}")
    return selected


def allocate_sqrt_by_group(
    total: int,
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    weights = {name: math.sqrt(len(items)) for name, items in groups.items() if items}
    denominator = sum(weights.values())
    raw = {name: total * weight / denominator for name, weight in weights.items()}
    allocated = {name: int(math.floor(value)) for name, value in raw.items()}
    residual = total - sum(allocated.values())
    for name in sorted(
        raw,
        key=lambda key: (raw[key] - allocated[key], key),
        reverse=True,
    )[:residual]:
        allocated[name] += 1
    return allocated


def exact_owner_candidate_counts_by_grid(
    inventory: list[dict[str, Any]],
    country_geometry: Any,
) -> dict[str, int]:
    """Count all boundary-valid owner-zone patch centers in each UTM grid."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for macro in inventory:
        groups.setdefault(str(macro["grid_id"]), []).append(macro)
    counts: dict[str, int] = {}
    offsets = (np.arange(MACRO_SIDE_PATCHES, dtype=np.float64) + 0.5) * PATCH_SIDE_METERS
    for grid_id, macros in sorted(groups.items()):
        epsg = int(macros[0]["grid_epsg"])
        transformer = _transformer(epsg, 4326)
        total = 0
        for start in range(0, len(macros), 1000):
            chunk = macros[start : start + 1000]
            left = np.asarray([item["utm_bounds"][0] for item in chunk], dtype=np.float64)
            bottom = np.asarray([item["utm_bounds"][1] for item in chunk], dtype=np.float64)
            x = (left[:, None, None] + offsets[None, None, :]).repeat(MACRO_SIDE_PATCHES, axis=1)
            y = (bottom[:, None, None] + offsets[None, :, None]).repeat(MACRO_SIDE_PATCHES, axis=2)
            longitude, latitude = transformer.transform(x.ravel(), y.ravel())
            longitude = np.asarray(longitude)
            latitude = np.asarray(latitude)
            owner_zone = np.floor((longitude + 180.0) / 6.0).astype(np.int16) + 1
            expected_epsg = np.where(latitude >= 0, 32600, 32700) + np.clip(owner_zone, 1, 60)
            inside = shapely.intersects_xy(country_geometry, longitude, latitude)
            total += int(np.count_nonzero(inside & (expected_epsg == epsg)))
        counts[grid_id] = total
    return counts


def allocate_sqrt_by_counts(total: int, counts: dict[str, int]) -> dict[str, int]:
    weights = {name: math.sqrt(value) for name, value in counts.items() if value > 0}
    denominator = sum(weights.values())
    raw = {name: total * weight / denominator for name, weight in weights.items()}
    allocated = {name: int(math.floor(value)) for name, value in raw.items()}
    residual = total - sum(allocated.values())
    for name in sorted(
        raw,
        key=lambda key: (raw[key] - allocated[key], key),
        reverse=True,
    )[:residual]:
        allocated[name] += 1
    return allocated


def choose_spatial_balance_candidates(
    inventory: list[dict[str, Any]],
    country_geometry: Any,
    existing: list[dict[str, Any]],
    count: int,
    seed: int,
    forbid_supplement_macros: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for macro in inventory:
        groups.setdefault(str(macro["grid_id"]), []).append(macro)
    exact_candidate_counts = exact_owner_candidate_counts_by_grid(inventory, country_geometry)
    allocations = allocate_sqrt_by_counts(count, exact_candidate_counts)
    selected: list[dict[str, Any]] = []
    used_macros = set(forbid_supplement_macros)
    for group in sorted(allocations):
        quota = allocations[group]
        group_selected = choose_replacements(
            inventory=groups[group],
            country_geometry=country_geometry,
            existing=existing + selected,
            count=quota,
            seed=seed + int(group.removeprefix("utm").removesuffix("n")),
            sampling_layer="utm_spatial_balance_supplement",
            primary_reasons=[f"sqrt_utm_balance:{group}"] * quota,
            forbid_supplement_macros=used_macros,
        )
        for item in group_selected:
            item["sampling_tier"] = "spatial_stratified_supplement"
            item["sampling_layer"] = "utm_spatial_balance_supplement"
        selected.extend(group_selected)
        used_macros.update(str(item["macro_id"]) for item in group_selected)
    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} spatial points, expected {count}")
    return selected, allocations, exact_candidate_counts


def choose_coastal_candidates(
    inventory: list[dict[str, Any]],
    country_geometry: Any,
    exclusive_land_geometry: Any,
    exclusive_ocean_geometry: Any,
    existing: list[dict[str, Any]],
    count: int,
    seed: int,
    forbid_supplement_macros: set[str],
) -> list[dict[str, Any]]:
    used_patch_ids = {str(item["patch_id"]) for item in existing}
    used_identities = {
        (
            int(item["grid_epsg"]),
            int(item["grid_col"]),
            int(item["grid_row"]),
        )
        for item in existing
    }
    prepared_ocean = prep(exclusive_ocean_geometry)
    to_equal_area = _transformer(4326, 6933)
    exclusive_land_equal_area = transform_geometry(to_equal_area.transform, exclusive_land_geometry)
    prepared_land_equal_area = prep(exclusive_land_equal_area)
    prepared_ocean_wgs84 = prep(exclusive_ocean_geometry)
    ordered = sorted(
        (
            item
            for item in inventory
            if prepared_ocean.intersects(box(*[float(value) for value in item["wgs84_bounds"]]))
            and str(item["macro_id"]) not in forbid_supplement_macros
        ),
        key=lambda item: stable_hash(seed, f"coastal:macro:{item['macro_id']}"),
    )
    selected: list[dict[str, Any]] = []
    used_macros = set(forbid_supplement_macros)

    def is_land_ocean_intersection(candidate: dict[str, Any]) -> bool:
        footprint = patch_geometry_wgs84(candidate)
        if not prepared_ocean_wgs84.intersects(footprint):
            return False
        footprint_equal_area = transform_geometry(to_equal_area.transform, footprint)
        if not prepared_land_equal_area.intersects(footprint_equal_area):
            return False
        minimum_area = footprint_equal_area.area * COAST_MIN_FRACTION
        land_area = footprint_equal_area.intersection(exclusive_land_equal_area).area
        outside_area = footprint_equal_area.area - land_area
        return land_area >= minimum_area and outside_area >= minimum_area

    for macro in ordered:
        if len(selected) == count:
            break
        macro_id = str(macro["macro_id"])
        if macro_id in used_macros:
            continue
        candidate = _candidate_from_macro(
            macro,
            country_geometry,
            seed + len(selected),
            used_patch_ids,
            used_identities,
            candidate_predicate=is_land_ocean_intersection,
        )
        if candidate is None:
            continue
        candidate.update(
            {
                "status": "design_candidate_not_quality_eligible",
                "sampling_tier": "coastal_ocean_adjacent_supplement",
                "sampling_layer": "coastal_supplement",
                "sampling_primary_reason": "verified_boundary_crossing_and_ocean",
                "sampling_reasons": [
                    "country_inside_and_outside_each_at_least_0.1pct",
                    "intersects_independent_ocean",
                ],
                "coastal_geometry_status": "verified_boundary_crossing_ocean",
            }
        )
        selected.append(candidate)
        used_patch_ids.add(str(candidate["patch_id"]))
        used_identities.add(
            (
                int(candidate["grid_epsg"]),
                int(candidate["grid_col"]),
                int(candidate["grid_row"]),
            )
        )
        used_macros.add(macro_id)
    strict_count = len(selected)
    if strict_count < count:
        country_boundary = country_geometry.boundary
        near_ocean = prep(exclusive_ocean_geometry.buffer(0.05))
        near_boundary = prep(country_boundary.buffer(0.05))
        fallback_macros = sorted(
            (
                item
                for item in inventory
                if str(item["macro_id"]) not in used_macros
                and near_ocean.intersects(box(*[float(value) for value in item["wgs84_bounds"]]))
                and near_boundary.intersects(box(*[float(value) for value in item["wgs84_bounds"]]))
            ),
            key=lambda item: stable_hash(seed, f"near-coast-fallback:{item['macro_id']}"),
        )

        def is_near_coast(candidate: dict[str, Any]) -> bool:
            center = Point(candidate["longitude"], candidate["latitude"])
            return (
                center.distance(exclusive_ocean_geometry) <= 0.035
                and center.distance(country_boundary) <= 0.035
            )

        for macro in fallback_macros:
            if len(selected) == count:
                break
            macro_id = str(macro["macro_id"])
            candidate = _candidate_from_macro(
                macro,
                country_geometry,
                seed + len(selected),
                used_patch_ids,
                used_identities,
                candidate_predicate=is_near_coast,
            )
            if candidate is None:
                continue
            candidate.update(
                {
                    "status": "design_candidate_not_quality_eligible",
                    "sampling_tier": "coastal_ocean_adjacent_supplement",
                    "sampling_layer": "coastal_supplement",
                    "sampling_primary_reason": "near_coast_requires_detailed_review",
                    "sampling_reasons": [
                        "center_within_0.035_degree_of_boundary_and_ocean",
                        "strict_boundary_crossing_not_satisfied",
                    ],
                    "coastal_geometry_status": "near_coast_pending",
                }
            )
            selected.append(candidate)
            used_patch_ids.add(str(candidate["patch_id"]))
            used_identities.add(
                (
                    int(candidate["grid_epsg"]),
                    int(candidate["grid_col"]),
                    int(candidate["grid_row"]),
                )
            )
            used_macros.add(macro_id)
    if len(selected) != count:
        raise RuntimeError(
            f"selected {strict_count} strict and {len(selected) - strict_count} "
            f"near-coast points, expected {count}"
        )
    return selected


def enforce_macro_limits(
    records: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep at most one base and one total supplement per macrocell."""
    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for item in records:
        is_base = str(item["sampling_tier"]) == "base_spatial"
        grouped.setdefault((str(item["macro_id"]), is_base), []).append(item)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for (macro_id, is_base), items in sorted(grouped.items()):
        ordered = sorted(
            items,
            key=lambda item: stable_hash(
                seed,
                f"macro-limit:{macro_id}:{is_base}:{item['patch_id']}",
            ),
        )
        kept.append(ordered[0])
        dropped.extend(ordered[1:])
    return kept, dropped


def _canonicalize_to_owner_grid(
    record: dict[str, Any],
    country_geometry: Any,
) -> dict[str, Any] | None:
    owner_epsg = utm_owner_epsg(record["longitude"], record["latitude"])
    transformer = _transformer(4326, owner_epsg)
    inverse = _transformer(owner_epsg, 4326)
    x, y = transformer.transform(record["longitude"], record["latitude"])
    grid_col = math.floor(x / PATCH_SIDE_METERS)
    grid_row = math.floor(y / PATCH_SIDE_METERS)
    longitude, latitude = inverse.transform(
        (grid_col + 0.5) * PATCH_SIDE_METERS,
        (grid_row + 0.5) * PATCH_SIDE_METERS,
    )
    if owner_epsg != utm_owner_epsg(longitude, latitude):
        return None
    if not country_geometry.covers(Point(longitude, latitude)):
        return None
    zone = owner_epsg % 100
    fixed = dict(record)
    fixed.update(
        {
            "grid_epsg": owner_epsg,
            "grid_id": f"utm{zone}{'n' if owner_epsg < 32700 else 's'}",
            "grid_col": grid_col,
            "grid_row": grid_row,
            "macro_id": (
                f"utm{zone}{'n' if owner_epsg < 32700 else 's'}"
                f"_c{grid_col // MACRO_SIDE_PATCHES}_r{grid_row // MACRO_SIDE_PATCHES}"
            ),
            "patch_id": (f"ownerfix_epsg{owner_epsg}_c{grid_col}_r{grid_row}"),
            "longitude": round(longitude, 7),
            "latitude": round(latitude, 7),
            "owner_zone_repaired": True,
        }
    )
    return fixed


def repair_owner_zones(
    records: list[dict[str, Any]],
    country_geometry: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    repaired: list[dict[str, Any]] = []
    seen: set[str] = set()
    counters = Counter()
    for record in records:
        item = dict(record)
        if int(item["grid_epsg"]) != utm_owner_epsg(item["longitude"], item["latitude"]):
            counters["owner_zone_mismatch"] += 1
            candidate = _canonicalize_to_owner_grid(item, country_geometry)
            if candidate is None:
                counters["dropped_outside_after_repair"] += 1
                continue
            item = candidate
            counters["owner_zone_repaired"] += 1
        if item["patch_id"] in seen:
            counters["dropped_duplicate_patch_id"] += 1
            continue
        seen.add(str(item["patch_id"]))
        repaired.append(item)
    return repaired, dict(counters)


def normalize_record(
    record: dict[str, Any],
    registry_version: str,
) -> dict[str, Any]:
    item = dict(record)
    sampling_layer = item.get(
        "sampling_layer",
        TIER_TO_LAYER.get(str(item.get("sampling_tier")), "unknown"),
    )
    geometry_utm = patch_geometry_utm(item)
    geometry_wgs84 = patch_geometry_wgs84(item)
    canonical_geometry = normalize_geometry(set_precision(geometry_wgs84, 1e-9))
    bounds_utm = [round(value, 3) for value in geometry_utm.bounds]
    bounds_wgs84 = [round(value, 8) for value in geometry_wgs84.bounds]
    identity = f"{item['grid_epsg']}:{item['grid_col']}:{item['grid_row']}:" f"{PATCH_SIDE_METERS}"
    item.update(
        {
            "schema_version": SCHEMA_VERSION,
            "registry_version": registry_version,
            "registry_status": "design_candidate_not_quality_eligible",
            "sampling_layer": sampling_layer,
            "sampling_primary_reason": item.get(
                "sampling_primary_reason", str(item.get("sampling_tier", "unknown"))
            ),
            "sampling_reasons": item.get(
                "sampling_reasons", [str(item.get("sampling_tier", "unknown"))]
            ),
            "sampling_design_type": item.get("sampling_design_type", "fixed_size_design_candidate"),
            "nominal_inclusion_probability": (
                0.01 if sampling_layer == "base_expected_1pct" else None
            ),
            "design_weight_nullable": None,
            "identity_hash": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "canonical_wgs84_footprint_hash": hashlib.sha256(canonical_geometry.wkb).hexdigest(),
            "utm_bounds": bounds_utm,
            "wgs84_bounds": bounds_wgs84,
            "boundary_version": "geoBoundaries-CHN-ADM0-local-202607",
            "ecoregion": item.get("ecoregion", "pending"),
            "source_coverage_by_quarter": "pending",
            "quality_summary_by_quarter": "pending",
            "split": "unassigned",
            "semantic_verification_status": item.get(
                "semantic_verification_status", "not_applicable"
            ),
            "selection_subseed": item.get("selection_subseed", item.get("sampling_seed")),
        }
    )
    return item


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    scalar_fields = [
        "patch_id",
        "identity_hash",
        "grid_id",
        "grid_epsg",
        "grid_row",
        "grid_col",
        "macro_id",
        "longitude",
        "latitude",
        "admin1",
        "sampling_layer",
        "sampling_primary_reason",
        "sampling_seed",
        "registry_status",
        "registry_version",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in scalar_fields})


def _shape_attributes(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "PATCH_ID": [item["patch_id"] for item in records],
        "LAYER": [item["sampling_layer"] for item in records],
        "REASON": [item["sampling_primary_reason"] for item in records],
        "UTM_EPSG": [item["grid_epsg"] for item in records],
        "GRID_COL": [item["grid_col"] for item in records],
        "GRID_ROW": [item["grid_row"] for item in records],
        "ADMIN1": [item["admin1"] for item in records],
        "STATUS": ["not_train_eligible" for _ in records],
        "ID_HASH": [item["identity_hash"] for item in records],
        "FP_HASH": [item["canonical_wgs84_footprint_hash"] for item in records],
        "LON": [item["longitude"] for item in records],
        "LAT": [item["latitude"] for item in records],
    }


def assign_admin1(
    records: list[dict[str, Any]],
    admin1_path: Path,
) -> int:
    """Recompute every province name from the final center coordinates."""
    record_indices = list(range(len(records)))
    if not record_indices:
        return 0
    admin1 = gpd.read_file(admin1_path).to_crs("EPSG:4326")
    points = gpd.GeoDataFrame(
        {"record_index": record_indices},
        geometry=[
            Point(records[index]["longitude"], records[index]["latitude"])
            for index in record_indices
        ],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        points,
        admin1[["shapeName", "geometry"]],
        how="left",
        predicate="intersects",
    )
    assigned = 0
    for index in record_indices:
        records[index]["admin1"] = "ADM1_NOT_COVERED_BY_FROZEN_SOURCE"
    for row in joined.itertuples():
        if isinstance(row.shapeName, str) and row.shapeName:
            records[int(row.record_index)]["admin1"] = row.shapeName
            assigned += 1
    return assigned


def write_spatial_outputs(output_dir: Path, records: list[dict[str, Any]]) -> None:
    attrs = _shape_attributes(records)
    point_frame = gpd.GeoDataFrame(
        attrs,
        geometry=gpd.points_from_xy(attrs["LON"], attrs["LAT"]),
        crs="EPSG:4326",
    )
    footprint_frame = gpd.GeoDataFrame(
        attrs,
        geometry=[patch_geometry_wgs84(item) for item in records],
        crs="EPSG:4326",
    )
    point_frame.to_file(
        output_dir / "china_quarterly_62000_candidate_centers.shp",
        driver="ESRI Shapefile",
        encoding="UTF-8",
    )
    footprint_frame.to_file(
        output_dir / "china_quarterly_62000_candidate_footprints.shp",
        driver="ESRI Shapefile",
        encoding="UTF-8",
    )
    gpkg_path = output_dir / "china_quarterly_62000_candidate.gpkg"
    point_frame.to_file(gpkg_path, layer="centers", driver="GPKG")
    footprint_frame.to_file(gpkg_path, layer="footprints", driver="GPKG")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "worktree_dirty": dirty}


def write_package_readme(output_dir: Path, validation: dict[str, Any]) -> None:
    text = f"""# 中国季度嵌入 62,000 点采样候选版

版本：`china-quarterly-candidate-2026-07-26`

本目录用于 2020 年和 2021 年季度多源数据的下载、覆盖统计和质量检查规划。
当前文件是空间设计候选版，不是已经通过影像质量门槛的最终训练清单。

## 核心文件

1. `china_quarterly_62000_candidate_centers.shp`
   采样中心点，WGS84 / EPSG:4326。
2. `china_quarterly_62000_candidate_footprints.shp`
   每个样本的 1,280 m x 1,280 m UTM 原生方格转换到 WGS84 后的范围。
3. `china_quarterly_62000_candidate.gpkg`
   推荐使用的无字段名截断版本，包含 `centers` 和 `footprints` 两个图层。
4. `china_quarterly_62000_candidate_registry.jsonl`
   候选注册表，包含身份哈希、边界、分层原因和待办状态。
5. `china_quarterly_62000_candidate_registry.csv`
   便于筛选和任务调度的平面表。
6. `china_quarterly_62000_candidate_audit.json`
   修复、去重、跨 UTM 重叠和验收结果。
7. `SHA256SUMS.json`
   文件完整性校验值。

## 已通过的空间检查

- 总数：{validation["exact_total"]:,}
- 基础 1% 候选：{validation["layer_counts"]["base_expected_1pct"]:,}
- UTM/空间平衡补样：{validation["layer_counts"]["utm_spatial_balance_supplement"]:,}
- 海岸补样：{validation["layer_counts"]["coastal_supplement"]:,}
- 语义与困难样本复核池：{validation["layer_counts"]["semantic_and_difficult_supplement"]:,}
- 重复 patch / 网格 / footprint：0
- 中心落在国界外：{validation["centers_outside_boundary"]}
- UTM 归属错误：{validation["owner_zone_mismatches"]}
- 跨 UTM 分区重叠超过 1%：{validation["cross_zone_overlaps_over_1pct"]}
- 国界内外各占至少 0.1% 且与独立海洋面相交的严格海岸样本：
  {validation["coastal_exclusive_land_ocean_area_pass"]:,}
- 近海待高精度复核样本：
  {validation["coastal_geometry_status_counts"].get("near_coast_pending", 0):,}
- ADM1 冻结面未覆盖点：{validation["admin1_not_covered_by_frozen_source"]:,}

## 尚未完成

1. 1,500 个语义与困难样本现在只是确定性复核池，需用全国 WorldCover、历史
   OSM、DEM 和源质量统计重新排序。
2. 基础层来自历史候选骨架修复与确定性补齐，尚未用完整整数候选 atlas
   重新执行获批的逐宏网格概率抽样。
3. `ADM1_NOT_COVERED_BY_FROZEN_SOURCE` 表示 ADM0 内、但冻结 ADM1 数据未覆盖的点，
   不应被当作一个真实省级类别。
4. 海岸层中 `near_coast_pending` 点需用更高精度海岸线复核后才可冻结。
5. 2020Q1 至 2021Q4 的 S2、S1、Landsat 覆盖、云、配准等数值质量门槛尚未执行。
6. 上述检查完成后，才可冻结最终训练 registry 和 train/validation/test 划分。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def validate_registry(
    records: list[dict[str, Any]],
    country_geometry: Any,
    exclusive_land_geometry: Any,
    exclusive_ocean_geometry: Any,
    expected_seed: int,
    required_fields: list[str],
    cross_zone_overlap_count: int = 0,
) -> dict[str, Any]:
    layer_counts = Counter(item["sampling_layer"] for item in records)
    base_macro_counts = Counter(
        item["macro_id"] for item in records if item["sampling_layer"] == "base_expected_1pct"
    )
    supplement_macro_counts = Counter(
        item["macro_id"] for item in records if item["sampling_layer"] != "base_expected_1pct"
    )
    owner_mismatches = sum(
        int(item["grid_epsg"]) != utm_owner_epsg(item["longitude"], item["latitude"])
        for item in records
    )
    outside_centers = sum(
        not country_geometry.covers(Point(item["longitude"], item["latitude"])) for item in records
    )
    coastal_records = [item for item in records if item["sampling_layer"] == "coastal_supplement"]
    to_equal_area = _transformer(4326, 6933)
    exclusive_land_equal_area = transform_geometry(to_equal_area.transform, exclusive_land_geometry)
    prepared_ocean = prep(exclusive_ocean_geometry)
    coastal_exclusive_area_pass = 0
    for item in coastal_records:
        footprint = transform_geometry(to_equal_area.transform, patch_geometry_wgs84(item))
        minimum_area = footprint.area * COAST_MIN_FRACTION
        land_area = footprint.intersection(exclusive_land_equal_area).area
        outside_area = footprint.area - land_area
        if (
            land_area >= minimum_area
            and outside_area >= minimum_area
            and prepared_ocean.intersects(patch_geometry_wgs84(item))
        ):
            coastal_exclusive_area_pass += 1
    semantic_reason_counts = Counter(
        item["sampling_primary_reason"]
        for item in records
        if item["sampling_layer"] == "semantic_and_difficult_supplement"
    )
    coastal_status_counts = Counter(
        item.get("coastal_geometry_status", "missing") for item in coastal_records
    )
    missing_required_fields = sorted(
        {field for item in records for field in required_fields if field not in item}
    )
    report = {
        "exact_total": len(records),
        "layer_counts": dict(sorted(layer_counts.items())),
        "unique_patch_ids": len({item["patch_id"] for item in records}),
        "unique_identity_hashes": len({item["identity_hash"] for item in records}),
        "unique_footprint_hashes": len(
            {item["canonical_wgs84_footprint_hash"] for item in records}
        ),
        "owner_zone_mismatches": owner_mismatches,
        "centers_outside_boundary": outside_centers,
        "cross_zone_overlaps_over_1pct": cross_zone_overlap_count,
        "base_macros_over_one": sum(value > 1 for value in base_macro_counts.values()),
        "supplement_macros_over_one": sum(value > 1 for value in supplement_macro_counts.values()),
        "unknown_admin1": sum(item.get("admin1") == "unknown" for item in records),
        "admin1_not_covered_by_frozen_source": sum(
            item.get("admin1") == "ADM1_NOT_COVERED_BY_FROZEN_SOURCE" for item in records
        ),
        "sampling_seed_values": sorted({int(item["sampling_seed"]) for item in records}),
        "coastal_exclusive_land_ocean_area_pass": coastal_exclusive_area_pass,
        "coastal_geometry_status_counts": dict(sorted(coastal_status_counts.items())),
        "semantic_candidate_reason_counts": dict(sorted(semantic_reason_counts.items())),
        "missing_required_registry_fields": missing_required_fields,
        "semantic_atlas_status": "pending",
        "quarterly_source_quality_status": "pending",
        "quality_eligible_for_training": False,
    }
    if len(records) != 62000:
        raise ValueError(f"registry contains {len(records)} records, expected 62000")
    if report["unique_patch_ids"] != len(records):
        raise ValueError("duplicate patch_id found")
    if report["unique_identity_hashes"] != len(records):
        raise ValueError("duplicate identity_hash found")
    if report["unique_footprint_hashes"] != len(records):
        raise ValueError("duplicate footprint found")
    if owner_mismatches:
        raise ValueError(f"{owner_mismatches} owner-zone mismatches remain")
    if outside_centers:
        raise ValueError(f"{outside_centers} centers are outside the boundary")
    if cross_zone_overlap_count:
        raise ValueError(f"{cross_zone_overlap_count} cross-zone overlaps over 1 percent remain")
    if report["base_macros_over_one"]:
        raise ValueError("more than one base point remains in a macrocell")
    if report["supplement_macros_over_one"]:
        raise ValueError("more than one supplement remains in a macrocell")
    if report["unknown_admin1"]:
        raise ValueError(f"{report['unknown_admin1']} records have unknown admin1")
    if report["sampling_seed_values"] != [expected_seed]:
        raise ValueError("sampling_seed must be the single approved global seed")
    if sum(coastal_status_counts.values()) != len(coastal_records):
        raise ValueError("coastal geometry status count does not match the coastal layer")
    if missing_required_fields:
        raise ValueError(f"required registry fields missing: {missing_required_fields}")
    return report


def cross_zone_overlap_violations(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find cross-zone footprint intersections exceeding one percent."""
    frame = gpd.GeoDataFrame(
        {
            "patch_id": [item["patch_id"] for item in records],
            "grid_epsg": [item["grid_epsg"] for item in records],
            "sampling_layer": [item["sampling_layer"] for item in records],
        },
        geometry=[patch_geometry_wgs84(item) for item in records],
        crs="EPSG:4326",
    )
    pairs = frame.sindex.query(frame.geometry, predicate="intersects")
    projected = frame.to_crs(6933)
    layer_removal_priority = {
        "base_expected_1pct": 0,
        "utm_spatial_balance_supplement": 1,
        "coastal_supplement": 2,
        "semantic_and_difficult_supplement": 3,
    }
    violations: list[dict[str, Any]] = []
    for left_index, right_index in zip(*pairs):
        left_index = int(left_index)
        right_index = int(right_index)
        if left_index >= right_index:
            continue
        if int(frame.iloc[left_index]["grid_epsg"]) == int(frame.iloc[right_index]["grid_epsg"]):
            continue
        left_geometry = projected.geometry.iloc[left_index]
        right_geometry = projected.geometry.iloc[right_index]
        overlap_area = left_geometry.intersection(right_geometry).area
        overlap_fraction = overlap_area / min(left_geometry.area, right_geometry.area)
        if overlap_fraction <= 0.01:
            continue
        candidates = [left_index, right_index]
        remove_index = max(
            candidates,
            key=lambda index: (
                layer_removal_priority.get(str(frame.iloc[index]["sampling_layer"]), 99),
                stable_hash(0, str(frame.iloc[index]["patch_id"])),
            ),
        )
        violations.append(
            {
                "left_index": left_index,
                "right_index": right_index,
                "left_patch_id": str(frame.iloc[left_index]["patch_id"]),
                "right_patch_id": str(frame.iloc[right_index]["patch_id"]),
                "overlap_fraction": round(float(overlap_fraction), 8),
                "remove_index": remove_index,
                "remove_patch_id": str(frame.iloc[remove_index]["patch_id"]),
            }
        )
    return violations


def render_preview(
    country_path: Path,
    adm1_path: Path,
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    country = gpd.read_file(country_path).to_crs("EPSG:4326")
    admin1 = gpd.read_file(adm1_path).to_crs("EPSG:4326")
    colors = {
        "base_expected_1pct": "#b2182b",
        "utm_spatial_balance_supplement": "#2166ac",
        "coastal_supplement": "#762a83",
        "semantic_and_difficult_supplement": "#1b7837",
    }
    fig, axis = plt.subplots(figsize=(15, 10), dpi=220)
    country.plot(ax=axis, color="#f7f7f7", edgecolor="#202020", linewidth=0.7)
    admin1.boundary.plot(ax=axis, color="#aaaaaa", linewidth=0.22)
    for layer, color in colors.items():
        subset = [item for item in records if item["sampling_layer"] == layer]
        axis.scatter(
            [item["longitude"] for item in subset],
            [item["latitude"] for item in subset],
            s=0.45 if layer == "base_expected_1pct" else 1.8,
            color=color,
            alpha=0.72,
            linewidths=0,
            label=f"{layer} ({len(subset):,})",
        )
    axis.set_axis_off()
    axis.set_aspect("equal")
    axis.set_title(
        "Xuannv China quarterly embedding: 62,000-location sampling candidate",
        fontsize=14,
    )
    axis.legend(loc="upper left", fontsize=8, markerscale=5, frameon=True)
    axis.text(
        0.01,
        0.015,
        "Design candidate only: semantic re-ranking and 2020-2021 quarterly source QA pending.",
        transform=axis.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#777777", "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-points", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--country", type=Path, required=True)
    parser.add_argument("--adm1", type=Path, required=True)
    parser.add_argument("--ocean", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    input_paths = {
        "existing_points": args.existing_points,
        "macrocell_inventory": args.inventory,
        "country_boundary": args.country,
        "admin1_boundary": args.adm1,
        "ocean_boundary": args.ocean,
        "sampling_policy": args.policy,
    }
    input_provenance = {
        name: {"path": str(path), "sha256": _sha256(path)} for name, path in input_paths.items()
    }
    seed = int(policy["sampling"]["sampling_seed"])
    layer_targets = {
        str(layer["id"]): int(layer["count"]) for layer in policy["sampling"]["layers"]
    }
    semantic_subquotas = next(
        layer["subquotas"]
        for layer in policy["sampling"]["layers"]
        if layer["id"] == "semantic_and_difficult_supplement"
    )
    semantic_reasons = [
        reason for reason, quota in semantic_subquotas.items() for _ in range(int(quota))
    ]
    if sum(layer_targets.values()) != int(policy["sampling"]["target_total"]):
        raise ValueError("policy layer quotas do not sum to target_total")
    registry_version = f"china-quarterly-candidate-{policy['version']}"
    country = gpd.read_file(args.country).to_crs("EPSG:4326").geometry.union_all()
    ocean = gpd.read_file(args.ocean).to_crs("EPSG:4326").geometry.union_all()
    # Use the frozen country interior as the land side and only the part of
    # the independent ocean polygon outside that country as the ocean side.
    # These two classes are disjoint without erasing coastline disagreement.
    exclusive_land = country
    exclusive_ocean = ocean
    existing = _read_jsonl(args.existing_points)
    repaired, repair_report = repair_owner_zones(existing, country)

    # Drop any identity collisions introduced by snapping legacy cross-zone
    # points onto their unique owner-zone grid. They are replenished below.
    deduplicated: list[dict[str, Any]] = []
    seen_identities: set[tuple[int, int, int]] = set()
    dropped_by_layer: Counter[str] = Counter()
    for item in repaired:
        identity = (
            int(item["grid_epsg"]),
            int(item["grid_col"]),
            int(item["grid_row"]),
        )
        if identity in seen_identities:
            dropped_by_layer[TIER_TO_LAYER[str(item["sampling_tier"])]] += 1
            continue
        seen_identities.add(identity)
        deduplicated.append(item)

    # Rebuild every supplement layer in policy order so the shared one-point
    # supplement macrocell cap is enforceable. Keep only the repaired base.
    legacy_coastal_removed = sum(
        item["sampling_tier"] == "coastal_ocean_adjacent_supplement" for item in deduplicated
    )
    legacy_spatial_removed = sum(
        item["sampling_tier"]
        in {
            "spatial_stratified_supplement",
            "spatial_stratified_supplement_fallback",
        }
        for item in deduplicated
    )
    deduplicated = [item for item in deduplicated if item["sampling_tier"] == "base_spatial"]
    deduplicated, macro_limit_dropped = enforce_macro_limits(deduplicated, seed=seed)

    inventory = _read_jsonl(args.inventory)
    target_legacy_layers = {
        key: layer_targets[key]
        for key in (
            "base_expected_1pct",
            "utm_spatial_balance_supplement",
            "coastal_supplement",
        )
    }
    base_macros = {
        str(item["macro_id"])
        for item in deduplicated
        if TIER_TO_LAYER[str(item["sampling_tier"])] == "base_expected_1pct"
    }
    missing_base = layer_targets["base_expected_1pct"] - len(deduplicated)
    if missing_base > 0:
        replacements = choose_replacements(
            inventory=inventory,
            country_geometry=country,
            existing=deduplicated,
            count=missing_base,
            seed=seed + layer_targets["base_expected_1pct"],
            sampling_layer="base_expected_1pct",
            primary_reasons=["base_expected_1pct_owner_zone_replenishment"] * missing_base,
            forbid_supplement_macros=base_macros,
        )
        for item in replacements:
            item["sampling_tier"] = "base_spatial"
            item["sampling_layer"] = "base_expected_1pct"
        deduplicated.extend(replacements)
        base_macros.update(str(item["macro_id"]) for item in replacements)

    coastal = choose_coastal_candidates(
        inventory=inventory,
        country_geometry=country,
        exclusive_land_geometry=exclusive_land,
        exclusive_ocean_geometry=exclusive_ocean,
        existing=deduplicated,
        count=layer_targets["coastal_supplement"],
        seed=seed + layer_targets["coastal_supplement"],
        forbid_supplement_macros=set(),
    )
    deduplicated.extend(coastal)
    supplement_macros = {str(item["macro_id"]) for item in coastal}

    spatial, spatial_allocation, exact_owner_candidate_counts = choose_spatial_balance_candidates(
        inventory=inventory,
        country_geometry=country,
        existing=deduplicated,
        count=layer_targets["utm_spatial_balance_supplement"],
        seed=seed + layer_targets["utm_spatial_balance_supplement"],
        forbid_supplement_macros=supplement_macros,
    )
    deduplicated.extend(spatial)
    supplement_macros.update(str(item["macro_id"]) for item in spatial)

    semantic = choose_replacements(
        inventory=inventory,
        country_geometry=country,
        existing=deduplicated,
        count=layer_targets["semantic_and_difficult_supplement"],
        seed=seed + 62000,
        sampling_layer="semantic_and_difficult_supplement",
        primary_reasons=semantic_reasons,
        forbid_supplement_macros=supplement_macros,
    )
    raw_records = deduplicated + semantic
    assigned_admin1 = assign_admin1(raw_records, args.adm1)
    for item in raw_records:
        item["selection_subseed"] = item.get("sampling_seed")
        item["sampling_seed"] = seed
    records = [normalize_record(item, registry_version=registry_version) for item in raw_records]
    boundary_version = f"sha256:{input_provenance['country_boundary']['sha256'][:16]}"
    for item in records:
        item["boundary_version"] = boundary_version
    overlap_resolution: list[dict[str, Any]] = []
    blocked_overlap_macros: set[str] = set()
    target_all_layers = {
        **target_legacy_layers,
        "semantic_and_difficult_supplement": layer_targets["semantic_and_difficult_supplement"],
    }
    for attempt in range(5):
        violations = cross_zone_overlap_violations(records)
        if not violations:
            break
        remove_indices = {item["remove_index"] for item in violations}
        removed = [records[index] for index in sorted(remove_indices)]
        overlap_resolution.extend(violations)
        blocked_overlap_macros.update(str(item["macro_id"]) for item in removed)
        records = [item for index, item in enumerate(records) if index not in remove_indices]
        missing_reasons: dict[str, list[str]] = {}
        for item in removed:
            missing_reasons.setdefault(str(item["sampling_layer"]), []).append(
                str(item["sampling_primary_reason"])
            )
        current_supplement_macros = {
            str(item["macro_id"])
            for item in records
            if item["sampling_layer"] != "base_expected_1pct"
        }
        current_base_macros = {
            str(item["macro_id"])
            for item in records
            if item["sampling_layer"] == "base_expected_1pct"
        }
        for layer, target in target_all_layers.items():
            missing = target - sum(item["sampling_layer"] == layer for item in records)
            if missing <= 0:
                continue
            reasons = missing_reasons.get(layer, [])
            reasons.extend([f"{layer}_overlap_replenishment"] * (missing - len(reasons)))
            forbidden_macros = blocked_overlap_macros | (
                current_base_macros if layer == "base_expected_1pct" else current_supplement_macros
            )
            if layer == "coastal_supplement":
                replacements = choose_coastal_candidates(
                    inventory=inventory,
                    country_geometry=country,
                    exclusive_land_geometry=exclusive_land,
                    exclusive_ocean_geometry=exclusive_ocean,
                    existing=records,
                    count=missing,
                    seed=seed + 70000 + attempt * 100 + target,
                    forbid_supplement_macros=forbidden_macros,
                )
            else:
                replacements = choose_replacements(
                    inventory=inventory,
                    country_geometry=country,
                    existing=records,
                    count=missing,
                    seed=seed + 70000 + attempt * 100 + target,
                    sampling_layer=layer,
                    primary_reasons=reasons,
                    forbid_supplement_macros=forbidden_macros,
                )
            for item in replacements:
                item["sampling_tier"] = {
                    "base_expected_1pct": "base_spatial",
                    "utm_spatial_balance_supplement": "spatial_stratified_supplement",
                    "coastal_supplement": "coastal_ocean_adjacent_supplement",
                    "semantic_and_difficult_supplement": "semantic_audit_reserve",
                }[layer]
                item["sampling_layer"] = layer
                item["selection_subseed"] = item.get("sampling_seed")
                item["sampling_seed"] = seed
            records.extend(
                normalize_record(item, registry_version=registry_version) for item in replacements
            )
            for item in records[-len(replacements) :]:
                item["boundary_version"] = boundary_version
    assigned_admin1 += assign_admin1(records, args.adm1)
    final_overlap_violations = cross_zone_overlap_violations(records)
    records.sort(
        key=lambda item: (
            int(item["grid_epsg"]),
            int(item["grid_row"]),
            int(item["grid_col"]),
            str(item["patch_id"]),
        )
    )
    validation = validate_registry(
        records,
        country,
        exclusive_land,
        exclusive_ocean,
        seed,
        list(policy["registry"]["required_fields"]),
        cross_zone_overlap_count=len(final_overlap_violations),
    )

    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "china_quarterly_62000_candidate_registry.jsonl", records)
    _write_csv(output_dir / "china_quarterly_62000_candidate_registry.csv", records)
    write_spatial_outputs(output_dir, records)
    render_preview(
        args.country,
        args.adm1,
        records,
        output_dir / "china_quarterly_62000_candidate_preview.png",
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": registry_version,
        "source_points": str(args.existing_points),
        "policy": str(args.policy),
        "input_provenance": input_provenance,
        "generator": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "git": _git_revision(Path(__file__).resolve().parents[2]),
            "command": " ".join(
                [
                    "python",
                    str(Path(__file__).resolve()),
                    "--existing-points",
                    str(args.existing_points),
                    "--inventory",
                    str(args.inventory),
                    "--country",
                    str(args.country),
                    "--adm1",
                    str(args.adm1),
                    "--ocean",
                    str(args.ocean),
                    "--policy",
                    str(args.policy),
                    "--output-dir",
                    str(args.output_dir),
                ]
            ),
        },
        "repair_report": repair_report,
        "identity_collisions_after_owner_repair": dict(dropped_by_layer),
        "legacy_coastal_records_rebuilt": legacy_coastal_removed,
        "legacy_spatial_records_rebuilt": legacy_spatial_removed,
        "utm_spatial_balance_allocation": spatial_allocation,
        "exact_owner_candidate_counts_by_utm": exact_owner_candidate_counts,
        "macro_limit_records_replenished": len(macro_limit_dropped),
        "admin1_values_filled_from_adm1": assigned_admin1,
        "cross_zone_overlap_resolution": overlap_resolution,
        "validation": validation,
        "limitations": [
            (
                "The 1,500 semantic/difficult locations are deterministic audit-reserve "
                "candidates, not semantically verified samples."
            ),
            (
                "The 57,405 base layer repairs the historical candidate skeleton; it is "
                "not a fresh exact-integer candidate-atlas redraw."
            ),
            (
                "Quarterly 2020-2021 source coverage and cloud/registration quality "
                "gates have not been applied."
            ),
            (
                "The coastal layer contains strict boundary-crossing samples plus a "
                "small near-coast reserve marked near_coast_pending; the latter must "
                "be checked against a higher-resolution coastline before registry freeze."
            ),
            (
                "ADM1_NOT_COVERED_BY_FROZEN_SOURCE marks ADM0 locations absent from "
                "the frozen ADM1 polygons."
            ),
            (
                "The registry is suitable for acquisition planning, not yet the "
                "frozen training registry."
            ),
        ],
    }
    audit_path = output_dir / "china_quarterly_62000_candidate_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_package_readme(output_dir, validation)
    checksums = {
        path.name: _sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()
    }
    (output_dir / "SHA256SUMS.json").write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "archive": archive,
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
