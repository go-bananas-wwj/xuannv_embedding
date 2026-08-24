"""Build a static, imagery-free China macrocell inventory for China V1.

The inventory stores 12.8 km (10 x 10 patch) cells in their owning UTM zone.
It is deliberately *not* a final candidate atlas: `estimated_patch_count` is
only a land-area estimate. Exact quality-eligible patch counts are added after
scene-quality processing, before the registry builder is allowed to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import geopandas as gpd
from pyproj import Transformer
from shapely import segmentize
from shapely.geometry import box
from shapely.prepared import prep

PATCH_SIDE_METERS = 1280
MACRO_SIDE_PATCHES = 10
MACRO_SIDE_METERS = PATCH_SIDE_METERS * MACRO_SIDE_PATCHES
WGS84_PROJECTION_SEGMENT_LENGTH_DEGREES = 0.01


def _hash_geometry(geometry: Any) -> str:
    return hashlib.sha256(geometry.wkb).hexdigest()


def _utm_zones(min_lon: float, max_lon: float) -> range:
    return range(
        max(1, math.floor((min_lon + 180) / 6) + 1),
        min(60, math.floor((max_lon + 180) / 6) + 1) + 1,
    )


def _admin_name(macro: Any, admin_geometries: list[tuple[str, Any]]) -> str:
    best_name = "unknown"
    best_area = 0.0
    for name, geometry in admin_geometries:
        if not geometry.intersects(macro):
            continue
        overlap_area = macro.intersection(geometry).area
        if overlap_area > best_area:
            best_name = name
            best_area = overlap_area
    return best_name


def build_inventory(
    adm0_path: Path, adm1_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    country = gpd.read_file(adm0_path).to_crs("EPSG:4326")
    admin1 = gpd.read_file(adm1_path).to_crs("EPSG:4326")
    geometry_wgs84 = country.geometry.union_all()
    min_lon, min_lat, max_lon, max_lat = geometry_wgs84.bounds
    records: list[dict[str, Any]] = []

    for zone in _utm_zones(min_lon, max_lon):
        epsg = 32600 + zone
        zone_west = -180 + (zone - 1) * 6
        zone_east = zone_west + 6
        # Never project all of China into every UTM zone: far-away geometry
        # explodes its projected bounds and can create millions of empty cells.
        zone_strip = box(zone_west, min_lat - 0.01, zone_east, max_lat + 0.01)
        zone_country_wgs84 = geometry_wgs84.intersection(zone_strip)
        if zone_country_wgs84.is_empty:
            continue
        # A zone-edge meridian is curved in UTM. Densify before reprojection so a
        # long geographic edge cannot become an inward chord that drops edge cells.
        country_zone_wgs84 = segmentize(
            zone_country_wgs84, max_segment_length=WGS84_PROJECTION_SEGMENT_LENGTH_DEGREES
        )
        country_zone = gpd.GeoSeries([country_zone_wgs84], crs="EPSG:4326").to_crs(epsg).iloc[0]
        prepared_country = prep(country_zone)
        admin_zone = admin1[admin1.geometry.intersects(zone_strip)].to_crs(epsg)
        admin_geometries = [(str(row.shapeName), row.geometry) for row in admin_zone.itertuples()]
        left, bottom, right, top = country_zone.bounds
        col_start = math.floor(left / MACRO_SIDE_METERS)
        col_end = math.ceil(right / MACRO_SIDE_METERS)
        row_start = math.floor(bottom / MACRO_SIDE_METERS)
        row_end = math.ceil(top / MACRO_SIDE_METERS)
        to_wgs84 = Transformer.from_crs(epsg, 4326, always_xy=True)

        for macro_col in range(col_start, col_end):
            for macro_row in range(row_start, row_end):
                bounds = (
                    macro_col * MACRO_SIDE_METERS,
                    macro_row * MACRO_SIDE_METERS,
                    (macro_col + 1) * MACRO_SIDE_METERS,
                    (macro_row + 1) * MACRO_SIDE_METERS,
                )
                macro = box(*bounds)
                if not prepared_country.intersects(macro):
                    continue
                land_fraction = macro.intersection(country_zone).area / macro.area
                if land_fraction <= 0:
                    continue
                west, south = to_wgs84.transform(bounds[0], bounds[1])
                east, north = to_wgs84.transform(bounds[2], bounds[3])
                grid_id = f"utm{zone:02d}n"
                records.append(
                    {
                        "schema_version": "china_v1_macrocell_inventory_v1",
                        "macro_id": f"{grid_id}_c{macro_col}_r{macro_row}",
                        "grid_id": grid_id,
                        "grid_epsg": epsg,
                        "macro_col": macro_col,
                        "macro_row": macro_row,
                        "utm_bounds": [round(value, 3) for value in bounds],
                        "wgs84_bounds": [
                            round(west, 7),
                            round(south, 7),
                            round(east, 7),
                            round(north, 7),
                        ],
                        "geometry_hash": _hash_geometry(macro),
                        "land_fraction": round(land_fraction, 6),
                        "estimated_patch_count": round(land_fraction * MACRO_SIDE_PATCHES**2, 3),
                        "candidate_count_status": "estimate_only_requires_exact_quality_atlas",
                        "admin1": _admin_name(macro, admin_geometries),
                    }
                )
        zone_macrocell_count = sum(record["grid_epsg"] == epsg for record in records)
        print(
            f"UTM zone {zone:02d}: {zone_macrocell_count} macrocells",
            file=sys.stderr,
            flush=True,
        )

    records.sort(key=lambda item: item["macro_id"])
    by_zone = Counter(record["grid_id"] for record in records)
    by_admin = Counter(record["admin1"] for record in records)
    summary = {
        "schema_version": "china_v1_macrocell_inventory_summary_v1",
        "patch_side_meters": PATCH_SIDE_METERS,
        "macro_side_patches": MACRO_SIDE_PATCHES,
        "macro_side_meters": MACRO_SIDE_METERS,
        "num_macrocells": len(records),
        "estimated_land_patches": round(
            sum(record["estimated_patch_count"] for record in records), 3
        ),
        "estimated_one_percent_base_samples": round(
            sum(record["estimated_patch_count"] for record in records) / 100, 3
        ),
        "macrocells_by_grid": dict(sorted(by_zone.items())),
        "macrocells_by_admin1": dict(sorted(by_admin.items())),
        "input_files": {"adm0": str(adm0_path), "adm1": str(adm1_path)},
    }
    return records, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adm0", type=Path, required=True)
    parser.add_argument("--adm1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    records, summary = build_inventory(args.adm0, args.adm1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
