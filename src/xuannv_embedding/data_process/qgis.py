#!/usr/bin/env python3
"""Export exact China tenfold boundaries and 1280 m grid cells as QGIS Shapefiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

GRID_COLUMNS = ["parent_key", "shard_id", "grid_id", "grid_col", "grid_row", "geometry"]
WGS84 = "EPSG:4326"


def make_polygonal_valid(geometry: shapely.Geometry) -> shapely.Geometry:
    """Repair rare topology artifacts while retaining only polygonal coverage."""
    if geometry.is_valid:
        return geometry
    repaired = shapely.make_valid(geometry, method="structure", keep_collapsed=False)
    if repaired.geom_type in {"Polygon", "MultiPolygon"}:
        return repaired
    polygon_parts = [
        part
        for part in shapely.get_parts(repaired)
        if part.geom_type in {"Polygon", "MultiPolygon"} and not part.is_empty
    ]
    if not polygon_parts:
        raise ValueError("Topology repair produced no polygonal coverage")
    return shapely.union_all(np.asarray(polygon_parts, dtype=object))


def write_qgis_shapefile(frame: gpd.GeoDataFrame, destination: Path) -> None:
    """Write a UTF-8 ESRI Shapefile that QGIS can open directly."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(
        destination, driver="ESRI Shapefile", engine="pyogrio", index=False, encoding="UTF-8"
    )


def write_qgis_geopackage(frame: gpd.GeoDataFrame, destination: Path, layer: str) -> None:
    """Write a GeoPackage layer for complex polygon boundaries without ring loss."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(destination, layer=layer, driver="GPKG", engine="pyogrio", index=False)


def write_boundary_exports(
    frame: gpd.GeoDataFrame,
    shapefile_destination: Path,
    geopackage_destination: Path,
    layer: str,
) -> None:
    """Write each exact boundary in both requested Shapefile and GeoPackage formats."""
    write_qgis_shapefile(frame, shapefile_destination)
    write_qgis_geopackage(frame, geopackage_destination, layer=layer)


def read_shard_grid_cells(shard_root: Path) -> gpd.GeoDataFrame:
    """Read a shard's exact WGS84 cell polygons from its GeoParquet pieces."""
    frames: list[pd.DataFrame] = []
    for path in sorted(shard_root.glob("utm*/part-*.parquet")):
        table = pq.read_table(path, columns=GRID_COLUMNS)
        frame = table.to_pandas()
        frame["geometry"] = shapely.from_wkb(frame["geometry"].to_numpy())
        frames.append(frame)
    if not frames:
        raise ValueError(f"No GeoParquet grid files found under {shard_root}")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=WGS84)


def dissolve_region_boundaries(
    cells: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Dissolve exact cells by shard and calculate their true national footprint."""
    if cells.empty:
        raise ValueError("Cannot dissolve an empty cell table")
    if "shard_id" not in cells:
        raise ValueError("Grid cells must have a shard_id column")

    records: list[dict[str, object]] = []
    for shard_id, shard_cells in cells.groupby("shard_id", sort=True):
        geometry = make_polygonal_valid(
            shapely.union_all(np.asarray(shard_cells.geometry.array, dtype=object))
        )
        records.append(
            {"shard_id": int(shard_id), "cell_count": len(shard_cells), "geometry": geometry}
        )
    regions = gpd.GeoDataFrame(records, geometry="geometry", crs=cells.crs)
    national_geometry = make_polygonal_valid(
        shapely.union_all(np.asarray(regions.geometry.array, dtype=object))
    )
    national = gpd.GeoDataFrame(
        [{"coverage": "china_full_grid_ten_regions", "geometry": national_geometry}],
        geometry="geometry",
        crs=cells.crs,
    )
    return regions, national


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delivery-root",
        type=Path,
        required=True,
        help="Existing V4 delivery directory to extend with qgis_shapes/.",
    )
    parser.add_argument(
        "--boundaries-only",
        action="store_true",
        help="Create canonical GeoPackage boundaries alongside existing grid-cell Shapefiles.",
    )
    return parser.parse_args()


def write_delivery_readmes(output_root: Path) -> None:
    """Describe the exact Shape exports without conflating them with bbox indices."""
    root_readme = output_root / "README.md"
    existing = root_readme.read_text(encoding="utf-8")
    marker = "## QGIS 精确 Shape 文件"
    if marker in existing:
        existing = existing.split(marker, maxsplit=1)[0].rstrip() + "\n"
    root_readme.write_text(
        existing
        + "\n## QGIS 精确 Shape 文件\n\n"
        + "`qgis_shapes/china_ten_regions_exact.shp` 含十个大区的精确裁剪边界；"
        + "`qgis_shapes/china_full_grid_coverage_exact.shp` 是十区合并后的全国网格覆盖边界。"
        + "二者都有同名 `.gpkg` 副本，可由 QGIS 直接打开。\n\n"
        + "每个 `qgis_shapes/shards/shard_XX/` 目录含：\n\n"
        + "- `shard_XX_boundary_exact.shp`：该大区所有 1280 m 网格的精确并集边界；\n"
        + "- `shard_XX_grid_cells_1280m.shp`：该大区全部精确 1280 m × 1280 m 网格。\n\n"
        + "这些文件都由 GeoParquet 的实际 `geometry` 直接导出或精确并集而来；"
        + "边界网格已经遵循原始中国范围裁剪。"
        + "`region_bounds.geojson` 仍只是一份经纬度外接矩形索引。\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    delivery_root = args.delivery_root.resolve()
    if not delivery_root.is_dir():
        raise FileNotFoundError(f"Delivery directory does not exist: {delivery_root}")
    shapes_root = delivery_root / "qgis_shapes"
    if shapes_root.exists() and not args.boundaries_only:
        raise FileExistsError(f"Refusing to overwrite existing Shape directory: {shapes_root}")

    manifest = json.loads((delivery_root / "tenfold_partition_manifest.json").read_text())
    summaries = {int(item["shard_id"]): item for item in manifest["shards"]}
    if sorted(summaries) != list(range(1, 11)):
        raise ValueError("Delivery manifest must contain shard_01 through shard_10")

    shapes_root.mkdir(parents=True, exist_ok=args.boundaries_only)
    region_records: list[dict[str, object]] = []
    for shard_id in range(1, 11):
        shard_root = delivery_root / "shards" / f"shard_{shard_id:02d}"
        cells = read_shard_grid_cells(shard_root)
        if len(cells) != int(summaries[shard_id]["shape_count"]):
            raise ValueError(f"Unexpected cell count in shard_{shard_id:02d}")
        if cells["shard_id"].nunique() != 1 or int(cells["shard_id"].iloc[0]) != shard_id:
            raise ValueError(f"Incorrect shard_id values in shard_{shard_id:02d}")

        shard_shape_root = shapes_root / "shards" / f"shard_{shard_id:02d}"
        if not args.boundaries_only:
            write_qgis_shapefile(
                cells, shard_shape_root / f"shard_{shard_id:02d}_grid_cells_1280m.shp"
            )
        region, _ = dissolve_region_boundaries(cells)
        boundary = region.assign(src_shard=int(summaries[shard_id]["source_shard_id"]))[
            ["shard_id", "src_shard", "cell_count", "geometry"]
        ]
        write_boundary_exports(
            boundary,
            shard_shape_root / f"shard_{shard_id:02d}_boundary_exact.shp",
            shard_shape_root / f"shard_{shard_id:02d}_boundary_exact.gpkg",
            layer="boundary",
        )
        region_records.append(boundary.iloc[0].to_dict())

    regions = gpd.GeoDataFrame(region_records, geometry="geometry", crs=WGS84)
    national_geometry = make_polygonal_valid(
        shapely.union_all(np.asarray(regions.geometry.array, dtype=object))
    )
    national = gpd.GeoDataFrame(
        [{"coverage": "china_full_grid_ten_regions", "geometry": national_geometry}],
        geometry="geometry",
        crs=WGS84,
    )
    write_boundary_exports(
        regions,
        shapes_root / "china_ten_regions_exact.shp",
        shapes_root / "china_ten_regions_exact.gpkg",
        layer="ten_regions",
    )
    write_boundary_exports(
        national,
        shapes_root / "china_full_grid_coverage_exact.shp",
        shapes_root / "china_full_grid_coverage_exact.gpkg",
        layer="national_coverage",
    )
    write_delivery_readmes(delivery_root)


if __name__ == "__main__":
    main()
