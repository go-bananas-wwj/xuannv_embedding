from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

from xuannv_embedding.data_process import qgis


def load_script():
    return qgis


def test_dissolved_boundaries_follow_cells_not_bounding_boxes() -> None:
    """A concave shard boundary must preserve the grid-cell footprint, not become a bbox."""
    module = load_script()
    cells = gpd.GeoDataFrame(
        {"shard_id": [1, 1, 1, 2]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
            Polygon([(3, 0), (4, 0), (4, 1), (3, 1)]),
        ],
        crs="EPSG:4326",
    )

    regions, national = module.dissolve_region_boundaries(cells)

    shard_one = regions.loc[regions["shard_id"] == 1].geometry.iloc[0]
    assert len(regions) == 2
    assert shard_one.area == 3
    assert not shard_one.equals(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    assert national.geometry.iloc[0].area == 4
    assert national.crs == cells.crs


def test_write_grid_cells_shapefile_keeps_exact_cell_features(tmp_path: Path) -> None:
    """QGIS export must retain one feature per 1280 m grid cell and its parent key."""
    module = load_script()
    cells = gpd.GeoDataFrame(
        {
            "parent_key": ["32648:1:2", "32648:1:3"],
            "shard_id": [1, 1],
            "grid_id": ["utm48n", "utm48n"],
            "grid_col": [1, 1],
            "grid_row": [2, 3],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        ],
        crs="EPSG:4326",
    )
    destination = tmp_path / "shard_01_grid_cells.shp"

    module.write_qgis_shapefile(cells, destination)

    result = gpd.read_file(destination)
    assert len(result) == 2
    assert result["parent_key"].tolist() == ["32648:1:2", "32648:1:3"]
    assert result.geometry.iloc[0].equals(cells.geometry.iloc[0])


def test_polygonal_validity_repair_removes_self_intersections() -> None:
    """A topology repair must keep exported national boundaries valid for QGIS."""
    module = load_script()
    self_intersecting = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])

    repaired = module.make_polygonal_valid(self_intersecting)

    assert not self_intersecting.is_valid
    assert repaired.is_valid
    assert repaired.geom_type in {"Polygon", "MultiPolygon"}


def test_qgis_readme_section_is_not_duplicated_on_regeneration(tmp_path: Path) -> None:
    """Regenerating an interrupted local export must not duplicate README guidance."""
    module = load_script()
    readme = tmp_path / "README.md"
    readme.write_text("# Delivery\n", encoding="utf-8")

    module.write_delivery_readmes(tmp_path)
    module.write_delivery_readmes(tmp_path)

    assert readme.read_text(encoding="utf-8").count("## QGIS 精确 Shape 文件") == 1


def test_boundary_shapefile_preserves_polygon_holes_as_valid_geometry(tmp_path: Path) -> None:
    """ESRI Shapefile ring orientation must not turn holes into nested shells."""
    module = load_script()
    boundary = gpd.GeoDataFrame(
        {"shard_id": [1], "cell_count": [8]},
        geometry=[
            Polygon(
                [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)],
                holes=[[(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]],
            )
        ],
        crs="EPSG:4326",
    )
    destination = tmp_path / "boundary.shp"

    module.write_qgis_shapefile(boundary, destination)

    result = gpd.read_file(destination).geometry.iloc[0]
    assert result.is_valid
    assert result.area == 8


def test_boundary_geopackage_preserves_complex_polygon_validity(tmp_path: Path) -> None:
    """The canonical QGIS boundary format must retain polygon holes without topology loss."""
    module = load_script()
    boundary = gpd.GeoDataFrame(
        {"shard_id": [1], "cell_count": [8]},
        geometry=[
            Polygon(
                [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)],
                holes=[[(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]],
            )
        ],
        crs="EPSG:4326",
    )
    destination = tmp_path / "boundary_exact.gpkg"

    module.write_qgis_geopackage(boundary, destination, layer="boundary")

    result = gpd.read_file(destination, layer="boundary").geometry.iloc[0]
    assert result.is_valid
    assert result.area == 8


def test_boundary_export_writes_qgis_shapefile_and_geopackage(tmp_path: Path) -> None:
    """Every large boundary export must include both requested .shp and robust .gpkg files."""
    module = load_script()
    boundary = gpd.GeoDataFrame(
        {"shard_id": [1], "cell_count": [8]},
        geometry=[
            Polygon(
                [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)],
                holes=[[(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]],
            )
        ],
        crs="EPSG:4326",
    )
    shp_path = tmp_path / "boundary_exact.shp"
    gpkg_path = tmp_path / "boundary_exact.gpkg"

    module.write_boundary_exports(boundary, shp_path, gpkg_path, layer="boundary")

    assert shp_path.exists()
    assert gpkg_path.exists()
    assert gpd.read_file(shp_path).geometry.iloc[0].is_valid
    assert gpd.read_file(gpkg_path, layer="boundary").geometry.iloc[0].is_valid
