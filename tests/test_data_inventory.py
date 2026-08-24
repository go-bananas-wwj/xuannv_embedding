from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box

from xuannv_embedding.data_process import inventory as MODULE


def test_build_inventory_marks_candidate_counts_as_estimates(tmp_path: Path) -> None:
    adm0 = tmp_path / "adm0.geojson"
    adm1 = tmp_path / "adm1.geojson"
    geometry = box(116.0, 39.0, 116.2, 39.2)
    gpd.GeoDataFrame({"shapeName": ["test"], "geometry": [geometry]}, crs="EPSG:4326").to_file(
        adm0, driver="GeoJSON"
    )
    gpd.GeoDataFrame(
        {"shapeName": ["test-province"], "geometry": [geometry]}, crs="EPSG:4326"
    ).to_file(adm1, driver="GeoJSON")

    records, summary = MODULE.build_inventory(adm0, adm1)
    assert records
    assert all(record["grid_epsg"] == 32650 for record in records)
    assert all(
        record["candidate_count_status"] == "estimate_only_requires_exact_quality_atlas"
        for record in records
    )
    assert summary["estimated_one_percent_base_samples"] > 0


def test_build_inventory_keeps_near_zone_edge_macro_after_projection(tmp_path: Path) -> None:
    """A long 114E edge must remain curved when projected into UTM 50N."""
    adm0 = tmp_path / "adm0.geojson"
    adm1 = tmp_path / "adm1.geojson"
    epsg = 32650
    macro_col, macro_row = 17, 301
    center_x = (macro_col + 0.5) * MODULE.MACRO_SIDE_METERS
    center_y = (macro_row + 0.5) * MODULE.MACRO_SIDE_METERS
    longitude, latitude = Transformer.from_crs(epsg, 4326, always_xy=True).transform(
        center_x, center_y
    )
    geometry = box(113.99, 20.0, longitude + 0.02, 50.0)
    gpd.GeoDataFrame({"shapeName": ["test"], "geometry": [geometry]}, crs="EPSG:4326").to_file(
        adm0, driver="GeoJSON"
    )
    gpd.GeoDataFrame(
        {"shapeName": ["test-province"], "geometry": [geometry]}, crs="EPSG:4326"
    ).to_file(adm1, driver="GeoJSON")

    records, _ = MODULE.build_inventory(adm0, adm1)

    assert f"utm50n_c{macro_col}_r{macro_row}" in {record["macro_id"] for record in records}
