import json

import numpy as np
import pytest

from xuannv_embedding.data_process.osm_weak_targets import (
    AREA_CLASSES,
    categories_for_tags,
    compose_labels,
    overpass_geometry,
    prepare,
)
from xuannv_embedding.export.context import dump, sha


def test_missing_string_tags_are_unknown_not_buildings_or_water():
    for value in [None, float("nan"), "nan", "None", "null", "", "no", "false", "0"]:
        assert "building" not in categories_for_tags({"building": value})
        assert "water" not in categories_for_tags({"water": value})
    assert categories_for_tags({"building": "yes"}) == {"building"}
    assert "forest_tag" in categories_for_tags({"landuse": "forest"})
    assert "wood_tag" in categories_for_tags({"natural": "wood"})


def test_area_priority_and_ambiguous_geometry_keep_unknown_masks():
    masks = {name: np.zeros((2, 3), bool) for name in AREA_CLASSES[1:]}
    masks["forest_tag"][0, :2] = True
    masks["wood_tag"][0, :2] = True
    masks["park_tag"][0, 1] = True
    masks["grass_tag"][1, 0] = True
    masks["farmland_tag"][1, 0] = True
    masks["building"] = np.zeros((2, 3), bool)
    masks["road_corridor"] = np.zeros((2, 3), bool)
    masks["water"] = np.zeros((2, 3), bool)
    masks["building"][0, 0] = masks["water"][0, 0] = True
    masks["building"][1, 1] = True
    area, geometry = compose_labels(masks)
    assert area.tolist() == [[0, 1, 0], [2, 0, 0]]
    assert geometry.tolist() == [[0, 0, 0], [0, 1, 0]]


def test_overpass_polygons_preserve_holes_and_never_invent_open_area():
    def line(points, role):
        return {"role": role, "geometry": [{"lon": x, "lat": y} for x, y in points]}

    outer = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
    inner = [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]
    geometry = overpass_geometry(
        {"type": "relation", "members": [line(outer, "outer"), line(inner, "inner")]}
    )
    assert geometry.area == 15
    open_way = overpass_geometry({"type": "way", **line(outer[:-1], "outer")})
    assert open_way.geom_type == "LineString"


def fixture_targets(tmp_path):
    def ref(p):
        return {"path": str(p), "sha256": sha(p)}

    # WGS84 geometry lies in a small EPSG:3857 metric image.
    source = tmp_path / "osm.json"
    elements = []
    for osm_id, tags, points in [
        (1, {"landuse": "forest"}, [(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001), (0, 0)]),
        (
            2,
            {"leisure": "park", "source": "LCDB 2024"},
            [(0, 0), (0.002, 0), (0.002, 0.002), (0, 0.002), (0, 0)],
        ),
    ]:
        elements.append(
            {
                "type": "way",
                "id": osm_id,
                "tags": tags,
                "geometry": [{"lon": x, "lat": y} for x, y in points],
            }
        )
    elements.append(
        {"type": "node", "id": 3, "lat": 0.0003, "lon": 0.0003, "tags": {"leisure": "park"}}
    )
    dump(source, {"version": 0.6, "elements": elements})
    index = tmp_path / "index.json"
    dump(
        index,
        {
            "crs": "EPSG:3857",
            "records": [
                {"patch_id": "p0", "crs": "EPSG:3857", "bounds": [0, 0, 128, 128]},
                {"patch_id": "p1", "crs": "EPSG:3857", "bounds": [256, 0, 384, 128]},
            ],
        },
    )
    spec = tmp_path / "spec.json"
    dump(
        spec,
        {
            "schema": "osm-weak-targets-v1",
            "index": ref(index),
            "sources": [{"format": "overpass_json", "input": ref(source)}],
            "grid_size": 16,
            "road_halfwidth_m": 5,
            "reserve_gib": 0,
            "output": str(tmp_path / "targets"),
        },
    )
    return spec


def test_prepare_excludes_primary_product_imports_points_and_unmapped_background(tmp_path):
    spec = fixture_targets(tmp_path)
    manifest = prepare(spec)
    meta = json.loads(manifest.read_text())
    area = np.load(meta["area"]["path"])
    assert area.shape == (2, 16, 16)
    assert np.count_nonzero(area[0] == 5) > 0
    assert np.count_nonzero(area == 1) == 0
    assert np.count_nonzero(area[1]) == 0
    assert meta["excluded_primary_source_features"] == 1
    assert meta["unknown_is_background"] is False
    assert meta["area_legend"] == list(AREA_CLASSES)
    assert prepare(spec) == manifest
    area[0, 0, 0] = 1
    np.save(meta["area"]["path"], area)
    with pytest.raises(ValueError, match="changed"):
        prepare(spec)
