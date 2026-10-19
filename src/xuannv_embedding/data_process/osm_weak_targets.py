"""Known OSM tag targets; unmapped locations are never negative background."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

from xuannv_embedding.export.context import dump, sha

AREA_CLASSES = (
    "unknown",
    "park_tag",
    "farmland_tag",
    "grass_tag",
    "construction_tag",
    "forest_tag",
    "wood_tag",
)
GEOMETRY_CLASSES = ("unknown", "building", "road_corridor", "water")
TAG_KEYS = {"building", "highway", "water", "waterway", "landuse", "natural", "leisure"}


def clean_tag(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return (
        None
        if text.lower() in {"", "nan", "none", "null", "<na>", "nat", "no", "false", "0"}
        else text
    )


def categories_for_tags(tags: dict) -> set[str]:
    t = {k: clean_tag(v) for k, v in tags.items()}
    result = set()
    lu, nat, leisure = t.get("landuse"), t.get("natural"), t.get("leisure")
    for name, selected in [
        ("forest_tag", lu == "forest"),
        ("wood_tag", nat == "wood"),
        ("park_tag", leisure == "park"),
        ("farmland_tag", lu == "farmland"),
        ("grass_tag", lu in {"grass", "meadow"} or nat == "grassland"),
        ("construction_tag", lu in {"construction", "quarry", "landfill"}),
        ("building", t.get("building") is not None),
        (
            "water",
            nat == "water"
            or t.get("water") is not None
            or t.get("waterway") in {"river", "stream", "canal", "riverbank", "drain"}
            or lu in {"reservoir", "basin"},
        ),
    ]:
        if selected:
            result.add(name)
    road = t.get("highway") or ""
    main = {"motorway", "trunk", "primary", "secondary", "tertiary"}
    if road.removesuffix("_link") in main or road in {
        "residential",
        "unclassified",
        "service",
        "living_street",
        "pedestrian",
        "track",
        "path",
        "footway",
        "cycleway",
        "steps",
        "bridleway",
    }:
        result.add("road_corridor")
    return result


def _primary_source(tags: dict) -> bool:
    for key, value in tags.items():
        if key.startswith("source") or key in {"note", "description", "ref"}:
            text = re.sub(r"[^a-z0-9]", "", str(value).lower())
            if any(word in text for word in ["lcdb", "clcd", "dynamicworld"]):
                return True
    return False


def overpass_geometry(element: dict):
    from shapely import make_valid
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union

    def coordinates(v):
        return [(p["lon"], p["lat"]) for p in v.get("geometry", []) if "lon" in p and "lat" in p]

    points = coordinates(element)
    if len(points) >= 4 and points[0] == points[-1]:
        return make_valid(Polygon(points))
    if len(points) >= 2:
        return LineString(points)
    if element.get("type") == "relation":
        outer, inner = [], []
        for member in element.get("members", []):
            points = coordinates(member)
            if len(points) >= 2:
                (inner if member.get("role") == "inner" else outer).append(LineString(points))
        if outer:
            geometry = unary_union(list(polygonize(unary_union(outer))))
            holes = unary_union(list(polygonize(unary_union(inner)))) if inner else None
            return make_valid(geometry.difference(holes) if holes is not None else geometry)
    return None


def compose_labels(masks: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    shape = next(iter(masks.values())).shape
    if len(shape) != 2 or any(a.shape != shape or a.dtype != bool for a in masks.values()):
        raise ValueError("OSM category masks must be aligned Boolean images")

    def get(k):
        return masks.get(k, np.zeros(shape, bool))

    area = np.zeros(shape, np.uint8)
    area[get("grass_tag")] = 3
    area[get("farmland_tag")] = 2
    forest, wood = get("forest_tag"), get("wood_tag")
    area[forest & ~wood] = 5
    area[wood & ~forest] = 6
    area[forest & wood] = 0
    area[get("construction_tag")] = 4
    area[get("park_tag")] = 1
    geometry = np.zeros(shape, np.uint8)
    types = np.stack([get(k) for k in GEOMETRY_CLASSES[1:]])
    unique = types.sum(0) == 1
    for i, mask in enumerate(types, 1):
        geometry[mask & unique] = i
    return area, geometry


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate weak-target JSON key: " + key)
        result[key] = value
    return result


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(), object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise ValueError("Weak-target JSON must be an object")
    return value


def _verified(ref: dict) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Invalid weak-target source reference")
    path = Path(ref["path"])
    if not path.is_absolute() or sha(path) != ref["sha256"]:
        raise ValueError("Weak-target source identity changed")
    return path


def _polygonal(geometry):
    from shapely.ops import unary_union

    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if geometry.geom_type == "GeometryCollection":
        return unary_union(
            [g for g in geometry.geoms if g.geom_type in {"Polygon", "MultiPolygon"}]
        )
    return None


def _features(source: dict, crs, roi):
    from pyproj import Transformer
    from shapely import make_valid
    from shapely.ops import transform

    path = _verified(source["input"])
    if source["format"] == "overpass_json":
        data = _json(path)
        if not isinstance(data.get("elements"), list) or data.get("remark"):
            raise ValueError("A complete Overpass snapshot with elements is required")
        project = Transformer.from_crs(4326, crs, always_xy=True).transform
        for element in data["elements"]:
            tags = element.get("tags", {})
            if not categories_for_tags(tags) and not _primary_source(tags):
                continue
            geometry = overpass_geometry(element)
            if geometry is not None and not geometry.is_empty:
                geometry = transform(project, geometry)
            yield str(element.get("type")) + ":" + str(element.get("id")), tags, geometry
    elif source["format"] == "geopackage":
        import geopandas as gpd
        import pyogrio

        info = pyogrio.read_info(path)
        fields = set(info["fields"])
        if not {"element", "id"}.issubset(fields) or not info["crs"]:
            raise ValueError("Geopackage must contain identified OSM objects and CRS")
        names = [
            k
            for k in fields
            if k in TAG_KEYS | {"element", "id", "note", "description", "ref"}
            or k.startswith("source")
        ]
        source_roi = transform(
            Transformer.from_crs(crs, info["crs"], always_xy=True).transform, roi
        )
        frame = gpd.read_file(path, bbox=source_roi.bounds, columns=names).to_crs(crs)
        for _, row in frame.iterrows():
            tags = {k: v for k, v in row.items() if k not in {"geometry", "element", "id"}}
            geometry = row.geometry
            if geometry is not None and not geometry.is_empty:
                geometry = make_valid(geometry)
            yield str(row["element"]) + ":" + str(row["id"]), tags, geometry
    else:
        raise ValueError("Unknown OSM source format")


def prepare(path: str | Path) -> Path:
    from pyproj import CRS
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import box
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    path = Path(path)
    spec = _json(path)
    if (
        set(spec)
        != {"schema", "index", "sources", "grid_size", "road_halfwidth_m", "reserve_gib", "output"}
        or spec["schema"] != "osm-weak-targets-v1"
        or type(spec["grid_size"]) is not int
        or spec["grid_size"] < 1
        or type(spec["reserve_gib"]) is not int
        or spec["reserve_gib"] < 0
        or type(spec["road_halfwidth_m"]) not in {int, float}
        or not math.isfinite(spec["road_halfwidth_m"])
        or spec["road_halfwidth_m"] <= 0
        or not isinstance(spec["sources"], list)
        or not spec["sources"]
        or not Path(spec["output"]).is_absolute()
        or any(set(s) != {"format", "input"} for s in spec["sources"])
    ):
        raise ValueError("Invalid self-contained OSM target specification")
    index = _json(_verified(spec["index"]))
    records = index["records"]
    if not records or len({r["patch_id"] for r in records}) != len(records):
        raise ValueError("Unique geographic image records required")
    crs = CRS.from_user_input(index.get("crs") or records[0]["crs"])
    if not crs.is_projected or any(
        a.unit_name.lower() not in {"metre", "meter"} for a in crs.axis_info[:2]
    ):
        raise ValueError("OSM buffers require a projected metric CRS")
    if any(CRS.from_user_input(r.get("crs") or crs) != crs for r in records):
        raise ValueError("OSM image records must share one CRS")
    for source in spec["sources"]:
        _verified(source["input"])
    root = Path(spec["output"])
    ident = {"spec_sha256": sha(path), "implementation_sha256": sha(__file__)}
    root.mkdir(parents=True, exist_ok=True)
    header = root / "identity.json"
    if not header.exists() and any(p.name != "prepare.lock" for p in root.iterdir()):
        raise ValueError("Unowned OSM target output")
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if header.exists() and _json(header) != ident:
            raise ValueError("OSM target specification or implementation changed")
        dump(header, ident)
        manifest = root / "manifest.json"
        if manifest.exists():
            saved = _json(manifest)
            if saved["state"] != "complete":
                raise ValueError("Invalid completed OSM target state")
            for key in ["area", "geometry"]:
                _verified(saved[key])
            return manifest
        required = len(records) * spec["grid_size"] ** 2 * 2
        if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30 + required:
            raise RuntimeError("OSM target preparation disk reserve reached")
        roi = unary_union([box(*r["bounds"]) for r in records])
        entries, blocked = [], set()
        for source in spec["sources"]:
            for key, tags, geometry in _features(source, crs, roi):
                if _primary_source(tags):
                    blocked.add(key)
                entries.append((key, tags, geometry))
        names = list(AREA_CLASSES[1:]) + list(GEOMETRY_CLASSES[1:])
        shapes = {k: [] for k in names}
        seen, unknown = set(), set()
        for key, tags, geometry in entries:
            if (
                key in blocked
                or geometry is None
                or geometry.is_empty
                or not geometry.intersects(roi)
            ):
                continue
            if not any(clean_tag(v) for k, v in tags.items() if k.startswith("source")):
                unknown.add(key)
            for category in categories_for_tags(tags):
                if (key, category) in seen:
                    continue
                seen.add((key, category))
                if category in {"road_corridor", "water"} and geometry.geom_type in {
                    "LineString",
                    "MultiLineString",
                }:
                    area = geometry.buffer(spec["road_halfwidth_m"])
                else:
                    area = _polygonal(geometry)
                if area is not None and not area.is_empty and area.area > 0:
                    shapes[category].append(area)
        trees = {k: STRtree(v) for k, v in shapes.items()}
        arrays = [
            np.lib.format.open_memmap(
                root / (k + ".partial.npy"),
                mode="w+",
                dtype=np.uint8,
                shape=(len(records), spec["grid_size"], spec["grid_size"]),
            )
            for k in ["area", "geometry"]
        ]
        counts = [np.zeros(len(AREA_CLASSES), np.int64), np.zeros(len(GEOMETRY_CLASSES), np.int64)]
        for i, record in enumerate(records):
            window = box(*record["bounds"])
            masks = {}
            for category in names:
                selected = trees[category].query(window, predicate="intersects")
                selected_shapes = [(shapes[category][j], 1) for j in selected]
                masks[category] = (
                    rasterize(
                        selected_shapes,
                        out_shape=(spec["grid_size"],) * 2,
                        transform=from_bounds(
                            *record["bounds"], spec["grid_size"], spec["grid_size"]
                        ),
                        fill=0,
                        dtype="uint8",
                        all_touched=False,
                    ).astype(bool)
                    if selected_shapes
                    else np.zeros((spec["grid_size"],) * 2, bool)
                )
            for a, target, c in zip(arrays, compose_labels(masks), counts):
                a[i] = target
                c += np.bincount(target.ravel(), minlength=len(c))
        for a, key in zip(arrays, ["area", "geometry"]):
            a.flush()
            (root / (key + ".partial.npy")).replace(root / (key + ".npy"))
        dump(
            manifest,
            {
                "state": "complete",
                "schema": "osm-weak-targets-v1",
                "dataset": "OpenStreetMap",
                "index": spec["index"],
                "records": [
                    {
                        k: r.get(k, str(crs) if k == "crs" else None)
                        for k in ["patch_id", "bounds", "crs"]
                    }
                    for r in records
                ],
                "area": {"path": str(root / "area.npy"), "sha256": sha(root / "area.npy")},
                "geometry": {
                    "path": str(root / "geometry.npy"),
                    "sha256": sha(root / "geometry.npy"),
                },
                "area_legend": list(AREA_CLASSES),
                "geometry_legend": list(GEOMETRY_CLASSES),
                "area_counts": counts[0].tolist(),
                "geometry_counts": counts[1].tolist(),
                "source_inputs": spec["sources"],
                "excluded_primary_source_features": len(blocked),
                "unknown_source_provenance_objects": len(unknown),
                "unknown_is_background": False,
                "primary_reference_maps_read": False,
                "forest_wood_are_tag_classes": True,
            },
        )
        return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv experiment osm-weak-targets")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.spec)
    return 0
