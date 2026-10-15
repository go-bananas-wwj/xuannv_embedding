"""Extract dated OSM geometry for a locked annual spatial confirmation set."""

from __future__ import annotations

import json
import math
from datetime import date
from importlib.metadata import version
from pathlib import Path


def prepare_confirmation_osm(plan_path: Path, snapshot: Path, snapshot_date: str) -> None:
    import osmium
    from pyproj import Transformer
    from shapely import wkb
    from shapely.geometry import box
    from shapely.strtree import STRtree

    from xuannv_embedding.data_process.observation_raster import parent_geometry
    from xuannv_embedding.training.annual_experiments import digest, write_json

    date.fromisoformat(snapshot_date)
    plan = json.loads(plan_path.read_text())
    if not plan.get("confirmation_only"):
        raise ValueError("Fine confirmation requires a previously frozen confirmation plan")
    root = plan_path.parent
    destination = root / "plan_fine.json"
    if destination.exists():
        raise ValueError("Fine confirmation plan is already frozen")
    records = json.loads((Path(plan["labels"]) / "labels.json").read_text())["records"]
    ids = sorted(row["patch_id"] for row in records)
    split_of = {row["patch_id"]: row["annual_split"] for row in records}
    snapshot_sha = digest(snapshot)
    geometries, metadata, groups, transforms = [], {}, {}, {}
    for patch_id in ids:
        epsg, (x0, y0, x1, y1) = parent_geometry(patch_id.removeprefix("parent_"))
        if epsg not in transforms:
            transforms[epsg] = Transformer.from_crs(epsg, 4326, always_xy=True)
        project = transforms[epsg]
        xs, ys = zip(
            *[project.transform(x, y) for x in (x0 - 50, x1 + 50) for y in (y0 - 50, y1 + 50)]
        )
        geometries.append(box(min(xs), min(ys), max(xs), max(ys)))
        longitude, latitude = project.transform(x0 + 640, y0 + 640)
        groups[patch_id] = f"block_{math.floor(longitude / 2)}:{math.floor(latitude / 2)}"
        metadata[patch_id] = {
            "epsg": epsg,
            "bounds": [x0, y0, x1, y1],
            "source": f"OSM snapshot {snapshot_date}",
            "snapshot_sha256": snapshot_sha,
        }
    tree = STRtree(geometries)
    factory = osmium.geom.WKBFactory()
    output = {patch_id: {task: [] for task in ["building", "road", "water"]} for patch_id in ids}
    skipped = {"road": 0, "area": 0}

    class Handler(osmium.SimpleHandler):
        def add(self, task, geometry, category=None):
            for index in tree.query(geometry):
                if geometries[index].intersects(geometry):
                    output[ids[index]][task].append((geometry.wkb_hex, category))

        def way(self, way):
            tags = way.tags
            if "highway" not in tags or tags.get("area") == "yes":
                return
            if tags.get("tunnel") in {"yes", "building_passage"}:
                return
            try:
                geometry = wkb.loads(factory.create_linestring(way), hex=True)
            except Exception:
                skipped["road"] += 1
                return
            self.add("road", geometry, tags.get("highway"))

        def area(self, area):
            tags = area.tags
            water = (
                tags.get("natural") == "water"
                or tags.get("landuse") == "reservoir"
                or tags.get("waterway") == "riverbank"
            )
            task = "building" if "building" in tags else "water" if water else None
            if task is None:
                return
            try:
                geometry = wkb.loads(factory.create_multipolygon(area), hex=True)
            except Exception:
                skipped["area"] += 1
                return
            self.add(task, geometry)

    print(json.dumps({"extracting_osm": len(ids), "snapshot": snapshot_date}), flush=True)
    Handler().apply_file(str(snapshot), locations=True, idx="flex_mem")
    osm_root = root / "osm"
    osm_root.mkdir(exist_ok=True)
    for patch_id in ids:
        write_json(osm_root / f"{patch_id}.json", {"_meta": metadata[patch_id], **output[patch_id]})
    city_rows = [[groups[patch_id], split_of[patch_id], patch_id, 0] for patch_id in ids]
    write_json(root / "spatial_patches.json", city_rows)
    result = {
        **plan,
        "fine_confirmation": True,
        "osm": str(osm_root.resolve()),
        "cities": str((root / "spatial_patches.json").resolve()),
        "osm_snapshot_date": snapshot_date,
        "osm_snapshot_sha256": snapshot_sha,
        "osmium_version": version("osmium"),
        "skipped_snapshot_geometries": skipped,
        "fine_label_limitations": (
            "Dated OSM is incomplete weak annotation; scope filtering does not certify "
            "exhaustive negatives or independent 5m ground truth."
        ),
    }
    write_json(destination, result)
    print(
        json.dumps(
            {
                "fine_confirmation_plan": str(destination),
                "geometry_counts": {
                    task: sum(len(row[task]) for row in output.values())
                    for task in ["building", "road", "water"]
                },
                "skipped": skipped,
            }
        ),
        flush=True,
    )
