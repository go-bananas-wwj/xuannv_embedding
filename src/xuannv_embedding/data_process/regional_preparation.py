"""Materialize frozen regional composites as independent training/evaluation inputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT

from xuannv_embedding.data_process.regional_acquisition import Budget, regular_grid
from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest


def _tile(source: Path, target: Path, bounds, crs, resolution, quality=False, categorical=False):
    size = math.ceil((bounds[2] - bounds[0]) / resolution)
    transform = from_bounds(*bounds, size, size)
    with (
        rasterio.open(source) as src,
        WarpedVRT(
            src,
            crs=crs,
            transform=transform,
            width=size,
            height=size,
            nodata=-9999,
            dtype="float32",
            src_nodata=None if resolution == 3 else src.nodata,
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        ) as vrt,
    ):
        values = vrt.read(masked=True)
        valid = ~np.ma.getmaskarray(values).any(axis=0)
        data = values.filled(-9999).astype("float32")
    if quality:
        data = np.concatenate([data, np.zeros((1, size, size), dtype="float32")])
    if categorical:
        converted = np.zeros(data.shape, dtype="float32")
        for index, code in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], start=1):
            converted[data == code] = index
        data = converted
        valid &= data[0] != 255
    data[:, ~valid] = 0 if categorical else -9999
    target.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=size,
        height=size,
        count=len(data),
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=0 if categorical else -9999,
        tiled=True,
        compress="deflate",
        predictor=3,
    )
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data)
    mask = target.with_name(target.stem + "_mask.tif")
    with rasterio.open(
        mask, "w", **{**profile, "count": 1, "dtype": "uint8", "nodata": 0, "predictor": 1}
    ) as dst:
        dst.write(valid.astype("uint8"), 1)
    return data, valid


def materialize_region(spec: dict) -> dict:
    """Publish a manifest only after every monthly source and static input is present."""
    root = Path(spec["root"])
    months = spec["months"]
    crs = spec["crs"]
    budget = Budget(
        root, int(spec.get("max_gib", 150) * 1024**3), int(spec.get("min_free_gib", 100) * 1024**3)
    )
    inputs = {
        source: [(month, root / "monthly" / source / (month + ".tif")) for month in months]
        for source in ["s2", "s1", "landsat"]
    }
    inputs["highres_optical"] = [("static_" + months[0][:4], root / "static/aerial.tif")]
    inputs["worldcover"] = [("static_2021", root / "weak_labels/worldcover_2021.tif")]
    missing = [str(path) for group in inputs.values() for _, path in group if not path.is_file()]
    if missing:
        raise ValueError(f"regional sources are incomplete ({len(missing)} missing rasters)")
    grid = regular_grid(tuple(spec["bounds"]), spec["patch_size_m"])
    records = []
    coverage = []
    stats = {}
    processed = root / "processed"
    processed.mkdir(exist_ok=True)
    for cell in grid:
        budget.check(20 * 1024**2)
        sources = {}
        fractions = {}
        for source, observations in inputs.items():
            sources[source] = []
            fractions[source] = {}
            for month, path in observations:
                stamp = month.replace("-", "") + "01" if month[:4].isdigit() else month
                target = processed / "patches" / source / f'{source}_{stamp}_{cell["patch_id"]}.tif'
                data, valid = _tile(
                    path,
                    target,
                    cell["bounds"],
                    crs,
                    3 if source == "highres_optical" else 10,
                    quality=source in {"s2", "landsat"},
                    categorical=source == "worldcover",
                )
                sources[source].append(str(target.relative_to(root)))
                fractions[source][month] = float(valid.mean())
                if source != "worldcover" and valid.any():
                    sample = data[:, valid].astype("float64")
                    n = sample.shape[1]
                    sums, squares, counts = stats.setdefault(
                        source, (np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data)))
                    )
                    sums += sample.sum(axis=1)
                    squares += (sample**2).sum(axis=1)
                    counts += n
        records.append(
            ManifestRecord(
                cell["patch_id"],
                spec.get("region", "regional_2024"),
                sources,
                grid={"crs": crs, "bounds": cell["bounds"]},
                quality={"valid_fraction": fractions},
                provenance={
                    "static_highres": "one annual observation, not a monthly time series",
                    "months": months,
                    "weak_worldcover_year": 2021,
                },
            )
        )
        coverage.append({"patch_id": cell["patch_id"], "sources": fractions})
    statistics = root / "statistics"
    statistics.mkdir(exist_ok=True)
    for source, (sums, squares, counts) in stats.items():
        means = sums / counts
        std = np.sqrt(np.maximum(0, squares / counts - means**2))
        std[std < 1e-8] = 1
        (statistics / (source + "_stats.json")).write_text(
            json.dumps(
                {
                    "mean": means.tolist(),
                    "std": std.tolist(),
                    "count": counts.astype(int).tolist(),
                    "scope": (
                        "all representation-training patches; " "no downstream reference labels"
                    ),
                },
                indent=2,
            )
        )
    (processed / "patch_grid.json").write_text(json.dumps({"crs": crs, "patches": grid}, indent=2))
    write_manifest(
        processed / "manifest.json",
        records,
        months=months,
        generator_version="xuannv/regional-preparation-v1",
    )
    result = {
        "state": "materialized",
        "patches": len(records),
        "months": len(months),
        "coverage": coverage,
        "static_highres_observations_per_patch": 1,
        "highres_resolution_m": "1280/427, nominal 3 m; exact patch bounds retained",
        "training_started": False,
        "evaluation_references_in_training": False,
    }
    (root / "materialization-report.json").write_text(json.dumps(result, indent=2))
    return result


def prepare_weak_semantics(spec: dict) -> dict:
    """OSM positives and explicitly identified weak negatives; other pixels stay unknown."""
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely import make_valid
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, transform, unary_union

    root = Path(spec["root"])
    snapshot = root / spec.get("osm_snapshot_path", "weak_labels/osm_snapshot.json")
    elements = json.loads(snapshot.read_text())["elements"]
    project = Transformer.from_crs("EPSG:4326", spec["crs"], always_xy=True).transform

    def geometry(element):
        coordinates = [
            (p["lon"], p["lat"]) for p in element.get("geometry", []) if "lon" in p and "lat" in p
        ]
        if len(coordinates) >= 3 and coordinates[0] == coordinates[-1]:
            return make_valid(Polygon(coordinates))
        if len(coordinates) >= 2:
            return LineString(coordinates)
        if element.get("type") == "relation":
            outer = []
            inner = []
            for member in element.get("members", []):
                points = [
                    (p["lon"], p["lat"])
                    for p in member.get("geometry", [])
                    if "lon" in p and "lat" in p
                ]
                if len(points) >= 2:
                    (inner if member.get("role") == "inner" else outer).append(LineString(points))
            if outer:
                outlines = unary_union(list(polygonize(unary_union(outer))))
                holes = unary_union(list(polygonize(unary_union(inner)))) if inner else None
                return outlines.difference(holes) if holes is not None else outlines
        return None

    negatives = {
        "building": [10, 20, 30, 40, 80],
        "major_road": [80],
        "minor_road": [80],
        "rail": [80],
        "water": [10, 40, 50],
        "green": [50, 80],
        "agriculture": [50, 80],
        "residential": [80],
        "commercial": [80],
        "industrial": [80],
        "construction": [10, 80],
        "path_walk": [80],
        "playground": [80],
    }
    shapes = {name: [] for name in negatives}
    for element in elements:
        tags = element.get("tags", {})
        categories = []
        road = tags.get("highway", "")
        if tags.get("building") not in {None, "no"}:
            categories.append("building")
        if road in {"motorway", "trunk", "primary", "secondary", "tertiary"} or road.endswith(
            "_link"
        ):
            categories.append("major_road")
        if road in {"residential", "unclassified", "service", "living_street"}:
            categories.append("minor_road")
        if road in {"path", "footway", "pedestrian", "cycleway", "steps", "bridleway"}:
            categories.append("path_walk")
        if tags.get("railway") in {"rail", "tram", "light_rail", "subway"}:
            categories.append("rail")
        if (
            tags.get("natural") in {"water", "wetland"}
            or tags.get("waterway") in {"river", "stream", "canal", "riverbank", "drain"}
            or tags.get("landuse") in {"reservoir", "basin"}
        ):
            categories.append("water")
        if (
            tags.get("natural") in {"wood", "scrub", "grassland", "heath"}
            or tags.get("landuse") in {"forest", "grass", "meadow", "recreation_ground"}
            or tags.get("leisure") in {"park", "garden"}
        ):
            categories.append("green")
        if tags.get("landuse") in {
            "farmland",
            "farmyard",
            "orchard",
            "vineyard",
            "plant_nursery",
            "greenhouse_horticulture",
        }:
            categories.append("agriculture")
        for category, values in {
            "residential": {"residential"},
            "commercial": {"commercial", "retail"},
            "industrial": {"industrial"},
            "construction": {"construction", "quarry", "landfill"},
        }.items():
            if tags.get("landuse") in values:
                categories.append(category)
        if tags.get("leisure") == "playground":
            categories.append("playground")
        if not categories:
            continue
        shape = geometry(element)
        if shape is None or shape.is_empty:
            continue
        shape = transform(project, shape)
        for category in categories:
            value = shape
            if shape.geom_type in {"LineString", "MultiLineString"}:
                radius = {
                    "major_road": 12,
                    "minor_road": 5,
                    "rail": 4,
                    "water": 5,
                    "path_walk": 2,
                }.get(category)
                if radius is None:
                    continue
                value = shape.buffer(radius)
            if value.geom_type in {"Polygon", "MultiPolygon"}:
                shapes[category].append((value, 1))
    with rasterio.open(root / "weak_labels/worldcover_2021.tif") as src:
        wc = src.read(1)
        profile = src.profile
        affine = src.transform
    grid = regular_grid(tuple(spec["bounds"]), spec["patch_size_m"])
    summary = {}
    for task, values in shapes.items():
        positive = (
            rasterize(values, out_shape=wc.shape, transform=affine, dtype="uint8")
            if values
            else np.zeros(wc.shape, "uint8")
        )
        negative = np.isin(wc, negatives[task])
        padded = np.pad(negative, 1, constant_values=False)
        eroded = np.ones(wc.shape, bool)
        for row in range(3):
            for col in range(3):
                eroded &= padded[row : row + wc.shape[0], col : col + wc.shape[1]]
        valid = (eroded | (positive > 0)) if values else np.zeros(wc.shape, bool)
        folder = root / "processed/labels" / ("osm_" + task)
        for cell in grid:
            row = cell["row"] * int(spec["patch_size_m"] / 10)
            col = cell["col"] * int(spec["patch_size_m"] / 10)
            n = int(spec["patch_size_m"] / 10)
            b = cell["bounds"]
            for kind, array in [("masks", positive), ("valid_masks", valid.astype("uint8"))]:
                path = folder / kind / (cell["patch_id"] + ".tif")
                path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(
                    path,
                    "w",
                    **{
                        **profile,
                        "count": 1,
                        "dtype": "uint8",
                        "width": n,
                        "height": n,
                        "transform": from_bounds(*b, n, n),
                        "nodata": None,
                    },
                ) as dst:
                    dst.write(array[row : row + n, col : col + n], 1)
        summary["osm_" + task] = {
            "positive_pixels": int(positive.sum()),
            "valid_pixels": int(valid.sum()),
            "weak_negative_worldcover_codes": negatives[task],
        }
    report = {
        "tasks": summary,
        "unmapped_osm_pixels": "unknown unless selected by declared weak-negative rules",
        "negative_boundary_erosion_m": 10,
        "line_buffer_radius_m": {
            "major_road": 12,
            "minor_road": 5,
            "rail": 4,
            "water": 5,
            "path_walk": 2,
        },
        "source_snapshot": str(snapshot),
        "evaluation_labels_used": False,
        "training_started": False,
    }
    (root / "weak-semantics-report.json").write_text(json.dumps(report, indent=2))
    return report


def prepare_evaluation_targets(spec: dict) -> dict:
    """Derive held-out land-cover, surface-height and terrain references, never model inputs."""
    import csv
    import warnings

    from rasterio.features import rasterize
    from rasterio.windows import from_bounds as window_from_bounds
    from shapely.geometry import box, shape
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    root = Path(spec["root"])
    output = root / "evaluation"
    output.mkdir(exist_ok=True)
    grid = regular_grid(tuple(spec["bounds"]), spec["patch_size_m"])
    width = int((spec["bounds"][2] - spec["bounds"][0]) / 10)
    height = int((spec["bounds"][3] - spec["bounds"][1]) / 10)
    affine = from_bounds(*spec["bounds"], width, height)
    fields = np.full((4, height, width), np.nan, dtype="float32")
    with (
        rasterio.open(root / "static/dem.tif") as dem,
        rasterio.open(root / "static/dsm.tif") as dsm,
    ):
        if (dem.crs, dem.transform, dem.width, dem.height) != (
            dsm.crs,
            dsm.transform,
            dsm.width,
            dsm.height,
        ):
            raise ValueError("DEM and DSM must share their native grid before differencing")
        for cell in grid:
            window = window_from_bounds(*cell["bounds"], transform=dem.transform)
            ground = dem.read(1, window=window, masked=True).filled(np.nan)
            surface = dsm.read(1, window=window, masked=True).filled(np.nan)
            n = int(spec["patch_size_m"] / 10)
            if ground.shape != (n * 10, n * 10):
                raise ValueError("LiDAR labels require a 1 m native grid")
            valid = np.isfinite(ground) & np.isfinite(surface)
            ground[~valid] = np.nan
            delta = np.maximum(surface - ground, 0)
            delta[~valid] = np.nan

            def blocks(array):
                return array.reshape(n, 10, n, 10).transpose(0, 2, 1, 3).reshape(n, n, 100)

            row = cell["row"] * n
            col = cell["col"] * n
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fields[0, row : row + n, col : col + n] = np.nanmean(blocks(ground), axis=-1)
                fields[2, row : row + n, col : col + n] = np.nanmean(blocks(delta), axis=-1)
                fields[3, row : row + n, col : col + n] = np.nanpercentile(
                    blocks(delta), 90, axis=-1
                )
    dy, dx = np.gradient(fields[0], 10, 10)
    fields[1] = np.degrees(np.arctan(np.hypot(dx, dy)))
    profile = dict(
        driver="GTiff",
        width=width,
        height=height,
        count=4,
        dtype="float32",
        crs=spec["crs"],
        transform=affine,
        nodata=-9999,
        tiled=True,
        compress="deflate",
    )
    with rasterio.open(output / "lidar_2024_targets.tif", "w", **profile) as dst:
        dst.write(np.nan_to_num(fields, nan=-9999))
        for i, name in enumerate(
            [
                "DEM_mean_m",
                "terrain_slope_degrees",
                "surface_height_mean_m",
                "surface_height_p90_m",
            ],
            1,
        ):
            dst.set_band_description(i, name)
    lcdb = json.loads((root / "evaluation_sources/lcdb_v6_generalized15m.geojson").read_text())
    polygons = []
    names = {}
    for feature in lcdb["features"]:
        properties = feature["properties"]
        code = properties.get("Class_2023")
        if code is None or not feature.get("geometry"):
            continue
        polygons.append((feature["geometry"], int(code)))
        names[int(code)] = properties.get("Name_2023")
    labels = rasterize(
        polygons, out_shape=(height, width), transform=affine, fill=0, dtype="uint16"
    )
    interior = labels > 0
    padded = np.pad(labels.astype("int32"), 2, constant_values=-1)
    for row in range(5):
        for col in range(5):
            interior &= padded[row : row + height, col : col + width] == labels
    with rasterio.open(
        output / "lcdb_2023_2024_classes.tif",
        "w",
        **{**profile, "count": 1, "dtype": "uint16", "nodata": 0},
    ) as dst:
        dst.write(labels, 1)
    with rasterio.open(
        output / "lcdb_interior_mask.tif",
        "w",
        **{**profile, "count": 1, "dtype": "uint8", "nodata": 0},
    ) as dst:
        dst.write(interior.astype("uint8"), 1)
    buildings = json.loads(
        (root / "evaluation_sources/buildings_mixed_2017_2022.geojson").read_text()
    )
    roofs = []
    for feature in buildings["features"]:
        if feature.get("geometry"):
            roof = shape(feature["geometry"])
            if not roof.is_empty and roof.area > 0:
                roofs.append(roof)
    tree = STRtree(roofs)
    counts = np.zeros(len(grid), int)
    nx = int((spec["bounds"][2] - spec["bounds"][0]) / spec["patch_size_m"])
    for roof in roofs:
        center = roof.centroid
        col = int((center.x - spec["bounds"][0]) / spec["patch_size_m"])
        row = int((spec["bounds"][3] - center.y) / spec["patch_size_m"])
        if 0 <= col < nx and 0 <= row < len(grid) // nx:
            counts[row * nx + col] += 1
    with (output / "building_legacy_patch_targets.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "patch_id",
                "coverage_fraction",
                "count",
                "density_per_km2",
                "reference_years",
            ],
        )
        writer.writeheader()
        for i, cell in enumerate(grid):
            polygon = box(*cell["bounds"])
            candidates = [roofs[j] for j in tree.query(polygon)]
            area = unary_union(candidates).intersection(polygon).area if candidates else 0
            writer.writerow(
                {
                    "patch_id": cell["patch_id"],
                    "coverage_fraction": area / polygon.area,
                    "count": int(counts[i]),
                    "density_per_km2": counts[i] * 1e6 / polygon.area,
                    "reference_years": "2017;2022",
                }
            )
    report = {
        "training_inputs": False,
        "LCDB": {
            "source_period": "2023/24",
            "product": "v6.0",
            "geometry_simplification_m": 15,
            "interior_exclusion_pixels": 2,
            "grid_spacing_m": 10,
            "classes": names,
            "valid_interior_fraction": float(interior.mean()),
        },
        "LiDAR": {
            "capture_period": "2024 Apr-Jun, Part 1",
            "vertical_datum": "NZVD2016",
            "height_definition": "nonnegative DSM minus DEM at 1 m, aggregated to 10 m",
            "height_is_not_building_only": True,
            "valid_height_fraction": float(np.isfinite(fields[2]).mean()),
        },
        "buildings": {
            "reference_years": [2017, 2022],
            "role": "historical diagnostic reference only",
            "eligible_for_claims_about_2024_building_accuracy": False,
        },
        "optional_soil_luc_sports": "not admitted without verified source/license/coverage",
    }
    (root / "evaluation-reference-report.json").write_text(json.dumps(report, indent=2))
    return report


def finalize_region(spec: dict) -> dict:
    """Finish data preparation and validate both loaders; never invoke training."""
    import hashlib

    from xuannv_embedding.config import Config
    from xuannv_embedding.data.raster_dataset import RegionRasterDataset

    root = Path(spec["root"])
    from xuannv_embedding.data_process.regional_acquisition import LANDSAT_QUALITY_VERSION

    for month in spec["months"]:
        receipt = root / "monthly" / "landsat" / (month + ".json")
        if (
            not receipt.exists()
            or json.loads(receipt.read_text()).get("quality_mask_version")
            != LANDSAT_QUALITY_VERSION
        ):
            raise ValueError("Landsat quality-mask version is not verified")
    materialized = materialize_region(spec)
    for receipt in (root / "monthly" / "s1").glob("*.json"):
        source_metadata = json.loads(receipt.read_text())
        source_metadata.update(
            composite="per-pixel validity-masked monthly median",
            quantity="gamma-naught",
            units="linear",
            native_grid_spacing_m=10,
            effective_sar_resolution_m=[20, 22],
        )
        source_metadata.pop("native_source_resolution_m", None)
        receipt.write_text(json.dumps(source_metadata, indent=2))
    weak = prepare_weak_semantics(spec)
    evaluation = prepare_evaluation_targets(spec)
    geographic_audit = audit_geography_and_panels(spec, materialized["coverage"])
    audits = []
    for name in ["public_three_source", "linz_static_rgb"]:
        config = Config.from_yaml(root / "configs" / (name + ".yaml"))
        dataset = RegionRasterDataset(config, config.data.datasets[0], max_records=2)
        shapes = []
        for index in range(len(dataset)):
            sample = dataset[index]
            assert sample["timestamps"].numel() == len(spec["months"])
            for tensor in sample["source_frames"].values():
                if not bool(tensor.isfinite().all()):
                    raise ValueError("nonfinite regional source")
            for head_name, head in config.model.target_heads.items():
                if head.loss_type == "categorical":
                    target = sample["targets"][head_name]
                    valid = sample["target_masks"][head_name] > 0
                    if bool(((target < 0) | (target >= head.channels)).any()):
                        raise ValueError("categorical target exceeds decoder class indices")
                    if bool((target[valid] == 0).any()):
                        raise ValueError(
                            "valid land-cover classes cannot use the ignored zero slot"
                        )
            if name == "linz_static_rgb":
                assert sample["highres_frames"]["highres_optical"].dim() == 3
            shapes.append({k: list(v.shape) for k, v in sample["source_frames"].items()})
        audits.append({"config": name, "input_shapes": shapes, "loaded_without_training": True})
    provenance = {}
    for path in (root / "catalogs").rglob("*.json"):
        provenance[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    from xuannv_embedding.data_process.regional_visualization import render_data_visualizations

    visualization = render_data_visualizations(
        spec, require_complete=True, label=spec.get("visualization_label", "研究区")
    )
    report = {
        "state": "data_ready",
        "visualization": visualization["overview"],
        "visualization_paths": visualization["path_inventory"],
        "patches": materialized["patches"],
        "months": len(spec["months"]),
        "loader_audits": audits,
        "geography_audit": geographic_audit,
        "evaluation": evaluation,
        "weak_semantic_tasks": weak["tasks"],
        "source_metadata_sha256": provenance,
        "training_started": False,
        "overleaf_modified": False,
        "legacy_building_labels_are_not_2024_ground_truth": True,
        "public_and_highres_versions_share_public_inputs_and_weak_labels": True,
    }
    (root / "readiness.json").write_text(json.dumps(report, indent=2))
    return report


def audit_geography_and_panels(spec: dict, coverage: list[dict]) -> dict:
    """Check spatial headers independently and save source panels for visual review."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root = Path(spec["root"])
    grid = regular_grid(tuple(spec["bounds"]), spec["patch_size_m"])
    records = json.loads((root / "processed/manifest.json").read_text())
    chosen = sorted({0, len(grid) // 2, len(grid) - 1})
    max_error = 0
    checked = 0
    for index in chosen:
        expected = np.array(grid[index]["bounds"])
        for values in records[index]["sources"].values():
            for relative in values:
                with rasterio.open(root / relative) as src:
                    if str(src.crs) != spec["crs"]:
                        raise ValueError("prepared source CRS differs")
                    max_error = max(max_error, float(np.abs(np.array(src.bounds) - expected).max()))
                    checked += 1
    if max_error > 1e-6:
        raise ValueError("prepared source bounds differ from frozen grid")
    candidates = [
        i
        for i, r in enumerate(coverage)
        if r["sources"]["highres_optical"]["static_" + spec["months"][0][:4]] >= 0.8
        and max(r["sources"]["s2"].values()) >= 0.5
    ]
    panels = []
    directory = root / "quality_panels"
    directory.mkdir(exist_ok=True)
    if candidates:
        for index in sorted({candidates[0], candidates[len(candidates) // 2], candidates[-1]}):
            month = max(coverage[index]["sources"]["s2"], key=coverage[index]["sources"]["s2"].get)
            month_index = spec["months"].index(month)
            paths = [
                records[index]["sources"]["highres_optical"][0],
                records[index]["sources"]["s2"][month_index],
            ]
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            from xuannv_embedding.data_process.regional_visualization import rgb_band_order

            for source, ax, relative, title in zip(
                ["aerial", "s2"], axes, paths, ["static aerial RGB", "S2 " + month]
            ):
                with rasterio.open(root / relative) as src:
                    image = src.read(rgb_band_order(source)).transpose(1, 2, 0)
                    mask = src.read_masks(1) > 0
                good = image[mask]
                if good.size:
                    lo, hi = np.percentile(good, [2, 98])
                    image = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
                image[~mask] = 0
                ax.imshow(image)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(grid[index]["patch_id"] + " | identical geographic bounds")
            fig.tight_layout()
            path = directory / (grid[index]["patch_id"] + ".png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            panels.append(str(path))
    report = {
        "header_checks": checked,
        "max_bounds_difference_m": max_error,
        "crs": spec["crs"],
        "panels": panels,
        "visual_review": "panels provided; no claim of measured subpixel registration accuracy",
    }
    (root / "co_registration_report.json").write_text(json.dumps(report, indent=2))
    return report
