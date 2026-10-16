"""Bounded, resumable regional raster acquisition without training side effects."""

from __future__ import annotations

import json
import math
import os
from calendar import monthrange
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urljoin

import numpy as np
import rasterio
import requests
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

GDAL_CACHE_BYTES = 128 * 1024**2
LANDSAT_QUALITY_VERSION = "radsat-zero-valid-v1"


@dataclass(frozen=True)
class Budget:
    root: Path
    max_bytes: int = 150 * 1024**3
    reserve_bytes: int = 100 * 1024**3

    def check(self, additional_bytes: int = 0) -> None:
        used = 0
        seen = set()
        for directory, _, names in os.walk(self.root):
            for name in names:
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                try:
                    info = path.stat()
                except FileNotFoundError:
                    continue
                key = (info.st_dev, info.st_ino)
                if key not in seen:
                    seen.add(key)
                    used += info.st_blocks * 512
        if used + additional_bytes > self.max_bytes:
            raise RuntimeError("regional preparation budget would be exceeded")
        space = os.statvfs(self.root)
        available = space.f_bavail * space.f_frsize
        if available - additional_bytes < self.reserve_bytes:
            raise RuntimeError("regional preparation free-space reserve would be exceeded")


def regular_grid(bounds: tuple, size: int) -> list[dict]:
    left, bottom, right, top = bounds
    width, height = (right - left) / size, (top - bottom) / size
    if size <= 0 or width <= 0 or height <= 0 or width != int(width) or height != int(height):
        raise ValueError("bounds must comprise complete equal-sized grid cells")
    return [
        {
            "patch_id": f"patch_{row*int(width)+col:06d}",
            "row": row,
            "col": col,
            "bounds": [
                left + col * size,
                top - (row + 1) * size,
                left + (col + 1) * size,
                top - row * size,
            ],
        }
        for row in range(int(height))
        for col in range(int(width))
    ]


def optical_valid(source: str, quality: np.ndarray, saturated: np.ndarray | None = None):
    if source == "s2":
        return np.isin(quality, [2, 4, 5, 6, 7])
    if source != "landsat":
        raise ValueError("unknown optical quality convention")
    valid = (quality.astype(np.uint32) & 0b111111) == 0
    return valid if saturated is None else valid & (saturated == 0)


def select_optical_items(items: list[dict], per_tile: int = 3) -> list[dict]:
    groups = {}
    for item in items:
        p = item["properties"]
        tile = p.get("s2:mgrs_tile") or str(p.get("landsat:wrs_path", "")) + "_" + str(
            p.get("landsat:wrs_row", "")
        )
        groups.setdefault(tile, []).append(item)
    result = []
    for values in groups.values():
        dates = set()
        for item in sorted(
            values, key=lambda x: (x["properties"].get("eo:cloud_cover", 100), x["id"])
        ):
            day = item["properties"]["datetime"][:10]
            if day not in dates:
                result.append(item)
                dates.add(day)
            if len(dates) >= per_tile:
                break
    return result


def fetch_json(url: str, params: dict | None = None) -> dict:
    if params and len(urlencode(params)) > 1800:
        response = requests.post(url, data=params, timeout=(15, 120))
    else:
        response = requests.get(url, params=params, timeout=(15, 120))
    response.raise_for_status()
    return response.json()


def catalog_items(url: str, bbox: tuple, interval: str, collection: str) -> list[dict]:
    params = {
        "bbox": ",".join(map(str, bbox)),
        "datetime": interval,
        "collections": collection,
        "limit": 100,
    }
    items = []
    while url:
        page = fetch_json(url, params)
        items.extend(page.get("features", []))
        url = next((x["href"] for x in page.get("links", []) if x["rel"] == "next"), None)
        params = None
    return items


def local_grid(spec: dict, resolution: float):
    left, bottom, right, top = spec["bounds"]
    return (
        from_origin(left, top, resolution, resolution),
        math.ceil((right - left) / resolution),
        math.ceil((top - bottom) / resolution),
    )


def window_mosaic(
    assets: list[str],
    destination: Path,
    spec: dict,
    resolution: float,
    count: int,
    dtype: str,
    budget: Budget,
    resampling=Resampling.average,
    *,
    source_overviews: bool = False,
) -> dict:
    """Read COG windows only; retain holes via nodata and an explicit validity mask."""
    receipt = destination.with_suffix(".json")
    if destination.exists() and receipt.exists():
        return json.loads(receipt.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    transform, width, height = local_grid(spec, resolution)
    budget.check(width * height * count * np.dtype(dtype).itemsize)
    temporary = destination.with_suffix(".partial.tif")
    nodata = 0 if np.issubdtype(np.dtype(dtype), np.integer) else -9999
    profile = dict(
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        crs=spec["crs"],
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )
    import contextlib

    valid_pixels = 0
    with (
        rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MAX_RETRY=3,
            GDAL_HTTP_TIMEOUT=90,
            GDAL_CACHEMAX=GDAL_CACHE_BYTES,
        ),
        contextlib.ExitStack() as stack,
    ):
        readers = []
        overview_records = []
        for url in assets:
            src = stack.enter_context(rasterio.open(url))
            if source_overviews:
                native_resolution = max(abs(v) for v in src.res)
                eligible = [
                    (i, factor)
                    for i, factor in enumerate(src.overviews(1))
                    if native_resolution * factor <= resolution
                ]
                if eligible:
                    level, factor = eligible[-1]
                    src = stack.enter_context(rasterio.open(url, overview_level=level))
                    overview_records.append(
                        {
                            "source": url,
                            "factor": factor,
                            "native_resolution_m": native_resolution,
                            "read_resolution_m": max(abs(v) for v in src.res),
                        }
                    )
            readers.append(
                stack.enter_context(
                    WarpedVRT(
                        src,
                        crs=spec["crs"],
                        transform=transform,
                        width=width,
                        height=height,
                        resampling=resampling,
                        nodata=None if ColorInterp.alpha in src.colorinterp else nodata,
                        src_nodata=None if ColorInterp.alpha in src.colorinterp else src.nodata,
                    )
                )
            )
        with rasterio.open(temporary, "w", **profile) as target:
            for row in range(0, height, 512):
                budget.check()
                for col in range(0, width, 512):
                    w = Window(col, row, min(512, width - col), min(512, height - row))
                    result = np.full((count, int(w.height), int(w.width)), nodata, dtype=dtype)
                    valid = np.zeros((int(w.height), int(w.width)), bool)
                    for source in readers:
                        values = source.read(list(range(1, count + 1)), window=w, masked=True)
                        good = ~np.ma.getmaskarray(values).any(axis=0)
                        use = good & ~valid
                        result[:, use] = values.data[:, use]
                        valid |= good
                    target.write(result, window=w)
                    target.write_mask(valid.astype("uint8") * 255, window=w)
                    valid_pixels += int(valid.sum())
        # GDAL may use an external mask; move it with its raster atomically.
    os.replace(temporary, destination)
    mask = Path(str(temporary) + ".msk")
    if mask.exists():
        os.replace(mask, Path(str(destination) + ".msk"))
    result = {
        "state": "complete",
        "path": str(destination),
        "shape": [count, height, width],
        "resolution_m": resolution,
        "valid_fraction": valid_pixels / (width * height),
        "source_overviews": overview_records,
        "sources": assets,
        "crs": spec["crs"],
        "bounds": spec["bounds"],
    }
    receipt.write_text(json.dumps(result, indent=2))
    return result


def linz_assets(collection_url: str, bbox: tuple, metadata_dir: Path) -> tuple[list[str], dict]:
    """Discover intersecting LINZ STAC items; metadata and unsigned URLs are preserved."""
    from concurrent.futures import ThreadPoolExecutor

    from shapely.geometry import box, shape

    metadata_dir.mkdir(parents=True, exist_ok=True)
    collection_path = metadata_dir / "collection.json"
    if not collection_path.exists():
        collection_path.write_text(json.dumps(fetch_json(collection_url)))
    collection = json.loads(collection_path.read_text())
    links = [urljoin(collection_url, x["href"]) for x in collection["links"] if x["rel"] == "item"]
    roi = box(*bbox)

    def read(url):
        path = metadata_dir / Path(url).name
        if not path.exists():
            path.write_text(json.dumps(fetch_json(url)))
        item = json.loads(path.read_text())
        if not shape(item["geometry"]).intersects(roi):
            return None
        asset = item["assets"].get("visual") or item["assets"].get("data")
        if asset is None:
            asset = next(v for v in item["assets"].values() if ".tif" in v.get("href", ""))
        return urljoin(url, asset["href"])

    with ThreadPoolExecutor(6) as pool:
        assets = [x for x in pool.map(read, links) if x]
    return assets, collection


def read_optical_asset(item: dict, name: str, spec: dict, categorical=False):
    """Sign only at read time; never persist expiring SAS URLs."""
    import planetary_computer

    transform, width, height = local_grid(spec, 10)
    url = planetary_computer.sign(item["assets"][name]["href"])
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MAX_RETRY=3,
        GDAL_HTTP_TIMEOUT=90,
        GDAL_CACHEMAX=GDAL_CACHE_BYTES,
    ):
        with (
            rasterio.open(url) as src,
            WarpedVRT(
                src,
                crs=spec["crs"],
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.nearest if categorical else Resampling.bilinear,
                nodata=0,
            ) as vrt,
        ):
            return vrt.read(1, masked=True)


def acquire_optical_item(
    item: dict, source: str, destination: Path, spec: dict, budget: Budget
) -> Path:
    import xml.etree.ElementTree as ET

    import planetary_computer

    if destination.exists() and destination.with_suffix(".json").exists():
        cached = json.loads(destination.with_suffix(".json").read_text())
        if source != "landsat" or cached.get("quality_mask_version") == LANDSAT_QUALITY_VERSION:
            return destination
    bands = (
        ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        if source == "s2"
        else ["blue", "green", "red", "nir08", "swir16", "swir22"]
    )
    budget.check(300 * 1024**2)
    q = read_optical_asset(item, "SCL" if source == "s2" else "qa_pixel", spec, True)
    # Zero means no saturated bands. QA_RADSAT's zero-valued nodata mask must not
    # remove clear pixels; QA_PIXEL and reflectance masks determine availability.
    sat = None if source == "s2" else read_optical_asset(item, "qa_radsat", spec, True).filled(0)
    valid = optical_valid(source, q.filled(0), sat) & ~np.ma.getmaskarray(q)
    offsets = {}
    quantification = None
    if source == "s2":
        url = planetary_computer.sign(item["assets"]["product-metadata"]["href"])
        response = requests.get(url, timeout=(15, 120))
        response.raise_for_status()
        tree = ET.fromstring(response.content)
        for e in tree.iter():
            tag = e.tag.split("}")[-1]
            if tag == "BOA_QUANTIFICATION_VALUE":
                quantification = float(e.text)
            if tag == "BOA_ADD_OFFSET":
                offsets[int(e.attrib["band_id"])] = float(e.text)
        if quantification is None:
            raise ValueError("S2 radiometry metadata missing")
        baseline = float(item["properties"].get("s2:processing_baseline", 0))
        if baseline >= 4 and not offsets:
            raise ValueError("S2 radiometry offsets missing")
    transform, width, height = local_grid(spec, 10)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".partial.tif")
    band_ids = {
        n: i
        for i, n in enumerate(
            [
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                "B10",
                "B11",
                "B12",
            ]
        )
    }
    with rasterio.open(
        tmp,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=len(bands),
        dtype="float32",
        crs=spec["crs"],
        transform=transform,
        nodata=-9999,
        tiled=True,
        compress="deflate",
        predictor=3,
    ) as dst:
        for i, name in enumerate(bands, 1):
            values = read_optical_asset(item, name, spec)
            valid &= ~np.ma.getmaskarray(values)
            data = values.filled(0).astype("float32")
            if source == "s2":
                data = (data + offsets.get(band_ids[name], 0)) / quantification
            else:
                rb = item["assets"][name]["raster:bands"][0]
                data = data * rb["scale"] + rb["offset"]
            data[~valid] = -9999
            dst.write(data, i)
            dst.set_band_description(i, name)
        dst.write_mask(valid.astype("uint8") * 255)
    os.replace(tmp, destination)
    mask = Path(str(tmp) + ".msk")
    if mask.exists():
        os.replace(mask, Path(str(destination) + ".msk"))
    destination.with_suffix(".json").write_text(
        json.dumps(
            {
                "item_id": item["id"],
                "source": source,
                "datetime": item["properties"]["datetime"],
                "bands": bands,
                "s2_boa_quantification": quantification,
                "s2_boa_offsets": offsets,
                "valid_fraction": float(valid.mean()),
                "quality_mask_version": (
                    LANDSAT_QUALITY_VERSION if source == "landsat" else "scl-v1"
                ),
                "state": "complete",
            },
            indent=2,
        )
    )
    return destination


def monthly_optical(
    paths: list[Path],
    destination: Path,
    spec: dict,
    budget: Budget,
    *,
    quality_version: str | None = None,
) -> dict:
    """Cloud-masked per-pixel median; no borrowing observations from other months."""
    import contextlib
    import warnings

    if not paths:
        raise ValueError("month has no observations")
    receipt = destination.with_suffix(".json")
    if destination.exists() and receipt.exists():
        cached = json.loads(receipt.read_text())
        if quality_version is None or cached.get("quality_mask_version") == quality_version:
            return cached
    budget.check(400 * 1024**2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".partial.tif")
    valid_pixels = 0
    with contextlib.ExitStack() as stack:
        inputs = [stack.enter_context(rasterio.open(p)) for p in paths]
        profile = inputs[0].profile
        with rasterio.open(tmp, "w", **profile) as dst:
            for _, window in dst.block_windows(1):
                data = np.stack([s.read(window=window, masked=True).filled(np.nan) for s in inputs])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    median = np.nanmedian(data, axis=0)
                valid = np.isfinite(median).all(axis=0)
                median[:, ~valid] = -9999
                dst.write(median, window=window)
                dst.write_mask(valid.astype("uint8") * 255, window=window)
                valid_pixels += int(valid.sum())
    os.replace(tmp, destination)
    mask = Path(str(tmp) + ".msk")
    if mask.exists():
        os.replace(mask, Path(str(destination) + ".msk"))
    result = {
        "state": "complete",
        "path": str(destination),
        "sources": [str(p) for p in paths],
        "valid_fraction": valid_pixels / (profile["width"] * profile["height"]),
        "composite": "per-pixel cloud-masked monthly median",
        "resolution_m": 10,
        "native_source_resolution_m": {"s2": [10, 20, 60], "landsat": [30]},
    }
    if quality_version is not None:
        result["quality_mask_version"] = quality_version
    receipt.write_text(json.dumps(result, indent=2))
    return result


def arcgis_window(service: str, destination: Path, spec: dict, budget: Budget) -> dict:
    """Fetch only intersecting vector object IDs, checking pagination completeness."""
    from shapely import make_valid
    from shapely.geometry import box, mapping, shape

    if destination.exists():
        return json.loads(destination.read_text())
    metadata = fetch_json(service, {"f": "json"})
    if "error" in metadata:
        raise ValueError("vector service metadata is unavailable")
    ids = fetch_json(
        service + "/query",
        {
            "f": "json",
            "where": "1=1",
            "geometry": ",".join(map(str, spec["bounds"])),
            "geometryType": "esriGeometryEnvelope",
            "inSR": spec["crs"].split(":")[-1],
            "spatialRel": "esriSpatialRelIntersects",
            "returnIdsOnly": "true",
        },
    )
    if "objectIds" not in ids:
        raise ValueError("vector spatial query did not return IDs")
    values = sorted(ids["objectIds"] or [])
    features = []
    roi = box(*spec["bounds"])
    pages = destination.with_suffix(".pages")
    pages.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(values), 500):
        budget.check(10 * 1024**2)
        page_path = pages / f"{offset:09d}.json"
        expected_ids = values[offset : offset + 500]
        if page_path.exists():
            cached = json.loads(page_path.read_text())
            if cached["object_ids"] != expected_ids:
                raise ValueError("vector source IDs changed since the saved page")
            page = cached["response"]
        else:
            page = fetch_json(
                service + "/query",
                {
                    "f": "geojson",
                    "objectIds": ",".join(map(str, expected_ids)),
                    "outFields": "*",
                    "returnGeometry": "true",
                    "outSR": spec["crs"].split(":")[-1],
                },
            )
        batch = page.get("features")
        if batch is None or len(batch) != len(values[offset : offset + 500]):
            raise ValueError("vector page is incomplete")
        if not page_path.exists():
            page_path.write_text(json.dumps({"object_ids": expected_ids, "response": page}))
        for feature in batch:
            if feature.get("geometry"):
                geometry = shape(feature["geometry"])
                if not geometry.is_valid:
                    geometry = make_valid(geometry)
                    feature["properties"]["acquisition_geometry_repaired"] = True
                feature["geometry"] = mapping(geometry.intersection(roi))
        features.extend(batch)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": spec["crs"]}},
        "features": features,
        "source_service": service,
        "source_metadata": metadata,
    }
    tmp = destination.with_suffix(".partial")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, destination)
    return data


def validate_jobs(jobs: list[dict], root: Path) -> None:
    seen = set()
    for job in jobs:
        if job.get("kind") not in {
            "linz",
            "monthly_optical",
            "monthly_radar",
            "vector",
            "raster",
            "osm",
        }:
            raise ValueError("unsupported acquisition job kind")
        options = {
            "linz": {"url", "resolution", "count", "dtype", "categorical", "source_overviews"},
            "raster": {"assets", "resolution", "count", "dtype", "categorical", "source_overviews"},
            "vector": {"url"},
            "osm": {"url"},
            "monthly_optical": {"source", "catalog"},
            "monthly_radar": {"source", "catalog"},
        }
        unknown = set(job) - {"id", "kind", "output"} - options[job["kind"]]
        if unknown:
            raise ValueError("unknown acquisition job configuration fields")
        if "catalog" in job and not (root / job["catalog"]).resolve().is_relative_to(
            root.resolve()
        ):
            raise ValueError("acquisition catalog escapes data root")
        key = job.get("id", "")
        if not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in key):
            raise ValueError("invalid acquisition job id")
        if key in seen:
            raise ValueError("duplicate acquisition job id")
        seen.add(key)
        target = (root / job["output"]).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("acquisition output escapes data root")


def _execute_job(payload):
    """Serialize duplicate job IDs across independent data-preparation processes."""
    import fcntl

    spec, job = payload
    lock_path = Path(spec["root"]) / "status" / (job["id"] + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        return _execute_job_unlocked(payload)


def _execute_job_unlocked(payload):
    import resource
    import time

    from xuannv_embedding.data_process.cli import _atomic_json

    spec, job = payload
    root = Path(spec["root"]).resolve()
    status = root / "status"
    status.mkdir(parents=True, exist_ok=True)
    budget = Budget(
        root, int(spec.get("max_gib", 150) * 1024**3), int(spec.get("min_free_gib", 100) * 1024**3)
    )
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    path = status / (job["id"] + ".json")
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("state") == "complete" and (root / job["output"]).exists():
            return previous
    _atomic_json(path, {"state": "running", "started": time.time(), "worker_pid": os.getpid()})
    target = root / job["output"]
    try:
        budget.check()
        if job["kind"] == "osm":
            result = download_osm(spec, target, budget, job["url"])
        elif job["kind"] == "vector":
            result = arcgis_window(job["url"], target, spec, budget)
            result = {"features": len(result["features"]), "path": str(target)}
        elif job["kind"] in {"linz", "raster"}:
            if job["kind"] == "linz":
                assets, collection = linz_assets(
                    job["url"], tuple(spec["bbox_wgs84"]), root / "catalogs" / job["id"]
                )
                (root / "catalogs" / (job["id"] + "_sources.json")).write_text(
                    json.dumps({"collection": collection, "assets": assets}, indent=2)
                )
            else:
                assets = job["assets"]
            result = window_mosaic(
                assets,
                target,
                spec,
                job["resolution"],
                job["count"],
                job["dtype"],
                budget,
                Resampling.nearest if job.get("categorical") else Resampling.average,
                source_overviews=job.get("source_overviews", False),
            )
        else:
            items = json.loads((root / job["catalog"]).read_text())["selected"]
            paths = []
            for item in items:
                temporary = root / "temporary" / job["id"] / (item["id"] + ".tif")
                if job["kind"] == "monthly_radar":
                    paths.append(acquire_radar_item(item, temporary, spec, budget))
                else:
                    paths.append(acquire_optical_item(item, job["source"], temporary, spec, budget))
            result = monthly_optical(
                paths,
                target,
                spec,
                budget,
                quality_version=LANDSAT_QUALITY_VERSION if job.get("source") == "landsat" else None,
            )
            for temporary in paths:
                temporary.unlink()
                Path(str(temporary) + ".msk").unlink(missing_ok=True)
        result = {"state": "complete", "result": result, "finished": time.time()}
    except Exception as exc:
        # Avoid signed URLs and credentials in public progress reports.
        result = {
            "state": "incomplete",
            "error": type(exc).__name__,
            "finished": time.time(),
            "resume_job": job["id"],
        }
    result["worker_pid"] = os.getpid()
    _atomic_json(path, result)
    return result


def run_acquisition(spec: dict, jobs: list[dict], workers: int = 4) -> dict:
    """Execute only data jobs, retaining per-job receipts and restartable partial progress."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    root = Path(spec["root"]).resolve()
    if not root.is_relative_to(Path("/data")):
        raise ValueError("regional data preparation requires a /data output root")
    root.mkdir(parents=True, exist_ok=True)
    validate_jobs(jobs, root)
    budget = Budget(
        root, int(spec.get("max_gib", 150) * 1024**3), int(spec.get("min_free_gib", 100) * 1024**3)
    )
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    status = root / "status"
    status.mkdir(exist_ok=True)

    budget.check()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_execute_job, [(spec, job) for job in jobs]))
    report = {
        "state": "complete" if all(r["state"] == "complete" for r in results) else "incomplete",
        "completed": sum(r["state"] == "complete" for r in results),
        "total": len(results),
        "jobs": {job["id"]: result for job, result in zip(jobs, results)},
        "training_started": False,
    }
    (root / "acquisition-report.json").write_text(json.dumps(report, indent=2))
    return report


def acquire_radar_item(item: dict, destination: Path, spec: dict, budget: Budget) -> Path:
    """Crop provider-calibrated VV/VH gamma-naught in linear units."""
    receipt = destination.with_suffix(".json")
    if destination.exists() and receipt.exists():
        return destination
    budget.check(64 * 1024**2)
    transform, width, height = local_grid(spec, 10)
    values = [read_optical_asset(item, b, spec) for b in ["vv", "vh"]]
    valid = np.ones((height, width), bool)
    for value in values:
        valid &= ~np.ma.getmaskarray(value) & np.isfinite(value.filled(0)) & (value.filled(0) > 0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".partial.tif")
    with rasterio.open(
        tmp,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=2,
        dtype="float32",
        crs=spec["crs"],
        transform=transform,
        nodata=-9999,
        tiled=True,
        compress="deflate",
        predictor=3,
    ) as dst:
        for i, value in enumerate(values, 1):
            data = value.filled(0).astype("float32")
            data[~valid] = -9999
            dst.write(data, i)
            dst.set_band_description(i, ["VV_gamma0_linear", "VH_gamma0_linear"][i - 1])
        dst.write_mask(valid.astype("uint8") * 255)
    os.replace(tmp, destination)
    mask = Path(str(tmp) + ".msk")
    if mask.exists():
        os.replace(mask, Path(str(destination) + ".msk"))
    receipt.write_text(
        json.dumps(
            {
                "state": "complete",
                "item_id": item["id"],
                "datetime": item["properties"]["datetime"],
                "provider": "Planetary Computer sentinel-1-rtc",
                "quantity": "gamma-naught",
                "units": "linear",
                "valid_fraction": float(valid.mean()),
                "native_grid_spacing_m": 10,
                "effective_sar_resolution_m": [20, 22],
                "relative_orbit": item["properties"].get("sat:relative_orbit"),
            },
            indent=2,
        )
    )
    return destination


def download_osm(spec: dict, destination: Path, budget: Budget, endpoint: str) -> dict:
    """Freeze an historical, bounded OSM weak-supervision snapshot."""
    if destination.exists():
        return {"state": "complete", "path": str(destination)}
    west, south, east, north = spec["bbox_wgs84"]
    bbox = f"{south},{west},{north},{east}"
    last_month = spec["months"][-1]
    year, month = map(int, last_month.split("-"))
    snapshot = f"{last_month}-{monthrange(year, month)[1]:02d}T23:59:59Z"
    query = f'[out:json][date:"{snapshot}"][timeout:180][maxsize:268435456];('
    for key in ["building", "highway", "railway", "waterway", "natural", "landuse", "leisure"]:
        query += f'way["{key}"]({bbox});relation["{key}"]({bbox});'
    query += ");out geom;"
    budget.check(512 * 1024**2)
    response = requests.post(
        endpoint,
        data={"data": query},
        timeout=(15, 240),
        headers={"User-Agent": "xuannv-embedding/1.0 regional research data preparation"},
    )
    response.raise_for_status()
    data = response.json()
    if data.get("remark") or "elements" not in data:
        raise ValueError("OSM snapshot request did not return a complete response")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data))
    destination.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "provider": "OpenStreetMap contributors",
                "license": "ODbL-1.0",
                "attribution": "© OpenStreetMap contributors",
                "snapshot": snapshot,
                "bbox_wgs84": spec["bbox_wgs84"],
                "query": query,
                "endpoint": endpoint,
                "role": "common weak supervision for both model variants",
            },
            indent=2,
        )
    )
    return {"state": "complete", "elements": len(data["elements"]), "path": str(destination)}
