"""Window official annual embeddings, dequantizing before any interpolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

from xuannv_embedding.export.context import dump, sha

_BUCKET = "alphaearth_foundations"
_PREFIX = "satellite_embedding/v1/annual"
_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_TIMEOUT": "45",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_CACHEMAX": 128 * 1024**2,
}


def dequantize(raw: np.ndarray) -> np.ndarray:
    if raw.dtype != np.int8 or raw.ndim != 3 or raw.shape[0] != 64:
        raise ValueError("Official embeddings require 64-band int8 pixels")
    missing = raw == -128
    if not np.array_equal(missing.any(0), missing.all(0)):
        raise ValueError("Inconsistent embedding band mask")
    values = raw.astype(np.float32) / 127.5
    values = np.sign(values) * values**2
    values[:, missing.all(0)] = np.nan
    return values


def normalize_vectors(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(values, axis=0)
    valid = np.isfinite(values).all(0) & np.isfinite(norm) & (norm > 1e-8)
    result = np.full_like(values, np.nan)
    np.divide(values, norm[None], out=result, where=valid[None])
    return result, valid


def source_window(bounds, transform, width, height, *, padding=2) -> Window:
    """Support both north-up and the official south-up COG affine transforms."""
    west, south, east, north = bounds
    corners = [~transform * (x, y) for x in (west, east) for y in (south, north)]
    col = max(0, math.floor(min(p[0] for p in corners)) - padding)
    row = max(0, math.floor(min(p[1] for p in corners)) - padding)
    right = min(width, math.ceil(max(p[0] for p in corners)) + padding)
    bottom = min(height, math.ceil(max(p[1] for p in corners)) + padding)
    if right <= col or bottom <= row:
        raise ValueError("Reference window is outside the source raster")
    return Window(col, row, right - col, bottom - row)


def _header(task: tuple) -> dict:
    item, target_crs, bounds, cache = task
    url = "https://storage.googleapis.com/" + _BUCKET + "/" + item["name"]
    path = Path(cache) / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("header_schema") == 2:
            return cached
    with rasterio.Env(**_ENV), rasterio.open(url) as raster:
        left, bottom, right, top = raster.bounds
        ordered = (min(left, right), min(bottom, top), max(left, right), max(bottom, top))
        box = transform_bounds(raster.crs, target_crs, *ordered, densify_pts=21)
        intersects = (
            box[0] < bounds[2] and box[2] > bounds[0] and box[1] < bounds[3] and box[3] > bounds[1]
        )
        result = {
            "header_schema": 2,
            "url": url,
            "generation": item.get("generation"),
            "md5Hash": item.get("md5Hash"),
            "source_bytes": int(item["size"]),
            "bounds": list(raster.bounds),
            "crs": str(raster.crs),
            "intersects": intersects,
            "count": raster.count,
            "dtype": raster.dtypes[0],
            "transform": list(raster.transform),
        }
    dump(path, result)
    return result


def acquire(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 100 * 2**30:
        raise RuntimeError("Disk reserve reached")
    bounds = args.bounds
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3] or args.resolution <= 0:
        raise ValueError("Invalid output grid")
    identity = {
        "year": args.year,
        "bounds": bounds,
        "crs": args.crs,
        "resolution": args.resolution,
        "provider": _BUCKET,
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Reference identity changed")
    dump(identity_path, identity)
    wgs = transform_bounds(args.crs, "EPSG:4326", *bounds, densify_pts=21)
    zones = range(max(1, int((wgs[0] + 180) // 6) + 1), min(60, int((wgs[2] + 180) // 6) + 1) + 1)
    hemispheres = ["S"] if wgs[3] < 0 else ["N"] if wgs[1] >= 0 else ["S", "N"]
    items = []
    for zone in zones:
        for hemisphere in hemispheres:
            prefix = f"{_PREFIX}/{args.year}/{zone}{hemisphere}/"
            page = None
            while True:
                response = requests.get(
                    f"https://storage.googleapis.com/storage/v1/b/{_BUCKET}/o",
                    params={
                        "prefix": prefix,
                        "maxResults": 1000,
                        **({"pageToken": page} if page else {}),
                    },
                    timeout=60,
                )
                response.raise_for_status()
                document = response.json()
                items.extend(
                    i for i in document.get("items", []) if i["name"].endswith((".tif", ".tiff"))
                )
                page = document.get("nextPageToken")
                if not page:
                    break
    if not items:
        raise ValueError("No official embedding assets for the requested year and zone")
    dump(output / "catalog.json", items)
    header_dir = output / "headers"
    header_dir.mkdir(exist_ok=True)
    selected = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        tasks = [(item, args.crs, bounds, str(header_dir)) for item in items]
        for count, header in enumerate(pool.map(_header, tasks), start=1):
            if header["intersects"]:
                selected.append(header)
            dump(
                output / "status.json",
                {
                    "state": "discovering",
                    "checked": count,
                    "total": len(items),
                    "selected": len(selected),
                },
            )
    if not selected:
        raise ValueError("No official embedding asset intersects the region")
    dump(output / "selected-sources.json", selected)
    height = round((bounds[3] - bounds[1]) / args.resolution)
    width = round((bounds[2] - bounds[0]) / args.resolution)
    transform = from_origin(bounds[0], bounds[3], args.resolution, args.resolution)
    mosaic = np.full((64, height, width), np.nan, dtype=np.float32)
    for i, source in enumerate(selected):
        with rasterio.Env(**_ENV), rasterio.open(source["url"]) as raster:
            region = transform_bounds(args.crs, raster.crs, *bounds, densify_pts=21)
            window = source_window(region, raster.transform, raster.width, raster.height)
            decoded = dequantize(raster.read(window=window))
            warped = np.full_like(mosaic, np.nan)
            reproject(
                decoded,
                warped,
                src_transform=raster.window_transform(window),
                src_crs=raster.crs,
                src_nodata=np.nan,
                dst_transform=transform,
                dst_crs=args.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
                num_threads=4,
            )
            valid = np.isfinite(warped).all(0) & ~np.isfinite(mosaic).all(0)
            mosaic[:, valid] = warped[:, valid]
        dump(
            output / "status.json",
            {"state": "windowing", "completed": i + 1, "total": len(selected)},
        )
    mosaic, valid = normalize_vectors(mosaic)
    if not valid.any():
        raise ValueError("Empty embedding reference")
    destination = output / f"aef_{args.year}_unit.tif"
    temporary = destination.with_suffix(".partial.tif")
    with rasterio.open(
        temporary,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=64,
        dtype="float32",
        crs=args.crs,
        transform=transform,
        nodata=np.nan,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="zstd",
        predictor=3,
        BIGTIFF="IF_SAFER",
    ) as raster:
        raster.write(mosaic)
        raster.descriptions = tuple(f"A{i:02d}" for i in range(64))
    temporary.replace(destination)
    report = {
        "state": "complete",
        **identity,
        "sources": selected,
        "path": str(destination),
        "sha256": sha(destination),
        "valid_fraction": float(valid.mean()),
        "native_grid_m": 10,
        "dequantization": "sign(x)*(x/127.5)^2; nodata=-128",
        "alignment": "dequantize then bilinear reproject then unit normalization",
        "attribution": "The AlphaEarth Foundations Satellite Embedding dataset is "
        "produced by Google and Google DeepMind.",
        "license": "CC-BY-4.0",
    }
    dump(output / "status.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv data reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bounds", type=float, nargs=4, required=True)
    parser.add_argument("--crs", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--resolution", type=float, default=10)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    args = parser.parse_args(argv)
    report = acquire(args)
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}))
    return 0
