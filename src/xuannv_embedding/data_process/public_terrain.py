"""Provenance-bound public surface-height and terrain coordinates on regional grids."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject

from xuannv_embedding.export.context import dump, sha

NODATA = -32767.0
NAMES = ["elevation_m", "slope_sin", "slope_cos", "downhill_aspect_sin", "downhill_aspect_cos"]


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def verified(value):
    if set(value) != {"path", "sha256"} or sha(value["path"]) != value["sha256"]:
        raise ValueError("Public terrain source identity changed")
    return Path(value["path"])


def terrain_coordinates(height, valid, resolution):
    gx, gy = np.zeros_like(height), np.zeros_like(height)
    gx[1:-1, 1:-1] = (height[1:-1, 2:] - height[1:-1, :-2]) / (2 * resolution)
    gy[1:-1, 1:-1] = (height[:-2, 1:-1] - height[2:, 1:-1]) / (2 * resolution)
    observed = np.zeros_like(valid)
    observed[1:-1, 1:-1] = (
        valid[1:-1, 1:-1] & valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    )
    length = np.sqrt(gx**2 + gy**2)
    normalizer = np.sqrt(1 + length**2)
    aspect_sin = np.divide(-gx, length, out=np.zeros_like(gx), where=length > 1e-6)
    aspect_cos = np.divide(-gy, length, out=np.zeros_like(gy), where=length > 1e-6)
    result = np.stack([height, length / normalizer, 1 / normalizer, aspect_sin, aspect_cos])
    return np.where(observed[None], result, 0).astype(np.float32), observed


def source_path(source):
    if source.get("kind") == "local" and set(source) == {"kind", "raster"}:
        return str(verified(source["raster"]))
    if source.get("kind") == "planetary-computer" and set(source) == {
        "kind",
        "href",
        "item_id",
        "collection",
    }:
        if (
            source["collection"] != "cop-dem-glo-30"
            or not source["href"].startswith("https://")
            or "?" in source["href"]
        ):
            raise ValueError("Public terrain requires an unsigned registered GLO-30 asset")
        import planetary_computer

        return planetary_computer.sign(source["href"])
    raise ValueError("Unknown public terrain source registration")


def prepare(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if (
        set(spec)
        != {
            "schema",
            "dataset",
            "index",
            "sources",
            "analysis_resolution_m",
            "halo_m",
            "patch_size",
            "output",
        }
        or spec["schema"] != "public-terrain-features-v1"
        or spec["dataset"] != "cop-dem-glo-30"
        or type(spec["patch_size"]) is not int
        or not 1 <= spec["patch_size"] <= 512
        or any(
            type(spec[k]) not in {int, float} or not math.isfinite(spec[k]) or spec[k] <= 0
            for k in ["analysis_resolution_m"]
        )
        or not isinstance(spec["sources"], list)
        or not spec["sources"]
    ):
        raise ValueError("Invalid public terrain feature specification")
    resolution = spec["analysis_resolution_m"]
    if (
        type(spec["halo_m"]) not in {int, float}
        or not math.isfinite(spec["halo_m"])
        or spec["halo_m"] < resolution
    ):
        raise ValueError("Terrain derivatives require at least one pixel halo")
    index = json.loads(verified(spec["index"]).read_text())
    records = index["records"]
    if not records or len({r["patch_id"] for r in records}) != len(records):
        raise ValueError("Terrain requires a unique regional geographic grid")
    crs = CRS.from_user_input(index.get("crs", records[0].get("crs")))
    if not crs.is_projected or crs.linear_units not in {"metre", "meter"}:
        raise ValueError("Terrain requires projected metric coordinates")
    for row in records:
        if row.get("crs") is not None and CRS.from_user_input(row["crs"]) != crs:
            raise ValueError("Terrain grid CRS differs")
    bounds = np.asarray([r["bounds"] for r in records])
    halo = math.ceil(spec["halo_m"] / resolution) * resolution
    left = math.floor(bounds[:, 0].min() / resolution) * resolution - halo
    top = math.ceil(bounds[:, 3].max() / resolution) * resolution + halo
    right = math.ceil(bounds[:, 2].max() / resolution) * resolution + halo
    bottom = math.floor(bounds[:, 1].min() / resolution) * resolution - halo
    width, height = round((right - left) / resolution), round((top - bottom) / resolution)
    transform = from_origin(left, top, resolution, resolution)
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {"spec_sha256": sha(spec_path), "implementation_sha256": sha(__file__)}
    header = root / "identity.json"
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError("Public terrain recipe changed")
    dump(header, identity)
    if (root / "status.json").exists():
        old = json.loads((root / "status.json").read_text())
        if old.get("state") == "complete":
            if (
                sha(root / "features.npy") != old["features_sha256"]
                or sha(root / "valid.npy") != old["valid_sha256"]
            ):
                raise ValueError("Public terrain completed arrays changed")
            return root / "features.npy"
    elevation = np.full((height, width), NODATA, np.float32)
    valid = np.zeros((height, width), bool)
    for number, source in enumerate(spec["sources"]):
        # Signed URLs are transient and never stored in provenance or printed.
        path = source_path(source)
        with (
            rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_TIMEOUT="45",
                GDAL_HTTP_MAX_RETRY="1",
            ),
            rasterio.open(path) as src,
            WarpedVRT(
                src,
                crs=crs,
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.bilinear,
                nodata=NODATA,
            ) as warped,
        ):
            values = warped.read(1, masked=True, out_dtype="float32")
            observed = (
                (~np.ma.getmaskarray(values)) & np.isfinite(values.data) & (values.data != NODATA)
            )
            take = observed & ~valid
            elevation[take] = values.data[take]
            valid |= observed
        dump(
            root / "status.json",
            {
                "state": "mosaicking",
                "completed_sources": number + 1,
                "sources": len(spec["sources"]),
            },
        )
    if not valid.any():
        raise ValueError("No public terrain observations cover the region")
    raw = root / "public-surface-height.tif"
    pending = raw.with_suffix(".partial.tif")
    with (
        rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),
        rasterio.open(
            pending,
            "w",
            driver="GTiff",
            count=1,
            dtype="float32",
            height=height,
            width=width,
            crs=crs,
            transform=transform,
            nodata=NODATA,
            compress="deflate",
            predictor=3,
        ) as dst,
    ):
        dst.write(elevation, 1)
        dst.write_mask(valid.astype(np.uint8) * 255)
    pending.replace(raw)
    coordinates, observed = terrain_coordinates(elevation, valid, resolution)
    size = spec["patch_size"]
    dest = np.lib.format.open_memmap(
        root / "features.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(records), 5, size, size),
    )
    coverage = np.lib.format.open_memmap(
        root / "valid.partial.npy", mode="w+", dtype=bool, shape=(len(records), size, size)
    )
    for i, row in enumerate(records):
        target = from_bounds(*row["bounds"], size, size)
        weight = np.zeros((size, size), np.float32)
        reproject(
            observed.astype(np.float32),
            weight,
            src_transform=transform,
            src_crs=crs,
            dst_transform=target,
            dst_crs=crs,
            resampling=Resampling.bilinear,
        )
        mask = weight > 0.999
        for channel in range(5):
            numerator = np.zeros((size, size), np.float32)
            reproject(
                coordinates[channel] * observed,
                numerator,
                src_transform=transform,
                src_crs=crs,
                dst_transform=target,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
            dest[i, channel] = np.where(mask, numerator / np.maximum(weight, 1e-6), 0)
        coverage[i] = mask
    dest.flush()
    coverage.flush()
    (root / "features.partial.npy").replace(root / "features.npy")
    (root / "valid.partial.npy").replace(root / "valid.npy")
    if not np.isfinite(dest).all():
        raise ValueError("Nonfinite terrain coordinates")
    dump(
        root / "metadata.json",
        {
            "dataset": "Copernicus GLO-30 DSM, public surface height",
            "feature_names": NAMES,
            "analysis_resolution_m": resolution,
            "output_sampling_m": "regional patch grid; interpolation adds no observed detail",
            "sources": spec["sources"],
            "index": spec["index"],
            "raw_crop": ref(raw),
            "labels_used": False,
            "LINZ_LiDAR_used": False,
            "source_role": (
                "candidate public auxiliary; direct input differs from AEF training-target role"
            ),
        },
    )
    dump(
        root / "status.json",
        {
            "state": "complete",
            "features_sha256": sha(root / "features.npy"),
            "valid_sha256": sha(root / "valid.npy"),
            "shape": list(dest.shape),
            "valid_pixels": int(coverage.sum()),
            "total_pixels": int(coverage.size),
        },
    )
    return root / "features.npy"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment public-terrain")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.spec)
    return 0
