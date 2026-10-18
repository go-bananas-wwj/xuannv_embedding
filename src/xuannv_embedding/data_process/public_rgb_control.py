"""Public S2 RGB inputs for a matched pretrained-capacity control; no label inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import warnings
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.shutil import copy as copy_raster
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def verified(value):
    if set(value) != {"path", "sha256"} or sha(value["path"]) != value["sha256"]:
        raise ValueError("Public RGB observation identity changed")
    return Path(value["path"])


def months_checked(values):
    if not isinstance(values, list) or not values or len(set(values)) != len(values):
        raise ValueError("Public RGB months must be nonempty and unique")
    try:
        dates = [dt.datetime.strptime(m, "%Y-%m") for m in values]
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid public RGB month") from error
    if any(m != d.strftime("%Y-%m") for m, d in zip(values, dates)) or dates != sorted(dates):
        raise ValueError("Public RGB months must follow their declared chronological order")


def public_rgb(record, observations, stats, selected, maximum):
    path = verified(record["input"])
    bands = observations["bands"]["s2"]
    if len(bands) != 11 or len(set(bands)) != 11 or set(bands) != set(PUBLIC_BANDS["s2"]):
        raise ValueError("Public RGB requires named S2 physical bands")
    mean = np.asarray(stats["s2"]["mean"][:11], np.float32)[None, :, None, None]
    std = np.asarray(stats["s2"]["std"][:11], np.float32)[None, :, None, None]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid archived S2 normalization")
    if observations["kind"] == "normalized_tensor":
        sample = torch.load(str(path), weights_only=True, mmap=True, map_location="cpu")
        value = np.asarray(sample["source_frames"]["s2"].numpy()[:, :11], np.float32)
        valid = sample["target_masks"]["s2_recon"].numpy() > 0
        frames = np.clip(value, -6, 6) * std + mean
    elif observations["kind"] == "physical_npz":
        with np.load(path) as stored:
            frames = np.asarray(stored["s2"][:, :11], np.float32)
            valid = stored["s2_valid"].astype(bool)
        frames = np.clip((frames - mean) / std, -6, 6) * std + mean
    else:
        raise ValueError("Unknown public RGB observation encoding")
    if (
        frames.ndim != 4
        or valid.shape != (len(frames), *frames.shape[-2:])
        or (not np.isfinite(frames.transpose(0, 2, 3, 1)[valid]).all())
    ):
        raise ValueError("Invalid public RGB source frames or validity")
    mapping = observations["radiometry"]["s2"]
    rgb = []
    for band in ["B04", "B03", "B02"]:
        item = mapping[band]
        if (
            set(item) != {"scale", "offset"}
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in item.values())
            or item["scale"] <= 0
        ):
            raise ValueError("Invalid named public RGB radiometry")
        rgb.append(frames[:, bands.index(band)] * item["scale"] + item["offset"])
    rgb = np.stack(rgb, 1)[selected]
    valid = valid[selected]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        composite = np.nanmedian(np.where(valid[:, None], rgb, np.nan), axis=0)
    return (
        np.clip(np.nan_to_num(composite) / maximum, 0, 1).astype(np.float32) * 255,
        valid.any(0),
        len(frames),
    )


def prepare(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if (
        set(spec)
        != {
            "schema",
            "observations",
            "index",
            "input_months",
            "months",
            "display_max_reflectance",
            "encoder_resolution_m",
            "output",
        }
        or spec["schema"] != "public-s2-rgb-control-v1"
        or any(
            type(spec[k]) not in {int, float} or not math.isfinite(spec[k]) or spec[k] <= 0
            for k in ["display_max_reflectance", "encoder_resolution_m"]
        )
    ):
        raise ValueError("Invalid public RGB control specification")
    months_checked(spec["input_months"])
    months_checked(spec["months"])
    if not set(spec["months"]).issubset(spec["input_months"]):
        raise ValueError("Requested public RGB month is unavailable")
    selected = [spec["input_months"].index(m) for m in spec["months"]]
    observation_path = verified(spec["observations"])
    observations = json.loads(observation_path.read_text())
    if observations.get("schema") != "named-public-statistics-v2":
        raise ValueError("Public RGB requires an observation-only registration")
    inputs = json.loads(verified(observations["inputs"]).read_text())["records"]
    stats = json.loads(verified(observations["statistics"]).read_text())
    index = json.loads(verified(spec["index"]).read_text())
    if len(inputs) != len(index["records"]):
        raise ValueError("Public RGB geographic count differs")
    root = Path(spec["output"])
    if root.resolve() in {
        observation_path.parent.resolve(),
        Path(observations["inputs"]["path"]).parent.resolve(),
    }:
        raise ValueError("Public RGB output must not overwrite its inputs")
    root.mkdir(parents=True, exist_ok=True)
    identity = {"spec_sha256": sha(spec_path), "implementation_sha256": sha(__file__)}
    header = root / "identity.json"
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError("Public RGB preparation identity changed")
    dump(header, identity)
    records = []
    dependencies = []
    coverage = []
    for i, (record, source) in enumerate(zip(index["records"], inputs)):
        crs = index.get("crs", record.get("crs"))
        if (
            source["patch_id"] != record["patch_id"]
            or source["bounds"] != record["bounds"]
            or source["crs"] != crs
        ):
            raise ValueError("Public RGB source geography differs")
        native, virtual = root / f"{i:06d}.tif", root / f"{i:06d}.vrt"
        receipt = root / f"receipt_{i:06d}.json"
        if receipt.exists():
            item = json.loads(receipt.read_text())
            if sha(native) != item["native"]["sha256"] or sha(virtual) != item["raster"]["sha256"]:
                raise ValueError("Public RGB cached image changed")
        else:
            rgb, valid, times = public_rgb(
                source, observations, stats, selected, spec["display_max_reflectance"]
            )
            if times != len(spec["input_months"]):
                raise ValueError("Public RGB monthly axis differs from registration")
            h, w = valid.shape
            bounds = record["bounds"]
            oh = (bounds[3] - bounds[1]) / spec["encoder_resolution_m"]
            ow = (bounds[2] - bounds[0]) / spec["encoder_resolution_m"]
            if abs(oh - round(oh)) > 1e-6 or abs(ow - round(ow)) > 1e-6:
                raise ValueError("Public RGB encoder grid is not integral")
            with (
                rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),
                rasterio.open(
                    native,
                    "w",
                    driver="GTiff",
                    count=3,
                    dtype="float32",
                    height=h,
                    width=w,
                    crs=crs,
                    transform=from_bounds(*bounds, w, h),
                    nodata=-1,
                    compress="deflate",
                    predictor=3,
                ) as dst,
            ):
                dst.write(np.where(valid[None], rgb, -1))
                dst.write_mask(valid.astype(np.uint8) * 255)
            with (
                rasterio.open(native) as src,
                WarpedVRT(
                    src,
                    crs=crs,
                    transform=from_bounds(*bounds, round(ow), round(oh)),
                    width=round(ow),
                    height=round(oh),
                    resampling=Resampling.bilinear,
                    src_nodata=-1,
                    nodata=-1,
                ) as warped,
            ):
                copy_raster(warped, virtual, driver="VRT")
            item = {
                "native": ref(native),
                "raster": ref(virtual),
                "valid_pixels": int(valid.sum()),
                "total_pixels": int(valid.size),
            }
            dump(receipt, item)
        records.append(
            {
                "patch_id": record["patch_id"],
                "bounds": record["bounds"],
                "crs": crs,
                "raster": item["raster"],
            }
        )
        dependencies.append(item["native"])
        coverage.append({k: item[k] for k in ["valid_pixels", "total_pixels"]})
        dump(root / "status.json", {"state": "preparing", "completed": i + 1, "total": len(inputs)})
    dump(
        root / "rasters.json",
        {"schema": "semantic-raster-inputs-v1", "records": records, "dependencies": dependencies},
    )
    dump(
        root / "audit.json",
        {
            "source": "public Sentinel-2 RGB only",
            "bands": ["B04", "B03", "B02"],
            "months": spec["months"],
            "input_months": spec["input_months"],
            "display_max_reflectance": spec["display_max_reflectance"],
            "upsampling_adds_observed_detail": False,
            "label_data_used": False,
            "coverage": coverage,
        },
    )
    dump(
        root / "status.json",
        {
            "state": "complete",
            "records": len(records),
            "manifest_sha256": sha(root / "rasters.json"),
        },
    )
    return root / "rasters.json"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment public-rgb-control")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.spec)
    return 0
