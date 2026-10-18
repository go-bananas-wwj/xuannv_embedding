"""Label-free temporal RGB inputs for a frozen optical semantic encoder."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as copy_raster
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT

from xuannv_embedding.export.context import dump, sha


def composite_rgb(images, masks, *, bands, low, high):
    images, masks = np.asarray(images), np.asarray(masks)
    if (
        images.ndim != 4
        or masks.shape != (images.shape[0], *images.shape[-2:])
        or masks.dtype != bool
        or not isinstance(bands, list)
        or len(bands) != 3
        or len(set(bands)) != 3
        or any(type(b) is not int or not 1 <= b <= images.shape[1] for b in bands)
        or not np.isfinite([low, high]).all()
        or high <= low
    ):
        raise ValueError("Invalid temporal RGB inputs, band mapping or fixed radiometry")
    rgb = images[:, np.array(bands) - 1].astype(np.float32)
    observed = masks & np.isfinite(rgb).all(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = np.nanmedian(np.where(observed[:, None], rgb, np.nan), axis=0)
    valid = observed.any(0)
    values = np.clip((np.nan_to_num(values) - low) / (high - low), 0, 1) * 255
    values[:, ~valid] = 0
    return values.astype(np.float32), valid


def _reference(ref):
    if set(ref) != {"path", "sha256"} or sha(ref["path"]) != ref["sha256"]:
        raise ValueError("RGB input registration changed")
    return Path(ref["path"])


def prepare(spec_path: Path):
    spec = json.loads(spec_path.read_text())
    if (
        set(spec) != {"schema", "index", "inputs", "rgb", "encoder_resolution_m", "output"}
        or spec["schema"] != "temporal-semantic-rgb-v1"
        or set(spec["rgb"]) != {"bands", "low", "high"}
        or type(spec["encoder_resolution_m"]) not in {int, float}
        or not 0.5 <= spec["encoder_resolution_m"] <= 10
    ):
        raise ValueError("Invalid temporal semantic RGB specification")
    index = json.loads(_reference(spec["index"]).read_text())
    inputs = json.loads(_reference(spec["inputs"]).read_text())
    if (
        set(inputs) != {"schema", "records"}
        or inputs["schema"] != "temporal-semantic-observations-v1"
        or len(inputs["records"]) != len(index["records"])
    ):
        raise ValueError("Invalid RGB observation manifest")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {"spec_sha256": sha(spec_path)}
    if (root / "identity.json").exists():
        if json.loads((root / "identity.json").read_text()) != identity:
            raise ValueError("RGB preparation identity changed")
    dump(root / "identity.json", identity)
    records, audit = [], []
    for position, (record, item) in enumerate(zip(index["records"], inputs["records"])):
        if set(item) != {"patch_id", "observations"} or item["patch_id"] != record["patch_id"]:
            raise ValueError("RGB observation order differs from regional geography")
        if not item["observations"]:
            raise ValueError("Every registered tile requires an observation or explicit empty mask")
        images, masks = [], []
        for observation in item["observations"]:
            if set(observation) != {"raster", "mask"}:
                raise ValueError("Each RGB observation requires a registered companion mask")
            raster, mask = (_reference(observation[k]) for k in ["raster", "mask"])
            with rasterio.open(raster) as source, rasterio.open(mask) as quality:
                if (
                    str(source.crs) != record["crs"]
                    or not np.allclose(source.bounds, record["bounds"], rtol=0, atol=1e-6)
                    or source.shape != quality.shape
                    or source.transform != quality.transform
                    or source.crs != quality.crs
                ):
                    raise ValueError("RGB and companion-mask geography differs")
                images.append(source.read(out_dtype="float32"))
                masks.append((quality.read(1) > 0) & (source.dataset_mask() > 0))
        rgb, valid = composite_rgb(np.stack(images), np.stack(masks), **spec["rgb"])
        height, width = valid.shape
        native = root / f"{position:06d}.tif"
        profile = dict(
            driver="GTiff",
            count=3,
            dtype="float32",
            height=height,
            width=width,
            crs=record["crs"],
            transform=from_bounds(*record["bounds"], width, height),
            compress="deflate",
            predictor=3,
            tiled=True,
            nodata=-1.0,
        )
        with (
            rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),
            rasterio.open(native, "w", **profile) as dst,
        ):
            dst.write(np.where(valid[None], rgb, -1.0))
            dst.write_mask(valid.astype(np.uint8) * 255)
        bounds, resolution = record["bounds"], spec["encoder_resolution_m"]
        shape = [
            round((bounds[3] - bounds[1]) / resolution),
            round((bounds[2] - bounds[0]) / resolution),
        ]
        virtual = root / f"{position:06d}.vrt"
        with (
            rasterio.open(native) as src,
            WarpedVRT(
                src,
                crs=record["crs"],
                transform=from_bounds(*bounds, shape[1], shape[0]),
                width=shape[1],
                height=shape[0],
                resampling=Resampling.bilinear,
                src_nodata=-1.0,
                nodata=-1.0,
            ) as warped,
        ):
            copy_raster(warped, virtual, driver="VRT")
        with rasterio.open(virtual) as check:
            if check.shape != tuple(shape) or check.count != 3:
                raise ValueError("Virtual encoder input geometry changed")
        records.append(
            {
                **{k: record[k] for k in ["patch_id", "bounds", "crs"]},
                "raster": {"path": str(virtual), "sha256": sha(virtual)},
            }
        )
        audit.append(
            {
                "patch_id": record["patch_id"],
                "native": {"path": str(native), "sha256": sha(native)},
                "source_shape": [height, width],
                "encoder_shape": shape,
                "valid_fraction": float(valid.mean()),
                "observations": len(images),
            }
        )
        dump(root / "status.json", {"state": "preparing", "completed": position + 1})
    dump(
        root / "rasters.json",
        {
            "schema": "semantic-raster-inputs-v1",
            "records": records,
            "dependencies": [row["native"] for row in audit],
        },
    )
    dump(
        root / "audit.json",
        {
            "label_data_used": False,
            "records": audit,
            "rgb": spec["rgb"],
            "resampling_adds_observed_detail": False,
        },
    )
    dump(root / "status.json", {"state": "complete", "completed": len(records)})
    return root / "rasters.json"
