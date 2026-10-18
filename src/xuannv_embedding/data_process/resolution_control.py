"""Degrade only local RGB detail while retaining the exact encoder observation mask."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.semantic_features import registered_rasters


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def verified(value):
    if set(value) != {"path", "sha256"} or sha(value["path"]) != value["sha256"]:
        raise ValueError("Resolution-control input identity changed")
    return Path(value["path"])


def write_rgb(path, values, valid, record):
    profile = {
        "driver": "GTiff",
        "count": 3,
        "dtype": "float32",
        "height": valid.shape[0],
        "width": valid.shape[1],
        "crs": record["crs"],
        "transform": from_bounds(*record["bounds"], valid.shape[1], valid.shape[0]),
        "nodata": -1.0,
        "compress": "deflate",
        "predictor": 3,
        "zlevel": 4,
    }
    pending = path.with_suffix(".partial.tif")
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(pending, "w", **profile) as dst:
        dst.write(np.where(valid[None], values, -1).astype(np.float32))
        dst.write_mask(valid.astype(np.uint8) * 255)
    pending.replace(path)


def prepare(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if (
        set(spec) != {"schema", "index", "rasters", "resolution_m", "output"}
        or spec["schema"] != "matched-resolution-control-v1"
        or type(spec["resolution_m"]) not in {int, float}
        or not math.isfinite(spec["resolution_m"])
        or spec["resolution_m"] <= 0
    ):
        raise ValueError("Invalid resolution-control specification")
    index = json.loads(verified(spec["index"]).read_text())
    manifest_path = verified(spec["rasters"])
    paths = registered_rasters({"rasters": spec["rasters"]}, index)
    root = Path(spec["output"])
    if root.resolve() == manifest_path.parent.resolve() or any(
        root.resolve() == Path(p).parent.resolve() for p in paths
    ):
        raise ValueError("Resolution control must not overwrite an input directory")
    root.mkdir(parents=True, exist_ok=True)
    identity = {"spec_sha256": sha(spec_path), "implementation_sha256": sha(__file__)}
    header = root / "identity.json"
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError("Resolution-control recipe changed")
    dump(header, identity)
    output, dependencies, audits = [], [spec["rasters"]], []
    for position, (record, path) in enumerate(zip(index["records"], paths)):
        fine_file, coarse_file = (
            root / f"encoder_{position:06d}.tif",
            root / f"coarse_{position:06d}.tif",
        )
        receipt = root / f"receipt_{position:06d}.json"
        if receipt.exists():
            row = json.loads(receipt.read_text())
            if any(sha(x["path"]) != x["sha256"] for x in [row["fine"], row["coarse"]]):
                raise ValueError("Resolution-control image changed")
        else:
            with rasterio.open(path) as src:
                if str(src.crs) != record["crs"] or not src.crs.is_projected or src.count != 3:
                    raise ValueError(
                        "Resolution control requires corresponding projected RGB grids"
                    )
                window = window_from_bounds(*record["bounds"], transform=src.transform)
                rounded = window.round_offsets().round_lengths()
                if any(abs(a - b) > 1e-6 for a, b in zip(window.flatten(), rounded.flatten())):
                    raise ValueError("Resolution control grid is not aligned to its source")
                h, w = int(rounded.height), int(rounded.width)
                bounds, resolution = record["bounds"], spec["resolution_m"]
                dh, dw = (bounds[3] - bounds[1]) / resolution, (bounds[2] - bounds[0]) / resolution
                ch, cw = round(dh), round(dw)
                if (
                    min(ch, cw) < 1
                    or abs(ch - dh) > 1e-6
                    or abs(cw - dw) > 1e-6
                    or ch > h
                    or cw > w
                    or h % ch
                    or w % cw
                ):
                    raise ValueError("Control resolution must form an integral coarser grid")
                values = src.read(window=rounded, out_dtype="float32")
                valid = (src.dataset_mask(window=rounded) > 0) & np.isfinite(values).all(0)
                if values.shape != (3, h, w):
                    raise ValueError("Source does not cover the requested control grid")
            mask = torch.from_numpy(valid)[None, None].float()
            rgb = torch.from_numpy(np.where(valid[None], np.clip(values, 0, 255), 0))[None]
            with torch.inference_mode():
                coverage = F.adaptive_avg_pool2d(mask, (ch, cw))
                coarse = F.adaptive_avg_pool2d(rgb, (ch, cw)) / coverage.clamp_min(1e-8)
                observed = coverage > 0
                numerator = F.interpolate(
                    coarse * observed, size=(h, w), mode="bilinear", align_corners=False
                )
                denominator = F.interpolate(
                    observed.float(), size=(h, w), mode="bilinear", align_corners=False
                )
                if valid.any() and not (denominator[0, 0].numpy()[valid] > 0).all():
                    raise ValueError("Degradation lost an original observation")
                fine = (numerator / denominator.clamp_min(1e-8))[0].numpy()
            write_rgb(coarse_file, coarse[0].numpy(), observed[0, 0].numpy(), record)
            write_rgb(fine_file, fine, valid, record)
            with rasterio.open(fine_file) as check:
                if not np.array_equal(check.dataset_mask() > 0, valid):
                    raise ValueError("Resolution control changed the observation footprint")
            row = {
                "fine": ref(fine_file),
                "coarse": ref(coarse_file),
                "fine_shape": [h, w],
                "coarse_shape": [ch, cw],
                "observed_pixels": int(valid.sum()),
                "footprint_exact": True,
                "effective_resolution_m": resolution,
            }
            dump(receipt, row)
        output.append(
            {**{k: record[k] for k in ["patch_id", "bounds", "crs"]}, "raster": row["fine"]}
        )
        dependencies.append(row["coarse"])
        audits.append({"patch_id": record["patch_id"], **row})
        dump(
            root / "status.json",
            {"state": "preparing", "completed": position + 1, "total": len(paths)},
        )
    original = json.loads(manifest_path.read_text())
    dependencies.extend(original.get("dependencies", []))
    dependencies.extend(r["raster"] for r in original["records"])
    dump(
        root / "rasters.json",
        {"schema": "semantic-raster-inputs-v1", "records": output, "dependencies": dependencies},
    )
    dump(
        root / "audit.json",
        {
            "effective_resolution_m": spec["resolution_m"],
            "source_dates_changed": False,
            "observation_masks_exact": True,
            "upsampling_adds_detail": False,
            "records": audits,
        },
    )
    dump(
        root / "status.json",
        {
            "state": "complete",
            "records": len(output),
            "manifest_sha256": sha(root / "rasters.json"),
        },
    )
    return root / "rasters.json"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment resolution-control")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.spec)
    return 0
