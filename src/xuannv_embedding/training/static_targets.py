"""Materialize a categorical reference for every image-cache record, without image copies."""

import json
from pathlib import Path

import numpy as np
import rasterio

from xuannv_embedding.downstream.worldcover_reference import CLASSES, project_tile
from xuannv_embedding.export.context import dump, sha


def encode_classes(codes):
    if not np.isin(codes, [0, *CLASSES]).all():
        raise ValueError("unknown ESA WorldCover class code")
    result = np.zeros(codes.shape, np.uint8)
    for index, code in enumerate(CLASSES, 1):
        result[codes == code] = index
    return result


def prepare(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if (
        set(spec)
        != {"protocol", "raster", "provenance", "reference_cache", "reference_crs", "output"}
        or spec["protocol"] != "worldcover-training-target-v1"
    ):
        raise ValueError("invalid static reference preparation specification")
    for key in ["raster", "provenance", "reference_cache"]:
        if sha(Path(spec[key]["path"])) != spec[key]["sha256"]:
            raise ValueError("static reference source identity changed")
    provenance = json.loads(Path(spec["provenance"]["path"]).read_text())
    if (
        provenance.get("dataset") != "ESA WorldCover"
        or provenance.get("sha256") != spec["raster"]["sha256"]
    ):
        raise ValueError("WorldCover provenance mismatch")
    cache = json.loads(Path(spec["reference_cache"]["path"]).read_text())
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=False)
    records, totals = [], np.zeros(12, np.int64)
    with rasterio.open(spec["raster"]["path"]) as source:
        for record in cache["records"]:
            codes, _ = project_tile(
                source, record["bounds"], cache["data"]["patch_size"], spec["reference_crs"]
            )
            labels = encode_classes(codes)
            counts = np.bincount(labels.ravel(), minlength=12)
            totals += counts
            path = output / (record["patch_id"] + ".npz")
            np.savez_compressed(path, labels=labels, valid=labels > 0)
            records.append(
                {
                    "patch_id": record["patch_id"],
                    "bounds": record["bounds"],
                    "path": str(path),
                    "sha256": sha(path),
                    "counts": counts.tolist(),
                }
            )
    dump(
        output / "manifest.json",
        {
            "protocol": spec["protocol"],
            "state": "complete",
            "target": "esa_worldcover",
            "classes": 12,
            "ignore_index": 0,
            "image_cache_sha256": spec["reference_cache"]["sha256"],
            "legend": {
                str(i): {"official_code": code, "name": name}
                for i, (code, name) in enumerate(CLASSES.items(), 1)
            },
            "source": provenance,
            "records": records,
            "class_counts": totals.tolist(),
            "resampling": "nearest",
            "crs": spec["reference_crs"],
            "spec_sha256": sha(spec_path),
            "implementation_sha256": sha(Path(__file__)),
        },
    )
