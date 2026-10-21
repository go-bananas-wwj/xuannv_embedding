"""Subpixel building composition from one fixed regional representation."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from xuannv_embedding.downstream.frozen_readouts import (
    fit_regression,
    load_readout,
    save_readout,
)
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.frozen_fusion import verified


def footprint_fraction(mask, factor):
    if (
        mask.ndim != 2
        or type(factor) is not int
        or factor < 1
        or any(n % factor for n in mask.shape)
    ):
        raise ValueError("Footprint reference must contain complete output cells")
    h, w = mask.shape
    return (
        mask.reshape(h // factor, factor, w // factor, factor)
        .mean((1, 3), dtype=np.float64)
        .astype("float32")
    )


def prepare_reference(spec):
    import pyogrio
    import shapely
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds

    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    p = root / "reference.json"
    if p.exists():
        result = json.loads(p.read_text())
        if result["spec_sha256"] != spec["_spec_sha256"]:
            raise ValueError("Composition reference specification changed")
        verified(result["targets"])
        return result
    data = json.loads(verified(spec["reference"]).read_text())
    shape = (len(data["records"]), data["size"], data["size"])
    if data["size"] != 128 or data["resolution_m"] != 10:
        raise ValueError("Composition reference requires the native 10m grid")
    source = verified(spec["footprints"])
    info = pyogrio.read_info(source)
    bounds = np.array([x["bounds"] for x in data["records"]])
    bbox = [bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()]
    geographic = transform_bounds(data["crs"], info["crs"], *bbox, densify_pts=21)
    frame = pyogrio.read_dataframe(source, bbox=geographic, columns=[])
    invalid = ~frame.geometry.is_valid
    frame.loc[invalid, "geometry"] = frame.loc[invalid, "geometry"].buffer(0)
    frame = frame[~frame.geometry.is_empty].to_crs(data["crs"])
    geometry = frame.geometry.values
    tree = shapely.STRtree(geometry)
    target = root / "fractions.npy"
    values = np.lib.format.open_memmap(
        root / "fractions.partial.npy", mode="w+", dtype="float32", shape=shape
    )
    started = time.monotonic()
    for i, record in enumerate(data["records"]):
        ids = tree.query(shapely.box(*record["bounds"]), predicate="intersects")
        fine = np.zeros((640, 640), np.uint8)
        if len(ids):
            fine = rasterize(
                [(geometry[j], 1) for j in ids],
                out_shape=(640, 640),
                transform=from_bounds(*record["bounds"], 640, 640),
                fill=0,
                dtype="uint8",
            )
        values[i] = footprint_fraction(fine, 5)
        if i % 40 == 0:
            print("fraction reference", i + 1, len(values), flush=True)
    values.flush()
    del values
    (root / "fractions.partial.npy").replace(target)
    result = {
        "state": "complete",
        "targets": {"path": str(target), "sha256": sha(target)},
        "source": spec["footprints"],
        "source_kind": "machine-derived CLSM footprints; acquisition years vary",
        "fine_reference_grid_m": 2,
        "output_grid_m": 10,
        "fraction_samples_per_cell": 25,
        "geometry_repaired": int(invalid.sum()),
        "polygons_in_bbox": len(frame),
        "wall_seconds": time.monotonic() - started,
        "spec_sha256": spec["_spec_sha256"],
    }
    dump(p, result)
    return result


def _summary(counts):
    count, square, absolute, bias = counts
    if not count:
        return {"observations": 0, "rmse": None, "mae": None, "bias": None}
    return {
        "observations": int(count),
        "rmse": float(np.sqrt(square / count)),
        "mae": float(absolute / count),
        "bias": float(bias / count),
    }


def score_model(spec, name, descriptor, reference):
    root = Path(spec["output"]) / name
    root.mkdir(exist_ok=True)
    data = json.loads(verified(spec["reference"]).read_text())
    targets = np.load(verified(reference["targets"]), mmap_mode="r")
    if "feature_receipt" in descriptor:
        p = Path(descriptor["feature_receipt"])
        if not p.exists():
            return False
        result = json.loads(p.read_text())
        if result["state"] != "complete":
            return False
        ref = result["features"]
    else:
        ref = descriptor["features"]
    path = verified(ref)
    features = np.load(path, mmap_mode="r")
    if descriptor["layout"] == "nchw":
        features = features.transpose(0, 2, 3, 1)
    if features.shape[:3] != targets.shape:
        raise ValueError("Composition feature and reference grids differ")
    identity = {
        "feature_sha256": ref["sha256"],
        "targets": reference["targets"],
        "code": sha(__file__),
        "spec_sha256": spec["_spec_sha256"],
    }
    header = root / "identity.json"
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError("Composition run input identity changed")
    dump(header, identity)
    if (root / "result.json").exists():
        old = json.loads((root / "result.json").read_text())
        if old["identity"] != identity:
            raise ValueError("Composition source binding changed")
        return True
    ids = [r["patch_id"] for r in data["records"]]
    ordered = sorted(
        data["train_positions"],
        key=lambda i: hashlib.sha256(f"20260922:{ids[i]}".encode()).digest(),
    )
    val = data["calibration_positions"]
    val_x = np.asarray(features[val], np.float32).reshape(-1, features.shape[-1])
    val_y = targets[val].ravel()
    rows = []
    for budget in [5, 20]:
        begin = time.monotonic()
        support = ordered[:budget]
        readout = root / f"readout-{budget}"
        if readout.exists():
            model = load_readout(readout, sha(readout / "identity.json"))
        else:
            model = fit_regression(
                np.asarray(features[support], np.float32).reshape(-1, features.shape[-1]),
                targets[support].ravel(),
                val_x,
                val_y,
            )
            save_readout(model, readout)
        excluded = set(support) | set(val)
        counts = {
            (scope, stratum): np.zeros(4, np.float64)
            for scope in ["all_region", "outside_support_and_calibration"]
            for stratum in ["all_cells", "mixed_cells", "interior_cells"]
        }
        tile_errors = []
        for i in range(len(targets)):
            truth = targets[i].ravel()
            scores = model.predict(
                np.asarray(features[i], np.float32).reshape(-1, features.shape[-1])
            )
            error = scores - truth
            mixed = (truth > 0) & (truth < 1)
            tile = {}
            for stratum, mask in [
                ("all_cells", np.ones(len(truth), bool)),
                ("mixed_cells", mixed),
                ("interior_cells", ~mixed),
            ]:
                e = error[mask]
                statistics = np.array(
                    [
                        len(e),
                        np.square(e).sum(dtype=np.float64),
                        np.abs(e).sum(dtype=np.float64),
                        e.sum(dtype=np.float64),
                    ]
                )
                counts["all_region", stratum] += statistics
                if i not in excluded:
                    counts["outside_support_and_calibration", stratum] += statistics
                tile[stratum] = statistics.tolist()
            tile_errors.append(tile)
        rows.append(
            {
                "budget_tiles": budget,
                "support_tile_ids": [data["indices"][i] for i in support],
                "calibration_tile_ids": [data["indices"][i] for i in val],
                "metrics": {
                    scope: {
                        s: _summary(counts[scope, s])
                        for s in ["all_cells", "mixed_cells", "interior_cells"]
                    }
                    for scope in ["all_region", "outside_support_and_calibration"]
                },
                "wall_seconds": time.monotonic() - begin,
                "tile_error_statistics": tile_errors,
            }
        )
    dump(
        root / "result.json",
        {
            "state": "complete",
            "identity": identity,
            "model": name,
            "rows": rows,
            "encoder_refitted": False,
            "target": "10m building fraction from 2m footprint rasterization",
            "primary_stratum": "mixed_cells",
        },
    )
    return True


def run(path):
    from xuannv_embedding.export.public_visual_subspace import read_json

    spec = read_json(path)
    if set(spec) != {
        "schema",
        "reference",
        "index",
        "footprints",
        "footprint_dependencies",
        "models",
        "output",
    } or (spec["schema"] != "journal-mixed-composition-v1" or not spec["models"]):
        raise ValueError("Invalid mixed-cell composition study")
    for dependency in spec["footprint_dependencies"]:
        verified(dependency)
    data = json.loads(verified(spec["reference"]).read_text())
    index = json.loads(verified(spec["index"]).read_text())
    if [r["patch_id"] for r in data["records"]] != [r["patch_id"] for r in index["records"]]:
        raise ValueError("Composition feature tile ordering differs")
    for a, b in zip(data["records"], index["records"], strict=True):
        if not np.allclose(a["bounds"], b["bounds"], rtol=0, atol=1e-6):
            raise ValueError("Composition physical grids differ")
    spec["_spec_sha256"] = sha(path)
    reference = prepare_reference(spec)
    started = time.monotonic()
    done = set()
    while len(done) < len(spec["models"]):
        if time.monotonic() - started > 21600:
            raise TimeoutError("Composition study waiting for registered feature producer")
        for name, descriptor in spec["models"].items():
            if name not in done and score_model(spec, name, descriptor, reference):
                done.add(name)
        root = Path(spec["output"])
        dump(
            root / "status.json",
            {
                "state": "complete" if len(done) == len(spec["models"]) else "running",
                "completed_models": sorted(done),
                "pending_models": sorted(set(spec["models"]) - done),
            },
        )
        if len(done) < len(spec["models"]):
            time.sleep(60)
    return reference
