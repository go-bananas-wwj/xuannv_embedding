"""Remove actual acquisition periods before compositing, fusion and image guidance."""

from __future__ import annotations

import copy
import fcntl
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from xuannv_embedding.data_process import semantic_composite as composite
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.journal_observations import (
    fuse_reencoded,
    guide_reencoded,
    reencode,
)
from xuannv_embedding.export.jstars_revision import score_frozen
from xuannv_embedding.export.public_visual_subspace import read_json
from xuannv_embedding.training.frozen_fusion import verified


def subset_inputs(original, views, slots):
    if not slots or slots != sorted(set(slots)) or any(i < 0 or i >= len(views) for i in slots):
        raise ValueError("Invalid physical observation subset")
    result = copy.deepcopy(original)

    def key(value):
        return json.dumps(value, sort_keys=True)

    for i, row in enumerate(original["records"]):
        source = [key(o) for o in row["observations"]]
        inventory = []
        retained = []
        for k, view in enumerate(views):
            value = view["records"][i]
            if value["patch_id"] != row["patch_id"]:
                raise ValueError("Temporal source geography differs")
            obs = [key(o) for o in value["observations"]]
            inventory.extend(obs)
            if k in slots:
                retained.extend(obs)
        if sorted(source) != sorted(inventory) or len(set(source)) != len(source):
            raise ValueError("Temporal views do not bind the distinct original observations")
        keep = set(retained)
        result["records"][i]["observations"] = [o for o in row["observations"] if key(o) in keep]
        if not result["records"][i]["observations"]:
            raise ValueError("Explicit empty image required for a wholly absent tile")
    return result


def _bind(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError("Temporal intervention binding changed")
    dump(path, value)
    return {"path": str(path), "sha256": sha(path)}


def _empty_assets(root, index, raw, shape):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "status.json").exists():
            return read_json(root / "status.json")
        refs = {}
        for name, target_shape, dtype in [
            ("native", shape, "float16"),
            ("native-mask", (shape[0], 1, *shape[-2:]), "float32"),
        ]:
            p = root / (name + ".npy")
            array = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=target_shape)
            array.flush()
            del array
            if np.count_nonzero(np.load(p, mmap_mode="r")):
                raise ValueError("Empty acquisition features contain values")
            refs[name] = {"path": str(p), "sha256": sha(p)}
        h, w = raw["source_shape"]
        rows = []
        for i, record in enumerate(index["records"]):
            p = root / f"{i:04d}.tif"
            crs = record.get("crs", index.get("crs"))
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(
                    p,
                    "w",
                    driver="GTiff",
                    height=h,
                    width=w,
                    count=3,
                    dtype="uint8",
                    crs=crs,
                    transform=from_bounds(*record["bounds"], w, h),
                    compress="deflate",
                    tiled=True,
                ) as dst:
                    dst.write(np.zeros((3, h, w), np.uint8))
                    dst.write_mask(np.zeros((h, w), np.uint8))
            rows.append(
                {
                    "patch_id": record["patch_id"],
                    "bounds": record["bounds"],
                    "crs": crs,
                    "raster": {"path": str(p), "sha256": sha(p)},
                }
            )
        p = root / "rasters.json"
        dump(p, {"schema": "semantic-raster-inputs-v1", "records": rows})
        result = {
            "state": "complete",
            "arrays": refs,
            "rasters": {"path": str(p), "sha256": sha(p)},
            "explicit_empty_acquisition": True,
        }
        dump(root / "status.json", result)
        return result


def run(path, device):
    spec = read_json(path)
    required = {
        "schema",
        "slots",
        "rgb_spec",
        "view_rgb_specs",
        "feature_specs",
        "projection",
        "original",
        "original_guide",
        "region",
        "readers",
        "output",
    }
    if set(spec) != required or spec["schema"] != "journal-temporal-case-v1":
        raise ValueError("Invalid complete temporal intervention")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    rgb = read_json(verified(spec["rgb_spec"]))
    inputs = read_json(verified(rgb["inputs"]))
    view_inputs = [
        read_json(verified(read_json(verified(ref))["inputs"])) for ref in spec["view_rgb_specs"]
    ]
    selected = subset_inputs(inputs, view_inputs, spec["slots"])
    inputs_ref = _bind(root / "observations.json", selected)
    rgb.update(inputs=inputs_ref, output=str(root / "composite"))
    rgb_ref = _bind(root / "rgb-spec.json", rgb)
    manifest = root / "composite/rasters.json"
    status = root / "composite/status.json"
    if not status.exists() or read_json(status)["state"] != "complete":
        composite.prepare(Path(rgb_ref["path"]))
    composite_ref = {"path": str(manifest), "sha256": sha(manifest)}
    sources = {}
    for name in ["detail", "context"]:
        feature = read_json(verified(spec["feature_specs"][name]))
        feature["rasters"] = composite_ref
        feature["output"] = str(root / "feature-inputs" / name)
        feature_ref = _bind(root / (name + "-feature.json"), feature)
        value = {
            "schema": "jstars-raw-vision-v1",
            "feature_spec": feature_ref,
            "projection": spec["projection"] if name == "detail" else None,
            "fraction": 1,
            "seed": 41,
            "raw_output": str(root / "raw-inputs"),
            "output": str(root / "encoded" / name),
        }
        source = _bind(root / (name + ".json"), value)
        reencode(Path(source["path"]), device)
        p = root / "encoded" / name / "status.json"
        sources[name] = {"path": str(p), "sha256": sha(p)}
    original = read_json(verified(spec["original"]))
    geometry = read_json(verified(original["geometry"]))
    guide = read_json(verified(spec["original_guide"]))
    raw = read_json(verified(guide["rgb_sources"]))
    index = read_json(verified(guide["index"]))
    count = len(raw["views"])
    if count != len(geometry["temporal"]["views"]) or count != len(view_inputs):
        raise ValueError("Registered temporal slot counts differ")
    first = geometry["temporal"]["views"][0]["features"]
    shape = np.load(first["path"], mmap_mode="r").shape
    empty = _empty_assets(root / "empty", index, raw, shape)
    sources["views"] = []
    for k, view in enumerate(raw["views"]):
        producer = view["producer_spec"]
        rasters = view["rasters"]
        if k in spec["slots"]:
            arrays = {
                label: {"path": ref["path"], "sha256": ref["sha256"]}
                for label, ref in [
                    ("native", geometry["temporal"]["views"][k]["features"]),
                    ("native-mask", geometry["temporal"]["views"][k]["coverage"]),
                ]
            }
        else:
            arrays = empty["arrays"]
            rasters = empty["rasters"]
            feature = read_json(verified(producer))
            feature.update(rasters=rasters, output=str(root / "empty" / f"slot{k}"))
            producer = _bind(root / f"empty-producer-{k}.json", feature)
        receipt = {
            "state": "complete",
            "arrays": arrays,
            "feature_spec": producer,
            "rasters": rasters,
            "raw_mask_before_visual_encoder": True,
            "pca_refitted": False,
            "model_weights_refitted": False,
            "slot": k,
            "actual_observation_retained": k in spec["slots"],
        }
        sources["views"].append(_bind(root / f"view-{k}.json", receipt))
    fusion = _bind(
        root / "fusion.json",
        {
            "schema": "jstars-raw-fusion-v1",
            "original": spec["original"],
            "sources": sources,
            "output": str(root / "fusion"),
        },
    )
    fuse_reencoded(Path(fusion["path"]), device)
    p = root / "fusion/status.json"
    guided = _bind(
        root / "guidance.json",
        {
            "schema": "jstars-raw-guidance-v1",
            "original_guide": spec["original_guide"],
            "fusion": {"path": str(p), "sha256": sha(p)},
            "views": sources["views"],
            "output": str(root / "guidance"),
        },
    )
    result = guide_reencoded(Path(guided["path"]), device)
    score = _bind(
        root / "score.json",
        {
            "schema": "jstars-frozen-intervention-v1",
            "region": spec["region"],
            "feature": result["features"],
            "readers": spec["readers"],
            "output": str(root / "evaluation"),
        },
    )
    return score_frozen(Path(score["path"]), device)
