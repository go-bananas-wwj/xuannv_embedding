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


def empty_slot_rasters(reference, output):
    """Give each withheld calendar slot a distinct verified zero-coverage identity."""
    import os

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    original = read_json(verified(reference))
    result = copy.deepcopy(original)
    for i, row in enumerate(result["records"]):
        source = verified(row["raster"])
        with rasterio.open(source) as dataset:
            if dataset.dataset_mask().any():
                raise ValueError("Calendar placeholder must have zero coverage")
        target = root / f"{i:04d}.tif"
        if not target.exists():
            os.link(source, target)
        if sha(target) != row["raster"]["sha256"]:
            raise ValueError("Empty slot image identity changed")
        row["raster"] = {"path": str(target), "sha256": row["raster"]["sha256"]}
    return _bind(root / "rasters.json", result)


def prepare_guidance_continuation(path):
    """Retain completed encoders/fusion and repair only empty calendar path identities."""
    from xuannv_embedding.data_process import subpixel_rgb as rgb

    spec = read_json(path)
    if set(spec) != {"schema", "case", "output"} or spec["schema"] != "journal-temporal-guide-v1":
        raise ValueError("Invalid temporal guidance continuation")
    case = read_json(verified(spec["case"]))
    if case.get("schema") != "journal-temporal-case-v1":
        raise ValueError("Continuation requires a registered whole-source temporal case")
    original_root, root = Path(case["output"]), Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    fusion_path = original_root / "fusion/status.json"
    fusion = read_json(fusion_path)
    if fusion["state"] != "complete" or not fusion["all_local_visual_sources_reencoded"]:
        raise ValueError("Temporal continuation requires completed frozen upstream fusion")
    fusion_spec = original_root / "fusion.json"
    if (
        fusion["identity"]["spec"] != sha(fusion_spec)
        or read_json(fusion_spec)["original"] != case["original"]
    ):
        raise ValueError("Temporal upstream fusion binding changed")
    verified(fusion["features"])
    guide = read_json(verified(case["original_guide"]))
    raw = read_json(verified(guide["rgb_sources"]))
    views = []
    for k in range(len(raw["views"])):
        source = original_root / f"view-{k}.json"
        value = read_json(source)
        if value["slot"] != k or value["actual_observation_retained"] != (k in case["slots"]):
            raise ValueError("Temporal continuation slots differ from original intervention")
        if k not in case["slots"]:
            if np.count_nonzero(np.load(verified(value["arrays"]["native-mask"]), mmap_mode="r")):
                raise ValueError("Withheld temporal feature has nonzero coverage")
            rasters = empty_slot_rasters(value["rasters"], root / "empty-slots" / str(k))
            producer = read_json(verified(value["feature_spec"]))
            producer.update(rasters=rasters, output=str(root / "empty-producers" / str(k)))
            producer_ref = _bind(root / f"producer-{k}.json", producer)
            value.update(rasters=rasters, feature_spec=producer_ref)
        value["original_view"] = {"path": str(source), "sha256": sha(source)}
        reference = _bind(root / f"view-{k}.json", value)
        views.append(reference)
        raw["views"][k].update(producer_spec=value["feature_spec"], rasters=value["rasters"])
    raw["output"] = str(root / "preflight-only")
    raw_ref = _bind(root / "preflight-rgb.json", raw)
    checked, index = rgb.specification(Path(raw_ref["path"]))
    rgb.Sources(checked, index)
    guidance = _bind(
        root / "guidance.json",
        dict(
            schema="jstars-raw-guidance-v1",
            original_guide=case["original_guide"],
            fusion={"path": str(fusion_path), "sha256": sha(fusion_path)},
            views=views,
            output=str(root / "guidance"),
        ),
    )
    return case, guidance


def continue_guidance(path, device):
    case, reference = prepare_guidance_continuation(path)
    result = guide_reencoded(Path(reference["path"]), device)
    root = Path(case["output"])
    score = _bind(
        root / "score-recovery.json",
        dict(
            schema="jstars-frozen-intervention-v1",
            region=case["region"],
            feature=result["features"],
            readers=case["readers"],
            output=str(root / "evaluation"),
        ),
    )
    return score_frozen(Path(score["path"]), device)


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
            rasters = empty_slot_rasters(empty["rasters"], root / "empty" / f"slot{k}")
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
