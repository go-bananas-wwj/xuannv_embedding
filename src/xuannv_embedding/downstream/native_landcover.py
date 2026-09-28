"""Prepare an external categorical product on its native grid for frozen readouts."""

import copy
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import Resampling, reproject, transform, transform_bounds
from rasterio.windows import Window

from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.downstream import review_readouts as review
from xuannv_embedding.export.context import dump, sha


def interior_cells(bounds, source_crs, target_transform, target_crs, size):
    """Only cells wholly inside the original tile; shared edges cannot duplicate cells."""
    row, col = np.indices((size, size))
    inside = np.ones((size, size), bool)
    for dy, dx in [(0, 0), (0, 0.5), (0, 1), (0.5, 0), (0.5, 1), (1, 0), (1, 0.5), (1, 1)]:
        xx, yy = target_transform * (col + dx, row + dy)
        sx, sy = transform(target_crs, source_crs, xx.ravel(), yy.ravel())
        sx, sy = np.asarray(sx).reshape(row.shape), np.asarray(sy).reshape(row.shape)
        inside &= (
            (sx >= bounds[0] - 1e-7)
            & (sx <= bounds[2] + 1e-7)
            & (sy >= bounds[1] - 1e-7)
            & (sy <= bounds[3] + 1e-7)
        )
    return inside


def aggregate_features(
    values, valid, source_transform, source_crs, target_transform, target_crs, size
):
    values, valid = np.asarray(values), np.asarray(valid)
    if (
        values.ndim != 3
        or valid.dtype != bool
        or valid.shape != values.shape[:2]
        or not np.isfinite(values[valid]).all()
    ):
        raise ValueError("finite HWC features and matching boolean validity required")
    channels = values.shape[-1]
    coverage = np.zeros((size, size), np.float32)
    kw = dict(
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.average,
        num_threads=2,
    )
    reproject(valid.astype(np.float32), coverage, **kw)
    out = np.full((channels, size, size), np.nan, np.float32)
    source = np.where(valid[..., None], values, 0).transpose(2, 0, 1).astype(np.float32)
    reproject(source, out, dst_nodata=np.nan, **kw)
    mask = (coverage >= 1 - 1e-6) & np.isfinite(out).all(0)
    out[:, ~mask] = 0
    return out.transpose(1, 2, 0), mask


def _reference(path):
    return {"path": str(path), "sha256": sha(Path(path))}


def prepare(spec_path):
    spec = primary._load(spec_path)
    fields = {
        "protocol",
        "parent_cohort",
        "shared",
        "raster",
        "provenance",
        "source_crs",
        "product",
        "year",
        "legend",
        "size",
        "budgets",
        "support_seeds",
        "output",
    }
    if (
        set(spec) != fields
        or spec["protocol"] != "native-landcover-cohort-v1"
        or type(spec["size"]) is not int
        or spec["size"] < 16
        or spec["size"] % 16
        or not spec["legend"]
        or not spec["budgets"]
        or not spec["support_seeds"]
    ):
        raise ValueError("invalid native product specification")
    primary._registered(spec["parent_cohort"])
    parent, cache = primary._spec(spec["parent_cohort"]["path"])
    for key in ["raster", "provenance"]:
        if sha(Path(spec[key]["path"])) != spec[key]["sha256"]:
            raise ValueError("native product input changed")
    if (
        any(type(v) is not int or v not in parent["budgets"] for v in spec["budgets"])
        or any(v not in parent["support_seeds"] for v in spec["support_seeds"])
        or len(set(spec["budgets"])) != len(spec["budgets"])
        or len(set(spec["support_seeds"])) != len(spec["support_seeds"])
    ):
        raise ValueError("native cohort must use registered support budgets and seeds")
    root, size = Path(spec["output"]), spec["size"]
    root.mkdir(parents=True, exist_ok=False)
    models = list(parent["models"])
    native_cache = copy.deepcopy(cache)
    native_cache["data"]["patch_size"] = size
    layouts, label_maps, feature_masks = {}, {}, {}
    masks_path = root / "masks"
    masks_path.mkdir()
    exports = {
        model: copy.deepcopy(primary._load(parent["models"][model]["manifest_path"]))
        for model in models
    }
    hashes = {model: {} for model in models}
    owners = {}
    eligibility = {}
    observed = set()
    with rasterio.open(spec["raster"]["path"]) as source:
        if source.count != 1 or source.crs is None or source.transform.b or source.transform.d:
            raise ValueError("categorical product requires a north-up single-band raster")
        native_cache["data"]["crs"] = source.crs.to_wkt()
        for i, original in enumerate(cache["records"]):
            box = transform_bounds(
                spec["source_crs"], source.crs, *original["bounds"], densify_pts=21
            )
            c0, r0 = (~source.transform) * (box[0], box[3])
            c1, r1 = (~source.transform) * (box[2], box[1])
            left, top = math.floor(c0), math.floor(r0)
            if math.ceil(c1) - left > size or math.ceil(r1) - top > size:
                raise ValueError("fixed native window does not cover the original tile")
            win = Window(left, top, size, size)
            tr = source.window_transform(win)
            bounds = list(array_bounds(size, size, tr))
            native_cache["records"][i]["bounds"] = bounds
            layouts[i] = {"window": win, "transform": tr, "bounds": bounds}
            for model in models:
                # Preserve record IDs and ordering, but never keep an old-grid file path.
                exports[model]["records"][i] = {"patch_id": original["patch_id"], "bounds": bounds}
        cache_path = root / "cache.json"
        dump(cache_path, native_cache)
        for phase in ["calibration", "test"]:
            batches = {}
            for model in models:
                batches[model], _, common = review._shared(
                    {"cohort": spec["parent_cohort"], "shared": spec["shared"], "model": model},
                    phase,
                )
            indices = list(batches[models[0]].indices)
            for local, i in enumerate(indices):
                original = cache["records"][i]
                layout = layouts[i]
                tr = from_bounds(
                    *original["bounds"], cache["data"]["patch_size"], cache["data"]["patch_size"]
                )
                values = {}
                mask = interior_cells(
                    original["bounds"], spec["source_crs"], layout["transform"], source.crs, size
                )
                for model in models:
                    values[model], valid = aggregate_features(
                        batches[model].values[local],
                        common[local],
                        tr,
                        spec["source_crs"],
                        layout["transform"],
                        source.crs,
                        size,
                    )
                    mask &= valid
                labels = source.read(1, window=layout["window"], boundless=True, fill_value=0)
                if not np.isin(labels, [0, *map(int, spec["legend"])]).all():
                    raise ValueError("unknown source product class code")
                mask &= labels != 0
                rr, cc = np.where(mask)
                for row, col in zip(
                    rr + int(layout["window"].row_off), cc + int(layout["window"].col_off)
                ):
                    key = (int(row), int(col))
                    if key in owners:
                        raise ValueError("native cell is shared by multiple support/query tiles")
                    owners[key] = i
                label_maps[i] = labels
                feature_masks[i] = mask
                observed.add(i)
                np.save(masks_path / (original["patch_id"] + ".npy"), mask)
                for model in models:
                    path = root / "features" / model / (original["patch_id"] + ".npz")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    values[model][~mask] = 0
                    data = dict(embedding=values[model].transpose(2, 0, 1)[None], valid_mask=mask)
                    selection = parent["models"][model]["selection"]
                    if selection["kind"] == "monthly":
                        data["timestamps"] = np.array([int(selection["period"].replace("-", ""))])
                    np.savez_compressed(path, **data)
                    digest = sha(path)
                    hashes[model][original["patch_id"]] = digest
                    exports[model]["records"][i].update(path=str(path), sha256=digest)
                dump(
                    root / "status.json",
                    {
                        "state": "preparing",
                        "phase": phase,
                        "tiles_complete": len(observed),
                        "native_cells": len(owners),
                    },
                )
            if phase == "calibration":
                tasks = []
                for code, name in spec["legend"].items():
                    task = spec["product"] + "_" + name
                    counts = {}
                    for split in ["train", "validation"]:
                        pos = neg = eligible = 0
                        for i in cache["split"][split]:
                            p = int(((label_maps[i] == int(code)) & feature_masks[i]).sum())
                            n = int(((label_maps[i] != int(code)) & feature_masks[i]).sum())
                            pos += p
                            neg += n
                            eligible += int(p > 0 and n > 0)
                        counts[split] = {
                            "positive": pos,
                            "negative": neg,
                            "eligible_tiles": eligible,
                        }
                    usable = (
                        counts["train"]["eligible_tiles"] >= max(spec["budgets"])
                        and counts["validation"]["positive"] > 0
                        and counts["validation"]["negative"] > 0
                    )
                    eligibility[task] = {"code": int(code), "included": usable, "counts": counts}
                    if usable:
                        tasks.append(task)
                if not tasks:
                    raise ValueError("no classes meet the registered support availability rule")
                dump(
                    root / "class_registration.json",
                    {
                        "state": "locked",
                        "tasks": tasks,
                        "classes": eligibility,
                        "query_labels_read": False,
                    },
                )
        for model in models:
            manifest = exports[model]
            selection = parent["models"][model]["selection"]
            # Only the already registered selected representation is aggregated.
            manifest["months"] = (
                [f"annual_{selection['period']}"]
                if selection["kind"] == "annual"
                else (
                    [f"mean_{selection['period'].replace('/', '_')}"]
                    if selection["kind"] == "temporal_mean"
                    else [selection["period"]]
                )
            )
            manifest["cache_sha256"] = sha(cache_path)
            manifest["exported_indices"] = sorted(observed)
            manifest.pop("exported_splits", None)
            manifest.update(
                native_product=spec["product"],
                native_crs=source.crs.to_wkt(),
                native_resolution=list(source.res),
                feature_aggregation="area average",
                parent_manifest_sha256=parent["models"][model]["manifest_sha256"],
            )
            path = root / "features" / model / "manifest.json"
            dump(path, manifest)
        label_refs = {}
        for split in ["train", "validation", "test"]:
            indices = cache["split"][split]
            labels = np.stack([label_maps[i] for i in indices])
            mask = np.stack([feature_masks[i] for i in indices])
            arrays = {
                task: np.where(mask, labels == eligibility[task]["code"], -1).astype(np.int8)
                for task in tasks
            }
            path = root / (split + "_labels.npz")
            np.savez_compressed(
                path, indices=np.array(indices), cache_sha256=np.array(sha(cache_path)), **arrays
            )
            label_refs[split] = _reference(path)
        new_models = {}
        for model in models:
            path = root / "features" / model / "manifest.json"
            new_models[model] = {
                **parent["models"][model],
                "manifest_path": str(path),
                "manifest_sha256": sha(path),
                "cache_path": str(cache_path),
                "cache_sha256": sha(cache_path),
                "tile_sha256": hashes[model],
            }
        evidence = root / "native_grid_audit.json"
        dump(
            evidence,
            {
                "state": "verified",
                "spec": _reference(spec_path),
                "source": spec["raster"],
                "source_provenance": spec["provenance"],
                "native_crs": source.crs.to_wkt(),
                "native_resolution": list(source.res),
                "original_crs": spec["source_crs"],
                "unique_native_cells": len(owners),
                "duplicate_native_cells": 0,
                "label_resampling": "none; native cells",
                "feature_resampling": "area-weighted average; fully valid cells only",
                "original_split_preserved": True,
                "tile_valid_pixels": {str(i): int(v.sum()) for i, v in feature_masks.items()},
                "labels_used_to_train_encoder": False,
                "implementation_sha256": sha(Path(__file__)),
            },
        )
    geographic = root / "geographic_audit.json"
    dump(
        geographic,
        {
            "state": "verified",
            "reference_cache_sha256": sha(cache_path),
            "model_manifest_sha256": {k: v["manifest_sha256"] for k, v in new_models.items()},
            "evidence": [_reference(evidence)],
        },
    )
    cohort = copy.deepcopy(parent)
    cohort.update(
        output=str(root / "cohort_context"),
        reference_cache=_reference(cache_path),
        labels=label_refs,
        models=new_models,
        budgets=spec["budgets"],
        support_seeds=spec["support_seeds"],
        geographic_audit=_reference(geographic),
        task_schema={fam: {spec["product"]: tasks} for fam in ["C", "R", "Q"]},
    )
    lock = root / "lock.json"
    dump(lock, {"state": "locked", "contract_sha256": primary.contract_sha256(cohort)})
    cohort["lock"] = _reference(lock)
    dump(root / "cohort.json", cohort)
    primary._spec(root / "cohort.json")
    dump(
        root / "status.json",
        {
            "state": "complete",
            "classes": tasks,
            "tiles": len(observed),
            "native_cells": len(owners),
        },
    )
    return cohort
