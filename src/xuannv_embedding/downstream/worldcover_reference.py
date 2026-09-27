"""Prepare provenance-bound ESA WorldCover labels on an existing comparison grid."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform

from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.export.context import dump, sha

CLASSES = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "crop",
    50: "built",
    60: "bare",
    70: "snow",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss",
}


def project_tile(source, bounds, size, crs):
    output = np.zeros((size, size), np.uint8)
    affine = from_bounds(*bounds, size, size)
    reproject(
        rasterio.band(source, 1),
        output,
        src_transform=source.transform,
        src_crs=source.crs,
        dst_transform=affine,
        dst_crs=crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    if not np.isin(output, [0, *CLASSES]).all():
        raise ValueError("source contains values outside the ESA WorldCover class legend")
    return output, affine


def prepare(spec_path):
    spec = primary._load(spec_path)
    expected = {
        "protocol",
        "raster",
        "provenance",
        "reference_cache",
        "reference_crs",
        "osm_labels",
        "training_validation_domain",
        "min_training_tiles",
        "output",
    }
    if set(spec) != expected or spec["protocol"] != "worldcover-reference-v1":
        raise ValueError("invalid WorldCover preparation specification")
    provenance = primary._registered(spec["provenance"])
    if (
        provenance.get("dataset") != "ESA WorldCover"
        or provenance.get("year") not in [2020, 2021]
        or provenance.get("version") != {2020: "v100", 2021: "v200"}.get(provenance.get("year"))
        or provenance.get("sha256") != spec["raster"]["sha256"]
        or sha(Path(spec["raster"]["path"])) != spec["raster"]["sha256"]
        or type(spec["min_training_tiles"]) is not int
        or spec["min_training_tiles"] < 1
    ):
        raise ValueError("WorldCover source identity or class eligibility rule differs")
    cache = primary._registered(spec["reference_cache"])
    primary.multitask_features._grid(cache, cache)
    if set(spec["osm_labels"]) != {"train", "validation", "test"}:
        raise ValueError("three registered OSM label partitions are required")
    domain = spec["training_validation_domain"]
    if sha(Path(domain["path"])) != domain["sha256"]:
        raise ValueError("training/validation feature domain changed")
    indices = cache["split"]["train"] + cache["split"]["validation"]
    valid = np.load(domain["path"], allow_pickle=False)
    size = cache["data"]["patch_size"]
    if valid.dtype != np.bool_ or valid.shape != (len(indices), size, size):
        raise ValueError("invalid training/validation feature domain")
    feature_valid = dict(zip(indices, valid))
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=False)
    dump(root / "status.json", {"state": "preparing", "phase": "train_validation"})
    codes, sample_checks = {}, 0
    with rasterio.open(spec["raster"]["path"]) as source:
        if source.count != 1 or source.crs is None:
            raise ValueError("WorldCover must be a georeferenced single-band class raster")

        def load(index):
            nonlocal sample_checks
            bounds = cache["records"][index]["bounds"]
            image, affine = project_tile(source, bounds, size, spec["reference_crs"])
            positions = [(0, 0), (size // 2, size // 2), (size - 1, size - 1)]
            xy = [affine * (x + 0.5, y + 0.5) for y, x in positions]
            sx, sy = transform(spec["reference_crs"], source.crs, *zip(*xy))
            sampled = list(source.sample(list(zip(sx, sy))))
            for (y, x), value in zip(positions, sampled):
                if image[y, x] != value[0]:
                    raise ValueError("categorical warp differs from native nearest-pixel sampling")
                sample_checks += 1
            codes[index] = image

        for index in indices:
            load(index)
        eligibility, selected = {}, []
        for code, name in CLASSES.items():
            task = "worldcover_" + name
            counts = {}
            for split in ["train", "validation"]:
                positive = negative = eligible_tiles = 0
                for index in cache["split"][split]:
                    mask = feature_valid[index] & (codes[index] != 0)
                    p = int(((codes[index] == code) & mask).sum())
                    n = int(((codes[index] != code) & mask).sum())
                    positive += p
                    negative += n
                    eligible_tiles += int(p > 0 and n > 0)
                counts[split] = dict(
                    positive_pixels=positive,
                    negative_pixels=negative,
                    eligible_tiles=eligible_tiles,
                )
            usable = (
                counts["train"]["eligible_tiles"] >= spec["min_training_tiles"]
                and counts["validation"]["positive_pixels"] > 0
                and counts["validation"]["negative_pixels"] > 0
            )
            eligibility[task] = {"code": code, "included": usable, "counts": counts}
            if usable:
                selected.append(task)
        if not selected:
            raise ValueError("no WorldCover classes support the registered validation protocol")
        # Class list is frozen before opening held-out query labels.
        schema = {
            "C": {"osm": list(primary.OSM_TASKS), "worldcover": selected},
            "R": {"worldcover": selected},
            "Q": {"osm": list(primary.OSM_TASKS)},
        }
        dump(
            root / "class_registration.json",
            {
                "state": "locked",
                "task_schema": schema,
                "classes": eligibility,
                "selection_scope": "training and validation labels plus fixed feature domain only",
                "test_labels_opened_for_selection": False,
            },
        )
        for index in cache["split"]["test"]:
            load(index)
    labels, test_counts = {}, {}
    for split in ["train", "validation", "test"]:
        selected_indices = cache["split"][split]
        original = spec["osm_labels"][split]
        if sha(Path(original["path"])) != original["sha256"]:
            raise ValueError("registered OSM label bundle changed")
        with np.load(original["path"], allow_pickle=False) as old:
            if (
                not np.array_equal(old["indices"], selected_indices)
                or str(old["cache_sha256"].item()) != spec["reference_cache"]["sha256"]
            ):
                raise ValueError("OSM reference partition differs")
            arrays = {task: old[task] for task in primary.OSM_TASKS}
            if any(
                y.shape != (len(selected_indices), size, size) or not np.isin(y, [-1, 0, 1]).all()
                for y in arrays.values()
            ):
                raise ValueError("OSM reference geometry or binary labels differ")
        arrays.update(
            indices=np.asarray(selected_indices, np.int64),
            cache_sha256=np.array(spec["reference_cache"]["sha256"]),
        )
        raw = np.stack([codes[i] for i in selected_indices])
        for task in selected:
            code = eligibility[task]["code"]
            arrays[task] = np.where(raw == 0, -1, raw == code).astype(np.int8)
            if split == "test":
                test_counts[task] = int((arrays[task] == 1).sum())
        path = root / (split + ".npz")
        np.savez_compressed(path, **arrays)
        labels[split] = {"path": str(path), "sha256": sha(path)}
        np.savez_compressed(
            root / (split + "_worldcover_codes.npz"),
            indices=np.asarray(selected_indices),
            codes=raw,
        )
    result = {
        "state": "complete",
        "protocol": spec["protocol"],
        "source": provenance,
        "reference_crs": spec["reference_crs"],
        "resampling": "nearest",
        "native_pixel_crosschecks": sample_checks,
        "labels": labels,
        "task_schema": schema,
        "class_registration_sha256": sha(root / "class_registration.json"),
        "test_positive_counts_after_class_freeze": test_counts,
        "spec_sha256": sha(Path(spec_path)),
        "implementation_sha256": sha(Path(__file__)),
        "trained_models": False,
    }
    dump(root / "verification.json", result)
    dump(root / "status.json", {"state": "complete"})
    return result
