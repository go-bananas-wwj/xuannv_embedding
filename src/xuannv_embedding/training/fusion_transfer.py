"""Fixed-recipe regional fusion checks; no encoder training or recipe search."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from rasterio.transform import from_bounds

from xuannv_embedding.data_process.semantic_composite import prepare as prepare_rgb
from xuannv_embedding.downstream import full_region_readouts as full
from xuannv_embedding.downstream.native_landcover import aggregate_features
from xuannv_embedding.export import semantic_features
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.representation_search import export_recipe


def reference(path):
    return {"path": str(path), "sha256": sha(path)}


def verified(ref):
    if set(ref) != {"path", "sha256"} or sha(ref["path"]) != ref["sha256"]:
        raise ValueError("Fixed fusion transfer input identity changed")
    return Path(ref["path"])


def read_spec(path):
    spec = json.loads(Path(path).read_text())
    if (
        set(spec)
        != {
            "schema",
            "index",
            "base",
            "aef",
            "extra",
            "rgb_spec",
            "weights",
            "datasets",
            "output",
            "support_seed",
        }
        or spec["schema"] != "fixed-regional-fusion-transfer-v1"
        or type(spec["support_seed"]) is not int
        or not spec["datasets"]
    ):
        raise ValueError("Invalid fixed regional fusion transfer specification")
    for key in ["index", "base", "aef", "rgb_spec", "weights"]:
        verified(spec[key])
    for ref in spec["extra"].values():
        verified(ref)
    for item in spec["datasets"]:
        if set(item) != {"name", "data"}:
            raise ValueError("Invalid fixed fusion evaluation dataset")
        verified(item["data"])
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {"spec_sha256": sha(path)}
    if (root / "identity.json").exists():
        if json.loads((root / "identity.json").read_text()) != identity:
            raise ValueError("Fixed fusion transfer contract changed")
    dump(root / "identity.json", identity)
    return spec


def prepare_inputs(spec):
    root = Path(spec["output"])
    rasters = prepare_rgb(Path(spec["rgb_spec"]["path"]))
    index = json.loads(Path(spec["index"]["path"]).read_text())
    cache = root / "semantic-index.json"
    dump(cache, {"state": "complete", "months": index["months"], "records": index["records"]})
    statistics = root / "rgb-statistics.json"
    dump(statistics, {"mean": [0, 0, 0], "std": [1, 1, 1]})
    semantic = {
        "schema": "regional-semantic-features-v1",
        "cache": reference(cache),
        "statistics": reference(statistics),
        "weights": spec["weights"],
        "model": "vit_large_patch16_dinov3.sat493m",
        "source": "highres_optical",
        "output": str(root / "semantic"),
        "seed": 41,
        "fit_samples": 65536,
        "components": 64,
        "rasters": reference(rasters),
        "output_shape": [128, 128],
        "max_pixels": 512,
        "crop_batch": 8,
    }
    dump(root / "semantic-spec.json", semantic)


def feature_exports(spec):
    root = Path(spec["output"])
    index = json.loads(Path(spec["index"]["path"]).read_text())
    count = len(index["records"])
    registered = {"base": spec["base"], "aef": spec["aef"], **spec["extra"]}
    sources = {}
    for name, ref in registered.items():
        array = np.load(verified(ref), mmap_mode="r")
        if array.shape != (count, 128, 128, 64) or not np.isfinite(array).all():
            raise ValueError("Regional feature shape or finiteness differs")
        path = root / "sources" / f"{name}.npy"
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            pending = path.with_suffix(".partial.npy")
            result = np.lib.format.open_memmap(
                pending, mode="w+", dtype=np.float32, shape=(count, 64, 128, 128)
            )
            for i in range(count):
                result[i] = array[i].transpose(2, 0, 1)
            result.flush()
            pending.replace(path)
        sources[name] = reference(path)
    # Preserve the original full-precision arrays for the original baselines.
    features = dict(registered)
    semantic = root / "semantic/annual.npy"
    state = json.loads((root / "semantic/status.json").read_text())
    if state["stage"] != "complete" or sha(semantic) != state["annual_sha256"]:
        raise ValueError("Semantic source is incomplete or changed")
    sources["semantic"] = reference(semantic)
    recipes = [
        {
            "name": "fusion_local",
            "kind": "blocks",
            "sources": ["public", "semantic"],
            "dimensions": [40, 24],
        },
        {
            "name": "fusion_context",
            "kind": "blocks",
            "sources": ["public", "semantic"],
            "dimensions": [40, 24],
            "post_pool": 13,
            "affinity_temperature": 1.0,
        },
        *[
            {
                "name": f"{name}_context",
                "kind": "identity",
                "sources": [name],
                "post_pool": 13,
                "affinity_temperature": 1.0,
            }
            for name in ["base", "aef"]
        ],
        {"name": "semantic", "kind": "identity", "sources": ["semantic"]},
    ]
    fusion = {
        "sources": {**sources, "public": sources["base"]},
        "index": spec["index"],
        "seed": 41,
        "fit_samples": 65536,
    }
    for recipe in recipes:
        output = root / "representations" / recipe["name"]
        export_recipe(fusion, recipe, output)
        array = np.load(output / "annual.npy", mmap_mode="r")
        path = output / "features.npy"
        if not path.exists():
            pending = path.with_suffix(".partial.npy")
            result = np.lib.format.open_memmap(
                pending, mode="w+", dtype=np.float32, shape=(count, 128, 128, 64)
            )
            for i in range(count):
                result[i] = array[i].transpose(1, 2, 0)
            result.flush()
            pending.replace(path)
        features[recipe["name"]] = reference(path)
        dump(root / "feature-status.json", {"state": "exporting", "completed": recipe["name"]})
    dump(root / "features.json", features)
    return features


def register_datasets(spec):
    root = Path(spec["output"])
    index = json.loads(Path(spec["index"]["path"]).read_text())
    features = json.loads((root / "features.json").read_text())
    first_data = json.loads(Path(spec["datasets"][0]["data"]["path"]).read_text())
    source_valid = np.load(verified(first_data["valid"]), mmap_mode="r")
    jobs = []
    for dataset in spec["datasets"]:
        data = json.loads(Path(dataset["data"]["path"]).read_text())
        if [r["patch_id"] for r in data["records"]] != [r["patch_id"] for r in index["records"]]:
            raise ValueError("Evaluation feature geography order differs")
        valid = np.load(verified(data["valid"]), mmap_mode="r")
        verified(data["labels"])
        data = copy.deepcopy(data)
        data["features"], data["feature_identities"] = {}, {}
        directory = root / "datasets" / dataset["name"]
        directory.mkdir(parents=True, exist_ok=True)
        for name, ref in features.items():
            values = np.load(verified(ref), mmap_mode="r")
            if data["size"] == 128 and data["crs"] == index["crs"]:
                if any(
                    "bounds" in b and a["bounds"] != b["bounds"]
                    for a, b in zip(index["records"], data["records"])
                ):
                    raise ValueError("Evaluation tile bounds differ")
                data["features"][name] = ref
            else:
                path = directory / f"{name}.npy"
                if not path.exists():
                    n = data["size"]
                    pending = path.with_suffix(".partial.npy")
                    result = np.lib.format.open_memmap(
                        pending, mode="w+", dtype=np.float32, shape=(len(values), n, n, 64)
                    )
                    for i, record in enumerate(data["records"]):
                        projected, mask = aggregate_features(
                            values[i],
                            source_valid[i],
                            from_bounds(*index["records"][i]["bounds"], 128, 128),
                            index["crs"],
                            from_bounds(*record["bounds"], n, n),
                            data["crs"],
                            n,
                        )
                        if not mask[valid[i]].all():
                            raise ValueError(
                                "Projected features do not cover fixed evaluation mask"
                            )
                        projected[~valid[i]] = 0
                        result[i] = projected
                    result.flush()
                    pending.replace(path)
                data["features"][name] = reference(path)
            data["feature_identities"][name] = {
                "channels": 64,
                "kind": "fixed_fusion_transfer",
                "source": ref,
                "encoder_retrained": False,
                "labels_used_for_projection": False,
            }
        path = directory / "data.json"
        dump(path, data)
        for name in features:
            for head in ["mlp", "conv3x3"]:
                job = {
                    "protocol": full.PROTOCOL,
                    "data": reference(path),
                    "model": name,
                    "head": head,
                    "budget": 20,
                    "seed": spec["support_seed"],
                    "device": "npu:0",
                    "archives": {},
                    "output": str(root / "readouts" / dataset["name"] / name / head),
                }
                sp = root / "jobs" / f"{dataset['name']}_{name}_{head}.json"
                dump(sp, job)
                full.specification(sp)
                jobs.append(
                    {
                        "spec": str(sp),
                        "dataset": dataset["name"],
                        "model": name,
                        "head": head,
                        "output": job["output"],
                    }
                )
    dump(root / "jobs.json", jobs)
    dump(root / "feature-status.json", {"state": "complete", "jobs": len(jobs)})


def report(spec):
    root = Path(spec["output"])
    rows, supports, summaries = [], {}, []
    jobs = json.loads((root / "jobs.json").read_text())
    for job in jobs:
        directory = Path(job["output"]) / "full_region"
        identity = json.loads((directory / "identity.json").read_text())
        if (
            identity["state"] != "complete"
            or identity["spec_sha256"] != sha(job["spec"])
            or identity["results_sha256"] != sha(directory / "results.json")
        ):
            raise ValueError("Fixed regional readout is incomplete or changed")
        results = json.loads((directory / "results.json").read_text())
        for item in results:
            record = json.loads((directory / "records" / f"{item['key']}.json").read_text())
            key = (job["dataset"], job["head"], item["task"])
            evidence = {
                "support": record["support"],
                "observations": item["metrics"]["observations"],
            }
            if key in supports and supports[key] != evidence:
                raise ValueError("Paired downstream support or scoring observations differ")
            supports[key] = evidence
            rows.append(
                {
                    "dataset": job["dataset"],
                    "model": job["model"],
                    "head": job["head"],
                    "task": item["task"],
                    "source": item["source"],
                    "metrics": item["metrics"],
                }
            )
        summaries.append(
            {
                "dataset": job["dataset"],
                "model": job["model"],
                "head": job["head"],
                "mean_task_f1": float(np.mean([r["metrics"]["f1"] for r in results])),
            }
        )
    dump(
        root / "results.json",
        {
            "state": "complete",
            "rows": rows,
            "summaries": summaries,
            "paired_support_verified": True,
            "all_results_retained": True,
            "scope": "overlapping full-region map agreement, not blind testing",
        },
    )
    lines = [
        "# Fixed regional fusion transfer",
        "",
        "|Dataset|Model|Head|Mean task F1 %|",
        "|---|---|---|---:|",
    ]
    for row in summaries:
        lines.append(
            f"|{row['dataset']}|{row['model']}|{row['head']}|{100*row['mean_task_f1']:.3f}|"
        )
    lines += [
        "",
        "All task F1/AP/IoU and declines remain in results.json. Full-region scores include",
        "support/calibration pixels; OSM/WorldCover targets may have been seen by the encoder.",
        "DINOv3 adds global satellite pretraining. No new encoder training, no recipe selection.",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment fusion-transfer")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["prepare", "semantic", "features", "report"], required=True
    )
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args(argv)
    spec = read_spec(args.spec)
    if args.phase == "prepare":
        prepare_inputs(spec)
    elif args.phase == "semantic":
        semantic_features.main(
            ["--spec", str(Path(spec["output"]) / "semantic-spec.json"), "--device", args.device]
        )
    elif args.phase == "features":
        feature_exports(spec)
        register_datasets(spec)
    else:
        report(spec)
    return 0
