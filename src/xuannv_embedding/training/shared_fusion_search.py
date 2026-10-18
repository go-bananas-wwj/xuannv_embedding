"""Shared, label-free feature recipes selected by their worst regional validation gain."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
from rasterio.transform import from_bounds
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from xuannv_embedding.downstream import full_region_readouts as full
from xuannv_embedding.downstream.native_landcover import aggregate_features
from xuannv_embedding.export.anchored_fusion import (
    align_samples,
    blend_aligned,
    fit_tail,
    inject_tail,
    minimax_rank,
)
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.representation_search import export_recipe


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, value)


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def verified(value):
    if set(value) != {"path", "sha256"} or sha(value["path"]) != value["sha256"]:
        raise ValueError("Shared search source identity changed")
    return Path(value["path"])


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if (
        set(plan) != {"schema", "regions", "anchor", "recipes", "output", "seed", "fit_samples"}
        or plan["schema"] != "shared-fusion-search-v1"
        or type(plan["seed"]) is not int
        or type(plan["fit_samples"]) is not int
        or plan["fit_samples"] < 65
    ):
        raise ValueError("Invalid shared fusion search plan")
    regions = json.loads(verified(plan["regions"]).read_text())["regions"]
    anchor = json.loads(verified(plan["anchor"]).read_text())
    if (
        len(regions) != 3
        or len({r["name"] for r in regions}) != 3
        or {r["name"] for r in regions} != set(anchor["values"])
        or any(
            not isinstance(r["name"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", r["name"])
            for r in regions
        )
    ):
        raise ValueError("All three registered regions and their anchors are required")
    if (
        anchor.get("state") != "complete"
        or anchor.get("selection_uses_validation_only") is not True
        or anchor.get("candidate_scores_used", False) is not False
    ):
        raise ValueError("A complete validation-only baseline anchor is required")
    try:
        minimax_rank(
            {
                r: {h: anchor["values"][r][h]["public"] for h in ["mlp", "conv3x3"]}
                for r in anchor["values"]
            },
            anchor["values"],
        )
    except KeyError as error:
        raise ValueError("Incomplete regional validation anchor") from error
    if not plan["recipes"] or not any(
        r.get("role", "candidate") == "candidate" for r in plan["recipes"]
    ):
        raise ValueError("At least one high-resolution candidate is required")
    names = set()
    for recipe in plan["recipes"]:
        if recipe["name"] in names or not recipe["name"].replace("_", "").isalnum():
            raise ValueError("Recipe names must be unique and path-safe")
        names.add(recipe["name"])
        if recipe["kind"] == "anchored":
            if (
                set(recipe) != {"name", "kind", "strength", "kernel", "ridge"}
                or type(recipe["kernel"]) is not int
                or recipe["kernel"] not in [1, 3]
                or type(recipe["strength"]) not in {int, float}
                or not 0 < recipe["strength"] <= 1
                or type(recipe["ridge"]) not in {int, float}
                or not 0 < recipe["ridge"] <= 1
            ):
                raise ValueError("Invalid public-preserving residual recipe")
        elif recipe["kind"] == "tail":
            if (
                set(recipe) != {"name", "kind", "strength", "semantic_dimensions"}
                or type(recipe["strength"]) not in {int, float}
                or not 0 < recipe["strength"] <= 1
                or type(recipe["semantic_dimensions"]) is not int
                or not 1 <= recipe["semantic_dimensions"] < 32
            ):
                raise ValueError("Invalid public-tail residual recipe")
        elif recipe["kind"] == "blocks":
            if (
                set(recipe) != {"name", "kind", "sources", "dimensions"}
                or recipe["sources"] != ["public", "semantic", "raw"]
                or len(recipe["dimensions"]) != 3
                or sum(recipe["dimensions"]) != 64
                or any(type(n) is not int or not 1 <= n <= 64 for n in recipe["dimensions"])
            ):
                raise ValueError("Invalid shared three-source recipe")
        elif recipe["kind"] == "standardized_joint":
            if set(recipe) != {"name", "kind", "sources", "role"} or (
                recipe["sources"],
                recipe["role"],
            ) not in [(["public", "semantic", "raw"], "candidate"), (["raw"], "control")]:
                raise ValueError("Invalid joint recipe or high-resolution contribution role")
        else:
            raise ValueError("Unknown shared fusion family")
    return plan, regions, anchor


def prepare_sources(plan, region):
    root = Path(plan["output"]) / region["name"] / "inputs"
    root.mkdir(parents=True, exist_ok=True)
    refs = {}
    for name in ["public", "semantic", "raw"]:
        value = region[name]
        source = verified({k: value[k] for k in ["path", "sha256"]})
        array = np.load(source, mmap_mode="r")
        if value["layout"] == "nchw":
            refs[name] = ref(source)
            continue
        if value["layout"] != "nhwc" or array.ndim != 4:
            raise ValueError("Invalid shared source layout")
        dest = root / f"{name}.npy"
        identity = {"source": ref(source), "layout": "nchw"}
        header = root / f"{name}-identity.json"
        if dest.exists():
            if not header.exists():
                raise ValueError("Unverified shared source conversion")
            saved = json.loads(header.read_text())
            if any(saved.get(k) != v for k, v in identity.items()) or sha(dest) != saved.get(
                "sha256"
            ):
                raise ValueError("Shared source conversion changed")
        if not dest.exists():
            pending = dest.with_suffix(".partial.npy")
            out = np.lib.format.open_memmap(
                pending,
                mode="w+",
                dtype=np.float32,
                shape=(len(array), array.shape[-1], *array.shape[1:3]),
            )
            for i in range(len(array)):
                out[i] = array[i].transpose(2, 0, 1)
            out.flush()
            pending.replace(dest)
            save(header, {**identity, "sha256": sha(dest)})
        refs[name] = ref(dest)
    arrays = {k: np.load(v["path"], mmap_mode="r") for k, v in refs.items()}
    index = json.loads(verified(region["index"]).read_text())
    shape = arrays["public"].shape
    if (
        len(shape) != 4
        or shape[1] != 64
        or len(index["records"]) != shape[0]
        or arrays["semantic"].shape != shape
        or arrays["raw"].shape != (shape[0], 95, *shape[-2:])
    ):
        raise ValueError("Shared input index, footprint or channel dimensions differ")
    return refs, index


def export_candidate(plan, region, recipe):
    """Only observation features and geography enter this function, never reference maps."""
    sources, index = prepare_sources(plan, region)
    root = Path(plan["output"]) / region["name"] / recipe["name"] / "embedding"
    root.mkdir(parents=True, exist_ok=True)
    from xuannv_embedding.export import anchored_fusion
    from xuannv_embedding.training import representation_search

    execution = {
        "recipe": recipe,
        "sources": sources,
        "seed": plan["seed"],
        "fit_samples": plan["fit_samples"],
        "implementation": {
            "shared": sha(__file__),
            "anchored": sha(anchored_fusion.__file__),
            "blocks": sha(representation_search.__file__),
        },
    }
    header = root / "execution.json"
    if header.exists() and json.loads(header.read_text()) != execution:
        raise ValueError("Shared candidate implementation or inputs changed")
    save(header, execution)
    if recipe["kind"] == "tail":
        return export_tail(plan, recipe, sources, root)
    if recipe["kind"] == "standardized_joint":
        return export_joint(plan, recipe, sources, root)
    if recipe["kind"] == "blocks":
        inputs = {
            "sources": {**sources, "raw": {**sources["raw"], "standardize": True}},
            "index": region["index"],
            "seed": plan["seed"],
            "fit_samples": plan["fit_samples"],
        }
        export_recipe(inputs, recipe, root)
        return root / "annual.npy"
    identity = {
        "recipe": recipe,
        "public": sources["public"],
        "semantic": sources["semantic"],
        "seed": plan["seed"],
        "fit_samples": plan["fit_samples"],
    }
    if (root / "identity.json").exists() and json.loads(
        (root / "identity.json").read_text()
    ) != identity:
        raise ValueError("Anchored recipe changed")
    save(root / "identity.json", identity)
    if (root / "status.json").exists() and (root / "annual.npy").exists():
        state = json.loads((root / "status.json").read_text())
        if state.get("stage") == "complete" and sha(root / "annual.npy") == state["sha256"]:
            return root / "annual.npy"
    u = np.load(sources["public"]["path"], mmap_mode="r")
    v = np.load(sources["semantic"]["path"], mmap_mode="r")
    rng = np.random.default_rng(plan["seed"])
    paired = []
    for i in range(len(u)):
        a, b = (
            np.asarray(u[i], np.float32).reshape(64, -1).T,
            np.asarray(v[i], np.float32).reshape(64, -1).T,
        )
        available = np.flatnonzero(
            (np.linalg.norm(a, axis=1) > 1e-8) & (np.linalg.norm(b, axis=1) > 1e-8)
        )
        pos = rng.choice(available, min(256, len(available)), replace=False)
        paired.append(np.concatenate([a[pos], b[pos]], 1))
    samples = np.concatenate(paired)
    if len(samples) > plan["fit_samples"]:
        samples = samples[rng.choice(len(samples), plan["fit_samples"], replace=False)]
    with threadpool_limits(limits=4):
        if len(samples):
            model = align_samples(samples[:, :64], samples[:, 64:], ridge=recipe["ridge"])
        else:
            model = {
                "matrix": np.zeros((64, 64), np.float32),
                "public_mean": np.zeros(64, np.float32),
                "semantic_mean": np.zeros(64, np.float32),
            }
    np.savez_compressed(root / "alignment.npz", **model)
    pending = root / "annual.partial.npy"
    result = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float32, shape=u.shape)
    for i in range(len(u)):
        result[i] = blend_aligned(
            u[i], v[i], model, strength=recipe["strength"], kernel=recipe["kernel"]
        )
    result.flush()
    pending.replace(root / "annual.npy")
    save(
        root / "status.json",
        {
            "stage": "complete",
            "shape": list(u.shape),
            "fit_samples": len(samples),
            "map_labels_used": False,
            "public_fallback_exact": True,
            "sha256": sha(root / "annual.npy"),
        },
    )
    return root / "annual.npy"


def export_tail(plan, recipe, sources, root):
    destination = root / "annual.npy"
    if destination.exists() and (root / "status.json").exists():
        status = json.loads((root / "status.json").read_text())
        if status.get("stage") == "complete" and sha(destination) == status["sha256"]:
            return destination
    arrays = [
        np.load(sources[name]["path"], mmap_mode="r") for name in ["public", "semantic", "raw"]
    ]
    rng, samples = np.random.default_rng(plan["seed"]), []
    for i in range(len(arrays[0])):
        count = arrays[0].shape[-1] * arrays[0].shape[-2]
        positions = rng.choice(count, min(256, count), replace=False)
        samples.append(
            np.concatenate(
                [
                    np.asarray(a[i], np.float32).reshape(a.shape[1], -1)[:, positions].T
                    for a in arrays
                ],
                1,
            )
        )
    sample = np.concatenate(samples)
    if len(sample) > plan["fit_samples"]:
        sample = sample[rng.choice(len(sample), plan["fit_samples"], replace=False)]
    with threadpool_limits(limits=4):
        model = fit_tail(
            sample[:, :64],
            sample[:, 64:128],
            sample[:, 128:],
            semantic_dimensions=recipe["semantic_dimensions"],
            seed=plan["seed"],
        )
    np.savez_compressed(root / "tail.npz", **model)
    result = np.lib.format.open_memmap(
        root / "annual.partial.npy", mode="w+", dtype=np.float32, shape=arrays[0].shape
    )
    for i in range(len(result)):
        with threadpool_limits(limits=4):
            result[i] = inject_tail(*(a[i] for a in arrays), model, strength=recipe["strength"])
    result.flush()
    (root / "annual.partial.npy").replace(destination)
    save(
        root / "status.json",
        {
            "stage": "complete",
            "shape": list(result.shape),
            "sha256": sha(destination),
            "model": ref(root / "tail.npz"),
            "map_labels_used": False,
            "fit_samples": len(sample),
            "public_coordinates_retained": 64,
            "residual_subspace_dimensions": 32,
            "public_tail_variance_fraction": float(model["public_tail_variance_fraction"]),
        },
    )
    return destination


def export_joint(plan, recipe, sources, root):
    destination = root / "annual.npy"
    if destination.exists() and (root / "status.json").exists():
        status = json.loads((root / "status.json").read_text())
        if status.get("stage") == "complete" and sha(destination) == status["sha256"]:
            return destination
    arrays = [np.load(sources[name]["path"], mmap_mode="r") for name in recipe["sources"]]
    shape = arrays[0].shape
    rng, samples = np.random.default_rng(plan["seed"]), []
    for i in range(len(arrays[0])):
        positions = rng.choice(
            shape[-1] * shape[-2], min(256, shape[-1] * shape[-2]), replace=False
        )
        samples.append(
            np.concatenate(
                [
                    np.asarray(a[i], np.float32).reshape(a.shape[1], -1)[:, positions].T
                    for a in arrays
                ],
                axis=1,
            )
        )
    sample = np.concatenate(samples)
    if len(sample) > plan["fit_samples"]:
        sample = sample[rng.choice(len(sample), plan["fit_samples"], replace=False)]
    with threadpool_limits(limits=4):
        scaler = StandardScaler().fit(sample)
        pca = PCA(n_components=64, svd_solver="randomized", random_state=plan["seed"])
        pca.fit(scaler.transform(sample))
    np.savez_compressed(
        root / "projection.npz",
        mean=pca.mean_,
        components=pca.components_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    result = np.lib.format.open_memmap(
        root / "annual.partial.npy", mode="w+", dtype=np.float32, shape=(shape[0], 64, *shape[-2:])
    )
    for i in range(len(result)):
        joined = np.concatenate(
            [np.asarray(a[i], np.float32).reshape(a.shape[1], -1).T for a in arrays], 1
        )
        with threadpool_limits(limits=4):
            values = pca.transform(scaler.transform(joined)).T.reshape(64, *shape[-2:])
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite shared joint projection")
        result[i] = values
    result.flush()
    (root / "annual.partial.npy").replace(destination)
    save(
        root / "status.json",
        {
            "stage": "complete",
            "shape": list(result.shape),
            "sha256": sha(destination),
            "map_labels_used": False,
            "fit_samples": len(sample),
            "per_pixel_unit_normalization": False,
            "role": recipe["role"],
        },
    )
    return destination


def reader_jobs(plan, region, recipe, feature):
    # Reference handling is deliberately downstream of the label-free exporter.
    data = json.loads(verified(region["evaluation"]).read_text())
    index = json.loads(verified(region["index"]).read_text())
    if [r["patch_id"] for r in index["records"]] != [r["patch_id"] for r in data["records"]]:
        raise ValueError("Evaluation patch order differs")
    data = copy.deepcopy(data)
    source = region["task_source"]
    data["tasks"] = {"C": {source: data["tasks"]["C"][source]}, "R": {}, "Q": {}}
    root = Path(plan["output"]) / region["name"] / recipe["name"]
    values = np.load(feature, mmap_mode="r")
    valid = np.load(verified(data["valid"]), mmap_mode="r")
    target = root / "evaluation-features.npy"
    if not target.exists():
        pending = target.with_suffix(".partial.npy")
        n = data["size"]
        result = np.lib.format.open_memmap(
            pending, mode="w+", dtype=np.float32, shape=(len(values), n, n, 64)
        )
        crs = index.get("crs", index["records"][0].get("crs"))
        for i, record in enumerate(data["records"]):
            x = np.asarray(values[i], np.float32).transpose(1, 2, 0)
            if (
                x.shape[:2] == (n, n)
                and data["crs"] == crs
                and ("bounds" not in record or record["bounds"] == index["records"][i]["bounds"])
            ):
                output = x.copy()
            else:
                h, w = x.shape[:2]
                output, mask = aggregate_features(
                    x,
                    np.ones((h, w), bool),
                    from_bounds(*index["records"][i]["bounds"], w, h),
                    crs,
                    from_bounds(*record["bounds"], n, n),
                    data["crs"],
                    n,
                )
                if not mask[valid[i]].all():
                    raise ValueError("Projected candidate does not cover fixed reference mask")
            if not np.isfinite(output[valid[i]]).all():
                raise ValueError("Nonfinite candidate within the fixed evaluation region")
            output[~valid[i]] = 0
            result[i] = output
        result.flush()
        pending.replace(target)
    data["features"] = {"candidate": ref(target)}
    data["feature_identities"] = {"candidate": {"recipe": recipe, "source": ref(feature)}}
    datafile = root / "data.json"
    save(datafile, data)
    jobs = []
    for head in ["mlp", "conv3x3"]:
        sp = root / f"{head}.json"
        job = {
            "protocol": full.PROTOCOL,
            "data": ref(datafile),
            "model": "candidate",
            "head": head,
            "budget": 20,
            "seed": region["support_seed"],
            "device": "npu:0",
            "archives": {},
            "output": str(root / "readouts" / head),
        }
        save(sp, job)
        full.specification(sp)
        jobs.append(
            {
                "region": region["name"],
                "recipe": recipe["name"],
                "head": head,
                "spec": str(sp),
                "output": job["output"],
            }
        )
    return jobs


def select_shared_winner(results):
    candidates = [r for r in results if r["recipe"].get("role", "candidate") == "candidate"]
    if not candidates:
        raise ValueError("No high-resolution candidate was evaluated")
    eligible = [r for r in candidates if r["validation_eligible_for_query"]]
    return max(eligible or candidates, key=lambda r: (r["minimum_gain"], r["mean_gain"]))


def summarize(plan, regions, anchor):
    results = []
    for recipe in plan["recipes"]:
        scores, records = {}, []
        for region in regions:
            scores[region["name"]] = {}
            for head in ["mlp", "conv3x3"]:
                root = Path(plan["output"]) / region["name"] / recipe["name"]
                path = root / "readouts" / head / "calibration"
                identity = json.loads((path / "identity.json").read_text())
                if (
                    identity["state"] != "complete"
                    or identity["implementation"] != full.code()
                    or identity["spec_sha256"] != sha(root / f"{head}.json")
                    or identity["results_sha256"] != sha(path / "results.json")
                ):
                    raise ValueError("Incomplete or changed shared validation run")
                rows = json.loads((path / "results.json").read_text())
                scores[region["name"]][head] = float(np.mean([r["metrics"]["f1"] for r in rows]))
                records.extend({"region": region["name"], "head": head, **r} for r in rows)
        rank = minimax_rank(scores, anchor["values"])
        eligible = (
            recipe.get("role", "candidate") == "candidate"
            and rank[0] > 0
            and all(
                np.mean([scores[r][h] - anchor["values"][r][h]["aef"] for h in ["mlp", "conv3x3"]])
                >= 0.005
                for r in scores
            )
        )
        results.append(
            {
                "recipe": recipe,
                "scores": scores,
                "minimum_gain": rank[0],
                "mean_gain": rank[1],
                "validation_eligible_for_query": bool(eligible),
                "all_task_records": records,
            }
        )
    best = select_shared_winner(results)
    selected = best["recipe"]["name"] if best["validation_eligible_for_query"] else None
    save(
        Path(plan["output"]) / "result.json",
        {
            "stage": "validation_complete",
            "candidates": results,
            "best_validation_recipe": best["recipe"]["name"],
            "selected_for_query": selected,
            "goal_complete": False,
            "selection": "worst shared region-reader gain, then mean; validation only",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment shared-fusion")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "summarize"], required=True)
    parser.add_argument("--region")
    args = parser.parse_args(argv)
    plan, regions, anchor = read_plan(args.plan)
    if args.phase == "summarize":
        summarize(plan, regions, anchor)
        return 0
    region = next((r for r in regions if r["name"] == args.region), None)
    if region is None:
        raise ValueError("Choose an explicitly registered region")
    jobs = []
    for recipe in plan["recipes"]:
        feature = export_candidate(plan, region, recipe)
        jobs.extend(reader_jobs(plan, region, recipe, feature))
        save(
            Path(plan["output"]) / region["name"] / "preparation.json",
            {"state": "preparing", "completed_recipe": recipe["name"]},
        )
    save(Path(plan["output"]) / region["name"] / "jobs.json", jobs)
    save(
        Path(plan["output"]) / region["name"] / "preparation.json",
        {"state": "complete", "jobs": len(jobs)},
    )
    return 0
