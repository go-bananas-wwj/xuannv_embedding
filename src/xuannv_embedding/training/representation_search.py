"""Validation-only search over fixed-dimension, label-free regional fusion recipes."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.regional_tuning import (
    candidate_benchmark,
    evaluate_selected,
    publish,
    validate_candidate,
)


def balanced_validation_rank(report: dict, reference: dict) -> tuple[float, float]:
    values = {h: report["heads"][h]["macro_f1"] for h in ["mlp", "conv3x3"]}
    if any(
        type(v) not in {int, float} or not math.isfinite(v) or not 0 <= v <= 1
        for v in [*values.values(), *reference.values()]
    ):
        raise ValueError("Invalid validation score")
    return min(values[h] - reference[h] for h in values), float(np.mean(list(values.values())))


def meets_acceptance(comparisons: dict, baseline: dict) -> bool:
    for head in ["mlp", "conv3x3"]:
        value = comparisons.get(f"highres:{head}:200", {}).get("macro_f1", float("nan"))
        threshold = max(
            baseline[f"{feature}:{head}:200"]["macro_f1"] for feature in ["public", "aef"]
        )
        if type(value) not in {float, int} or not math.isfinite(value) or value <= threshold:
            return False
    return True


def unit(values):
    norm = np.linalg.norm(values, axis=0, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm > 1e-8)


def compose_tile(recipe: dict, arrays: list, projections: list) -> np.ndarray:
    kind = recipe["kind"]
    if kind == "blend":
        alpha = recipe["alpha"]
        if type(alpha) not in {float, int} or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Invalid fusion weight")
        values = (1 - alpha) * arrays[0] + alpha * arrays[1]
    elif kind == "blocks":
        blocks = []
        for values, projection in zip(arrays, projections):
            mask = np.linalg.norm(values, axis=0) > 1e-8
            h, w = values.shape[-2:]
            projected = projection.transform(values.reshape(len(values), -1).T).T
            projected = projected.reshape(-1, h, w).astype(np.float32)
            blocks.append(unit(projected) * mask[None])
        values = np.concatenate(blocks)
    elif kind == "joint":
        joined = np.concatenate(arrays)
        h, w = joined.shape[-2:]
        values = projections[0].transform(joined.reshape(len(joined), -1).T).T
        values = values.reshape(-1, h, w).astype(np.float32)
    elif kind == "identity":
        values = arrays[0]
    else:
        raise ValueError("Unknown representation recipe")
    if values.shape[0] != 64 or not np.isfinite(values).all():
        raise ValueError("Every candidate must be finite and 64-dimensional")
    return unit(values).astype(np.float32)


def _reference(ref):
    if set(ref) != {"path", "sha256"} or sha(ref["path"]) != ref["sha256"]:
        raise ValueError("Registered representation-search source changed")
    return Path(ref["path"])


def _same_index(a, b):
    if a["months"] != b["months"] or len(a["records"]) != len(b["records"]):
        raise ValueError("Representation geographic/time index differs")
    for left, right in zip(a["records"], b["records"]):
        if any(left.get(k) != right.get(k) for k in ["patch_id", "bounds", "crs", "region"]):
            raise ValueError("Representation geographic index differs")


def registered_parent(spec):
    if "parent" not in spec:
        return None
    if set(spec["parent"]) != {"validation", "result"}:
        raise ValueError("Unknown parent-reference fields")
    for ref in spec["parent"].values():
        _reference(ref)
    report = json.loads(Path(spec["parent"]["validation"]["path"]).read_text())
    if report.get("test_scored") is not False or report.get("fold") != 0:
        raise ValueError("Parent selection must use the registered validation split")
    return report


def read_spec(path: Path):
    spec = json.loads(path.read_text())
    expected = {
        "schema",
        "output",
        "baseline",
        "index",
        "sources",
        "recipes",
        "semantic",
        "devices",
        "github",
        "fit_samples",
        "seed",
    }
    if (
        not expected.issubset(spec)
        or set(spec) - expected - {"parent"}
        or spec["schema"] != "regional-representation-search-v1"
    ):
        raise ValueError("Unknown representation-search specification")
    registered_parent(spec)
    _reference(spec["baseline"])
    _reference(spec["semantic"])
    common = json.loads(_reference(spec["index"]).read_text())
    if (
        not spec["devices"]
        or any(type(d) is not int or d < 1 for d in spec["devices"])
        or (len(spec["devices"]) != len(set(spec["devices"])))
    ):
        raise ValueError("Search devices must be disjoint positive NPU indices")
    if type(spec["fit_samples"]) is not int or spec["fit_samples"] < 64:
        raise ValueError("Projection sample budget must be at least 64")
    for source in spec["sources"].values():
        if set(source) == {"producer"}:
            producer = json.loads(_reference(source["producer"]).read_text())
            _same_index(common, json.loads(_reference(producer["cache"]).read_text()))
        elif set(source) == {"path", "sha256", "index"}:
            _reference({k: source[k] for k in ["path", "sha256"]})
            _same_index(common, json.loads(_reference(source["index"]).read_text()))
        else:
            raise ValueError("Unknown feature source registration")
    names = []
    for recipe in spec["recipes"]:
        required = {"name", "kind", "sources", "eligible"}
        if not required.issubset(recipe) or set(recipe) - required - {"alpha", "dimensions"}:
            raise ValueError("Unknown recipe fields")
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]*", recipe["name"])
            or recipe["name"] in names
            or recipe["name"] in {"public", "parent", "semantic"}
            or type(recipe["eligible"]) is not bool
        ):
            raise ValueError("Invalid recipe name or eligibility")
        names.append(recipe["name"])
        if not recipe["sources"] or any(s not in spec["sources"] for s in recipe["sources"]):
            raise ValueError("Unregistered fusion source")
        kind = recipe["kind"]
        if kind == "blocks":
            dims = recipe.get("dimensions", [])
            if (
                len(dims) != len(recipe["sources"])
                or sum(dims) != 64
                or any(type(v) is not int or not 1 <= v <= 64 for v in dims)
            ):
                raise ValueError("Block dimensions must sum to 64")
        elif kind == "blend":
            if len(recipe["sources"]) != 2 or not 0 < recipe.get("alpha", -1) < 1:
                raise ValueError("Blends require two sources and a nontrivial weight")
        elif kind == "identity":
            if len(recipe["sources"]) != 1:
                raise ValueError("Identity requires one source")
        elif kind != "joint":
            raise ValueError("Unknown recipe kind")
        if recipe["eligible"] and "public" not in recipe["sources"]:
            raise ValueError("Goal candidates must actually fuse the public representation")
    repo = Path(spec["github"]["repository"]).resolve()
    report = (repo / spec["github"]["report"]).resolve()
    if not report.is_relative_to(repo / "docs") or report.suffix != ".md":
        raise ValueError("Public progress must be inside repository docs")
    return spec


def adapter(spec):
    return {
        "output": spec["output"],
        "baseline": spec["baseline"],
        "github": spec["github"],
        "selection": {"fold": 0, "budget": 200, "heads": ["mlp", "conv3x3"]},
        "candidates": [{"name": r["name"]} for r in spec["recipes"]],
    }


def source_path(spec, name):
    source = spec["sources"][name]
    if "producer" not in source:
        return Path(source["path"])
    producer = json.loads(Path(source["producer"]["path"]).read_text())
    directory = Path(producer["output"])
    path = directory / "status.json"
    if not path.exists() or json.loads(path.read_text()).get("stage") != "complete":
        return None
    return directory / "annual.npy"


def prepare_recipe(spec, recipe):
    root = Path(candidate_benchmark(adapter(spec), {"name": recipe["name"]})["output"])
    output = root / "representations/highres"
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "recipe": recipe,
        "seed": spec["seed"],
        "fit_samples": spec["fit_samples"],
        "sources": {name: sha(source_path(spec, name)) for name in recipe["sources"]},
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Representation recipe identity changed")
    dump(identity_path, identity)
    if (output / "annual.npy").exists() and (output / "status.json").exists():
        return
    arrays = [np.load(source_path(spec, name), mmap_mode="r") for name in recipe["sources"]]
    if any(a.shape != arrays[0].shape or a.ndim != 4 or a.shape[1] != 64 for a in arrays):
        raise ValueError("Fusion sources must share the complete 64-D grid")
    public_shape = np.load(source_path(spec, "public"), mmap_mode="r").shape
    count = len(json.loads(Path(spec["index"]["path"]).read_text())["records"])
    reference_shape = (count, 64, *public_shape[-2:])
    if arrays[0].shape != reference_shape:
        raise ValueError("Fusion source grid differs from the frozen reference")
    rng = np.random.default_rng(spec["seed"])
    samples = []
    for i in range(len(arrays[0])):
        count = arrays[0].shape[-1] * arrays[0].shape[-2]
        positions = rng.choice(count, min(256, count), replace=False)
        samples.append(
            np.concatenate(
                [np.asarray(a[i], np.float32).reshape(64, -1)[:, positions].T for a in arrays],
                axis=1,
            )
        )
    sampled = np.concatenate(samples)
    if len(sampled) > spec["fit_samples"]:
        sampled = sampled[rng.choice(len(sampled), spec["fit_samples"], replace=False)]
    projections = []
    dimensions = recipe.get("dimensions", [64])
    with threadpool_limits(limits=8):
        if recipe["kind"] in {"joint", "blocks"}:
            for index, dim in enumerate(dimensions):
                values = (
                    sampled
                    if recipe["kind"] == "joint"
                    else sampled[:, index * 64 : (index + 1) * 64]
                )
                active = np.linalg.norm(values, axis=1) > 1e-8
                pca = PCA(n_components=dim, svd_solver="full")
                pca.fit(values[active])
                projections.append(pca)
                np.savez_compressed(
                    output / f"projection{index}.npz",
                    mean=pca.mean_,
                    components=pca.components_,
                    explained_variance_ratio=pca.explained_variance_ratio_,
                )
    path = output / "annual.partial.npy"
    result = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=arrays[0].shape)
    for i in range(len(result)):
        result[i] = compose_tile(
            recipe, [np.asarray(a[i], np.float32) for a in arrays], projections
        ).astype(np.float16)
    result.flush()
    path.replace(output / "annual.npy")
    dump(
        output / "status.json",
        {
            "stage": "complete",
            "shape": list(result.shape),
            "labels_used_for_projection": False,
            "recipe": recipe,
            "fit_samples": len(sampled),
            "annual_sha256": sha(output / "annual.npy"),
        },
    )


def validation_reference(spec, feature):
    baseline = json.loads(Path(spec["baseline"]["path"]).read_text())
    result = {}
    for head in ["mlp", "conv3x3"]:
        path = Path(baseline["output"]) / "scores" / f"{feature}_{head}_fold0_n200/result.json"
        item = json.loads(path.read_text())
        result[head] = max(v["validation_macro_f1"] for v in item["curve"])
    return result


def follow(spec_path):
    spec = read_spec(spec_path)
    plan = adapter(spec)
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    reference = validation_reference(spec, "aef")
    active, complete, retries, prepared = {}, {}, {}, set()
    identity = sha(spec_path)
    with (root / "search.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration = root / "search-identity.json"
        if (
            registration.exists()
            and json.loads(registration.read_text())["spec_sha256"] != identity
        ):
            raise ValueError("Cannot resume a changed search specification")
        dump(registration, {"spec_sha256": identity})
        try:
            while len(complete) < len(spec["recipes"]):
                if sha(spec_path) != identity or (root / "STOP").exists():
                    raise RuntimeError("Search specification changed or STOP requested")
                if shutil.disk_usage(root).free < 100 * 2**30:
                    raise RuntimeError("Storage reserve reached")
                for name, (process, device, start) in list(active.items()):
                    if process.poll() is None:
                        if time.time() - start > 3600:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        else:
                            continue
                    del active[name]
                    if process.returncode:
                        retries[name] = retries.get(name, 0) + 1
                        if retries[name] > 2:
                            raise RuntimeError(f"Representation job repeatedly failed: {name}")
                        continue
                    if name != "semantic":
                        path = root / name / "benchmark/validation-selection.json"
                        report = json.loads(path.read_text())
                        rank = balanced_validation_rank(report, reference)
                        complete[name] = {
                            "validation_macro_f1": report["validation_macro_f1"],
                            "minimum_margin": rank[0],
                            "heads": {h: v["macro_f1"] for h, v in report["heads"].items()},
                        }
                        publish(plan, {"stage": "validation", "completed": complete}, name)
                available = [
                    d for d in spec["devices"] if d not in [record[1] for record in active.values()]
                ]

                def launch(name, arguments, device):
                    directory = root / name
                    directory.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable,
                        "-m",
                        "xuannv_embedding.cli",
                        "experiment",
                        *arguments,
                    ]
                    with (directory / "process.log").open("ab") as log:
                        process = subprocess.Popen(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                    active[name] = process, device, time.time()

                semantic = json.loads(Path(spec["semantic"]["path"]).read_text())
                semantic_status = Path(semantic["output"]) / "status.json"
                semantic_done = (
                    semantic_status.exists()
                    and json.loads(semantic_status.read_text()).get("stage") == "complete"
                )
                if not semantic_done and "semantic" not in active and available:
                    device = available.pop(0)
                    launch(
                        "semantic",
                        [
                            "semantic-features",
                            "--spec",
                            spec["semantic"]["path"],
                            "--device",
                            f"npu:{device}",
                        ],
                        device,
                    )
                for recipe in spec["recipes"]:
                    name = recipe["name"]
                    if name in complete or name in active:
                        continue
                    existing = root / name / "benchmark/validation-selection.json"
                    if existing.exists():
                        report = json.loads(existing.read_text())
                        complete[name] = {
                            "validation_macro_f1": report["validation_macro_f1"],
                            "minimum_margin": balanced_validation_rank(report, reference)[0],
                            "heads": {h: v["macro_f1"] for h, v in report["heads"].items()},
                        }
                        continue
                    if not available or any(
                        source_path(spec, s) is None for s in recipe["sources"]
                    ):
                        continue
                    if name not in prepared:
                        dump(
                            root / "status.json",
                            {
                                "stage": "preparing",
                                "recipe": name,
                                "completed": complete,
                                "total": len(spec["recipes"]),
                            },
                        )
                        prepare_recipe(spec, recipe)
                        prepared.add(name)
                    device = available.pop(0)
                    launch(
                        name,
                        [
                            "representation-search",
                            "--spec",
                            str(spec_path),
                            "--phase",
                            "validate",
                            "--recipe",
                            name,
                            "--device",
                            f"npu:{device}",
                        ],
                        device,
                    )
                state = {
                    "stage": "validation_search",
                    "completed": complete,
                    "total": len(spec["recipes"]),
                    "active": {k: v[1] for k, v in active.items()},
                    "retries": retries,
                    "supervisor_pid": os.getpid(),
                    "utc": time.time(),
                }
                dump(root / "status.json", state)
                if not (root / "started.json").exists():
                    publish(plan, state, "representation_search_started")
                    dump(root / "started.json", {"utc": time.time()})
                if len(complete) < len(spec["recipes"]):
                    time.sleep(30)
            public = validation_reference(spec, "public")
            scores = {"public": {"heads": {h: {"macro_f1": v} for h, v in public.items()}}}
            if "parent" in spec:
                scores["parent"] = registered_parent(spec)
            for recipe in spec["recipes"]:
                if recipe["eligible"]:
                    scores[recipe["name"]] = json.loads(
                        (root / recipe["name"] / "benchmark/validation-selection.json").read_text()
                    )
            winner = max(scores, key=lambda name: balanced_validation_rank(scores[name], reference))
            selection = {
                "winner": winner,
                "criterion": "minimum_of_MLP_and_CNN_validation_margins",
                "validation_reference": reference,
                "scores": complete,
                "test_used_for_selection": False,
            }
            dump(root / "selection.json", selection)
            dump(root / "status.json", {"stage": "final_evaluation", "winner": winner})
            result = evaluate_selected(plan, winner)
            baseline = json.loads(Path(spec["baseline"]["path"]).read_text())
            comparisons = json.loads((Path(baseline["output"]) / "comparison.json").read_text())[
                "comparisons"
            ]
            accepted = winner not in {"public", "parent"} and meets_acceptance(
                result["comparisons"], comparisons
            )
            result["acceptance_met"] = accepted
            result["representation_kind"] = (
                "unchanged public baseline"
                if winner == "public"
                else (
                    "retained regional fusion reference"
                    if winner == "parent"
                    else "label-free regional 64-D feature fusion"
                )
            )
            dump(root / "result.json", result)
            final = {
                "stage": "complete",
                "winner": winner,
                "acceptance_met": accepted,
                "validation": complete,
                "overleaf_modified": False,
            }
            dump(root / "status.json", final)
            publish(plan, final, "representation_search_complete")
        except BaseException as error:
            for process, _, _ in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            dump(root / "status.json", {"stage": "failed", "reason": str(error)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment representation-search")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["follow", "validate"], required=True)
    parser.add_argument("--recipe")
    parser.add_argument("--device", default="npu:2")
    args = parser.parse_args(argv)
    spec = read_spec(args.spec)
    if args.phase == "follow":
        follow(args.spec)
    else:
        recipe = next(r for r in spec["recipes"] if r["name"] == args.recipe)
        result = validate_candidate(
            adapter(spec), {"name": recipe["name"]}, device_name=args.device
        )
        dump(
            Path(spec["output"]) / recipe["name"] / "status.json",
            {
                "stage": "complete",
                "validation_macro_f1": result["validation_macro_f1"],
                "selected_step": None,
                "kind": "label-free 64-D feature fusion",
            },
        )
    return 0
