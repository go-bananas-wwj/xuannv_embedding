"""Locked, paired primary C/R/Q calibration and held-out prediction workflows."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score

from xuannv_embedding.downstream import fixed_audit, frozen_readouts, multitask, multitask_features
from xuannv_embedding.downstream.fixed_audit import (
    balanced_positions,
    bf1,
    boundary_counts,
    nested_support,
)
from xuannv_embedding.downstream.frozen_readouts import (
    fit_classification,
    fit_regression,
    freeze_retrieval,
    load_readout,
    save_readout,
    verify_validation,
)
from xuannv_embedding.downstream.multitask import (
    block_regression_data,
    regression_metrics,
    retrieval_prototypes,
)
from xuannv_embedding.downstream.multitask_features import FeatureSelection, read_features
from xuannv_embedding.export.context import dump, sha

PROTOCOL = "multitask-final-primary-v1"
OSM_TASKS = ("osm_building", "osm_road", "osm_water", "osm_green")
ESRI_TASKS = tuple("esri_" + name for name in ("water", "trees", "range", "crops", "built", "bare"))


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs)


def _registered(record):
    if set(record) != {"path", "sha256"} or sha(Path(record["path"])) != record["sha256"]:
        raise ValueError("registered input digest changed")
    return _load(record["path"])


def contract_sha256(spec):
    contract = {k: v for k, v in spec.items() if k not in ("output", "lock")}
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _code():
    return {
        name: sha(Path(path))
        for name, path in {
            "workflow": __file__,
            "features": multitask_features.__file__,
            "readouts": frozen_readouts.__file__,
            "sampling_and_boundaries": fixed_audit.__file__,
            "task_metrics": multitask.__file__,
        }.items()
    }


def _spec(path):
    spec = _load(path)
    allowed = {
        "protocol",
        "output",
        "reference_cache",
        "labels",
        "models",
        "month",
        "support_seeds",
        "budgets",
        "retrieval_budgets",
        "method_groups",
        "geographic_audit",
        "lock",
    }
    if set(spec) != allowed or spec["protocol"] != PROTOCOL:
        raise ValueError("invalid paired primary specification")
    lock = _registered(spec["lock"])
    if lock.get("state") != "locked" or lock.get("contract_sha256") != contract_sha256(spec):
        raise ValueError("evaluation contract is not locked or has changed")
    cache = _registered(spec["reference_cache"])
    multitask_features._grid(cache, {"records": cache["records"], "split": cache["split"]})
    if spec["month"] not in cache["data"]["months"] or cache["data"]["patch_size"] % 16:
        raise ValueError("reference period or 160-m proxy block geometry differs")
    if set(spec["labels"]) != {"train", "validation", "test"} or len(spec["models"]) < 2:
        raise ValueError("three label splits and at least two models are required")
    for key in ("support_seeds", "budgets", "retrieval_budgets"):
        values = spec[key]
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not int or v < 1 for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("support seeds and budgets must be distinct positive integers")
    if max(spec["budgets"]) > len(cache["split"]["train"]):
        raise ValueError("support budget exceeds available training tiles")
    for name, model in spec["models"].items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("invalid model name")
        if set(model) != {
            "manifest_path",
            "manifest_sha256",
            "cache_path",
            "cache_sha256",
            "tile_sha256",
            "selection",
        }:
            raise ValueError("invalid model feature contract")
        selection = FeatureSelection(**model["selection"])
        if selection.evaluation_month != spec["month"]:
            raise ValueError("model evaluation month differs")
    members = [m for names in spec["method_groups"].values() for m in names]
    if (
        not spec["method_groups"]
        or any(not isinstance(v, list) or not v for v in spec["method_groups"].values())
        or sorted(members) != sorted(spec["models"])
    ):
        raise ValueError("method groups must partition the registered model realizations")
    geographic = _registered(spec["geographic_audit"])
    if (
        geographic.get("state") != "verified"
        or geographic.get("reference_cache_sha256") != spec["reference_cache"]["sha256"]
        or geographic.get("model_manifest_sha256")
        != {n: m["manifest_sha256"] for n, m in spec["models"].items()}
        or not geographic.get("evidence")
    ):
        raise ValueError("upstream geographic audit does not bind this comparison")
    for evidence in geographic["evidence"]:
        if sha(Path(evidence["path"])) != evidence["sha256"]:
            raise ValueError("upstream geographic evidence changed")
    return spec, cache


def _conditions(spec):
    result = []
    for family, tasks, budgets in (
        ("C", OSM_TASKS + ESRI_TASKS, spec["budgets"]),
        ("R", ESRI_TASKS, spec["budgets"]),
        ("Q", OSM_TASKS, spec["retrieval_budgets"]),
    ):
        for task in tasks:
            for seed in spec["support_seeds"]:
                for budget in budgets:
                    result.append(
                        dict(
                            key=f"{family}_{task}_{seed}_{budget}",
                            family=family,
                            task=task,
                            source=task.split("_")[0],
                            seed=seed,
                            budget=budget,
                        )
                    )
    return result


def _labels(spec, cache, splits):
    collected = {task: [] for task in OSM_TASKS + ESRI_TASKS}
    size = cache["data"]["patch_size"]
    for split in splits:
        record = spec["labels"][split]
        path = Path(record["path"])
        if sha(path) != record["sha256"]:
            raise ValueError("label bundle digest changed")
        with np.load(path, allow_pickle=False) as data:
            if (
                set(data.files) != {"indices", "cache_sha256", "esri", *OSM_TASKS}
                or str(data["cache_sha256"].item()) != spec["reference_cache"]["sha256"]
                or not np.issubdtype(data["indices"].dtype, np.integer)
                or not np.array_equal(data["indices"], cache["split"][split])
            ):
                raise ValueError("label bundle partition or cache binding differs")
            shape = (len(cache["split"][split]), size, size)
            for task in (*OSM_TASKS, "esri"):
                y = data[task]
                choices = (-1, 0, 1, 2, 3, 4, 5) if task == "esri" else (-1, 0, 1)
                if y.shape != shape or not np.isin(y, choices).all():
                    raise ValueError("label geometry or class encoding differs")
            for task in OSM_TASKS:
                collected[task].append(data[task].astype(np.int8))
            for i, task in enumerate(ESRI_TASKS):
                collected[task].append(
                    np.where(data["esri"] < 0, -1, data["esri"] == i).astype(np.int8)
                )
    return {task: np.concatenate(values) for task, values in collected.items()}


def _common(spec, cache, splits, stage):
    _, layout = multitask_features._grid(
        cache, {"records": cache["records"], "split": cache["split"]}
    )
    layout_sha = hashlib.sha256(
        json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    indices = tuple(i for split in splits for i in cache["split"][split])
    batches, valid = {}, None
    labels = _labels(spec, cache, splits)
    for name, model in spec["models"].items():
        arguments = dict(model, selection=FeatureSelection(**model["selection"]))
        batch = read_features(**arguments, splits=splits, output=stage / "features" / name)
        if batch.indices != indices or batch.identity["spatial_layout_sha256"] != layout_sha:
            raise ValueError("models do not share the reference layout and partition")
        batches[name] = batch
        valid = np.array(batch.valid, copy=True) if valid is None else valid & batch.valid
    labels = {task: np.where(valid, y, -1).astype(np.int8) for task, y in labels.items()}
    np.save(stage / "common_valid.npy", valid)
    np.savez_compressed(stage / "common_labels.npz", **labels)
    return batches, labels, indices


def _query(x, y, indices, family):
    if family == "R":
        size = y.shape[1]
        eligible = (y[indices] >= 0).reshape(len(indices), size // 16, 16, size // 16, 16).mean(
            (2, 4)
        ) >= 0.8
        if not eligible.any():
            return (
                np.empty((0, x.shape[-1]), x.dtype),
                np.empty(0),
                np.empty(0, np.int64),
                np.empty(0, np.int64),
            )
        features, truth, tiles = block_regression_data(x, y, indices)
        return features, truth, tiles, np.empty(0, np.int64)
    target = y[indices].ravel()
    keep = target >= 0
    features = x[indices].reshape(-1, x.shape[-1])[keep]
    tiles = np.repeat(indices, y.shape[1] * y.shape[2])[keep]
    return features, target[keep], tiles, np.flatnonzero(keep)


def _fit(x, y, train, val, ids, global_indices, condition):
    family, seed, budget = (condition[k] for k in ("family", "seed", "budget"))
    query, truth, _, _ = _query(x, y, val, family)
    if family == "C":
        support = nested_support(y, ids, train, budget, seed)
        labels = y[support].ravel()
        positions = balanced_positions(labels, 4096, seed)
        model = fit_classification(
            x[support].reshape(-1, x.shape[-1])[positions], labels[positions], query, truth
        )
        descriptor = {
            "support_tiles": [global_indices[i] for i in support],
            "support_positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
            "support_pixels": len(positions),
        }
    elif family == "R":
        support = sorted(train, key=lambda i: hashlib.sha256(f"{seed}:{ids[i]}".encode()).digest())[
            :budget
        ]
        features, target, tiles = block_regression_data(x, y, support)
        model = fit_regression(features, target, query, truth)
        descriptor = {
            "support_tiles": [global_indices[i] for i in support],
            "support_blocks": len(target),
            "training_mean": float(target.mean()),
            "block_pixels": 16,
            "minimum_valid_fraction": 0.8,
            "support_targets_sha256": frozen_readouts._digest(target, tiles),
        }
    else:
        prototypes, samples = retrieval_prototypes(x, y, train, ids, budget, seed)
        model = freeze_retrieval(prototypes, normalized=True)
        descriptor = {"queries": [{**s, "tile": global_indices[s["tile"]]} for s in samples]}
    return model, descriptor


def _metrics(kind, truth, values, readout, maps, positions, support):
    if kind == "R":
        if not len(truth):
            return {"rmse": None, "mae": None, "r2": None, "bias": None, "observations": 0}
        return {
            **regression_metrics(truth, values),
            "observations": len(truth),
            "training_mean_baseline": regression_metrics(
                truth, np.full_like(truth, support["training_mean"])
            ),
        }
    positives = int((truth == 1).sum())
    metrics = {
        "ap": float(average_precision_score(truth, values)) if positives else None,
        "observations": len(truth),
    }
    if kind == "Q":
        order = np.argsort(-values, kind="stable")
        for fraction in (0.01, 0.05):
            k = max(1, int(np.ceil(len(values) * fraction)))
            hits = int(truth[order[:k]].sum())
            metrics[f"precision_top{fraction}"] = hits / k if len(truth) else None
            metrics[f"recall_top{fraction}"] = hits / positives if positives else None
        return metrics
    predictions = values >= readout.metadata["threshold"]
    tp, fp, fn = (
        int(v.sum())
        for v in (
            predictions & (truth == 1),
            predictions & (truth == 0),
            ~predictions & (truth == 1),
        )
    )
    metrics.update(
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        iou=tp / (tp + fp + fn) if tp + fp + fn else None,
        ba=(
            float(balanced_accuracy_score(truth, predictions))
            if np.unique(truth).size == 2
            else None
        ),
        threshold=readout.metadata["threshold"],
    )
    prediction_maps = np.zeros(maps.shape, bool)
    prediction_maps.ravel()[positions] = predictions
    for radius in (1, 2):
        counts = np.sum(
            [boundary_counts(y, pred, radius) for y, pred in zip(maps, prediction_maps)], axis=0
        )
        metrics[f"boundary_f1_{radius}px"] = bf1(counts) if counts[1] + counts[3] else None
        metrics[f"boundary_counts_{radius}px"] = counts.tolist()
    return metrics


def _predict(stage, name, condition, batch, y, query_indices, indices, readout, support):
    query, truth, tiles, positions = _query(batch.values, y, query_indices, condition["family"])
    values = readout.predict(query)
    path = stage / "predictions" / name / (condition["key"] + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, scores=values, truth=truth, tiles=np.asarray(indices)[tiles], valid_indices=positions
    )
    metrics = _metrics(
        condition["family"], truth, values, readout, y[query_indices], positions, support
    )
    return {**condition, "metrics": metrics, "prediction_sha256": sha(path)}


def _identity(spec, spec_path, batches, stage, *, test):
    return {
        "state": "complete",
        "protocol": PROTOCOL,
        "spec_sha256": sha(Path(spec_path)),
        "contract_sha256": contract_sha256(spec),
        "implementation": _code(),
        "models": {name: batch.identity for name, batch in batches.items()},
        "method_groups": spec["method_groups"],
        "conditions": _conditions(spec),
        "common_valid_sha256": sha(stage / "common_valid.npy"),
        "common_labels_sha256": sha(stage / "common_labels.npz"),
        "results_sha256": sha(stage / "results.json"),
        "test_scored": test,
        "parameters_refitted": False if test else None,
    }


def calibrate(spec_path: Path):
    spec, cache = _spec(spec_path)
    stage = Path(spec["output"]) / "calibration"
    stage.mkdir(parents=True, exist_ok=False)
    dump(stage / "status.json", {"state": "running", "phase": "prepare", "test_scored": False})
    try:
        batches, labels, indices = _common(spec, cache, ("train", "validation"), stage)
        train = list(range(len(cache["split"]["train"])))
        val = list(range(len(train), len(indices)))
        ids = [cache["records"][i]["patch_id"] for i in indices]
        for y in labels.values():
            if np.unique(y[val][y[val] >= 0]).size != 2:
                raise ValueError("common validation domain lacks both classes")
            for seed in spec["support_seeds"]:
                nested_support(y, ids, train, max(spec["budgets"]), seed)
        rows, records = {name: [] for name in batches}, {}
        for condition in _conditions(spec):
            key, y = condition["key"], labels[condition["task"]]
            record, shared_support = {"models": {}}, None
            for name, batch in batches.items():
                model, support = _fit(batch.values, y, train, val, ids, indices, condition)
                if shared_support is not None and support != shared_support:
                    raise ValueError("models used different training support")
                shared_support = support
                directory = stage / "readouts" / key / name
                save_readout(model, directory)
                digest = sha(directory / "identity.json")
                loaded = load_readout(directory, digest)
                replay = None
                if condition["family"] != "Q":
                    query, truth, _, _ = _query(batch.values, y, val, condition["family"])
                    replay = verify_validation(loaded, query, truth)
                row = _predict(stage, name, condition, batch, y, val, indices, loaded, support)
                rows[name].append(row)
                record["models"][name] = {
                    "readout_identity_sha256": digest,
                    "validation_prediction_sha256": row["prediction_sha256"],
                    "validation_replay": replay,
                }
            support_path = stage / "readouts" / key / "support.json"
            dump(support_path, shared_support)
            record["support_sha256"] = sha(support_path)
            records[key] = record
            dump(
                stage / "status.json",
                {"state": "running", "conditions_complete": len(records), "test_scored": False},
            )
        dump(stage / "results.json", rows)
        identity = _identity(spec, spec_path, batches, stage, test=False)
        identity["readouts"] = records
        dump(stage / "identity.json", identity)
        dump(
            stage / "status.json",
            {"state": "complete", "conditions_complete": len(records), "test_scored": False},
        )
        return identity
    except BaseException as exc:
        dump(stage / "status.json", {"state": "failed", "error": repr(exc), "test_scored": False})
        raise


def score(spec_path: Path, calibration_identity_sha256: str):
    spec, cache = _spec(spec_path)
    calibration = Path(spec["output"]) / "calibration"
    identity_path = calibration / "identity.json"
    if sha(identity_path) != calibration_identity_sha256:
        raise ValueError("calibration identity changed")
    identity = _load(identity_path)
    if (
        identity.get("state") != "complete"
        or identity.get("test_scored") is not False
        or identity.get("contract_sha256") != contract_sha256(spec)
        or identity.get("implementation") != _code()
        or identity.get("conditions") != _conditions(spec)
        or _load(calibration / "status.json").get("state") != "complete"
    ):
        raise ValueError("calibration contract, implementation or completion differs")
    for filename, key in (
        ("results.json", "results_sha256"),
        ("common_valid.npy", "common_valid_sha256"),
        ("common_labels.npz", "common_labels_sha256"),
    ):
        if sha(calibration / filename) != identity[key]:
            raise ValueError("calibration data or report changed")
    frozen, supports = {}, {}
    for condition in _conditions(spec):
        key = condition["key"]
        record = identity["readouts"][key]
        support_path = calibration / "readouts" / key / "support.json"
        if sha(support_path) != record["support_sha256"]:
            raise ValueError("calibration support changed")
        supports[key], frozen[key] = _load(support_path), {}
        for name in spec["models"]:
            registered = record["models"][name]
            frozen[key][name] = load_readout(
                calibration / "readouts" / key / name, registered["readout_identity_sha256"]
            )
            if (
                sha(calibration / "predictions" / name / (key + ".npz"))
                != registered["validation_prediction_sha256"]
            ):
                raise ValueError("calibration validation predictions changed")
    stage = Path(spec["output"]) / "test"
    stage.mkdir(parents=True, exist_ok=False)
    dump(stage / "status.json", {"state": "running", "phase": "prepare", "test_scored": False})
    scored = False
    try:
        batches, labels, indices = _common(spec, cache, ("test",), stage)
        query_indices = list(range(len(indices)))
        rows = {name: [] for name in batches}
        for condition in _conditions(spec):
            key, y = condition["key"], labels[condition["task"]]
            for name, batch in batches.items():
                rows[name].append(
                    _predict(
                        stage,
                        name,
                        condition,
                        batch,
                        y,
                        query_indices,
                        indices,
                        frozen[key][name],
                        supports[key],
                    )
                )
                scored = True
            dump(
                stage / "status.json",
                {
                    "state": "running",
                    "conditions_complete": len(rows[next(iter(rows))]),
                    "test_scored": True,
                },
            )
        dump(stage / "results.json", rows)
        result = _identity(spec, spec_path, batches, stage, test=True)
        result["calibration_identity_sha256"] = calibration_identity_sha256
        dump(stage / "identity.json", result)
        dump(
            stage / "status.json",
            {
                "state": "complete",
                "conditions_complete": len(_conditions(spec)),
                "test_scored": True,
            },
        )
        return result
    except BaseException as exc:
        dump(
            stage / "status.json",
            {
                "state": "failed",
                "error": repr(exc),
                "test_scored": scored,
                "test_data_access_started": True,
            },
        )
        raise
