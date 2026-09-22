"""Locked, paired five-head classification calibration and held-out scoring."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from xuannv_embedding.downstream import neural_readouts, strong_classifiers
from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.export.context import dump, sha

PROTOCOL = "multitask-final-strong-v1"
HEADS = ("rf", "svm", "knn", "mlp", "conv3x3")


def contract_sha256(spec):
    return primary.contract_sha256(spec)


def _code():
    return {
        "workflow": sha(Path(__file__)),
        "primary": primary._code(),
        "classical": strong_classifiers._implementation(),
        "neural": neural_readouts._implementation(),
    }


def _spec(path):
    spec = primary._load(path)
    if (
        set(spec) != {"protocol", "primary_spec", "heads", "neural_device", "output", "lock"}
        or spec["protocol"] != PROTOCOL
        or spec["heads"] != list(HEADS)
        or not isinstance(spec["neural_device"], str)
        or not re.fullmatch(r"cpu|npu:[0-9]+", spec["neural_device"])
    ):
        raise ValueError("invalid five-head classification specification")
    lock = primary._registered(spec["lock"])
    if lock.get("state") != "locked" or lock.get("contract_sha256") != contract_sha256(spec):
        raise ValueError("strong evaluation contract is not locked or has changed")
    primary._registered(spec["primary_spec"])
    cohort, cache = primary._spec(spec["primary_spec"]["path"])
    if Path(spec["output"]).resolve() == Path(cohort["output"]).resolve():
        raise ValueError("strong and primary outputs must be separate")
    return spec, cohort, cache


def _conditions(cohort):
    return [
        {**c, "head": head, "key": c["key"] + "_" + head}
        for c in primary._conditions(cohort)
        if c["family"] == "C"
        for head in HEADS
    ]


def _maps(values, indices):
    # Contiguous validation/test partitions remain views of the on-disk feature array.
    if indices and indices == list(range(indices[0], indices[0] + len(indices))):
        selected = values[indices[0] : indices[-1] + 1]
    else:
        selected = values[indices]
    return selected.transpose(0, 3, 1, 2)


def _support(y, ids, train, indices, condition):
    tiles = primary.nested_support(y, ids, train, condition["budget"], condition["seed"])
    labels = y[tiles].ravel()
    if condition["head"] in neural_readouts.KINDS:
        positions = np.flatnonzero(labels >= 0)
        incoming = positions
    else:
        incoming = primary.balanced_positions(labels, 4096, condition["seed"])
        positions = incoming
        if condition["head"] == "knn":
            selected = np.concatenate(
                [np.flatnonzero(labels[incoming] == c)[:1024] for c in (0, 1)]
            )
            positions = incoming[selected]
    descriptor = {
        "support_tiles": [indices[i] for i in tiles],
        "fitted_pixels": len(positions),
        "incoming_pixels": len(incoming),
        "positions_sha256": primary.frozen_readouts._digest(positions),
        "targets_sha256": primary.frozen_readouts._digest(labels[positions]),
        "position_unit": "flattened pixels within ordered full support tiles",
    }
    return tiles, incoming, positions, descriptor


def _fit(spec, batch, y, valid, val, tiles, incoming, positions, condition):
    x, head = batch.values, condition["head"]
    if head in neural_readouts.KINDS:
        model = neural_readouts.fit_neural(
            head,
            _maps(x, tiles),
            y[tiles],
            valid[tiles],
            _maps(x, val),
            y[val],
            valid[val],
            device=spec["neural_device"],
        )
    else:
        query, truth, _, _ = primary._query(x, y, val, "C")
        model = strong_classifiers.fit_classifier(
            head,
            x[tiles].reshape(-1, x.shape[-1])[incoming],
            y[tiles].ravel()[incoming],
            query,
            truth,
            seed=condition["seed"],
        )
        if not np.array_equal(incoming[model.arrays["selected_input_positions"]], positions):
            raise ValueError("readout retained different registered support positions")
    if model.metadata["fitted_pixels"] != len(positions):
        raise ValueError("readout fitted pixel count differs from shared support")
    return model


def _save(model, directory):
    if model.kind in neural_readouts.KINDS:
        neural_readouts.save_neural(model, directory)
    else:
        strong_classifiers.save_classifier(model, directory)


def _load(directory, digest, head, device):
    if head in neural_readouts.KINDS:
        model = neural_readouts.load_neural(directory, digest, device=device)
    else:
        model = strong_classifiers.load_classifier(directory, digest)
    if model.kind != head:
        raise ValueError("saved readout kind differs from registered condition")
    return model


def _replay(model, batch, y, valid, val):
    if model.kind in neural_readouts.KINDS:
        return neural_readouts.verify_validation(
            model, _maps(batch.values, val), y[val], valid[val]
        )
    query, truth, _, _ = primary._query(batch.values, y, val, "C")
    return strong_classifiers.verify_validation(model, query, truth)


def _predict(stage, name, condition, batch, y, valid, query_indices, indices, model, support):
    if model.kind in neural_readouts.KINDS:
        maps = y[query_indices]
        positions = np.flatnonzero(maps.ravel() >= 0)
        truth = maps.ravel()[positions]
        tiles = np.repeat(query_indices, y.shape[1] * y.shape[2])[positions]
        values = model.predict(_maps(batch.values, query_indices), valid[query_indices]).ravel()[
            positions
        ]
    else:
        query, truth, tiles, positions = primary._query(batch.values, y, query_indices, "C")
        values = model.predict(query)
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite strong-head scores on the common labeled domain")
    path = stage / "predictions" / name / (condition["key"] + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, scores=values, truth=truth, tiles=np.asarray(indices)[tiles], valid_indices=positions
    )
    metrics = primary._metrics("C", truth, values, model, y[query_indices], positions, support)
    return {**condition, "metrics": metrics, "prediction_sha256": sha(path)}


def _identity(spec, cohort, spec_path, batches, stage, *, test, elapsed):
    return {
        "state": "complete",
        "protocol": PROTOCOL,
        "spec_sha256": sha(Path(spec_path)),
        "contract_sha256": contract_sha256(spec),
        "primary_contract_sha256": primary.contract_sha256(cohort),
        "implementation": _code(),
        "heads": list(HEADS),
        "conditions": _conditions(cohort),
        "models": {name: batch.identity for name, batch in batches.items()},
        "method_groups": cohort["method_groups"],
        "common_valid_sha256": sha(stage / "common_valid.npy"),
        "common_labels_sha256": sha(stage / "common_labels.npz"),
        "results_sha256": sha(stage / "results.json"),
        "elapsed_seconds": elapsed,
        "test_scored": test,
        "parameters_refitted": False if test else None,
        "context_domain": "all-model intersection, including neural convolution context",
    }


def calibrate(spec_path):
    spec, cohort, cache = _spec(spec_path)
    stage = Path(spec["output"]) / "calibration"
    stage.mkdir(parents=True, exist_ok=False)
    tic = time.monotonic()
    dump(stage / "status.json", {"state": "running", "phase": "prepare", "test_scored": False})
    try:
        batches, labels, indices = primary._common(cohort, cache, ("train", "validation"), stage)
        valid = np.load(stage / "common_valid.npy", allow_pickle=False)
        train = list(range(len(cache["split"]["train"])))
        val = list(range(len(train), len(indices)))
        ids = [cache["records"][i]["patch_id"] for i in indices]
        for y in labels.values():
            if np.unique(y[val][y[val] >= 0]).size != 2:
                raise ValueError("common validation domain lacks both classes")
            for seed in cohort["support_seeds"]:
                primary.nested_support(y, ids, train, max(cohort["budgets"]), seed)
        rows, records = {name: [] for name in batches}, {}
        for condition in _conditions(cohort):
            key, y = condition["key"], labels[condition["task"]]
            tiles, incoming, positions, support = _support(y, ids, train, indices, condition)
            directory = stage / "readouts" / key
            directory.mkdir(parents=True)
            dump(directory / "support.json", support)
            np.save(directory / "positions.npy", positions)
            record = {
                "support_sha256": sha(directory / "support.json"),
                "positions_sha256": sha(directory / "positions.npy"),
                "models": {},
            }
            for name, batch in batches.items():
                model = _fit(spec, batch, y, valid, val, tiles, incoming, positions, condition)
                _save(model, directory / name)
                del model
                digest = sha(directory / name / "identity.json")
                loaded = _load(directory / name, digest, condition["head"], spec["neural_device"])
                replay = _replay(loaded, batch, y, valid, val)
                row = _predict(
                    stage, name, condition, batch, y, valid, val, indices, loaded, support
                )
                rows[name].append(row)
                record["models"][name] = {
                    "readout_identity_sha256": digest,
                    "validation_prediction_sha256": row["prediction_sha256"],
                    "validation_replay": replay,
                }
                del loaded
            records[key] = record
            dump(stage / "status.json", {"state": "running", "conditions_complete": len(records)})
        dump(stage / "results.json", rows)
        identity = _identity(
            spec, cohort, spec_path, batches, stage, test=False, elapsed=time.monotonic() - tic
        )
        identity["readouts"] = records
        dump(stage / "identity.json", identity)
        dump(stage / "status.json", {"state": "complete", "test_scored": False})
        return identity
    except BaseException as exc:
        dump(stage / "status.json", {"state": "failed", "error": repr(exc), "test_scored": False})
        raise


def _calibration(spec, cohort, expected):
    stage = Path(spec["output"]) / "calibration"
    if sha(stage / "identity.json") != expected:
        raise ValueError("strong calibration identity changed")
    identity = primary._load(stage / "identity.json")
    if (
        identity.get("state") != "complete"
        or identity.get("protocol") != PROTOCOL
        or identity.get("test_scored") is not False
        or identity.get("contract_sha256") != contract_sha256(spec)
        or identity.get("primary_contract_sha256") != primary.contract_sha256(cohort)
        or identity.get("implementation") != _code()
        or identity.get("conditions") != _conditions(cohort)
        or identity.get("heads") != list(HEADS)
        or identity.get("method_groups") != cohort["method_groups"]
        or primary._load(stage / "status.json").get("state") != "complete"
    ):
        raise ValueError("strong calibration contract, implementation or completion differs")
    for filename, key in (
        ("results.json", "results_sha256"),
        ("common_valid.npy", "common_valid_sha256"),
        ("common_labels.npz", "common_labels_sha256"),
    ):
        if sha(stage / filename) != identity[key]:
            raise ValueError("strong calibration data or report changed")
    for condition in _conditions(cohort):
        key = condition["key"]
        record = identity["readouts"][key]
        directory = stage / "readouts" / key
        if (
            sha(directory / "support.json") != record["support_sha256"]
            or sha(directory / "positions.npy") != record["positions_sha256"]
            or set(record["models"]) != set(cohort["models"])
        ):
            raise ValueError("strong calibration support changed")
        for name, registered in record["models"].items():
            model = _load(
                directory / name,
                registered["readout_identity_sha256"],
                condition["head"],
                spec["neural_device"],
            )
            del model  # Validate every payload before test access without retaining all forests.
            if (
                sha(stage / "predictions" / name / (key + ".npz"))
                != registered["validation_prediction_sha256"]
            ):
                raise ValueError("strong calibration validation prediction changed")
    return identity


def score(spec_path, calibration_identity_sha256):
    spec, cohort, cache = _spec(spec_path)
    identity = _calibration(spec, cohort, calibration_identity_sha256)
    calibration = Path(spec["output"]) / "calibration"
    stage = Path(spec["output"]) / "test"
    stage.mkdir(parents=True, exist_ok=False)
    dump(stage / "status.json", {"state": "running", "phase": "prepare", "test_scored": False})
    tic, scored = time.monotonic(), False
    try:
        batches, labels, indices = primary._common(cohort, cache, ("test",), stage)
        valid = np.load(stage / "common_valid.npy", allow_pickle=False)
        query = list(range(len(indices)))
        rows = {name: [] for name in batches}
        for number, condition in enumerate(_conditions(cohort)):
            key, y = condition["key"], labels[condition["task"]]
            directory = calibration / "readouts" / key
            support = primary._load(directory / "support.json")
            for name, batch in batches.items():
                model = _load(
                    directory / name,
                    identity["readouts"][key]["models"][name]["readout_identity_sha256"],
                    condition["head"],
                    spec["neural_device"],
                )
                rows[name].append(
                    _predict(
                        stage, name, condition, batch, y, valid, query, indices, model, support
                    )
                )
                del model
                scored = True
            dump(
                stage / "status.json",
                {"state": "running", "conditions_complete": number + 1, "test_scored": True},
            )
        dump(stage / "results.json", rows)
        result = _identity(
            spec, cohort, spec_path, batches, stage, test=True, elapsed=time.monotonic() - tic
        )
        result["calibration_identity_sha256"] = calibration_identity_sha256
        dump(stage / "identity.json", result)
        dump(stage / "status.json", {"state": "complete", "test_scored": True})
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
