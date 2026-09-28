"""Explicit overlapping full-region map evaluation of frozen regional representations.

The full-region score intentionally includes downstream support and calibration
locations. It is never exported as an independent held-out test score.
"""

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np

from xuannv_embedding.downstream import (
    baseline_review,
    frozen_readouts,
    unet_readout,
)
from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.downstream import strong_multitask as strong
from xuannv_embedding.downstream.multitask_features import FeatureBatch
from xuannv_embedding.export.context import dump, sha

PROTOCOL = "full-region-readout-v1"
HEADS = ("linear", "regression", "retrieval", "mlp", "conv3x3", "rf", "unet")


def code():
    return {
        "full_region": sha(Path(__file__)),
        "strong": strong._code(),
        "primary": primary._code(),
        "unet": unet_readout.code(),
    }


def specification(path):
    spec = primary._load(path)
    if (
        set(spec)
        != {"protocol", "data", "model", "head", "budget", "seed", "device", "archives", "output"}
        or spec["protocol"] != PROTOCOL
        or spec["head"] not in HEADS
        or type(spec["budget"]) is not int
        or spec["budget"] < 1
        or type(spec["seed"]) is not int
        or not re.fullmatch(r"cpu|npu:[0-9]+", spec["device"])
    ):
        raise ValueError("invalid full-region job specification")
    data = primary._registered(spec["data"])
    if data["state"] != "complete" or data["protocol"] != "full-region-data-v1":
        raise ValueError("full-region data must be registered and complete")
    if spec["model"] not in data["features"] or spec["budget"] > len(data["train_positions"]):
        raise ValueError("unregistered model or excessive support budget")
    family = {"regression": "R", "retrieval": "Q"}.get(spec["head"], "C")
    conditions = []
    for source, tasks in data["tasks"][family].items():
        for task in tasks:
            key = f"{family}_{task}_{spec['seed']}_{spec['budget']}"
            if spec["head"] in ["mlp", "conv3x3", "rf", "unet"]:
                key += "_" + spec["head"]
            conditions.append(
                {
                    "family": family,
                    "source": source,
                    "task": task,
                    "seed": spec["seed"],
                    "budget": spec["budget"],
                    "head": spec["head"],
                    "key": key,
                }
            )
    if not conditions:
        raise ValueError("no registered tasks for this readout family")
    return spec, data, conditions


def _arrays(spec, data):
    refs = [data["features"][spec["model"]], data["valid"], data["labels"]]
    for r in refs:
        if sha(Path(r["path"])) != r["sha256"]:
            raise ValueError("full-region array changed")
    x = np.load(refs[0]["path"], mmap_mode="r")
    valid = np.load(refs[1]["path"], mmap_mode="r")
    with np.load(refs[2]["path"]) as z:
        labels = {k: z[k] for k in z.files}
    ids = data["indices"]
    train, val = data["train_positions"], data["calibration_positions"]
    if (
        x.ndim != 4
        or valid.shape != x.shape[:-1]
        or valid.dtype != bool
        or len(ids) != len(x)
        or len(set(ids)) != len(ids)
        or not train
        or not val
        or set(train) & set(val)
        or any(type(i) is not int or i < 0 or i >= len(x) for i in train + val)
        or len(set(train)) != len(train)
        or len(set(val)) != len(val)
    ):
        raise ValueError("invalid full-region feature or support geometry")
    for y in labels.values():
        if y.shape != valid.shape or not np.isin(y, [-1, 0, 1]).all() or (y[~valid] != -1).any():
            raise ValueError("invalid full-region labels")
    return FeatureBatch(x, valid, tuple(ids), {}), labels, valid


def load_head(path, digest, head, device):
    if head in ["linear", "regression", "retrieval"]:
        return frozen_readouts.load_readout(Path(path), digest)
    if head == "unet":
        return unet_readout.load(path, digest, device=device)
    return strong._load(path, digest, head, device)


def save_head(model, path, head):
    if head in ["linear", "regression", "retrieval"]:
        return frozen_readouts.save_readout(model, path)
    if head == "unet":
        return unet_readout.save(model, path)
    return strong._save(model, path)


def _primary_support(batch, y, train, ids, condition):
    family, seed, budget = (condition[k] for k in ["family", "seed", "budget"])
    if family == "C":
        tiles = primary.nested_support(y, ids, train, budget, seed)
        target = y[tiles].ravel()
        positions = primary.balanced_positions(target, 4096, seed)
        x = batch.values[tiles].reshape(-1, batch.values.shape[-1])[positions]
        target = target[positions]
        support = {
            "support_tiles": [batch.indices[i] for i in tiles],
            "support_positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
            "support_pixels": len(positions),
        }
    elif family == "R":
        tiles = sorted(train, key=lambda i: hashlib.sha256(f"{seed}:{ids[i]}".encode()).digest())[
            :budget
        ]
        x, target, blocks = primary.block_regression_data(batch.values, y, tiles)
        support = {
            "support_tiles": [batch.indices[i] for i in tiles],
            "support_blocks": len(target),
            "training_mean": float(target.mean()),
            "block_pixels": 16,
            "minimum_valid_fraction": 0.8,
            "support_targets_sha256": frozen_readouts._digest(target, blocks),
        }
    else:
        x, samples = primary.retrieval_prototypes(batch.values, y, train, ids, budget, seed)
        target = None
        support = {"queries": [{**s, "tile": batch.indices[s["tile"]]} for s in samples]}
    return x, target, support


def _reuse(model, batch, y, valid, train, val, ids, condition, expected_support):
    head = condition["head"]
    if head in ["linear", "regression", "retrieval"]:
        x, target, support = _primary_support(batch, y, train, ids, condition)
        if support != expected_support:
            raise ValueError("archived primary support differs")
        if head == "retrieval":
            if (
                frozen_readouts._digest(frozen_readouts._matrix(x))
                != model.metadata["prototype_sha256"]
            ):
                raise ValueError("archived retrieval prototypes differ")
            return support
        vx, vy, _, _ = primary._query(batch.values, y, val, condition["family"])
        sx = frozen_readouts._matrix(x)
        vx = frozen_readouts._matrix(vx, sx.dtype)
        sy = frozen_readouts._target(target, len(sx), classification=head == "linear")
        vy = frozen_readouts._target(vy, len(vx), classification=head == "linear")
        support_hash = frozen_readouts._digest(sx, sy)
        validation_hash = frozen_readouts._digest(vx, vy)
    else:
        proxy = {**condition, "head": "conv3x3" if head == "unet" else head}
        tiles, incoming, _, support = strong._support(y, ids, train, list(batch.indices), proxy)
        if support != expected_support:
            raise ValueError("archived support differs")
        if head in ["mlp", "conv3x3", "unet"]:
            support_hash = frozen_readouts._digest(
                strong._maps(batch.values, tiles), y[tiles], valid[tiles]
            )
            validation_hash = frozen_readouts._digest(
                strong._maps(batch.values, val), y[val], valid[val]
            )
        else:
            sx = (
                batch.values[tiles].reshape(-1, batch.values.shape[-1])[incoming].astype(np.float64)
            )
            vx, vy, _, _ = primary._query(batch.values, y, val, "C")
            support_hash = frozen_readouts._digest(sx, y[tiles].ravel()[incoming])
            validation_hash = frozen_readouts._digest(vx.astype(np.float64), vy)
    if (
        support_hash != model.metadata["support_sha256"]
        or validation_hash != model.metadata["validation_sha256"]
    ):
        raise ValueError("archived support or calibration observations changed")
    return support


def _fit(spec, batch, y, valid, train, val, ids, condition):
    head = spec["head"]
    if head in ["linear", "regression", "retrieval"]:
        return primary._fit(batch.values, y, train, val, ids, list(batch.indices), condition)
    proxy = {**condition, "head": "conv3x3" if head == "unet" else head}
    tiles, incoming, positions, support = strong._support(y, ids, train, list(batch.indices), proxy)
    if head == "unet":
        model = unet_readout.fit(
            strong._maps(batch.values, tiles),
            y[tiles],
            valid[tiles],
            strong._maps(batch.values, val),
            y[val],
            valid[val],
            device=spec["device"],
        )
    else:
        model = strong._fit(
            {"neural_device": spec["device"]},
            batch,
            y,
            valid,
            val,
            tiles,
            incoming,
            positions,
            condition,
        )
    return model, support


def _predict(path, spec, condition, batch, y, valid, query, model, support):
    if spec["head"] in ["linear", "regression", "retrieval"]:
        return primary._predict(
            path, spec["model"], condition, batch, y, query, list(batch.indices), model, support
        )
    return baseline_review.predict(
        path, spec, condition, batch, y, valid, query, list(batch.indices), model, support
    )


def run(spec_path, phase):
    if phase not in ["calibration", "full_region"]:
        raise ValueError("full-region workflow needs explicit calibration or full_region phase")
    spec, data, conditions = specification(spec_path)
    root = Path(spec["output"])
    path = root / phase
    if phase == "full_region" and not (root / "calibration/identity.json").exists():
        raise ValueError("calibration must finish before full-region scoring")
    signature = {"spec_sha256": sha(Path(spec_path)), "implementation": code()}
    root.mkdir(parents=True, exist_ok=True)
    if (root / "registration.json").exists():
        if primary._load(root / "registration.json") != signature:
            raise ValueError("full-region job changed")
    else:
        dump(root / "registration.json", signature)
    path.mkdir(exist_ok=True)
    batch, labels, valid = _arrays(spec, data)
    if (path / "identity.json").exists():
        previous = primary._load(path / "identity.json")
        if (
            previous["state"] != "complete"
            or previous["results_sha256"] != sha(path / "results.json")
            or any(previous[k] != v for k, v in signature.items())
        ):
            raise ValueError("completed full-region job changed")
        return previous
    train, val = data["train_positions"], data["calibration_positions"]
    query = val if phase == "calibration" else list(range(len(batch.indices)))
    ids = [r["patch_id"] for r in data["records"]]
    previous = {}
    if phase == "full_region":
        cal = primary._load(root / "calibration/identity.json")
        if cal["state"] != "complete" or any(cal[k] != v for k, v in signature.items()):
            raise ValueError("frozen calibration changed")
        for key, digest in cal["records"].items():
            p = root / "calibration/records" / (key + ".json")
            if sha(p) != digest:
                raise ValueError("calibration record changed")
            previous[key] = primary._load(p)
    rows, records, reused = [], {}, 0
    try:
        for condition in conditions:
            key = condition["key"]
            y = labels[condition["task"]]
            archive = spec["archives"].get(key)
            record_path = path / "records" / (key + ".json")
            if record_path.exists():
                item = primary._load(record_path)
                if (
                    item["condition"] != condition
                    or sha(path / "predictions" / spec["model"] / (key + ".npz"))
                    != item["row"]["prediction_sha256"]
                ):
                    raise ValueError("resumed prediction changed")
                model = load_head(
                    item["readout"], item["readout_identity_sha256"], spec["head"], spec["device"]
                )
                del model
            else:
                if phase == "calibration":
                    if archive:
                        readout = Path(archive["readout"])
                        digest = archive["readout_identity_sha256"]
                        model = load_head(readout, digest, spec["head"], spec["device"])
                        support = _reuse(
                            model, batch, y, valid, train, val, ids, condition, archive["support"]
                        )
                        source = archive["prediction"]
                        assert sha(Path(source["path"])) == source["sha256"]
                        target = path / "predictions" / spec["model"] / (key + ".npz")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source["path"], target)
                        row = {
                            **condition,
                            "metrics": archive["row"]["metrics"],
                            "prediction_sha256": sha(target),
                        }
                    else:
                        readout = path / "readouts" / key
                        model, support = _fit(spec, batch, y, valid, train, val, ids, condition)
                        save_head(model, readout, spec["head"])
                        digest = sha(readout / "identity.json")
                        row = _predict(
                            path, spec, condition, batch, y, valid, query, model, support
                        )
                else:
                    old = previous[key]
                    readout = Path(old["readout"])
                    digest = old["readout_identity_sha256"]
                    support = old["support"]
                    model = load_head(readout, digest, spec["head"], spec["device"])
                    row = _predict(path, spec, condition, batch, y, valid, query, model, support)
                item = {
                    "condition": condition,
                    "row": row,
                    "support": support,
                    "readout": str(readout),
                    "readout_identity_sha256": digest,
                    "reused": bool(archive) if phase == "calibration" else True,
                }
                record_path.parent.mkdir(exist_ok=True)
                dump(record_path, item)
                del model
            rows.append(item["row"])
            records[key] = sha(record_path)
            reused += int(item["reused"])
            dump(
                path / "status.json",
                {"state": "running", "completed_conditions": len(rows), "total": len(conditions)},
            )
        dump(path / "results.json", rows)
        result = {
            **signature,
            "state": "complete",
            "phase": phase,
            "records": records,
            "results_sha256": sha(path / "results.json"),
            "reused_conditions": reused,
            "full_region_tiles": len(batch.indices) if phase == "full_region" else None,
            "includes_downstream_support": phase == "full_region",
            "parameters_refitted": phase == "calibration" and reused < len(rows),
            "scope": (
                "regional mapping agreement; support/calibration/query/buffer included; "
                "not independent generalization"
            ),
        }
        if phase == "full_region":
            result["calibration_identity_sha256"] = sha(root / "calibration/identity.json")
        dump(path / "identity.json", result)
        dump(path / "status.json", {"state": "complete", "conditions": len(rows)})
        return result
    except BaseException as exc:
        dump(path / "status.json", {"state": "failed", "error": repr(exc)})
        raise
