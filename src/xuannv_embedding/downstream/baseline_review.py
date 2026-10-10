"""Additional RF/U-Net jobs reusing the frozen review domain and support schedule."""

import re
import time
from pathlib import Path

import numpy as np

from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.downstream import review_readouts as review
from xuannv_embedding.downstream import strong_multitask as strong
from xuannv_embedding.downstream import unet_readout as unet
from xuannv_embedding.export.context import dump, sha

PROTOCOL = "review-additional-baseline-v1"


def code():
    return {"workflow": sha(Path(__file__)), "review": review._code(), "unet": unet.code()}


def specification(path):
    spec = primary._load(path)
    if (
        set(spec)
        != {
            "protocol",
            "cohort",
            "shared",
            "model",
            "head",
            "device",
            "budgets",
            "support_seeds",
            "output",
        }
        or spec["protocol"] != PROTOCOL
        or spec["head"] not in ("rf", "unet")
        or not re.fullmatch(r"cpu|npu:[0-9]+", spec["device"])
    ):
        raise ValueError("invalid additional baseline specification")
    primary._registered(spec["cohort"])
    cohort, cache = primary._spec(spec["cohort"]["path"])
    if spec["model"] not in cohort["models"]:
        raise ValueError("unregistered model")
    for key in ["budgets", "support_seeds"]:
        if (
            not spec[key]
            or len(set(spec[key])) != len(spec[key])
            or any(type(v) is not int or v not in cohort[key] for v in spec[key])
        ):
            raise ValueError("unregistered budget or support seed")
    conditions = [
        {**c, "head": spec["head"], "key": c["key"] + "_" + spec["head"]}
        for c in primary._conditions(cohort)
        if c["family"] == "C"
        and c["budget"] in spec["budgets"]
        and c["seed"] in spec["support_seeds"]
    ]
    return spec, cache, conditions


def load_head(path, digest, spec):
    if spec["head"] == "unet":
        return unet.load(path, digest, device=spec["device"])
    return strong._load(path, digest, "rf", "cpu")


def predict(stage, spec, condition, batch, y, valid, query, indices, model, support):
    if spec["head"] == "rf":
        return strong._predict(
            stage, spec["model"], condition, batch, y, valid, query, indices, model, support
        )
    maps = y[query]
    positions = np.flatnonzero(maps.ravel() >= 0)
    truth = maps.ravel()[positions]
    tiles = np.repeat(query, y.shape[1] * y.shape[2])[positions]
    values = model.predict(strong._maps(batch.values, query), valid[query]).ravel()[positions]
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite U-Net prediction on labeled domain")
    p = stage / "predictions" / spec["model"] / (condition["key"] + ".npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p, scores=values, truth=truth, tiles=np.asarray(indices)[tiles], valid_indices=positions
    )
    return {
        **condition,
        "metrics": primary._metrics("C", truth, values, model, maps, positions, support),
        "prediction_sha256": sha(p),
    }


def run(spec_path, phase):
    if phase not in ("calibration", "test"):
        raise ValueError("invalid baseline phase")
    spec, cache, conditions = specification(spec_path)
    root = Path(spec["output"])
    signature = {"spec_sha256": sha(Path(spec_path)), "implementation": code()}
    if phase == "test" and not (root / "calibration/identity.json").exists():
        raise ValueError("calibration must be complete before testing")
    registration = root / "registration.json"
    if registration.exists() and primary._load(registration) != signature:
        raise ValueError("baseline registration changed")
    root.mkdir(parents=True, exist_ok=True)
    if not registration.exists():
        dump(registration, signature)
    batch, labels, valid = review._shared(spec, phase)
    stage = root / phase
    stage.mkdir(exist_ok=True)
    identity_path = stage / "identity.json"
    if identity_path.exists():
        identity = primary._load(identity_path)
        if (
            identity["state"] != "complete"
            or any(identity[k] != v for k, v in signature.items())
            or identity["results_sha256"] != sha(stage / "results.json")
            or identity["shared_identity_sha256"]
            != sha(Path(spec["shared"]) / phase / "identity.json")
        ):
            raise ValueError("completed baseline identity changed")
        if phase == "test" and identity["calibration_identity_sha256"] != sha(
            root / "calibration/identity.json"
        ):
            raise ValueError("completed baseline calibration changed")
        for key, digest in identity["records"].items():
            p = stage / "records" / (key + ".json")
            if sha(p) != digest:
                raise ValueError("completed baseline record changed")
            item = primary._load(p)
            model = load_head(item["readout"], item["readout_identity_sha256"], spec)
            del model
            if (
                sha(stage / "predictions" / spec["model"] / (key + ".npz"))
                != item["row"]["prediction_sha256"]
            ):
                raise ValueError("completed baseline prediction changed")
        return identity
    train = list(range(len(cache["split"]["train"])))
    indices = list(batch.indices)
    query = (
        list(range(len(train), len(indices)))
        if phase == "calibration"
        else list(range(len(indices)))
    )
    previous = {}
    if phase == "test":
        cal = primary._load(root / "calibration/identity.json")
        if cal["state"] != "complete" or any(cal[k] != v for k, v in signature.items()):
            raise ValueError("calibration identity changed")
        for key, digest in cal["records"].items():
            p = root / "calibration/records" / (key + ".json")
            if sha(p) != digest:
                raise ValueError("calibration record changed")
            previous[key] = primary._load(p)
    records, rows = {}, []
    start = time.monotonic()
    try:
        for condition in conditions:
            key, y = condition["key"], labels[condition["task"]]
            checkpoint = stage / "records" / (key + ".json")
            if phase == "calibration":
                # U-Net uses the exact full-tile support of the existing CNN.
                support_condition = {
                    **condition,
                    "head": "conv3x3" if spec["head"] == "unet" else "rf",
                }
                ids = [cache["records"][i]["patch_id"] for i in indices]
                tiles, incoming, positions, support = strong._support(
                    y, ids, train, indices, support_condition
                )
                readout = stage / "readouts" / key
                if not (readout / "identity.json").exists():
                    if spec["head"] == "unet":
                        model = unet.fit(
                            strong._maps(batch.values, tiles),
                            y[tiles],
                            valid[tiles],
                            strong._maps(batch.values, query),
                            y[query],
                            valid[query],
                            device=spec["device"],
                        )
                        unet.save(model, readout)
                    else:
                        model = strong._fit(
                            {"neural_device": "cpu"},
                            batch,
                            y,
                            valid,
                            query,
                            tiles,
                            incoming,
                            positions,
                            condition,
                        )
                        strong._save(model, readout)
                    del model
                digest = sha(readout / "identity.json")
            else:
                item = previous[key]
                support, readout = item["support"], Path(item["readout"])
                digest = item["readout_identity_sha256"]
            if checkpoint.exists():
                item = primary._load(checkpoint)
                if (
                    item["condition"] != condition
                    or item["support"] != support
                    or item["readout_identity_sha256"] != digest
                    or sha(stage / "predictions" / spec["model"] / (key + ".npz"))
                    != item["row"]["prediction_sha256"]
                ):
                    raise ValueError("resumed baseline condition changed")
                model = load_head(readout, digest, spec)
                del model
            else:
                model = load_head(readout, digest, spec)
                row = predict(
                    stage, spec, condition, batch, y, valid, query, indices, model, support
                )
                if (
                    phase == "calibration"
                    and abs(row["metrics"]["ap"] - model.metadata["validation_ap"]) > 1e-10
                ):
                    raise ValueError("saved baseline validation AP differs")
                item = {
                    "condition": condition,
                    "support": support,
                    "readout": str(readout),
                    "readout_identity_sha256": digest,
                    "row": row,
                }
                checkpoint.parent.mkdir(exist_ok=True)
                dump(checkpoint, item)
                del model
            rows.append(item["row"])
            records[key] = sha(checkpoint)
            dump(
                stage / "status.json",
                {
                    "state": "running",
                    "completed_conditions": len(rows),
                    "total_conditions": len(conditions),
                },
            )
        dump(stage / "results.json", rows)
        identity = {
            **signature,
            "state": "complete",
            "protocol": PROTOCOL,
            "phase": phase,
            "model": spec["model"],
            "head": spec["head"],
            "records": records,
            "results_sha256": sha(stage / "results.json"),
            "shared_identity_sha256": sha(Path(spec["shared"]) / phase / "identity.json"),
            "parameters_refitted": phase == "calibration",
            "elapsed_seconds": time.monotonic() - start,
        }
        if phase == "test":
            identity["calibration_identity_sha256"] = sha(root / "calibration/identity.json")
        dump(identity_path, identity)
        dump(stage / "status.json", {"state": "complete", "conditions": len(rows)})
        return identity
    except BaseException as exc:
        dump(stage / "status.json", {"state": "failed", "error": repr(exc)})
        raise
