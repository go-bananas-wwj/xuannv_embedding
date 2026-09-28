"""Independent neural jobs over a shared comparison domain, with audited archive reuse."""

import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from xuannv_embedding.downstream import neural_trajectory as trajectory
from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.downstream import strong_multitask as strong
from xuannv_embedding.downstream.multitask_features import FeatureBatch
from xuannv_embedding.export.context import dump, sha

PROTOCOL = "review-neural-job-v1"


def _code():
    return {
        "review": sha(Path(__file__)),
        "heads": strong._code(),
        "trajectory": trajectory._code(),
    }


def prepare_shared(cohort_path, output, phase):
    cohort, cache = primary._spec(cohort_path)
    if phase not in ("calibration", "test"):
        raise ValueError("unknown shared phase")
    root = Path(output) / phase
    root.mkdir(parents=True, exist_ok=False)
    splits = ("train", "validation") if phase == "calibration" else ("test",)
    batches, _, indices = primary._common(cohort, cache, splits, root)
    files = {name: sha(root / name) for name in ["common_valid.npy", "common_labels.npz"]}
    for model in batches:
        for name in ["features.npy", "valid.npy", "identity.json"]:
            relative = f"features/{model}/{name}"
            files[relative] = sha(root / relative)
    dump(
        root / "identity.json",
        {
            "state": "complete",
            "protocol": "review-shared-domain-v1",
            "phase": phase,
            "cohort_sha256": sha(Path(cohort_path)),
            "indices": list(indices),
            "files": files,
            "models": {k: v.identity for k, v in batches.items()},
            "implementation": _code(),
            "query_labels_read": phase == "test",
        },
    )


def _spec(path):
    spec = primary._load(path)
    if (
        set(spec) - {"reuse_only", "support_seeds", "trajectory_checkpoints"}
        != {"protocol", "cohort", "shared", "model", "head", "device", "budgets", "reuse", "output"}
        or spec["protocol"] != PROTOCOL
        or spec["head"] not in ("mlp", "conv3x3")
        or not re.fullmatch(r"cpu|npu:[0-9]+", spec["device"])
    ):
        raise ValueError("invalid review job specification")
    primary._registered(spec["cohort"])
    cohort, cache = primary._spec(spec["cohort"]["path"])
    if (
        spec["model"] not in cohort["models"]
        or not spec["budgets"]
        or len(set(spec["budgets"])) != len(spec["budgets"])
        or any(type(v) is not int or v not in cohort["budgets"] for v in spec["budgets"])
    ):
        raise ValueError("review model or budget is outside the registered cohort")
    for entry in spec["reuse"]:
        if set(entry) != {"spec", "model"}:
            raise ValueError("invalid reuse entry")
        primary._registered(entry["spec"])
    if "reuse_only" in spec and type(spec["reuse_only"]) is not bool:
        raise ValueError("reuse_only must be boolean")
    if "support_seeds" in spec and (
        not spec["support_seeds"]
        or len(set(spec["support_seeds"])) != len(spec["support_seeds"])
        or any(
            type(v) is not int or v not in cohort["support_seeds"] for v in spec["support_seeds"]
        )
    ):
        raise ValueError("job support seeds are outside the cohort")
    if "trajectory_checkpoints" in spec and (
        spec["head"] != "conv3x3" or spec["trajectory_checkpoints"] != [100, 300, 1000]
    ):
        raise ValueError("invalid integrated convergence checkpoints")
    return spec, cohort, cache


def _shared(spec, phase):
    root = Path(spec["shared"]) / phase
    identity = primary._load(root / "identity.json")
    if (
        identity["state"] != "complete"
        or identity["phase"] != phase
        or identity["cohort_sha256"] != spec["cohort"]["sha256"]
        or identity["implementation"] != _code()
    ):
        raise ValueError("shared comparison identity differs")
    selected = spec["model"]
    paths = ["common_valid.npy", "common_labels.npz"] + [
        f"features/{selected}/{name}" for name in ["features.npy", "valid.npy", "identity.json"]
    ]
    for name in paths:
        if sha(root / name) != identity["files"][name]:
            raise ValueError("shared comparison arrays changed")
    valid = np.load(root / "common_valid.npy", mmap_mode="r")
    with np.load(root / "common_labels.npz", allow_pickle=False) as z:
        labels = {k: z[k] for k in z.files}
    batch = FeatureBatch(
        np.load(root / "features" / selected / "features.npy", mmap_mode="r"),
        valid,
        tuple(identity["indices"]),
        identity["models"][selected],
    )
    return batch, labels, valid


def _conditions(cohort, spec):
    return [
        c
        for c in strong._conditions(cohort, [spec["head"]])
        if c["budget"] in spec["budgets"]
        and c["seed"] in spec.get("support_seeds", cohort["support_seeds"])
    ]


def _fit_condition(spec, path, batch, y, valid, query, tiles, incoming, positions, condition):
    if "trajectory_checkpoints" not in spec:
        return strong._fit(
            {"neural_device": spec["device"]},
            batch,
            y,
            valid,
            query,
            tiles,
            incoming,
            positions,
            condition,
        )
    directory = path / "trajectories" / condition["key"]
    selection = directory / "selection.json"
    if selection.exists():
        state = primary._load(selection)
        model = trajectory.load(
            directory / "100", state["curve"]["100"]["identity_sha256"], device=spec["device"]
        )
    else:
        models = trajectory.fit(
            spec["head"],
            strong._maps(batch.values, tiles),
            y[tiles],
            valid[tiles],
            strong._maps(batch.values, query),
            y[query],
            valid[query],
            checkpoints=spec["trajectory_checkpoints"],
            device=spec["device"],
        )
        curve = {}
        for step, checkpoint in models.items():
            snapshot = directory / str(step)
            trajectory.save(checkpoint, snapshot)
            curve[str(step)] = {
                "readout": str(snapshot),
                "identity_sha256": sha(snapshot / "identity.json"),
                "validation_ap": checkpoint.metadata["validation_ap"],
                "fit_seconds": checkpoint.metadata["fit_seconds"],
            }
        dump(
            selection,
            {
                "selected_steps": trajectory.choose(models),
                "curve": curve,
                "selection": "calibration AP; shortest exact tie; single continuous trajectory",
            },
        )
        model = models[100]
    model.metadata["training_implementation"] = trajectory._code()
    return model


def _same_domain(left, right):
    if not np.array_equal(np.load(left / "common_valid.npy"), np.load(right / "common_valid.npy")):
        return False
    with np.load(left / "common_labels.npz") as x, np.load(right / "common_labels.npz") as y:
        return set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)


def _archives(spec, cohort, phase):
    found = {}
    shared = Path(spec["shared"]) / phase
    for entry in spec["reuse"]:
        archived, old_cohort, _ = strong._spec(entry["spec"]["path"])
        old_model = entry["model"]
        if (
            old_model not in old_cohort["models"]
            or spec["head"] not in archived["heads"]
            or old_cohort["models"][old_model] != cohort["models"][spec["model"]]
            or old_cohort["labels"] != cohort["labels"]
            or old_cohort["reference_cache"] != cohort["reference_cache"]
        ):
            continue
        root = Path(archived["output"])
        stage = root / phase
        if not _same_domain(stage, shared):
            continue
        calibration = root / "calibration"
        identity = primary._load(stage / "identity.json")
        if (
            identity.get("state") != "complete"
            or identity.get("test_scored") is not (phase == "test")
            or identity.get("contract_sha256") != strong.contract_sha256(archived)
            or identity.get("primary_contract_sha256") != primary.contract_sha256(old_cohort)
            or identity.get("implementation") != strong._code()
            or identity.get("conditions") != strong._conditions(old_cohort, archived["heads"])
            or primary._load(stage / "status.json").get("state") != "complete"
        ):
            raise ValueError("archived strong producer contract differs")
        for filename, digest_key in [
            ("results.json", "results_sha256"),
            ("common_valid.npy", "common_valid_sha256"),
            ("common_labels.npz", "common_labels_sha256"),
        ]:
            if sha(stage / filename) != identity[digest_key]:
                raise ValueError("archived strong arrays or results changed")
        if (
            phase == "test"
            and sha(calibration / "identity.json") != identity["calibration_identity_sha256"]
        ):
            raise ValueError("archived frozen calibration identity changed")
        rows = primary._load(stage / "results.json")[old_model]
        cal_identity = primary._load(calibration / "identity.json")
        wanted = {c["key"] for c in _conditions(cohort, spec)}
        for row in rows:
            if row["head"] != spec["head"] or row["key"] not in wanted:
                continue
            key = row["key"]
            record = cal_identity["readouts"][key]["models"][old_model]
            readout = calibration / "readouts" / key / old_model
            _verify_readout(readout, record["readout_identity_sha256"])
            support_path = calibration / "readouts" / key / "support.json"
            if sha(support_path) != cal_identity["readouts"][key]["support_sha256"]:
                raise ValueError("archived support changed")
            if sha(stage / "predictions" / old_model / (key + ".npz")) != row["prediction_sha256"]:
                raise ValueError("archived prediction changed")
            found[key] = {
                "archive_spec": entry["spec"],
                "model": old_model,
                "row": row,
                "prediction": str(stage / "predictions" / old_model / (key + ".npz")),
                "source_identity_sha256": sha(stage / "identity.json"),
                "readout": str(readout),
                "readout_identity_sha256": record["readout_identity_sha256"],
                "support": primary._load(support_path),
            }
    return found


def _link_prediction(source, destination, digest):
    source, destination = Path(source), Path(destination)
    if sha(source) != digest:
        raise ValueError("archived prediction changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != digest:
            raise ValueError("existing imported prediction differs")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _verify_readout(path, digest):
    path = Path(path)
    if sha(path / "identity.json") != digest:
        raise ValueError("review readout identity changed")
    identity = primary._load(path / "identity.json")
    if sha(path / "parameters.npz") != identity["payload_sha256"]:
        raise ValueError("review readout payload changed")


def _task(
    path,
    spec,
    cohort,
    phase,
    condition,
    support,
    batch,
    y,
    valid,
    query,
    indices,
    archive=None,
    incoming=None,
    positions=None,
    tiles=None,
    calibration=None,
):
    key, name = condition["key"], spec["model"]
    checkpoint = path / "records" / (key + ".json")
    if checkpoint.exists():
        item = primary._load(checkpoint)
        if (
            item["condition"] != condition
            or item["support"] != support
            or sha(path / "predictions" / name / (key + ".npz")) != item["row"]["prediction_sha256"]
            or sha(Path(item["readout"]) / "identity.json") != item["readout_identity_sha256"]
        ):
            raise ValueError("completed review condition changed")
        _verify_readout(item["readout"], item["readout_identity_sha256"])
        return item
    if archive is not None:
        if archive["support"] != support:
            raise ValueError("archived support differs from current nested support")
        row = archive["row"]
        if any(row[k] != v for k, v in condition.items()):
            raise ValueError("archived condition differs")
        _link_prediction(
            archive["prediction"],
            path / "predictions" / name / (key + ".npz"),
            row["prediction_sha256"],
        )
        readout = Path(archive["readout"])
        digest = archive["readout_identity_sha256"]
        provenance = {k: v for k, v in archive.items() if k not in ("row", "support")}
    else:
        if spec.get("reuse_only", False):
            raise ValueError("reuse-only job has no matching completed archive condition")
        tic = time.monotonic()
        if phase == "calibration":
            readout = path / "readouts" / key / name
            if not (readout / "identity.json").exists():
                fitted = _fit_condition(
                    spec,
                    path,
                    batch,
                    y,
                    valid,
                    query,
                    tiles,
                    incoming,
                    positions,
                    condition,
                )
                strong._save(fitted, readout)
                del fitted
            digest = sha(readout / "identity.json")
        else:
            readout = Path(calibration["readout"])
            digest = calibration["readout_identity_sha256"]
        model = strong._load(readout, digest, condition["head"], spec["device"])
        row = strong._predict(
            path, name, condition, batch, y, valid, query, indices, model, support
        )
        if (
            phase == "calibration"
            and abs(row["metrics"]["ap"] - model.metadata["validation_ap"]) > 1e-12
        ):
            raise ValueError("saved head validation metric differs from fitted head")
        provenance = {
            "computed_here": True,
            "elapsed_seconds": time.monotonic() - tic,
            "parameters_refitted": phase == "calibration",
        }
        del model
    item = {
        "condition": condition,
        "row": row,
        "support": support,
        "readout": str(readout),
        "readout_identity_sha256": digest,
        "origin": provenance,
        "imported": archive is not None,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    dump(checkpoint, item)
    return item


def _run(spec_path, phase):
    spec, cohort, cache = _spec(spec_path)
    root = Path(spec["output"])
    path = root / phase
    path.mkdir(parents=True, exist_ok=True)
    signature = {"spec_sha256": sha(Path(spec_path)), "implementation": _code()}
    registration = root / "registration.json"
    if registration.exists():
        if primary._load(registration) != signature:
            raise ValueError("review resume identity differs")
    else:
        dump(registration, signature)
    batch, labels, valid = _shared(spec, phase)
    complete_path = path / "identity.json"
    if complete_path.exists():
        complete = primary._load(complete_path)
        if (
            complete["state"] != "complete"
            or any(complete[k] != v for k, v in signature.items())
            or complete["shared_identity_sha256"]
            != sha(Path(spec["shared"]) / phase / "identity.json")
            or complete["results_sha256"] != sha(path / "results.json")
            or set(complete["records"]) != {c["key"] for c in _conditions(cohort, spec)}
        ):
            raise ValueError("completed review phase changed")
        if phase == "test" and complete["calibration_identity_sha256"] != sha(
            root / "calibration/identity.json"
        ):
            raise ValueError("completed query calibration changed")
        for key, digest in complete["records"].items():
            record = path / "records" / (key + ".json")
            if sha(record) != digest:
                raise ValueError("completed review record changed")
            item = primary._load(record)
            _verify_readout(item["readout"], item["readout_identity_sha256"])
            if (
                sha(path / "predictions" / spec["model"] / (key + ".npz"))
                != item["row"]["prediction_sha256"]
            ):
                raise ValueError("completed prediction changed")
        return complete
    train = list(range(len(cache["split"]["train"])))
    indices = list(batch.indices)
    query = (
        list(range(len(train), len(indices)))
        if phase == "calibration"
        else list(range(len(indices)))
    )
    archives = _archives(spec, cohort, phase)
    previous = {}
    if phase == "test":
        cal = primary._load(root / "calibration/identity.json")
        if cal["state"] != "complete" or cal["spec_sha256"] != signature["spec_sha256"]:
            raise ValueError("review calibration must be complete and frozen before query")
        for key, record in cal["records"].items():
            p = root / "calibration/records" / (key + ".json")
            if sha(p) != record:
                raise ValueError("review calibration record changed")
            previous[key] = primary._load(p)
    rows, records, imported = [], {}, 0
    started = time.monotonic()
    try:
        for condition in _conditions(cohort, spec):
            key, y = condition["key"], labels[condition["task"]]
            tiles = incoming = positions = None
            if phase == "calibration":
                ids = [cache["records"][i]["patch_id"] for i in indices]
                tiles, incoming, positions, support = strong._support(
                    y, ids, train, indices, condition
                )
            else:
                support = previous[key]["support"]
            item = _task(
                path,
                spec,
                cohort,
                phase,
                condition,
                support,
                batch,
                y,
                valid,
                query,
                indices,
                archives.get(key),
                incoming,
                positions,
                tiles,
                previous.get(key),
            )
            rows.append(item["row"])
            imported += int(item["imported"])
            records[key] = sha(path / "records" / (key + ".json"))
            dump(
                path / "status.json",
                {
                    "state": "running",
                    "completed_conditions": len(rows),
                    "imported_conditions": imported,
                },
            )
        dump(path / "results.json", rows)
        identity = {
            "state": "complete",
            "protocol": PROTOCOL,
            **signature,
            "phase": phase,
            "model": spec["model"],
            "head": spec["head"],
            "records": records,
            "shared_identity_sha256": sha(Path(spec["shared"]) / phase / "identity.json"),
            "results_sha256": sha(path / "results.json"),
            "elapsed_seconds": time.monotonic() - started,
            "computed_conditions": len(rows) - imported,
            "imported_conditions": imported,
            "parameters_refitted": False if phase == "test" else None,
        }
        if phase == "test":
            identity["calibration_identity_sha256"] = sha(root / "calibration/identity.json")
        dump(path / "identity.json", identity)
        dump(path / "status.json", {"state": "complete", "conditions": len(rows)})
        return identity
    except BaseException as exc:
        dump(path / "status.json", {"state": "failed", "error": repr(exc)})
        raise


def calibrate(spec_path):
    return _run(spec_path, "calibration")


def score(spec_path):
    return _run(spec_path, "test")
