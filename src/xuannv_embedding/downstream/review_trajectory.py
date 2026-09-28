"""Sharded, bounded convergence checks for a predeclared neural baseline."""

import time
from pathlib import Path

from xuannv_embedding.downstream import neural_trajectory as trajectory
from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.downstream import review_readouts as review
from xuannv_embedding.downstream import strong_multitask as strong
from xuannv_embedding.export.context import dump, sha


def run(spec_path, phase):
    spec = primary._load(spec_path)
    if (
        set(spec) != {"protocol", "base_job", "condition_keys", "checkpoints", "device", "output"}
        or spec["protocol"] != "review-trajectory-job-v1"
        or phase not in ("calibration", "test")
        or not spec["condition_keys"]
        or len(set(spec["condition_keys"])) != len(spec["condition_keys"])
        or spec["checkpoints"] != [100, 300, 1000]
    ):
        raise ValueError("invalid convergence-check contract")
    primary._registered(spec["base_job"])
    job, cohort, cache = review._spec(spec["base_job"]["path"])
    job = {**job, "device": spec["device"]}
    available = {c["key"]: c for c in review._conditions(cohort, job)}
    if job["head"] != "conv3x3" or not set(spec["condition_keys"]) <= set(available):
        raise ValueError("convergence condition is outside the registered convolution cohort")
    root = Path(spec["output"])
    stage = root / phase
    stage.mkdir(parents=True, exist_ok=True)
    signature = {
        "spec_sha256": sha(Path(spec_path)),
        "implementation": {
            "worker": sha(Path(__file__)),
            "trajectory": trajectory._code(),
            "review": review._code(),
        },
    }
    if (root / "registration.json").exists():
        if primary._load(root / "registration.json") != signature:
            raise ValueError("convergence resume registration differs")
    else:
        dump(root / "registration.json", signature)
    batch, labels, valid = review._shared(job, phase)
    finished_path = stage / "identity.json"
    if finished_path.exists():
        finished = primary._load(finished_path)
        if (
            finished["state"] != "complete"
            or any(finished[k] != v for k, v in signature.items())
            or finished["results_sha256"] != sha(stage / "results.json")
            or set(finished["records"]) != set(spec["condition_keys"])
        ):
            raise ValueError("completed convergence identity changed")
        for key, digest in finished["records"].items():
            path = stage / "records" / (key + ".json")
            if sha(path) != digest:
                raise ValueError("completed convergence record changed")
            item = primary._load(path)
            for checkpoint in item["curve"].values():
                review._verify_readout(checkpoint["readout"], checkpoint["identity_sha256"])
            if (
                sha(stage / "predictions" / job["model"] / (key + ".npz"))
                != item["row"]["prediction_sha256"]
            ):
                raise ValueError("completed convergence prediction changed")
        return finished
    indices = list(batch.indices)
    train = list(range(len(cache["split"]["train"])))
    query = (
        list(range(len(train), len(indices)))
        if phase == "calibration"
        else list(range(len(indices)))
    )
    ids = [cache["records"][i]["patch_id"] for i in indices]
    previous = {}
    if phase == "test":
        frozen = primary._load(root / "calibration/identity.json")
        if frozen["state"] != "complete" or frozen["spec_sha256"] != signature["spec_sha256"]:
            raise ValueError("convergence selection must be frozen before query scoring")
        for key, digest in frozen["records"].items():
            path = root / "calibration/records" / (key + ".json")
            if sha(path) != digest:
                raise ValueError("convergence selection record changed")
            previous[key] = primary._load(path)
    rows, records = [], {}
    started = time.monotonic()
    for key in spec["condition_keys"]:
        condition = available[key]
        target = stage / "records" / (key + ".json")
        if target.exists():
            item = primary._load(target)
            review._verify_readout(item["readout"], item["readout_identity_sha256"])
            if (
                sha(stage / "predictions" / job["model"] / (key + ".npz"))
                != item["row"]["prediction_sha256"]
            ):
                raise ValueError("convergence prediction changed")
        else:
            y = labels[condition["task"]]
            if phase == "calibration":
                tiles, _, _, support = strong._support(y, ids, train, indices, condition)
                if job.get("trajectory_checkpoints") != spec["checkpoints"]:
                    raise ValueError(
                        "convergence requires the fixed-budget job's continuous trajectory"
                    )
                selection_path = (
                    Path(job["output"]) / "calibration/trajectories" / key / "selection.json"
                )
                captured = primary._load(selection_path)
                chosen, curve = captured["selected_steps"], captured["curve"]
                if set(curve) != {str(v) for v in spec["checkpoints"]}:
                    raise ValueError("captured trajectory checkpoints differ")
                for checkpoint in curve.values():
                    review._verify_readout(checkpoint["readout"], checkpoint["identity_sha256"])
                readout = Path(curve[str(chosen)]["readout"])
                identity_sha = curve[str(chosen)]["identity_sha256"]
                selected = trajectory.load(readout, identity_sha, device=job["device"])
                fixed = Path(job["output"]) / "calibration/records" / (key + ".json")
                fixed_record = primary._load(fixed)
                fixed_identity = primary._load(Path(fixed_record["readout"]) / "identity.json")
                captured_100 = primary._load(Path(curve["100"]["readout"]) / "identity.json")
                if (
                    fixed_identity["metadata"]["final_weights_sha256"]
                    != captured_100["metadata"]["final_weights_sha256"]
                ):
                    raise ValueError(
                        "fixed-budget result is not the captured 100-update checkpoint"
                    )
            else:
                entry = previous[key]
                chosen, curve, support = entry["selected_steps"], entry["curve"], entry["support"]
                readout, identity_sha = Path(entry["readout"]), entry["readout_identity_sha256"]
                selected = trajectory.load(readout, identity_sha, device=job["device"])
            row = strong._predict(
                stage, job["model"], condition, batch, y, valid, query, indices, selected, support
            )
            item = {
                "row": row,
                "condition": condition,
                "support": support,
                "curve": curve,
                "selected_steps": chosen,
                "selection": "calibration AP; shortest exact tie",
                "readout": str(readout),
                "readout_identity_sha256": identity_sha,
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            dump(target, item)
        rows.append(item["row"])
        records[key] = sha(target)
        dump(stage / "status.json", {"state": "running", "completed_conditions": len(rows)})
    dump(stage / "results.json", rows)
    dump(
        stage / "identity.json",
        {
            "state": "complete",
            **signature,
            "phase": phase,
            "records": records,
            "results_sha256": sha(stage / "results.json"),
            "parameters_refitted": False if phase == "test" else None,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    dump(stage / "status.json", {"state": "complete", "conditions": len(rows)})
