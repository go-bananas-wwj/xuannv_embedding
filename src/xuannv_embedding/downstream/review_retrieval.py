"""Extend registered positive-example budgets without rerunning completed queries."""

import time
from pathlib import Path

from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.downstream import paired_multitask_report as report
from xuannv_embedding.downstream import review_readouts as review
from xuannv_embedding.export.context import dump, sha


def run(spec_path):
    spec = primary._load(spec_path)
    if (
        set(spec) != {"protocol", "cohort", "shared", "reuse", "model", "output"}
        or spec["protocol"] != "review-retrieval-v1"
    ):
        raise ValueError("invalid review retrieval contract")
    primary._registered(spec["cohort"])
    cohort, cache = primary._spec(spec["cohort"]["path"])
    if spec["model"] not in cohort["models"]:
        raise ValueError("retrieval feature is outside the cohort")
    conditions = [c for c in primary._conditions(cohort) if c["family"] == "Q"]
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    signature = {
        "spec_sha256": sha(Path(spec_path)),
        "implementation": {
            "retrieval": sha(Path(__file__)),
            "primary": primary._code(),
            "shared": review._code(),
        },
    }
    if (root / "registration.json").exists():
        if primary._load(root / "registration.json") != signature:
            raise ValueError("retrieval resume identity differs")
    else:
        dump(root / "registration.json", signature)
    train_batch, train_labels, _ = review._shared(spec, "calibration")
    query_batch, query_labels, _ = review._shared(spec, "test")
    train = list(range(len(cache["split"]["train"])))
    calibration = list(range(len(train), len(train_batch.indices)))
    query = list(range(len(query_batch.indices)))
    ids = [cache["records"][i]["patch_id"] for i in train_batch.indices]
    archives = {}
    for entry in spec["reuse"]:
        primary._registered(entry["spec"])
        old_cohort, _ = primary._spec(entry["spec"]["path"])
        model = entry["model"]
        if (
            model not in old_cohort["models"]
            or old_cohort["models"][model] != cohort["models"][spec["model"]]
            or old_cohort["labels"] != cohort["labels"]
        ):
            continue
        old_root = Path(old_cohort["output"])
        if not all(
            review._same_domain(old_root / phase, Path(spec["shared"]) / phase)
            for phase in ["calibration", "test"]
        ):
            continue
        report._inputs(entry["spec"]["path"], sha(old_root / "test/identity.json"))
        cal_identity = primary._load(old_root / "calibration/identity.json")
        for row in primary._load(old_root / "test/results.json")[model]:
            if row["family"] != "Q":
                continue
            key = row["key"]
            directory = old_root / "calibration/readouts" / key
            if sha(directory / "support.json") != cal_identity["readouts"][key]["support_sha256"]:
                raise ValueError("archived retrieval support changed")
            readout = directory / model
            digest = cal_identity["readouts"][key]["models"][model]["readout_identity_sha256"]
            primary.load_readout(readout, digest)
            archives[key] = {
                "row": row,
                "prediction": str(old_root / "test/predictions" / model / (key + ".npz")),
                "support": primary._load(directory / "support.json"),
                "readout": str(readout),
                "readout_identity_sha256": digest,
                "source_test_identity_sha256": sha(old_root / "test/identity.json"),
            }
    rows, records, imported = [], {}, 0
    started = time.monotonic()
    for condition in conditions:
        key, task = condition["key"], condition["task"]
        path = root / "records" / (key + ".json")
        if path.exists():
            item = primary._load(path)
            if (
                item["row"]["key"] != key
                or sha(root / "predictions" / spec["model"] / (key + ".npz"))
                != item["row"]["prediction_sha256"]
            ):
                raise ValueError("completed retrieval condition changed")
        elif key in archives:
            a = archives[key]
            review._link_prediction(
                a["prediction"],
                root / "predictions" / spec["model"] / (key + ".npz"),
                a["row"]["prediction_sha256"],
            )
            item = {**a, "imported": True}
            path.parent.mkdir(parents=True, exist_ok=True)
            dump(path, item)
        else:
            y = train_labels[task]
            model, support = primary._fit(
                train_batch.values, y, train, calibration, ids, list(train_batch.indices), condition
            )
            readout = root / "readouts" / key
            primary.save_readout(model, readout)
            digest = sha(readout / "identity.json")
            frozen = primary.load_readout(readout, digest)
            row = primary._predict(
                root,
                spec["model"],
                condition,
                query_batch,
                query_labels[task],
                query,
                list(query_batch.indices),
                frozen,
                support,
            )
            item = {
                "row": row,
                "support": support,
                "readout": str(readout),
                "readout_identity_sha256": digest,
                "imported": False,
                "prototype_fit_scope": "training support components only",
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            dump(path, item)
        rows.append(item["row"])
        imported += int(item["imported"])
        records[key] = sha(path)
        dump(
            root / "status.json",
            {
                "state": "running",
                "completed_conditions": len(rows),
                "imported_conditions": imported,
            },
        )
    dump(root / "results.json", rows)
    identity = {
        "state": "complete",
        **signature,
        "records": records,
        "results_sha256": sha(root / "results.json"),
        "imported_conditions": imported,
        "computed_conditions": len(rows) - imported,
        "elapsed_seconds": time.monotonic() - started,
    }
    dump(root / "identity.json", identity)
    dump(root / "status.json", {"state": "complete", "conditions": len(rows)})
    return identity
