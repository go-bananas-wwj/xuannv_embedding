"""Same-representation classification, retrieval and coverage readouts."""

from __future__ import annotations

import copy
import fcntl
import json
import resource
import time
from pathlib import Path

from xuannv_embedding.downstream import full_region_readouts as full
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.frozen_fusion import verified
from xuannv_embedding.training.shared_fusion_search import reader_jobs


def run(path, device):
    spec = full.primary._load(path)
    required = {
        "schema",
        "reference",
        "index",
        "model",
        "head",
        "budget",
        "seed",
        "preparation",
        "output",
    }
    if set(spec) != required or spec["schema"] != "journal-auxiliary-job-v1":
        raise ValueError("Invalid journal auxiliary job")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    ident = {"spec": sha(path), "code": sha(__file__), "readout": full.code()}
    if (root / "result.json").exists():
        result = json.loads((root / "result.json").read_text())
        if result["identity"] != ident:
            raise ValueError("Auxiliary source or score changed")
        return result
    started = time.monotonic()
    preparation = Path(spec["preparation"])
    preparation.mkdir(parents=True, exist_ok=True)
    descriptor = spec["model"]
    name = descriptor["name"]
    with (preparation / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        datafile = preparation / "data.json"
        header = preparation / "identity.json"
        binding = {"reference": spec["reference"], "index": spec["index"], "model": descriptor}
        if header.exists() and json.loads(header.read_text()) != binding:
            raise ValueError("Auxiliary preparation binding changed")
        if not datafile.exists():
            data = copy.deepcopy(json.loads(verified(spec["reference"]).read_text()))
            if "legacy_feature" in descriptor:
                features = data["features"][descriptor["legacy_feature"]]
                verified(features)
            else:
                if "feature_receipt" in descriptor:
                    receipt = json.loads(Path(descriptor["feature_receipt"]).read_text())
                    if receipt["state"] != "complete":
                        raise ValueError("Final source feature incomplete")
                    feature = verified(receipt["features"])
                else:
                    feature = verified(descriptor["features"])
                region = {
                    "name": "registered_region",
                    "index": spec["index"],
                    "evaluation": spec["reference"],
                    "task_source": next(iter(data["tasks"]["C"])),
                    "support_seed": spec["seed"],
                }
                jobs = reader_jobs(
                    {"output": str(preparation / "projection")},
                    region,
                    {"name": name, "kind": "journal_multitask"},
                    feature,
                )
                projected = full.primary._load(full.primary._load(jobs[0]["spec"])["data"]["path"])
                features = projected["features"]["candidate"]
            data["features"] = {name: features}
            data["feature_identities"] = {name: {"source": descriptor, "preparation": binding}}
            dump(datafile, data)
            dump(header, binding)
        dataref = {"path": str(datafile), "sha256": sha(datafile)}
    prepared = time.monotonic()
    job = {
        "protocol": full.PROTOCOL,
        "data": dataref,
        "model": name,
        "head": spec["head"],
        "budget": spec["budget"],
        "seed": spec["seed"],
        "device": device,
        "archives": {},
        "output": str(root / "readout"),
    }
    jobpath = root / "readout.json"
    if jobpath.exists() and full.primary._load(jobpath) != job:
        raise ValueError("Auxiliary readout device or protocol changed")
    dump(jobpath, job)
    full.run(jobpath, "calibration")
    calibrated = time.monotonic()
    full.run(jobpath, "full_region")
    result = {
        "state": "complete",
        "identity": ident,
        "model": name,
        "head": spec["head"],
        "budget": spec["budget"],
        "rows": full.primary._load(root / "readout/full_region/results.json"),
        "prepare_seconds": prepared - started,
        "fit_and_calibrate_seconds": calibrated - prepared,
        "full_query_seconds": time.monotonic() - calibrated,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "same_representation_for_all_tasks": True,
        "new_encoder_fit": False,
    }
    dump(root / "result.json", result)
    return result
