"""Score missing-input curves with unchanged readouts, targets and query positions."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.downstream.multitask_features import FeatureSelection, read_features
from xuannv_embedding.export.context import dump, sha

PROTOCOL = "fixed-readout-highres-stress-v1"
RETENTION_RULE = "hashed edge, nested raster prefix of originally valid pixels per month"


def _manifest(contract):
    return primary._registered(
        {"path": contract["manifest_path"], "sha256": contract["manifest_sha256"]}
    )


def _variant_contract(base, variant, spec):
    source, changed = base["models"][variant["model"]], variant["features"]
    if set(changed) != set(source) or any(
        changed[k] != source[k] for k in ("cache_path", "cache_sha256", "selection")
    ):
        raise ValueError("stress features changed cache or feature selection")
    original, partial = _manifest(source), _manifest(changed)
    for key in ("checkpoint_sha256", "config_sha256", "training_git_sha"):
        if not original.get(key) or original[key] != partial.get(key):
            raise ValueError("stress export changed checkpoint, config or training code")
    for key in ("cache_sha256", "months", "split", "adaptation", "checkpoint_epoch", "dtype"):
        if original.get(key) != partial.get(key):
            raise ValueError("stress export changed model or input identity")
    if original.get("input_ablation"):
        raise ValueError("baseline must use its original available observations")
    expected = {
        "dropped_sources": [],
        "last_visible_month_index": None,
        "last_visible_month": None,
        "undated_static_inputs": "unchanged unless dropped",
        "training_time_causality_claim": False,
        "highres_retention": {
            "sources": spec["sources"],
            "fraction": variant["fraction"],
            "seed": spec["mask_seed"],
            "rule": RETENTION_RULE,
        },
    }
    if partial.get("input_ablation") != expected:
        raise ValueError("stress input mask differs from its registered retention rule")


def _spec(path):
    spec = primary._load(path)
    if (
        set(spec)
        != {
            "protocol",
            "base_spec",
            "calibration_identity_sha256",
            "baseline_identity_sha256",
            "split",
            "sources",
            "fractions",
            "mask_seed",
            "variants",
            "output",
        }
        or spec["protocol"] != PROTOCOL
        or spec["split"] not in ("validation", "test")
        or not spec["sources"]
        or spec["sources"] != sorted(set(spec["sources"]))
        or any(not isinstance(s, str) or not s for s in spec["sources"])
        or type(spec["mask_seed"]) is not int
        or spec["mask_seed"] < 0
        or not spec["variants"]
    ):
        raise ValueError("invalid fixed-readout stress specification")
    fractions = spec["fractions"]
    if (
        not isinstance(fractions, list)
        or any(
            type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in fractions
        )
        or fractions != sorted(set(fractions), reverse=True)
        or 1 not in fractions
        or 0 not in fractions
    ):
        raise ValueError("retention curve requires unique fractions and both endpoint controls")
    primary._registered(spec["base_spec"])
    base, cache = primary._spec(spec["base_spec"]["path"])
    curves = {}
    for name, variant in spec["variants"].items():
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or set(variant) != {"model", "fraction", "features"}
            or variant["model"] not in base["models"]
            or type(variant["fraction"]) not in (int, float)
            or variant["fraction"] not in fractions
        ):
            raise ValueError("invalid stress variant")
        curves.setdefault(variant["model"], []).append(variant["fraction"])
        _variant_contract(base, variant, spec)
    if any(sorted(values, reverse=True) != fractions for values in curves.values()):
        raise ValueError("every stress model must retain the complete registered curve")
    return spec, base, cache


def _stage(path, expected_sha, base, *, test):
    identity_path = path / "identity.json"
    if sha(identity_path) != expected_sha:
        raise ValueError("baseline stage identity changed")
    identity = primary._load(identity_path)
    if (
        identity.get("state") != "complete"
        or identity.get("test_scored") is not test
        or identity.get("contract_sha256") != primary.contract_sha256(base)
        or identity.get("implementation") != primary._code()
        or identity.get("conditions") != primary._conditions(base)
        or primary._load(path / "status.json").get("state") != "complete"
    ):
        raise ValueError("baseline evaluation contract, code or state changed")
    for filename, key in (
        ("results.json", "results_sha256"),
        ("common_labels.npz", "common_labels_sha256"),
        ("common_valid.npy", "common_valid_sha256"),
    ):
        if sha(path / filename) != identity[key]:
            raise ValueError("baseline results, truth or query domain changed")
    return identity


def run(spec_path):
    spec, base, cache = _spec(spec_path)
    calibration = Path(base["output"]) / "calibration"
    calibrated = _stage(calibration, spec["calibration_identity_sha256"], base, test=False)
    test = spec["split"] == "test"
    baseline = Path(base["output"]) / ("test" if test else "calibration")
    identity = _stage(baseline, spec["baseline_identity_sha256"], base, test=test)
    if test and identity.get("calibration_identity_sha256") != spec["calibration_identity_sha256"]:
        raise ValueError("test baseline uses a different calibration")
    offset = 0 if test else len(cache["split"]["train"])
    valid = np.load(baseline / "common_valid.npy", allow_pickle=False)[offset:]
    with np.load(baseline / "common_labels.npz", allow_pickle=False) as archive:
        labels = {k: archive[k][offset:] for k in archive.files}
    current = primary._labels(base, cache, (spec["split"],))
    if any(not np.array_equal(labels[k], np.where(valid, y, -1)) for k, y in current.items()):
        raise ValueError("baseline truth differs from registered labels")
    baseline_rows = primary._load(baseline / "results.json")
    stage = Path(spec["output"])
    stage.mkdir(parents=True, exist_ok=False)
    dump(stage / "status.json", {"state": "running", "phase": "prepare", "test_scored": False})
    scored = False
    try:
        indices = tuple(cache["split"][spec["split"]])
        query_indices = list(range(len(indices)))
        originals, readouts, supports, control = {}, {}, {}, []
        for model in sorted({v["model"] for v in spec["variants"].values()}):
            contract = base["models"][model]
            originals[model] = read_features(
                **dict(contract, selection=FeatureSelection(**contract["selection"])),
                splits=(spec["split"],),
                output=stage / "original_features" / model,
            )
            readouts[model] = {}
            for condition in primary._conditions(base):
                key = condition["key"]
                record = calibrated["readouts"][key]
                support_path = calibration / "readouts" / key / "support.json"
                if sha(support_path) != record["support_sha256"]:
                    raise ValueError("frozen training support changed")
                supports[key] = primary._load(support_path)
                readouts[model][key] = primary.load_readout(
                    calibration / "readouts" / key / model,
                    record["models"][model]["readout_identity_sha256"],
                )
        results, feature_identities = {}, {}
        for name, variant in spec["variants"].items():
            model, contract = variant["model"], variant["features"]
            batch = read_features(
                **dict(contract, selection=FeatureSelection(**contract["selection"])),
                splits=(spec["split"],),
                output=stage / "features" / name,
            )
            original = originals[model]
            if (
                batch.indices != indices
                or original.indices != indices
                or batch.identity["spatial_layout_sha256"]
                != original.identity["spatial_layout_sha256"]
                or not np.array_equal(batch.valid, original.valid)
                or (valid & ~batch.valid).any()
            ):
                raise ValueError("stress inference changed the fixed query domain")
            if variant["fraction"] == 1:
                if not np.array_equal(batch.values, original.values):
                    raise ValueError("full-retention export does not reproduce original features")
                control.append(model)
            previous = {r["key"]: r for r in baseline_rows[model]}
            rows = []
            for condition in primary._conditions(base):
                key = condition["key"]
                old_path = baseline / "predictions" / model / (key + ".npz")
                if sha(old_path) != previous[key]["prediction_sha256"]:
                    raise ValueError("baseline predictions changed")
                row = primary._predict(
                    stage,
                    name,
                    condition,
                    batch,
                    labels[condition["task"]],
                    query_indices,
                    indices,
                    readouts[model][key],
                    supports[key],
                )
                scored = True
                with (
                    np.load(old_path, allow_pickle=False) as old,
                    np.load(
                        stage / "predictions" / name / (key + ".npz"), allow_pickle=False
                    ) as new,
                ):
                    for field in ("truth", "tiles", "valid_indices"):
                        if not np.array_equal(old[field], new[field]):
                            raise ValueError("stress prediction changed truth or query positions")
                    delta = float(np.max(np.abs(new["scores"] - old["scores"]), initial=0))
                    if variant["fraction"] == 1 and delta != 0:
                        raise ValueError("full-retention predictions differ from frozen baseline")
                    row["maximum_absolute_score_change"] = delta
                row["baseline_metrics"] = previous[key]["metrics"]
                rows.append(row)
            results[name] = rows
            feature_identities[name] = batch.identity
            dump(stage / "status.json", {"state": "running", "variants_complete": len(results)})
        report = {
            "state": "complete",
            "protocol": PROTOCOL,
            "spec_sha256": sha(Path(spec_path)),
            "base_contract_sha256": primary.contract_sha256(base),
            "calibration_identity_sha256": spec["calibration_identity_sha256"],
            "baseline_identity_sha256": spec["baseline_identity_sha256"],
            "split": spec["split"],
            "test_scored": test,
            "parameters_refitted": False,
            "identity_control_verified": sorted(control),
            "features": feature_identities,
            "results": results,
            "implementation_sha256": sha(Path(__file__)),
            "readout_implementation": primary._code(),
        }
        dump(stage / "summary.json", report)
        dump(stage / "status.json", {"state": "complete", "test_scored": test})
        return report
    except BaseException as exc:
        dump(
            stage / "status.json",
            {"state": "failed", "error": repr(exc), "test_scored": test and scored},
        )
        raise
