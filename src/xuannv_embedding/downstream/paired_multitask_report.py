"""Identity-checked, paired task-family uncertainty for saved primary predictions."""

from __future__ import annotations

import itertools
from pathlib import Path

import numba
import numpy as np

from xuannv_embedding.downstream import multitask_bootstrap, paired_multitask, product_bootstrap
from xuannv_embedding.downstream.multitask_bootstrap import seed_metric_draws, tile_weights
from xuannv_embedding.export.context import dump, sha


def _summary(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values[1:])
    return {
        "observed": float(values[0]) if np.isfinite(values[0]) else None,
        "interval95": (np.percentile(values[1:], [2.5, 97.5]).tolist() if valid.all() else None),
        "defined_draws": int(valid.sum()),
        "total_draws": len(valid),
    }


def _condition_key(condition):
    return condition["family"], condition["task"], condition["budget"]


def _validate_conditions(conditions):
    tasks = {
        "C": paired_multitask.OSM_TASKS + paired_multitask.ESRI_TASKS,
        "R": paired_multitask.ESRI_TASKS,
        "Q": paired_multitask.OSM_TASKS,
    }
    keys = []
    for c in conditions:
        if (
            set(c) != {"family", "task", "source", "budget"}
            or c["family"] not in tasks
            or c["task"] not in tasks[c["family"]]
            or c["source"] != c["task"].split("_")[0]
            or type(c["budget"]) is not int
            or c["budget"] < 1
        ):
            raise ValueError("invalid task-family condition")
        keys.append(_condition_key(c))
    budgets = {
        family: {c["budget"] for c in conditions if c["family"] == family} for family in tasks
    }
    expected = {(f, t, b) for f, ts in tasks.items() for t in ts for b in budgets[f]}
    if (
        len(set(keys)) != len(keys)
        or set(keys) != expected
        or any(not b for b in budgets.values())
        or budgets["C"] != budgets["R"]
    ):
        raise ValueError("incomplete or duplicate registered task matrix")


def _family_mean(conditions, values, family):
    groups = ("osm", "esri") if family == "C" else ("esri" if family == "R" else "osm",)
    return np.mean(
        [
            values[
                [
                    i
                    for i, c in enumerate(conditions)
                    if c["family"] == family and c["source"] == source
                ]
            ].mean(0)
            for source in groups
        ],
        axis=0,
    )


def compare(conditions, baseline, candidate):
    """Summarize [task/budget, support seed, observed + shared bootstrap draws].

    Values already average metrics over model realizations, never predictions. Two-dimensional
    input represents one support seed. Normalize reference errors before averaging support seeds.
    This numerical rule is not a substitute for recipe, geography or multi-seed audits.
    """
    _validate_conditions(conditions)
    baseline, candidate = np.asarray(baseline, float), np.asarray(candidate, float)
    if baseline.ndim == 2:
        baseline = baseline[:, None, :]
    if candidate.ndim == 2:
        candidate = candidate[:, None, :]
    if (
        baseline.ndim != 3
        or baseline.shape != candidate.shape
        or baseline.shape[0] != len(conditions)
        or baseline.shape[1] < 1
        or baseline.shape[2] < 2
        or np.isinf(baseline).any()
        or np.isinf(candidate).any()
        or (baseline < 0).any()
        or (candidate < 0).any()
    ):
        raise ValueError("invalid paired metric arrays")
    ap = np.array([c["family"] != "R" for c in conditions])
    if (baseline[ap] > 1).any() or (candidate[ap] > 1).any():
        raise ValueError("AP must be in [0, 1]")
    base_error = np.where(ap[:, None, None], 1 - baseline, baseline)
    candidate_error = np.where(ap[:, None, None], 1 - candidate, candidate)
    normalized = ((base_error - candidate_error) / np.maximum(base_error, 1e-6)).mean(1)
    baseline, candidate = baseline.mean(1), candidate.mean(1)
    families, practical, nonnegative = {}, [], []
    for family in ("C", "R", "Q"):
        base = _family_mean(conditions, baseline, family)
        cand = _family_mean(conditions, candidate, family)
        gain = base - cand if family == "R" else cand - base
        record = {
            "metric": "rmse" if family == "R" else "ap",
            "baseline": _summary(base),
            "candidate": _summary(cand),
            "candidate_minus_baseline": _summary(cand - base),
            "improvement": _summary(gain),
        }
        nonnegative.append(np.isfinite(gain[0]) and gain[0] >= 0)
        if family == "R":
            relative = np.full(base.shape, np.nan)
            np.divide(gain, base, out=relative, where=base > 0)
            record["relative_improvement"] = _summary(relative)
            criterion, threshold = record["relative_improvement"], 0.02
        else:
            criterion, threshold = record["improvement"], 0.01
        meets = (
            criterion["observed"] is not None
            and criterion["observed"] >= threshold
            and criterion["interval95"] is not None
            and criterion["interval95"][0] > 0
        )
        record["practical_threshold"] = threshold
        record["practical_and_positive_interval"] = bool(meets)
        if meets:
            practical.append(family)
        families[family] = record
    normalized_families = {f: _family_mean(conditions, normalized, f) for f in ("C", "R", "Q")}
    conditions_report = [
        {
            **c,
            "baseline": _summary(baseline[i]),
            "candidate": _summary(candidate[i]),
            "candidate_minus_baseline": _summary(candidate[i] - baseline[i]),
        }
        for i, c in enumerate(conditions)
    ]
    return {
        "families": families,
        "conditions": conditions_report,
        "normalized_error_gain_by_family": {f: _summary(v) for f, v in normalized_families.items()},
        "normalized_error_gain": _summary(np.mean(list(normalized_families.values()), axis=0)),
        "near_zero_reference_errors": {
            "observed_support_conditions": int((base_error[:, :, 0] < 1e-6).sum()),
            "support_condition_draws": int((base_error[:, :, 1:] < 1e-6).sum()),
            "denominator_floor": 1e-6,
        },
        "numerical_primary_gate": {
            "met": bool(all(nonnegative) and practical),
            "all_observed_family_improvements_nonnegative": bool(all(nonnegative)),
            "families_meeting_practical_and_confidence_rule": practical,
            "scope": "primary numerical criteria only; does not certify the complete paper claim",
        },
    }


def _inputs(spec_path, expected):
    spec, cache = paired_multitask._spec(spec_path)
    root = Path(spec["output"])
    stage = root / "test"
    path = stage / "identity.json"
    if sha(path) != expected:
        raise ValueError("test identity changed")
    identity = paired_multitask._load(path)
    if (
        identity.get("state") != "complete"
        or identity.get("test_scored") is not True
        or identity.get("parameters_refitted") is not False
        or identity.get("contract_sha256") != paired_multitask.contract_sha256(spec)
        or identity.get("implementation") != paired_multitask._code()
        or identity.get("conditions") != paired_multitask._conditions(spec)
        or identity.get("method_groups") != spec["method_groups"]
        or paired_multitask._load(stage / "status.json").get("state") != "complete"
        or sha(root / "calibration/identity.json") != identity["calibration_identity_sha256"]
    ):
        raise ValueError("test producer contract or frozen calibration differs")
    for name, key in (
        ("results.json", "results_sha256"),
        ("common_valid.npy", "common_valid_sha256"),
        ("common_labels.npz", "common_labels_sha256"),
    ):
        if sha(stage / name) != identity[key]:
            raise ValueError("test result or common domain changed")
    rows = paired_multitask._load(stage / "results.json")
    if set(rows) != set(spec["models"]):
        raise ValueError("test model set differs")
    indexed = {}
    for model, records in rows.items():
        by_key = {r["key"]: r for r in records}
        if len(by_key) != len(records) or set(by_key) != {c["key"] for c in identity["conditions"]}:
            raise ValueError("test condition set differs")
        for c in identity["conditions"]:
            row = by_key[c["key"]]
            if any(row[k] != value for k, value in c.items()):
                raise ValueError("test condition metadata differs")
            if sha(stage / "predictions" / model / (c["key"] + ".npz")) != row["prediction_sha256"]:
                raise ValueError("test prediction digest changed")
        indexed[model] = by_key
    return spec, cache, stage, indexed


def _prediction(path, canonical_tiles):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"scores", "truth", "tiles", "valid_indices"}:
            raise ValueError("invalid saved prediction fields")
        score, truth, tiles, positions = (
            data[k] for k in ("scores", "truth", "tiles", "valid_indices")
        )
    if (
        score.ndim != 1
        or score.shape != truth.shape
        or score.shape != tiles.shape
        or not np.issubdtype(tiles.dtype, np.integer)
        or not np.isin(tiles, canonical_tiles).all()
        or positions.ndim != 1
        or not np.issubdtype(positions.dtype, np.integer)
    ):
        raise ValueError("saved prediction geometry differs")
    lookup = np.full(max(canonical_tiles) + 1, -1, dtype=np.int64)
    lookup[canonical_tiles] = np.arange(len(canonical_tiles))
    local_tiles = lookup[tiles]
    return score, truth, local_tiles, positions


def _verified_metric(observed, recorded):
    if recorded is None:
        if not np.isnan(observed):
            raise ValueError("saved undefined metric differs from recomputation")
    elif not np.isfinite(observed) or abs(observed - recorded) > 1e-12:
        raise ValueError("saved metric differs from independent prediction recomputation")


def run(spec_path, test_identity_sha256, output, *, threads=2, repeats=2000, seed=20260921):
    """Read archived predictions only; fixed CLI resampling, API overrides for fixtures."""
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("use one to four bootstrap threads")
    spec, cache, stage, rows = _inputs(spec_path, test_identity_sha256)
    canonical = cache["split"]["test"]
    tile_ids = [cache["records"][i]["patch_id"] for i in canonical]
    weights = tile_weights(tile_ids, repeats=repeats, seed=seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prior_threads = numba.get_num_threads()
    dump(output / "status.json", {"state": "running", "conditions_complete": 0})
    try:
        numba.set_num_threads(threads)
        np.save(output / "tile_weights.npy", weights)
        conditions = list(
            {
                _condition_key(c): {k: v for k, v in c.items() if k not in ("key", "seed")}
                for c in paired_multitask._conditions(spec)
            }.values()
        )
        _validate_conditions(conditions)
        seeds, groups = spec["support_seeds"], spec["method_groups"]
        means = {method: [] for method in groups}
        observations, saved = [], {}
        domains = {}
        for number, condition in enumerate(conditions):
            family, task, budget = _condition_key(condition)
            metric = "rmse" if family == "R" else "ap"
            for method_index, (method, models) in enumerate(groups.items()):
                predictions, records = [], []
                for model in models:
                    prediction_seeds, record_seeds = [], []
                    for support_seed in seeds:
                        key = f"{family}_{task}_{support_seed}_{budget}"
                        score, truth, tiles, positions = _prediction(
                            stage / "predictions" / model / (key + ".npz"), canonical
                        )
                        domain = multitask_bootstrap._digest(truth, tiles, positions)
                        # Domains must also match between budgets and support realizations.
                        if domains.setdefault((family, task), domain) != domain:
                            raise ValueError(
                                "paired prediction truth, positions or tile domains differ"
                            )
                        prediction_seeds.append(score)
                        record_seeds.append(rows[model][key]["metrics"][metric])
                    predictions.append(prediction_seeds)
                    records.append(record_seeds)
                draws = seed_metric_draws(
                    truth,
                    np.asarray(predictions),
                    tiles,
                    weights,
                    tile_ids=tile_ids,
                    training_seeds=list(range(len(models))),
                    support_seeds=seeds,
                    metric=metric,
                )
                for i, record in enumerate(records):
                    for j, value in enumerate(record):
                        _verified_metric(draws.observed[i, j], value)
                means[method].append(
                    np.concatenate([draws.observed.mean(0)[:, None], draws.draws.mean(0)], axis=1)
                )
                name = f"{method_index:02d}_{family}_{task}_{budget}.npz"
                np.savez_compressed(output / name, observed=draws.observed, draws=draws.draws)
                saved[name] = sha(output / name)
                observations.append(
                    {
                        **condition,
                        "method": method,
                        "model_realizations": models,
                        "support_seeds": seeds,
                        "observed_by_realization_and_support": [
                            [float(v) if np.isfinite(v) else None for v in row]
                            for row in draws.observed
                        ],
                        "domain_sha256": domain,
                        "draws_file": name,
                    }
                )
            dump(output / "status.json", {"state": "running", "conditions_complete": number + 1})
        arrays = {method: np.asarray(values) for method, values in means.items()}
        np.savez_compressed(output / "draws.npz", **arrays)
        comparisons = [
            {"baseline": a, "candidate": b, **compare(conditions, arrays[a], arrays[b])}
            for a, b in itertools.permutations(groups, 2)
        ]
        result = {
            "state": "complete",
            "protocol": "paired-primary-uncertainty-v1",
            "conditions": len(conditions),
            "observations": observations,
            "comparisons": comparisons,
            "method_groups": groups,
            "resampling": {
                "repeats": repeats,
                "seed": seed,
                "tile_ids": tile_ids,
                "registered_schedule": repeats == 2000 and seed == 20260921,
                "weights_sha256": sha(output / "tile_weights.npy"),
            },
            "refitted": False,
            "new_feature_data_read": False,
            "aggregation": (
                "metrics over realizations/support seeds within each common tile draw; "
                "C sources equally weighted"
            ),
            "scope": (
                "primary numerical evidence only; model recipe, multiple training seed "
                "provenance, strong heads and auxiliary evaluations need separate audit"
            ),
            "spec_sha256": sha(Path(spec_path)),
            "test_identity_sha256": test_identity_sha256,
            "contract_sha256": paired_multitask.contract_sha256(spec),
            "draws_sha256": sha(output / "draws.npz"),
            "individual_draws_sha256": saved,
            "implementation_sha256": {
                "report": sha(Path(__file__)),
                "metrics": sha(Path(multitask_bootstrap.__file__)),
                "weighted_ap": sha(Path(product_bootstrap.__file__)),
            },
            "runtime": {"numpy": np.__version__, "numba": numba.__version__},
        }
        dump(output / "summary.json", result)
        dump(
            output / "status.json",
            {
                "state": "complete",
                "conditions_complete": len(conditions),
                "summary_sha256": sha(output / "summary.json"),
            },
        )
        return result
    except BaseException as exc:
        dump(output / "status.json", {"state": "failed", "error": repr(exc)})
        raise
    finally:
        numba.set_num_threads(prior_threads)
