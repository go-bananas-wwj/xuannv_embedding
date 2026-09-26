"""Paired tile uncertainty for registered heads, using archived predictions only."""

from __future__ import annotations

import itertools
from pathlib import Path

import numba
import numpy as np

from xuannv_embedding.downstream import multitask_bootstrap, product_bootstrap
from xuannv_embedding.downstream import paired_multitask_report as primary_report
from xuannv_embedding.downstream import strong_multitask as workflow
from xuannv_embedding.export.context import dump, sha


def conditions(cohort, heads=workflow.HEADS):
    return list(
        {
            (c["head"], c["task"], c["budget"]): {
                k: v for k, v in c.items() if k not in ("key", "seed")
            }
            for c in workflow._conditions(cohort, heads)
        }.values()
    )


def compare(registered, baseline, candidate, *, heads=workflow.HEADS):
    """Compare AP arrays [head/task/budget, support seed, observed + paired draws]."""
    selected = workflow.selected_heads(heads)
    tasks = workflow.primary.OSM_TASKS + workflow.primary.ESRI_TASKS
    keys, budgets = [], set()
    for c in registered:
        if (
            set(c) != {"head", "family", "task", "source", "budget"}
            or c["head"] not in selected
            or c["family"] != "C"
            or c["task"] not in tasks
            or c["source"] != c["task"].split("_")[0]
            or type(c["budget"]) is not int
            or c["budget"] < 1
        ):
            raise ValueError("invalid strong-head condition")
        keys.append((c["head"], c["task"], c["budget"]))
        budgets.add(c["budget"])
    expected = {(h, t, b) for h in selected for t in tasks for b in budgets}
    if not budgets or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("incomplete or duplicate registered-head matrix")
    a, b = np.asarray(baseline, float), np.asarray(candidate, float)
    if (
        a.ndim != 3
        or a.shape != b.shape
        or a.shape[0] != len(registered)
        or a.shape[1] < 1
        or a.shape[2] < 2
        or any(np.isinf(x).any() or (x < 0).any() or (x > 1).any() for x in (a, b))
    ):
        raise ValueError("invalid paired strong-head AP arrays")
    a, b = a.mean(1), b.mean(1)  # Average metrics, never predictions; undefined stays undefined.

    def summary(x, y):
        return {
            "baseline": primary_report._summary(x),
            "candidate": primary_report._summary(y),
            "candidate_minus_baseline": primary_report._summary(y - x),
        }

    heads = {}
    for head in selected:
        sources, arrays = {}, []
        for source in ("osm", "esri"):
            row_indices = [
                i for i, c in enumerate(registered) if c["head"] == head and c["source"] == source
            ]
            x, y = a[row_indices].mean(0), b[row_indices].mean(0)
            sources[source] = summary(x, y)
            arrays.append((x, y))
        x, y = np.mean(arrays, axis=0)
        heads[head] = {**summary(x, y), "sources": sources}
    return {
        "heads": heads,
        "conditions": [{**c, **summary(a[i], b[i])} for i, c in enumerate(registered)],
        "scope": "supplementary per-head classification evidence; no automatic paper-claim gate",
    }


def _stage(root, name, expected, spec, cohort):
    stage = root / name
    if sha(stage / "identity.json") != expected:
        raise ValueError("strong stage identity changed")
    identity = workflow.primary._load(stage / "identity.json")
    if (
        identity.get("state") != "complete"
        or identity.get("protocol") != workflow.PROTOCOL
        or identity.get("test_scored") is not (name == "test")
        or (name == "test" and identity.get("parameters_refitted") is not False)
        or identity.get("contract_sha256") != workflow.contract_sha256(spec)
        or identity.get("primary_contract_sha256") != workflow.primary.contract_sha256(cohort)
        or identity.get("implementation") != workflow._code()
        or identity.get("heads") != list(spec["heads"])
        or identity.get("conditions") != workflow._conditions(cohort, spec["heads"])
        or identity.get("method_groups") != cohort["method_groups"]
        or workflow.primary._load(stage / "status.json").get("state") != "complete"
    ):
        raise ValueError("strong stage contract or completion differs")
    for filename, key in (
        ("results.json", "results_sha256"),
        ("common_valid.npy", "common_valid_sha256"),
        ("common_labels.npz", "common_labels_sha256"),
    ):
        if sha(stage / filename) != identity[key]:
            raise ValueError("strong stage data or report changed")
    return identity


def _payload(directory, record, head):
    if sha(directory / "identity.json") != record["readout_identity_sha256"]:
        raise ValueError("saved readout identity changed")
    identity = workflow.primary._load(directory / "identity.json")
    neural = head in workflow.neural_readouts.KINDS
    producer = workflow.neural_readouts if neural else workflow.strong_classifiers
    if (
        identity.get("format") != producer.FORMAT
        or identity.get("kind") != head
        or identity.get("implementation") != producer._implementation()
        or record.get("validation_replay", {}).get("state") != "verified"
    ):
        raise ValueError("saved readout producer or replay differs")
    payloads = {"parameters.npz": identity["payload_sha256"]} if neural else identity["payloads"]
    expected = {"parameters.npz"}
    if not neural and head != "knn":
        expected.add("estimator.joblib")
    if set(payloads) != expected or any(sha(directory / p) != h for p, h in payloads.items()):
        raise ValueError("saved readout payload changed")
    # Hash numeric/pickle bytes only. Reporting never deserializes models or initializes NPU.


def _inputs(spec_path, expected):
    spec, cohort, cache = workflow._spec(spec_path)
    root = Path(spec["output"])
    identity = _stage(root, "test", expected, spec, cohort)
    calibrated = _stage(root, "calibration", identity["calibration_identity_sha256"], spec, cohort)
    cal = root / "calibration"
    for c in workflow._conditions(cohort, spec["heads"]):
        key, head = c["key"], c["head"]
        record = calibrated["readouts"][key]
        directory = cal / "readouts" / key
        if (
            sha(directory / "support.json") != record["support_sha256"]
            or sha(directory / "positions.npy") != record["positions_sha256"]
            or set(record["models"]) != set(cohort["models"])
        ):
            raise ValueError("strong calibration support changed")
        for model, value in record["models"].items():
            _payload(directory / model, value, head)
            if (
                sha(cal / "predictions" / model / (key + ".npz"))
                != value["validation_prediction_sha256"]
            ):
                raise ValueError("strong validation prediction changed")
    stage = root / "test"
    rows = workflow.primary._load(stage / "results.json")
    if set(rows) != set(cohort["models"]):
        raise ValueError("strong test model set differs")
    indexed = {}
    for model, records in rows.items():
        by_key = {r["key"]: r for r in records}
        if len(by_key) != len(records) or set(by_key) != {c["key"] for c in identity["conditions"]}:
            raise ValueError("strong test condition set differs")
        for c in identity["conditions"]:
            row = by_key[c["key"]]
            if any(row[k] != v for k, v in c.items()):
                raise ValueError("strong test condition metadata differs")
            if sha(stage / "predictions" / model / (c["key"] + ".npz")) != row["prediction_sha256"]:
                raise ValueError("strong test prediction changed")
        indexed[model] = by_key
    return spec, cohort, cache, stage, indexed


def _domains(stage, cache):
    count, size = len(cache["split"]["test"]), cache["data"]["patch_size"]
    valid = np.load(stage / "common_valid.npy", allow_pickle=False)
    if valid.dtype != np.bool_ or valid.shape != (count, size, size):
        raise ValueError("invalid archived common feature domain")
    tasks = workflow.primary.OSM_TASKS + workflow.primary.ESRI_TASKS
    domains = {}
    with np.load(stage / "common_labels.npz", allow_pickle=False) as data:
        if set(data.files) != set(tasks):
            raise ValueError("archived common task labels differ")
        for task in tasks:
            y = data[task]
            if (
                y.shape != valid.shape
                or not np.isin(y, [-1, 0, 1]).all()
                or (y[~valid] != -1).any()
            ):
                raise ValueError("archived labels violate the common feature domain")
            positions = np.flatnonzero(y.ravel() >= 0)
            tiles = np.repeat(np.arange(count), size * size)[positions]
            domains[task] = (y.ravel()[positions], tiles, positions)
    return domains


def run(spec_path, test_identity_sha256, output, *, threads=2, repeats=2000, seed=20260921):
    """Fixed CLI resampling; explicit API overrides are only for synthetic checks."""
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("use one to four bootstrap threads")
    spec, cohort, cache, stage, rows = _inputs(spec_path, test_identity_sha256)
    canonical = cache["split"]["test"]
    tile_ids = [cache["records"][i]["patch_id"] for i in canonical]
    weights = multitask_bootstrap.tile_weights(tile_ids, repeats=repeats, seed=seed)
    domains = _domains(stage, cache)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prior_threads = numba.get_num_threads()
    dump(output / "status.json", {"state": "running", "conditions_complete": 0})
    try:
        numba.set_num_threads(threads)
        np.save(output / "tile_weights.npy", weights)
        registered = conditions(cohort, spec["heads"])
        seeds, groups = cohort["support_seeds"], cohort["method_groups"]
        means = {method: [] for method in groups}
        observations, saved = [], {}
        for number, c in enumerate(registered):
            task, head, budget = c["task"], c["head"], c["budget"]
            truth, tiles, positions = domains[task]
            for method_index, (method, models) in enumerate(groups.items()):
                predictions, recorded = [], []
                for model in models:
                    scores, metrics = [], []
                    for support_seed in seeds:
                        key = f"C_{task}_{support_seed}_{budget}_{head}"
                        score, target, tile, position = primary_report._prediction(
                            stage / "predictions" / model / (key + ".npz"), canonical
                        )
                        if any(
                            not np.array_equal(a, b)
                            for a, b in ((truth, target), (tiles, tile), (positions, position))
                        ):
                            raise ValueError("prediction domain differs from archived common truth")
                        scores.append(score)
                        metrics.append(rows[model][key]["metrics"]["ap"])
                    predictions.append(scores)
                    recorded.append(metrics)
                draws = multitask_bootstrap.seed_metric_draws(
                    truth,
                    np.asarray(predictions),
                    tiles,
                    weights,
                    tile_ids=tile_ids,
                    training_seeds=list(range(len(models))),
                    support_seeds=seeds,
                    metric="ap",
                )
                for i, values in enumerate(recorded):
                    for j, value in enumerate(values):
                        primary_report._verified_metric(draws.observed[i, j], value)
                means[method].append(
                    np.concatenate([draws.observed.mean(0)[:, None], draws.draws.mean(0)], axis=1)
                )
                name = f"{method_index:02d}_{head}_{task}_{budget}.npz"
                np.savez_compressed(output / name, observed=draws.observed, draws=draws.draws)
                saved[name] = sha(output / name)
                observations.append(
                    {
                        **c,
                        "method": method,
                        "model_realizations": models,
                        "support_seeds": seeds,
                        "observed_by_realization_and_support": [
                            [float(v) if np.isfinite(v) else None for v in row]
                            for row in draws.observed
                        ],
                        "domain_sha256": multitask_bootstrap._digest(truth, tiles, positions),
                        "draws_file": name,
                    }
                )
            dump(output / "status.json", {"state": "running", "conditions_complete": number + 1})
        arrays = {method: np.asarray(values) for method, values in means.items()}
        np.savez_compressed(output / "draws.npz", **arrays)
        comparisons = [
            {
                "baseline": a,
                "candidate": b,
                **compare(registered, arrays[a], arrays[b], heads=spec["heads"]),
            }
            for a, b in itertools.permutations(groups, 2)
        ]
        result = {
            "state": "complete",
            "protocol": "paired-strong-uncertainty-v1",
            "conditions": len(registered),
            "heads": list(spec["heads"]),
            "comparisons": comparisons,
            "observations": observations,
            "method_groups": groups,
            "resampling": {
                "repeats": repeats,
                "seed": seed,
                "tile_ids": tile_ids,
                "registered_schedule": repeats == 2000 and seed == 20260921,
                "weights_sha256": sha(output / "tile_weights.npy"),
                "interval": "pointwise 95% percentile; no multiplicity correction",
            },
            "refitted": False,
            "new_feature_data_read": False,
            "models_deserialized": False,
            "aggregation": (
                "mean AP over realizations/support seeds; OSM/ESRI equally weighted per head"
            ),
            "scope": (
                "supplementary classification evidence; no best-head selection or paper-claim gate"
            ),
            "spec_sha256": sha(Path(spec_path)),
            "test_identity_sha256": test_identity_sha256,
            "contract_sha256": workflow.contract_sha256(spec),
            "draws_sha256": sha(output / "draws.npz"),
            "individual_draws_sha256": saved,
            "implementation_sha256": {
                "report": sha(Path(__file__)),
                "workflow": workflow._code(),
                "primary_report": sha(Path(primary_report.__file__)),
                "metrics": sha(Path(multitask_bootstrap.__file__)),
                "weighted_ap": sha(Path(product_bootstrap.__file__)),
            },
            "runtime": {"numpy": np.__version__, "numba": numba.__version__},
        }
        dump(output / "summary.json", result)
        dump(
            output / "status.json",
            {"state": "complete", "summary_sha256": sha(output / "summary.json")},
        )
        return result
    except BaseException as exc:
        dump(output / "status.json", {"state": "failed", "error": repr(exc)})
        raise
    finally:
        numba.set_num_threads(prior_threads)
