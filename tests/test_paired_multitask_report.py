import json
from pathlib import Path

import numpy as np
import pytest
from test_paired_multitask import fixture_spec, write_json

from xuannv_embedding.downstream import paired_multitask as workflow
from xuannv_embedding.downstream import paired_multitask_report as reporting
from xuannv_embedding.downstream.multitask import normalized_score
from xuannv_embedding.export.context import sha


def numerical_conditions():
    spec = {"support_seeds": [7], "budgets": [1], "retrieval_budgets": [1]}
    return [
        {k: v for k, v in c.items() if k not in ("key", "seed")} for c in workflow._conditions(spec)
    ]


def numerical_values():
    conditions = numerical_conditions()
    base = np.array([0.4 if c["family"] == "R" else 0.5 for c in conditions])
    candidate = np.array(
        [
            (
                0.39
                if c["family"] == "R"
                else 0.51 if c["family"] == "Q" else 0.7 if c["source"] == "osm" else 0.5
            )
            for c in conditions
        ]
    )
    return conditions, np.repeat(base[:, None], 4, 1), np.repeat(candidate[:, None], 4, 1)


def test_family_aggregation_balances_sources_and_applies_practical_rule():
    conditions, base, candidate = numerical_values()
    result = reporting.compare(conditions, base, candidate)
    assert result["families"]["C"]["improvement"]["observed"] == pytest.approx(0.1)
    assert result["families"]["R"]["improvement"]["observed"] == pytest.approx(0.01)
    assert result["families"]["R"]["relative_improvement"]["observed"] == pytest.approx(0.025)
    assert result["families"]["Q"]["improvement"]["observed"] == pytest.approx(0.01)
    assert result["numerical_primary_gate"]["met"]
    # C contributes 0.2, R 0.025, Q 0.02 to the normalized score.
    assert result["normalized_error_gain"]["observed"] == pytest.approx((0.2 + 0.025 + 0.02) / 3)


def test_family_means_are_computed_within_each_paired_draw():
    conditions, base, candidate = numerical_values()
    indices = [i for i, c in enumerate(conditions) if c["family"] == "C"]
    candidate[indices, 1] = 0.4
    candidate[indices, 2] = 0.6
    candidate[indices, 3] = 0.9
    result = reporting.compare(conditions, base, candidate)
    np.testing.assert_allclose(
        result["families"]["C"]["improvement"]["interval95"],
        np.percentile([-0.1, 0.1, 0.4], [2.5, 97.5]),
    )


def test_normalized_score_keeps_per_support_reference_errors_before_averaging():
    conditions, base, candidate = numerical_values()
    base = np.repeat(base[:, None, :], 2, axis=1)
    candidate = np.repeat(candidate[:, None, :], 2, axis=1)
    c = [i for i, item in enumerate(conditions) if item["family"] == "C"]
    base[c, 0] = 0.2
    base[c, 1] = 0.8
    candidate[c, 0] = 0.3
    candidate[c, 1] = 0.85
    result = reporting.compare(conditions, base, candidate)
    assert result["families"]["C"]["improvement"]["observed"] == pytest.approx(0.075)
    assert result["normalized_error_gain_by_family"]["C"]["observed"] == pytest.approx(0.1875)
    rows = []
    for array in (base, candidate):
        rows.append(
            [
                {
                    "key": f"{i}_{seed}",
                    "family": item["family"],
                    "source": item["source"],
                    "error": array[i, seed, 0] if item["family"] == "R" else 1 - array[i, seed, 0],
                }
                for i, item in enumerate(conditions)
                for seed in range(2)
            ]
        )
    assert result["normalized_error_gain"]["observed"] == pytest.approx(
        normalized_score(*rows)["score"]
    )


def test_undefined_draws_are_not_dropped_and_regression_zero_reference_is_explicit():
    conditions, base, candidate = numerical_values()
    candidate[0, 1] = np.nan
    r = [i for i, c in enumerate(conditions) if c["family"] == "R"]
    base[r] = candidate[r] = 0
    result = reporting.compare(conditions, base, candidate)
    assert result["families"]["C"]["improvement"]["interval95"] is None
    assert result["families"]["C"]["improvement"]["defined_draws"] == 2
    assert result["families"]["R"]["relative_improvement"]["observed"] is None
    assert result["normalized_error_gain"]["interval95"] is None


def test_a_degraded_family_prevents_primary_gate_even_with_large_classification_gain():
    conditions, base, candidate = numerical_values()
    for i, c in enumerate(conditions):
        if c["family"] == "Q":
            candidate[i] = 0.49
    assert not reporting.compare(conditions, base, candidate)["numerical_primary_gate"]["met"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "source", "shape", "infinity"])
def test_incomplete_or_invalid_task_matrix_is_rejected(change):
    conditions, base, candidate = numerical_values()
    if change == "missing":
        conditions, base, candidate = conditions[:-1], base[:-1], candidate[:-1]
    elif change == "duplicate":
        conditions[-1] = conditions[0]
    elif change == "source":
        conditions[0]["source"] = "esri"
    elif change == "shape":
        candidate = candidate[:, :-1]
    else:
        candidate[0, 0] = np.inf
    with pytest.raises(ValueError):
        reporting.compare(conditions, base, candidate)


def test_file_backed_report_uses_saved_predictions_without_features_or_refitting(
    tmp_path, monkeypatch
):
    path, spec, test_paths = fixture_spec(tmp_path)
    # A published baseline can have one realization and the candidate several.
    spec["models"]["replica"] = dict(spec["models"]["candidate"])
    spec["method_groups"]["candidate"].append("replica")
    geographic_path = Path(spec["geographic_audit"]["path"])
    geographic = json.loads(geographic_path.read_text())
    geographic["model_manifest_sha256"]["replica"] = spec["models"]["replica"]["manifest_sha256"]
    spec["geographic_audit"] = write_json(geographic_path, geographic)
    spec["lock"] = write_json(
        Path(spec["lock"]["path"]),
        {"state": "locked", "contract_sha256": workflow.contract_sha256(spec)},
    )
    path.write_text(json.dumps(spec))
    workflow.calibrate(path)
    root = Path(spec["output"])
    workflow.score(path, sha(root / "calibration/identity.json"))
    identity = root / "test/identity.json"
    for p in test_paths:
        p.rename(p.with_suffix(".withheld"))
    before = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("uncertainty reporting cannot fit or load new feature data")

    monkeypatch.setattr(workflow, "read_features", forbidden)
    monkeypatch.setattr(workflow, "fit_classification", forbidden)
    out = tmp_path / "report"
    result = reporting.run(path, sha(identity), out, repeats=20)
    assert len(result["comparisons"]) == 2
    assert result["resampling"]["registered_schedule"] is False
    assert result["conditions"] == 20
    assert result["refitted"] is False
    assert before == {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    assert (out / "draws.npz").is_file()
    with np.load(out / "01_C_osm_building_1.npz") as draws:
        assert draws["observed"].shape == (2, 1)
    assert json.loads((out / "status.json").read_text())["state"] == "complete"
    with pytest.raises(FileExistsError):
        reporting.run(path, sha(identity), out, repeats=20)


def test_prediction_tampering_is_rejected_before_bootstrap(tmp_path, monkeypatch):
    path, spec, _ = fixture_spec(tmp_path)
    workflow.calibrate(path)
    root = Path(spec["output"])
    workflow.score(path, sha(root / "calibration/identity.json"))
    prediction = next((root / "test/predictions/base").glob("*.npz"))
    prediction.write_bytes(prediction.read_bytes() + b"changed")

    def forbidden(*args, **kwargs):
        raise AssertionError("tampered inputs must fail before metric resampling")

    monkeypatch.setattr(reporting, "seed_metric_draws", forbidden)
    with pytest.raises(ValueError, match="prediction"):
        reporting.run(path, sha(root / "test/identity.json"), tmp_path / "report", repeats=20)


def test_resigned_predictions_with_different_positions_are_not_a_paired_domain(tmp_path):
    path, spec, _ = fixture_spec(tmp_path)
    workflow.calibrate(path)
    root = Path(spec["output"])
    workflow.score(path, sha(root / "calibration/identity.json"))
    stage = root / "test"
    prediction = stage / "predictions/candidate/C_osm_building_7_1.npz"
    with np.load(prediction) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["valid_indices"][0] += 1
    np.savez(prediction, **arrays)
    result = json.loads((stage / "results.json").read_text())
    result["candidate"][0]["prediction_sha256"] = sha(prediction)
    (stage / "results.json").write_text(json.dumps(result))
    identity = json.loads((stage / "identity.json").read_text())
    identity["results_sha256"] = sha(stage / "results.json")
    (stage / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="domains differ"):
        reporting.run(path, sha(stage / "identity.json"), tmp_path / "report", repeats=20)


def test_report_cli_has_registered_defaults_and_dispatch(tmp_path, monkeypatch):
    from xuannv_embedding.training.experiment import main

    calls = []
    monkeypatch.setattr(reporting, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    path, out = tmp_path / "spec.json", tmp_path / "report"
    assert (
        main(
            [
                "summarize-primary",
                "--spec",
                str(path),
                "--test-identity-sha256",
                "a" * 64,
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert calls == [((path, "a" * 64, out), {"threads": 2})]
