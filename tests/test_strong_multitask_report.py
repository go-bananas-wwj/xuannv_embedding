import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_paired_multitask import write_json
from test_strong_multitask import strong_spec

from xuannv_embedding.downstream import strong_multitask as workflow
from xuannv_embedding.downstream import strong_multitask_report as reporting
from xuannv_embedding.export.context import sha


def conditions():
    return reporting.conditions({"support_seeds": [7], "budgets": [1], "retrieval_budgets": [1]})


def test_sources_are_balanced_and_all_heads_and_conditions_are_retained():
    c = conditions()
    a = np.full((len(c), 2, 5), 0.2)
    b = a.copy()
    for i, row in enumerate(c):
        b[i] += 0.2 if row["source"] == "osm" else 0.4
    result = reporting.compare(c, a, b)
    assert set(result["heads"]) == set(workflow.HEADS)
    assert len(result["conditions"]) == 50
    for value in result["heads"].values():
        assert value["candidate_minus_baseline"]["observed"] == pytest.approx(0.3)
        np.testing.assert_allclose(value["candidate_minus_baseline"]["interval95"], [0.3, 0.3])
        assert value["sources"]["osm"]["candidate_minus_baseline"]["observed"] == pytest.approx(0.2)
    assert "numerical_primary_gate" not in result


def test_pairing_uses_same_draws_and_undefined_task_is_not_dropped():
    c = conditions()
    a = np.broadcast_to([0.2, 0.8, 0.1, 0.5], (len(c), 2, 4)).copy()
    b = a + 0.1
    index = next(i for i, v in enumerate(c) if v["head"] == "rf" and v["source"] == "osm")
    b[index, 0, 2] = np.nan
    result = reporting.compare(c, a, b)
    assert result["heads"]["rf"]["candidate_minus_baseline"]["interval95"] is None
    assert result["heads"]["rf"]["candidate_minus_baseline"]["defined_draws"] == 2
    for head in ("svm", "knn", "mlp", "conv3x3"):
        np.testing.assert_allclose(
            result["heads"][head]["candidate_minus_baseline"]["interval95"], [0.1, 0.1]
        )


@pytest.mark.parametrize("change", ["missing", "duplicate", "negative", "infinite", "shape"])
def test_invalid_or_incomplete_strong_matrix_is_rejected(change):
    c = conditions()
    a = np.full((50, 2, 4), 0.5)
    b = a.copy()
    if change == "missing":
        c = c[:-1]
        a = a[:-1]
        b = b[:-1]
    if change == "duplicate":
        c[-1] = c[0]
    if change == "negative":
        b[0, 0, 0] = -0.1
    if change == "infinite":
        b[0, 0, 0] = np.inf
    if change == "shape":
        b = b[:, 0]
    with pytest.raises(ValueError):
        reporting.compare(c, a, b)


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("strong_reporting")
    path, spec, primary, _ = strong_spec(tmp)
    # Two distinct synthetic candidate realizations with five channels, one three-channel base.
    model = dict(primary["models"]["candidate"])
    manifest = json.loads(Path(model["manifest_path"]).read_text())
    model["tile_sha256"] = {}
    for i, record in enumerate(manifest["records"]):
        with np.load(record["path"], allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["embedding"] = -arrays["embedding"]
        target = tmp / f"replica_{i}.npz"
        np.savez(target, **arrays)
        record["path"] = str(target)
        model["tile_sha256"][record["patch_id"]] = sha(target)
    ref = write_json(tmp / "replica.json", manifest)
    model.update(manifest_path=ref["path"], manifest_sha256=ref["sha256"])
    primary["models"]["replica"] = model
    primary["method_groups"] = {"base": ["base"], "candidate": ["candidate", "replica"]}
    geo_path = Path(primary["geographic_audit"]["path"])
    geo = json.loads(geo_path.read_text())
    geo["model_manifest_sha256"]["replica"] = ref["sha256"]
    primary["geographic_audit"] = write_json(geo_path, geo)
    primary["lock"] = write_json(
        tmp / "lock.json",
        {"state": "locked", "contract_sha256": workflow.primary.contract_sha256(primary)},
    )
    spec["primary_spec"] = write_json(Path(spec["primary_spec"]["path"]), primary)
    spec["lock"] = write_json(
        tmp / "strong-lock.json",
        {"state": "locked", "contract_sha256": workflow.contract_sha256(spec)},
    )
    path.write_text(json.dumps(spec))
    workflow.calibrate(path)
    root = Path(spec["output"])
    workflow.score(path, sha(root / "calibration/identity.json"))
    for record in primary["labels"].values():
        Path(record["path"]).rename(Path(record["path"]).with_suffix(".unavailable"))
    for model in primary["models"].values():
        for record in json.loads(Path(model["manifest_path"]).read_text())["records"]:
            p = Path(record["path"])
            p.rename(p.with_suffix(".unavailable"))
    return path, spec


@pytest.fixture
def copied(archive, tmp_path):
    _, spec = archive
    new = dict(spec, output=str(tmp_path / "run"))
    shutil.copytree(spec["output"], new["output"])
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(new))
    return path, new, tmp_path / "report"


def test_report_uses_saved_predictions_without_deserializing_or_refitting(copied, monkeypatch):
    path, spec, out = copied
    root = Path(spec["output"])

    def forbidden(*a, **k):
        raise AssertionError("unexpected source/model access")

    monkeypatch.setattr(workflow, "_calibration", forbidden)
    monkeypatch.setattr(workflow.strong_classifiers, "load_classifier", forbidden)
    monkeypatch.setattr(workflow.neural_readouts, "load_neural", forbidden)
    monkeypatch.setattr(workflow.primary, "_common", forbidden)
    before = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
    result = reporting.run(path, sha(root / "test/identity.json"), out, repeats=8)
    assert result["state"] == "complete" and result["refitted"] is False
    assert not result["resampling"]["registered_schedule"]
    assert len(result["comparisons"]) == 2
    rows = json.loads((root / "test/results.json").read_text())
    head = result["comparisons"][0]["heads"]["rf"]
    expected = np.mean(
        [
            r["metrics"]["ap"]
            for name in ["candidate", "replica"]
            for r in rows[name]
            if r["head"] == "rf" and r["source"] == "esri"
        ]
    )
    assert head["sources"]["esri"]["candidate"]["observed"] == pytest.approx(expected)
    assert head["candidate"]["observed"] is None  # No osm_green truth in the fixture.
    assert head["candidate"]["interval95"] is None
    obs = next(v for v in result["observations"] if v["method"] == "candidate")
    assert obs["model_realizations"] == ["candidate", "replica"]
    with np.load(out / obs["draws_file"]) as data:
        assert data["observed"].shape == (2, 1)
    assert before == {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind", ["payload", "prediction", "calibration", "support"])
def test_modified_saved_evidence_is_rejected(copied, kind):
    path, spec, out = copied
    root = Path(spec["output"])
    digest = sha(root / "test/identity.json")
    name = {
        "payload": "calibration/readouts/C_osm_building_7_1_rf/base/estimator.joblib",
        "prediction": "test/predictions/base/C_osm_building_7_1_rf.npz",
        "calibration": "calibration/identity.json",
        "support": "calibration/readouts/C_osm_building_7_1_rf/positions.npy",
    }[kind]
    p = root / name
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        reporting.run(path, digest, out, repeats=4)
    assert not out.exists()


@pytest.mark.parametrize("kind", ["positions", "metric"])
def test_independent_domain_and_metric_checks_reject_internally_rehashed_errors(copied, kind):
    path, spec, out = copied
    stage = Path(spec["output"]) / "test"
    rows = json.loads((stage / "results.json").read_text())
    row = next(r for r in rows["base"] if r["key"] == "C_osm_building_7_1_rf")
    if kind == "positions":
        p = stage / "predictions/base" / f"{row['key']}.npz"
        with np.load(p) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["valid_indices"] = arrays["valid_indices"][::-1]
        np.savez_compressed(p, **arrays)
        row["prediction_sha256"] = sha(p)
    else:
        row["metrics"]["ap"] += 0.1
    (stage / "results.json").write_text(json.dumps(rows))
    identity = json.loads((stage / "identity.json").read_text())
    identity["results_sha256"] = sha(stage / "results.json")
    (stage / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError):
        reporting.run(path, sha(stage / "identity.json"), out, repeats=4)
    assert json.loads((out / "status.json").read_text())["state"] == "failed"
