import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from xuannv_embedding.downstream.strong_classifiers import (
    fit_classifier,
    load_classifier,
    save_classifier,
    verify_validation,
)
from xuannv_embedding.export.context import sha


def example():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(40, 5))
    query = rng.normal(size=(24, 5))
    return x, (x[:, 0] > 0).astype(int), query, (query[:, 0] > 0).astype(int)


@pytest.mark.parametrize("kind", ["rf", "svm", "knn"])
def test_saved_strong_classifier_replays_validation_and_never_refits(tmp_path, monkeypatch, kind):
    x, y, query, target = example()
    model = fit_classifier(kind, x, y, query, target, seed=7)
    scores = model.predict(query)
    assert model.metadata["validation_ap"] == average_precision_score(target, scores)
    save_classifier(model, tmp_path / "readout")
    digest = sha(tmp_path / "readout/identity.json")
    loaded = load_classifier(tmp_path / "readout", digest)
    before = {p.name: sha(p) for p in (tmp_path / "readout").iterdir()}
    if kind != "knn":

        def forbidden(*args, **kwargs):
            raise AssertionError("saved classifier must not refit")

        monkeypatch.setattr(loaded.estimator, "fit", forbidden)
    np.testing.assert_array_equal(loaded.predict(query), scores)
    assert verify_validation(loaded, query, target)["state"] == "verified"
    loaded.predict(query + 500)
    assert before == {p.name: sha(p) for p in (tmp_path / "readout").iterdir()}
    assert loaded.predict(np.empty((0, 5))).shape == (0,)
    with pytest.raises(ValueError, match="validation"):
        verify_validation(loaded, query + 1, target)


def test_svm_candidates_are_selected_only_by_validation_ap_and_small_C_wins_ties():
    x, y = np.zeros((8, 3)), np.tile([0, 1], 4)
    model = fit_classifier("svm", x, y, x, y, seed=7)
    assert model.metadata["selected_parameter"] == 0.1
    assert [t["parameter"] for t in model.metadata["trials"]] == [0.1, 1.0, 10.0]


def test_rf_and_svm_match_their_registered_estimator_predictions():
    x, y, query, target = example()
    for kind in ("rf", "svm"):
        model = fit_classifier(kind, x, y, query, target, seed=7)
        z = model.scaler.transform(query.astype(np.float64))
        expected = (
            model.estimator.predict_proba(z)[:, 1]
            if kind == "rf"
            else model.estimator.decision_function(z)
        )
        np.testing.assert_allclose(model.predict(query), expected, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(model.scaler.mean_, x.mean(0))


def test_knn_caps_each_class_and_has_deterministic_zero_query_ties():
    x = np.zeros((2200, 2))
    y = np.r_[np.zeros(1100), np.ones(1100)]
    model = fit_classifier("knn", x, y, x[:2], [0, 1], seed=7)
    assert model.metadata["fitted_pixels"] == 2048
    assert model.metadata["k"] == 5
    # Equal similarities choose earlier registered support positions, all class zero here.
    np.testing.assert_array_equal(model.predict(x[:2]), [0, 0])
    np.testing.assert_array_equal(model.arrays["selected_input_positions"][:3], [0, 1, 2])
    np.testing.assert_array_equal(
        model.arrays["selected_input_positions"][1024:1027], [1100, 1101, 1102]
    )


def test_knn_matches_independent_cosine_distance_ranking():
    from sklearn.metrics import pairwise_distances

    x, y, query, target = example()
    model = fit_classifier("knn", x, y, query, target, seed=7)
    selected = model.arrays["selected_input_positions"]
    distances = pairwise_distances(
        model.scaler.transform(query), model.scaler.transform(x[selected]), metric="cosine"
    )
    neighbors = np.argsort(distances, kind="stable", axis=1)[:, :5]
    expected = y[selected][neighbors].mean(1)
    np.testing.assert_array_equal(model.predict(query), expected)


@pytest.mark.parametrize("change", ["single_class", "nonfinite", "dimension", "kind"])
def test_invalid_strong_classifier_inputs_are_rejected(change):
    x, y, query, target = example()
    kind = "rf"
    if change == "single_class":
        target[:] = 1
    elif change == "nonfinite":
        query[0, 0] = np.nan
    elif change == "dimension":
        query = query[:, :-1]
    else:
        kind = "unknown"
    with pytest.raises(ValueError):
        fit_classifier(kind, x, y, query, target, seed=7)


@pytest.mark.parametrize("changed_file", ["parameters.npz", "estimator.joblib"])
def test_changed_payload_is_rejected_before_deserialization(tmp_path, monkeypatch, changed_file):
    import xuannv_embedding.downstream.strong_classifiers as strong

    model = fit_classifier("rf", *example(), seed=7)
    save_classifier(model, tmp_path / "readout")
    digest = sha(tmp_path / "readout/identity.json")
    path = tmp_path / "readout" / changed_file
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr(strong.joblib, "load", lambda *a, **k: pytest.fail("unsafe early load"))
    with pytest.raises(ValueError, match="payload"):
        load_classifier(tmp_path / "readout", digest)


def test_runtime_mismatch_and_existing_output_are_rejected(tmp_path):
    model = fit_classifier("knn", *example(), seed=7)
    save_classifier(model, tmp_path / "readout")
    with pytest.raises(FileExistsError):
        save_classifier(model, tmp_path / "readout")
    path = tmp_path / "readout/identity.json"
    data = json.loads(path.read_text())
    data["runtime"]["numpy"] = "changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="runtime"):
        load_classifier(tmp_path / "readout", sha(path))
