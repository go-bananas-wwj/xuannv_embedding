import json

import numpy as np
import pytest

from xuannv_embedding.downstream.fixed_audit import balanced_positions, nested_support
from xuannv_embedding.downstream.frozen_readouts import (
    fit_classification,
    fit_regression,
    freeze_retrieval,
    load_readout,
    save_readout,
    verify_validation,
)
from xuannv_embedding.downstream.multitask import (
    _classification,
    _regression,
    _retrieval,
    block_regression_data,
    retrieval_prototypes,
)
from xuannv_embedding.export.context import sha


def sample():
    rng = np.random.default_rng(7)
    y = np.tile((np.indices((32, 32))[1] >= 16), (3, 1, 1)).astype(np.int8)
    x = np.stack([y + 0.25, rng.normal(size=y.shape)], axis=-1).astype(np.float32)
    return x, y, ["a", "b", "c"]


def test_frozen_classification_matches_registered_validation_evaluator(tmp_path):
    x, y, ids = sample()
    prior = _classification(x, y, [0, 1], [2], ids, 1, 5, tmp_path / "prior.npz")
    support = nested_support(y, ids, [0, 1], 1, 5)
    labels = y[support].ravel()
    positions = balanced_positions(labels, 4096, 5)
    model = fit_classification(
        x[support].reshape(-1, 2)[positions], labels[positions], x[2].reshape(-1, 2), y[2].ravel()
    )
    assert model.metadata["alpha"] == prior["alpha"]
    assert model.metadata["threshold"] == prior["threshold"]
    assert model.metadata["validation_metrics"]["ap"] == prior["ap"]
    with np.load(tmp_path / "prior.npz") as expected:
        np.testing.assert_array_equal(model.predict(x[2].reshape(-1, 2)), expected["scores"])
    save_readout(model, tmp_path / "frozen")
    loaded = load_readout(tmp_path / "frozen", sha(tmp_path / "frozen/identity.json"))
    np.testing.assert_array_equal(
        loaded.predict(x[2].reshape(-1, 2)), model.predict(x[2].reshape(-1, 2))
    )


def test_frozen_regression_matches_registered_clipped_validation_predictions(tmp_path):
    x, y, ids = sample()
    prior = _regression(x, y, [0], [2], ids, 1, 5, tmp_path / "prior.npz")
    train_x, train_y, _ = block_regression_data(x, y, [0])
    val_x, val_y, _ = block_regression_data(x, y, [2])
    model = fit_regression(train_x, train_y, val_x, val_y)
    assert model.metadata["alpha"] == prior["alpha"]
    assert model.metadata["validation_metrics"]["rmse"] == prior["rmse"]
    with np.load(tmp_path / "prior.npz") as expected:
        np.testing.assert_array_equal(model.predict(val_x), expected["predictions"])
    save_readout(model, tmp_path / "frozen")
    loaded = load_readout(tmp_path / "frozen", sha(tmp_path / "frozen/identity.json"))
    assert np.isfinite(loaded.predict([[100, -100]])).all()
    assert 0 <= loaded.predict([[100, -100]])[0] <= 1


def test_frozen_retrieval_matches_training_prototypes_and_zero_query_policy(tmp_path):
    x, y, ids = sample()
    _retrieval(x, y, [0, 1], [2], ids, 1, 5, tmp_path / "prior.npz")
    prototypes, _ = retrieval_prototypes(x, y, [0, 1], ids, 1, 5)
    model = freeze_retrieval(prototypes, normalized=True)
    with np.load(tmp_path / "prior.npz") as expected:
        np.testing.assert_array_equal(model.predict(x[2].reshape(-1, 2)), expected["scores"])
    np.testing.assert_array_equal(model.predict(np.zeros((1, 2))), [0])
    save_readout(model, tmp_path / "frozen")
    loaded = load_readout(tmp_path / "frozen", sha(tmp_path / "frozen/identity.json"))
    np.testing.assert_array_equal(
        loaded.predict(x[2].reshape(-1, 2)), model.predict(x[2].reshape(-1, 2))
    )


def test_scaler_uses_only_support_and_prediction_does_not_mutate_frozen_state(tmp_path):
    x = np.array([[0, 2], [2, 4], [0, 2], [2, 4]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    model = fit_classification(x, y, x + 100, y)
    np.testing.assert_array_equal(model.arrays["mean"], [1, 3])
    save_readout(model, tmp_path / "frozen")
    hashes = {p.name: sha(p) for p in (tmp_path / "frozen").iterdir()}
    cut = model.metadata["threshold"]
    expected = model.predict(x)
    model.predict(x + 1000)
    np.testing.assert_array_equal(model.predict(x), expected)
    assert model.metadata["threshold"] == cut
    assert hashes == {p.name: sha(p) for p in (tmp_path / "frozen").iterdir()}


def test_equal_validation_scores_keep_strongest_registered_regularization():
    x, y = np.zeros((4, 2), dtype=np.float32), np.array([0, 1, 0, 1])
    assert fit_classification(x, y, x, y).metadata["alpha"] == 10
    assert fit_regression(x, y, x, y).metadata["alpha"] == 10


def test_readout_identity_or_array_tampering_is_rejected(tmp_path):
    x, y = np.array([[0], [1]], dtype=np.float32), np.array([0, 1])
    save_readout(fit_classification(x, y, x, y), tmp_path / "frozen")
    identity = tmp_path / "frozen/identity.json"
    expected = sha(identity)
    data = json.loads(identity.read_text())
    data["metadata"]["threshold"] += 1
    identity.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity"):
        load_readout(tmp_path / "frozen", expected)
    expected = sha(identity)
    arrays = tmp_path / "frozen/parameters.npz"
    arrays.write_bytes(arrays.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="parameters"):
        load_readout(tmp_path / "frozen", expected)


@pytest.mark.parametrize(
    "change", ["single_class", "nonfinite", "dimension", "alpha_order", "alpha_duplicate"]
)
def test_invalid_calibration_inputs_are_rejected(change):
    x, y = np.array([[0], [1]], dtype=np.float32), np.array([0, 1])
    val, target, alphas = x.copy(), y.copy(), (10, 1, 0.1)
    if change == "single_class":
        target[:] = 1
    elif change == "nonfinite":
        val[0] = np.nan
    elif change == "dimension":
        val = np.ones((2, 3))
    elif change == "alpha_order":
        alphas = (0.1, 1)
    else:
        alphas = (1, 1)
    with pytest.raises(ValueError):
        fit_classification(x, y, val, target, alphas=alphas)


def test_invalid_prototypes_and_existing_output_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        freeze_retrieval([[0, 0]])
    with pytest.raises(ValueError):
        freeze_retrieval([[2, 0]], normalized=True)
    model = freeze_retrieval([[2, 0]])
    save_readout(model, tmp_path / "frozen")
    with pytest.raises(FileExistsError):
        save_readout(model, tmp_path / "frozen")


def test_readout_requires_registered_implementation_and_rejects_wrong_query_dimension(tmp_path):
    model = freeze_retrieval([[1, 0]])
    with pytest.raises(ValueError, match="dimensions"):
        model.predict([[1, 0, 0]])
    save_readout(model, tmp_path / "frozen")
    path = tmp_path / "frozen/identity.json"
    data = json.loads(path.read_text())
    data["implementation_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="implementation"):
        load_readout(tmp_path / "frozen", sha(path))


def test_validation_replay_after_load_rejects_changed_observations_without_refitting(tmp_path):
    x, y = np.array([[0], [1]], dtype=np.float32), np.array([0, 1])
    save_readout(fit_classification(x, y, x, y), tmp_path / "frozen")
    readout = load_readout(tmp_path / "frozen", sha(tmp_path / "frozen/identity.json"))
    proof = verify_validation(readout, x, y)
    assert proof["state"] == "verified" and proof["parameters_refitted"] is False
    with pytest.raises(ValueError, match="observations"):
        verify_validation(readout, x + 1, y)
    with pytest.raises(ValueError, match="observations"):
        verify_validation(readout, x, 1 - y)
