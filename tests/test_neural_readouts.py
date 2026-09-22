import json

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from xuannv_embedding.downstream.neural_readouts import (
    fit_neural,
    load_neural,
    save_neural,
    verify_validation,
)
from xuannv_embedding.export.context import sha


def example(channels=3):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(3, channels, 4, 4)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int8)
    valid = np.ones(y.shape, bool)
    valid[0, 0, 0] = False
    y[1, 0, 0] = -1
    q = rng.normal(size=(2, channels, 4, 4)).astype(np.float32)
    target = (q[:, 0] > 0).astype(np.int8)
    qvalid = np.ones(target.shape, bool)
    qvalid[0, 0, 0] = False
    return x, y, valid, q, target, qvalid


@pytest.mark.parametrize("kind", ["mlp", "conv3x3"])
def test_neural_readout_replays_saved_predictions_without_training(tmp_path, monkeypatch, kind):
    x, y, valid, q, target, qvalid = example()
    before_rng = torch.get_rng_state().clone()
    model = fit_neural(kind, x, y, valid, q, target, qvalid)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert len(model.metadata["losses"]) == 100
    assert np.isfinite(model.metadata["losses"]).all()
    assert model.metadata["initial_weights_sha256"] != model.metadata["final_weights_sha256"]
    expected = model.predict(q, qvalid)
    assert np.isnan(expected[~qvalid]).all()
    assert np.isfinite(expected[qvalid]).all()
    weights = {k: v.clone() for k, v in model.head.state_dict().items()}
    save_neural(model, tmp_path / "model")
    loaded = load_neural(tmp_path / "model", sha(tmp_path / "model/identity.json"))
    monkeypatch.setattr(
        torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("unexpected fitting")
    )
    np.testing.assert_array_equal(loaded.predict(q, qvalid), expected)
    assert verify_validation(loaded, q, target, qvalid)["state"] == "verified"
    for k, v in loaded.head.state_dict().items():
        assert torch.equal(v, weights[k])
    assert loaded.predict(np.empty((0, 3, 4, 4), np.float32), np.empty((0, 4, 4), bool)).shape == (
        0,
        4,
        4,
    )
    with pytest.raises(ValueError, match="validation"):
        verify_validation(loaded, q + 1, target, qvalid)


@pytest.mark.parametrize("kind", ["mlp", "conv3x3"])
def test_training_support_scaling_and_invalid_context_are_explicit(kind):
    x, y, valid, q, target, qvalid = example()
    model = fit_neural(kind, x, y, valid, q, target, qvalid)
    expected = StandardScaler().fit(x.transpose(0, 2, 3, 1)[valid & (y >= 0)])
    np.testing.assert_array_equal(model.scaler.mean_, expected.mean_)
    np.testing.assert_array_equal(model.scaler.scale_, expected.scale_)
    changed = q.copy()
    changed.transpose(0, 2, 3, 1)[~qvalid] = 1e6
    np.testing.assert_array_equal(model.predict(q, qvalid), model.predict(changed, qvalid))
    assert model.metadata["fitted_pixels"] == int((valid & (y >= 0)).sum())


def test_batch_sampling_is_independent_of_input_dimensions_and_global_rng():
    a = fit_neural("mlp", *example(3))
    torch.rand(91)
    b = fit_neural("conv3x3", *example(5))
    np.testing.assert_array_equal(a.batches, b.batches)
    assert a.metadata["head_seed"] == b.metadata["head_seed"] == 41
    assert a.batches.shape == (100, 2)
    assert np.all(a.batches[:, 0] != a.batches[:, 1])


def test_validation_truth_only_changes_calibration_not_weights():
    args = list(example())
    a = fit_neural("mlp", *args)
    args[4] = 1 - args[4]
    b = fit_neural("mlp", *args)
    assert a.metadata["final_weights_sha256"] == b.metadata["final_weights_sha256"]
    np.testing.assert_array_equal(a.predict(args[3], args[5]), b.predict(args[3], args[5]))
    assert a.metadata["validation_ap"] != b.metadata["validation_ap"]


@pytest.mark.parametrize(
    "change", ["kind", "mask", "nonfinite", "labels", "empty_tile", "dimension"]
)
def test_neural_invalid_inputs_fail_before_training(change):
    args = list(example())
    kind = "mlp"
    if change == "kind":
        kind = "rf"
    if change == "mask":
        args[2] = args[2].astype(int)
    if change == "nonfinite":
        args[0][0, 0, 0, 0] = np.nan
    if change == "labels":
        args[1][0, 0, 1] = 2
    if change == "empty_tile":
        args[1][0] = -1
    if change == "dimension":
        args[3] = args[3][:, :2]
    with pytest.raises(ValueError):
        fit_neural(kind, *args)


def test_saved_payload_and_runtime_changes_are_rejected(tmp_path):
    model = fit_neural("mlp", *example())
    root = tmp_path / "model"
    save_neural(model, root)
    with pytest.raises(FileExistsError):
        save_neural(model, root)
    identity = root / "identity.json"
    digest = sha(identity)
    payload = root / "parameters.npz"
    original = payload.read_bytes()
    payload.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="payload"):
        load_neural(root, digest)
    payload.write_bytes(original)
    data = json.loads(identity.read_text())
    data["runtime"]["torch"] = "changed"
    identity.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="runtime"):
        load_neural(root, sha(identity))
