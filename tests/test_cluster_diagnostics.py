import numpy as np

from xuannv_embedding.downstream.cluster_diagnostics import (
    covariance_summary,
    fit,
    load,
    predict,
    save,
)
from xuannv_embedding.export.context import sha


def test_cluster_fit_uses_only_registered_unlabeled_positions_and_roundtrips(tmp_path):
    rng = np.random.default_rng(9)
    values = rng.normal(size=(3, 8, 8, 4)).astype(np.float32)
    valid = np.ones(values.shape[:-1], bool)
    valid[0, 0, 0] = False
    model = fit(values, valid, [0, 1], clusters=3, step=2, offset=0)
    changed = values.copy()
    changed[2] = 1000
    same = fit(changed, valid, [0, 1], clusters=3, step=2, offset=0)
    np.testing.assert_array_equal(model["centers"], same["centers"])
    assert model["metadata"]["sampled_positions"] == 31
    assert model["metadata"]["labels_used_for_fitting"] is False
    save(model, tmp_path / "cluster")
    restored = load(tmp_path / "cluster", sha(tmp_path / "cluster/identity.json"))
    np.testing.assert_array_equal(predict(restored, values, valid), predict(model, values, valid))
    assert predict(model, values, valid)[0, 0, 0] == -1


def test_covariance_summary_handles_constant_and_rank_one_features():
    constant = covariance_summary(np.ones((20, 4)))
    assert constant["effective_rank"] == 0 and constant["dimensions_for_95_percent"] == 0
    x = np.arange(20)[:, None] * np.array([[1, 2, 3, 4]])
    stats = covariance_summary(x)
    assert abs(stats["effective_rank"] - 1) < 1e-10
    assert stats["dimensions_for_95_percent"] == 1
