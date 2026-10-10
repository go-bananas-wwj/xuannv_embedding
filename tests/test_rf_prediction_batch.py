import numpy as np

from xuannv_embedding.downstream.strong_classifiers import fit_classifier


def test_forest_batch_size_is_recorded_and_predictions_match_legacy_chunks():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(200, 4))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = fit_classifier("rf", x[:120], y[:120], x[120:], y[120:], seed=41)
    assert model.metadata["prediction_chunk"] == 65536
    query = rng.normal(size=(33000, 4))
    large = model.predict(query)
    model.metadata["prediction_chunk"] = 1024
    small = model.predict(query)
    assert np.allclose(large, small, rtol=0, atol=1e-14)
