import json
from pathlib import Path

import numpy as np
import pytest
from test_paired_multitask import fixture_spec

from xuannv_embedding.downstream.compression_diagnostics import fit_pca, project, run
from xuannv_embedding.export.context import sha


def test_pca_projection_matches_discarded_variance_and_full_reconstruction():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 5)) * [1, 2, 3, 4, 5] + 10
    pca = fit_pca(x)
    z = project(x, pca, 5)
    np.testing.assert_allclose(z @ pca["components"] + pca["mean"], x, atol=2e-6)
    small = project(x, pca, 2)
    residual = x - pca["mean"] - small @ pca["components"][:2]
    assert np.sum(residual**2) / 49 == pytest.approx(pca["eigenvalues"][2:].sum(), rel=1e-6)
    np.testing.assert_array_equal(project(x.astype(np.float32), None, None), x.astype(np.float32))


@pytest.mark.parametrize("bad", [np.ones((1, 3)), np.full((3, 2), np.nan), np.ones(3)])
def test_pca_rejects_insufficient_or_nonfinite_samples(bad):
    with pytest.raises(ValueError):
        fit_pca(bad)


def test_projection_rejects_invalid_dimensions():
    x = np.arange(30).reshape(10, 3)
    pca = fit_pca(x)
    for d in [0, 4, True]:
        with pytest.raises(ValueError):
            project(x, pca, d)


def test_validation_compression_never_opens_test_and_saves_train_only_pca(tmp_path):
    _, primary, test_paths = fixture_spec(tmp_path)
    spec = {
        "protocol": "compression-validation-v1",
        "reference_cache": primary["reference_cache"],
        "labels": {k: primary["labels"][k] for k in ("train", "validation")},
        "models": primary["models"],
        "dimensions": [1, 2],
        "budgets": [1],
        "retrieval_budgets": [1],
        "support_seeds": [7],
        "sample_step": 4,
        "sample_offset": 2,
        "output": str(tmp_path / "compression"),
    }
    for path in test_paths:
        path.unlink()
    path = tmp_path / "compression_spec.json"
    path.write_text(json.dumps(spec))
    report = run(path)
    assert report["test_scored"] is False
    assert report["conditions_per_model"] == 42
    for name, model in primary["models"].items():
        record = report["models"][name]
        assert record["fit_global_indices"] == [0]
        pca = np.load(tmp_path / "compression" / name / "pca.npz")
        manifest = json.loads(Path(model["manifest_path"]).read_text())
        source = np.load(manifest["records"][0]["path"])["embedding"][-1].transpose(1, 2, 0)
        expected = source[2::4, 2::4].reshape(-1, source.shape[-1]).mean(0)
        np.testing.assert_allclose(pca["mean"], expected, atol=1e-7)
        assert set(record["variants"]) == {"native", "pca1", "pca2"}
        assert all(len(v["conditions"]) == 14 for v in record["variants"].values())
        assert record["pca_sha256"] == sha(tmp_path / "compression" / name / "pca.npz")
    with pytest.raises(FileExistsError):
        run(path)


def test_unregistered_test_labels_and_dimensions_are_rejected(tmp_path):
    _, primary, _ = fixture_spec(tmp_path)
    base = {
        "protocol": "compression-validation-v1",
        "reference_cache": primary["reference_cache"],
        "labels": primary["labels"],
        "models": primary["models"],
        "dimensions": [1],
        "budgets": [1],
        "retrieval_budgets": [1],
        "support_seeds": [7],
        "sample_step": 4,
        "sample_offset": 2,
        "output": str(tmp_path / "invalid"),
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError):
        run(path)
    assert not (tmp_path / "invalid").exists()
