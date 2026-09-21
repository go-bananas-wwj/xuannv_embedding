import json
from argparse import Namespace

import numpy as np
import pytest

from xuannv_embedding.downstream.embedding_diagnostics import (
    Moments,
    adjacent_change,
    run,
)


def test_streaming_moments_match_centered_covariance_and_ignore_batch_boundaries():
    values = np.random.default_rng(12).normal(size=(41, 5)) + 1e5
    accumulator = Moments(5)
    for chunk in np.array_split(values, 7):
        accumulator.add(chunk)
    report = accumulator.report()
    covariance = np.cov(values, rowvar=False, bias=True)
    np.testing.assert_allclose(report["variance_by_dimension"], np.diag(covariance), rtol=1e-10)
    assert report["sample_count"] == 41
    assert report["total_variance"] == pytest.approx(np.trace(covariance))


def test_covariance_effective_rank_handles_collapse_and_two_equal_axes():
    collapsed = Moments(3)
    collapsed.add(np.ones((10, 3)))
    assert collapsed.report()["covariance_effective_rank"] == 0
    assert collapsed.report()["largest_variance_fraction"] is None
    two_axes = Moments(3)
    two_axes.add(np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]))
    assert two_axes.report()["covariance_effective_rank"] == pytest.approx(2)


def test_diagnostics_reject_nonfinite_vectors_and_empty_sample_domain():
    moment = Moments(2)
    with pytest.raises(ValueError, match="finite"):
        moment.add(np.array([[np.nan, 0]]))
    with pytest.raises(ValueError, match="empty"):
        moment.report()


def test_identical_nonbinary_vectors_have_zero_variance_and_rank():
    moment = Moments(3)
    values = np.tile([0.1, 0.2, 0.3], (100, 1))
    moment.add(values)
    moment.add(values)
    assert moment.report()["total_variance"] == 0
    assert moment.report()["covariance_effective_rank"] == 0


def test_adjacent_changes_count_zero_norm_pairs_without_fabricated_cosines():
    left = np.array([[1, 0], [1, 0], [0, 0]])
    right = np.array([[1, 0], [0, 1], [1, 0]])
    report = adjacent_change(left, right)
    assert report["position_count"] == 3
    assert report["cosine_defined_count"] == 2
    assert report["mean_cosine_similarity"] == pytest.approx(0.5)
    assert report["mean_l2_difference"] == pytest.approx((np.sqrt(2) + 1) / 3)


def test_diagnostic_reads_only_validation_embeddings_and_preserves_months(tmp_path):
    records = [{"patch_id": str(i), "bounds": [i, 0, i + 1, 1]} for i in range(4)]
    cache = tmp_path / "cache.json"
    split = {"train": [0], "validation": [1], "test": [2], "buffer": [3]}
    cache.write_text(json.dumps({"records": records, "split": split}))
    embeddings = np.zeros((2, 2, 8, 8), dtype=np.float32)
    embeddings[0, 0], embeddings[1, 1] = 1, 1
    path = tmp_path / "validation.npz"
    np.savez_compressed(path, embedding=embeddings, timestamps=np.array([202601, 202602]))
    exported = [
        {**record, "path": str(path if i == 1 else tmp_path / "must-not-read.npz")}
        for i, record in enumerate(records)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"records": exported, "split": split, "months": ["2026-01", "2026-02"]})
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "cache": str(cache),
                "output": str(tmp_path / "out"),
                "sample_stride": 2,
                "models": {"model": {"manifest": str(manifest)}},
            }
        )
    )
    run(Namespace(spec=spec, model="model"))
    result = json.loads((tmp_path / "out/model/results.json").read_text())
    assert result["months"][0]["month"] == "2026-01"
    assert result["months"][0]["sample_count"] == 16
    assert result["months"][0]["full_grid_vector_count"] == 64
    assert result["months"][0]["mean_norm"] == 1
    assert result["adjacent_months"][0]["mean_cosine_similarity"] == 0
    identity = json.loads((tmp_path / "out/model/identity.json").read_text())
    assert identity["validation_indices"] == [1]
    assert identity["labels_used"] is False
    assert str(tmp_path) not in json.dumps(identity)
    with pytest.raises(FileExistsError):
        run(Namespace(spec=spec, model="model"))
