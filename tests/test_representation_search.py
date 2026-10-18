import numpy as np
import pytest

from xuannv_embedding.training.representation_search import (
    balanced_validation_rank,
    compose_tile,
    meets_acceptance,
)


def test_balanced_rank_does_not_reward_sacrificing_mlp_for_cnn():
    reference = {"mlp": 0.57, "conv3x3": 0.56}
    balanced = {"heads": {"mlp": {"macro_f1": 0.58}, "conv3x3": {"macro_f1": 0.57}}}
    unbalanced = {"heads": {"mlp": {"macro_f1": 0.53}, "conv3x3": {"macro_f1": 0.65}}}
    assert balanced_validation_rank(balanced, reference) > balanced_validation_rank(
        unbalanced, reference
    )
    unbalanced["test_macro_f1"] = 1.0
    assert balanced_validation_rank(balanced, reference) > balanced_validation_rank(
        unbalanced, reference
    )


def test_acceptance_requires_both_heads_above_both_frozen_references():
    candidate = {"highres:mlp:200": {"macro_f1": 0.58}, "highres:conv3x3:200": {"macro_f1": 0.55}}
    baseline = {
        "aef:mlp:200": {"macro_f1": 0.57},
        "aef:conv3x3:200": {"macro_f1": 0.56},
        "public:mlp:200": {"macro_f1": 0.54},
        "public:conv3x3:200": {"macro_f1": 0.53},
    }
    assert not meets_acceptance(candidate, baseline)
    candidate["highres:conv3x3:200"]["macro_f1"] = 0.57
    assert meets_acceptance(candidate, baseline)


def test_blend_is_unit_length_and_rejects_invalid_weights():
    left = np.zeros((64, 2, 2), np.float32)
    left[0] = 1
    right = np.zeros_like(left)
    right[1] = 1
    output = compose_tile({"kind": "blend", "alpha": 0.25}, [left, right], [])
    assert np.allclose(np.linalg.norm(output, axis=0), 1)
    assert np.allclose(output[1] / output[0], 1 / 3)
    with pytest.raises(ValueError):
        compose_tile({"kind": "blend", "alpha": 2}, [left, right], [])


def test_block_projection_keeps_missing_source_zero_and_total_dimensions_64():
    class Projection:
        def transform(self, values):
            return values[:, :32] - 0.1

    left = np.ones((64, 2, 2), np.float32)
    missing = np.zeros_like(left)
    output = compose_tile({"kind": "blocks"}, [left, missing], [Projection(), Projection()])
    assert output.shape == (64, 2, 2)
    assert np.count_nonzero(output[32:]) == 0
    assert np.allclose(np.linalg.norm(output, axis=0), 1)


def test_saved_unlabelled_projection_reproduces_feature_grid(tmp_path, monkeypatch):
    import json

    from xuannv_embedding.training import representation_search as search

    rng = np.random.default_rng(41)
    sources = {}
    arrays = []
    for name in ["public", "detail"]:
        values = rng.normal(size=(4, 64, 8, 8)).astype(np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        path = tmp_path / f"{name}.npy"
        np.save(path, values)
        sources[name] = {"path": str(path)}
        arrays.append(values)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [{}, {}, {}, {}]}))
    output = tmp_path / "benchmark"
    output.mkdir()
    monkeypatch.setattr(search, "candidate_benchmark", lambda *args: {"output": str(output)})
    recipe = {
        "name": "fusion",
        "kind": "blocks",
        "sources": ["public", "detail"],
        "dimensions": [32, 32],
        "eligible": True,
    }
    spec = {
        "sources": sources,
        "seed": 41,
        "fit_samples": 128,
        "index": {"path": str(index)},
        "output": str(tmp_path),
        "baseline": {},
        "github": {},
        "recipes": [recipe],
    }
    search.prepare_recipe(spec, recipe)
    directory = output / "representations/highres"
    result = np.load(directory / "annual.npy")
    assert result.shape == (4, 64, 8, 8)
    assert np.allclose(np.linalg.norm(result.astype(np.float32), axis=1), 1, atol=1e-3)
    blocks = []
    for i in range(2):
        with np.load(directory / f"projection{i}.npz") as p:
            values = (arrays[i][0].reshape(64, -1).T - p["mean"]) @ p["components"].T
        blocks.append(search.unit(values.T.reshape(32, 8, 8)))
    expected = search.unit(np.concatenate(blocks))
    assert np.allclose(expected, result[0], atol=1e-3)
    report = json.loads((directory / "status.json").read_text())
    assert report["labels_used_for_projection"] is False
