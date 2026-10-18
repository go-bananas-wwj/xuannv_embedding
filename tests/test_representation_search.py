import numpy as np
import pytest

from xuannv_embedding.training.representation_search import (
    balanced_validation_rank,
    compose_tile,
    meets_acceptance,
    registered_parent,
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


def test_spatial_pool_preserves_constant_feature_directions_and_grid_extent():
    left = np.ones((64, 7, 7), np.float32)
    right = left.copy()
    result = compose_tile({"kind": "blend", "alpha": 0.5, "post_pool": 5}, [left, right], [])
    assert result.shape == left.shape
    assert np.allclose(result, 1 / 8)
    with pytest.raises(ValueError):
        compose_tile({"kind": "blend", "alpha": 0.5, "post_pool": 4}, [left, right], [])


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


def test_parent_reference_rejects_test_metrics_even_with_a_valid_file_hash(tmp_path):
    import json

    from xuannv_embedding.export.context import sha

    validation = tmp_path / "validation.json"
    result = tmp_path / "result.json"
    result.write_text('{"state":"complete"}')
    validation.write_text(json.dumps({"test_scored": False, "fold": 0, "heads": {}}))
    spec = {
        "parent": {
            "validation": {"path": str(validation), "sha256": sha(validation)},
            "result": {"path": str(result), "sha256": sha(result)},
        }
    }
    assert registered_parent(spec)["test_scored"] is False
    validation.write_text(json.dumps({"test_scored": True, "fold": 0, "heads": {}}))
    spec["parent"]["validation"]["sha256"] = sha(validation)
    with pytest.raises(ValueError, match="validation split"):
        registered_parent(spec)


def test_spatial_residual_can_restore_local_directions_without_changing_extent():
    values = np.zeros((64, 5, 5), np.float32)
    values[0, :, :2] = 1
    values[1, :, 2:] = 1
    output = compose_tile({"kind": "identity", "post_pool": 5, "detail_mix": 1.0}, [values], [])
    assert np.array_equal(output, values)
    with pytest.raises(ValueError, match="detail"):
        compose_tile({"kind": "identity", "post_pool": 5, "detail_mix": float("nan")}, [values], [])


def test_branch_filters_preserve_missing_modalities_and_respect_source_weights():
    class Projection:
        def transform(self, values):
            return values[:, :32]

    values = np.ones((64, 5, 5), np.float32)
    recipe = {"kind": "blocks", "branch_pool": [1, 3], "block_weights": [1.0, 2.0]}
    output = compose_tile(recipe, [values, values], [Projection(), Projection()])
    assert np.allclose(output[32:] / output[:32], 2)
    missing = compose_tile(recipe, [values, np.zeros_like(values)], [Projection(), Projection()])
    assert np.count_nonzero(missing[32:]) == 0
    with pytest.raises(ValueError, match="branch"):
        compose_tile({**recipe, "branch_pool": [1]}, [values, values], [Projection(), Projection()])


def test_improvement_target_requires_both_heads_and_mean_gain_over_archived_result():
    from xuannv_embedding.training.representation_search import meets_improvement

    previous = {
        "highres:mlp:200": {"macro_f1": 0.5868},
        "highres:conv3x3:200": {"macro_f1": 0.5681},
    }
    values = {"highres:mlp:200": {"macro_f1": 0.5968}, "highres:conv3x3:200": {"macro_f1": 0.5681}}
    assert not meets_improvement(values, previous, minimum_mean_gain=0.005)
    values["highres:conv3x3:200"]["macro_f1"] = 0.5701
    assert meets_improvement(values, previous, minimum_mean_gain=0.005)
    values["highres:mlp:200"]["macro_f1"] = 0.5869
    assert not meets_improvement(values, previous, minimum_mean_gain=0.005)
    values["highres:conv3x3:200"]["macro_f1"] = float("nan")
    assert not meets_improvement(values, previous, minimum_mean_gain=0.005)


def test_improvement_validation_anchor_stays_fixed_when_parent_changes(tmp_path):
    import json

    from xuannv_embedding.export.context import sha
    from xuannv_embedding.training.representation_search import improvement_validation_reference

    path = tmp_path / "anchor.json"
    report = {
        "fold": 0,
        "test_scored": False,
        "heads": {"mlp": {"macro_f1": 0.60}, "conv3x3": {"macro_f1": 0.59}},
    }
    path.write_text(json.dumps(report))
    spec = {"improvement": {"validation": {"path": str(path), "sha256": sha(path)}}}
    assert improvement_validation_reference(spec) == {"mlp": 0.60, "conv3x3": 0.59}
    report["test_scored"] = True
    path.write_text(json.dumps(report))
    spec["improvement"]["validation"]["sha256"] = sha(path)
    with pytest.raises(ValueError, match="validation"):
        improvement_validation_reference(spec)


@pytest.mark.parametrize("kernel", [11, 13])
def test_large_context_preserves_unit_constant_features_at_tile_edges(kernel):
    values = np.ones((64, 15, 15), np.float32)
    result = compose_tile({"kind": "identity", "post_pool": kernel}, [values], [])
    assert result.shape == values.shape
    assert np.allclose(result, 1 / 8)


def test_standardized_mixed_channel_source_reloads_and_keeps_missing_pixels_zero(
    tmp_path, monkeypatch
):
    import json

    from xuannv_embedding.training import representation_search as search

    rng = np.random.default_rng(41)
    public = rng.normal(size=(4, 64, 8, 8)).astype(np.float32)
    raw = rng.normal(size=(4, 95, 8, 8)).astype(np.float32)
    raw[:, 0] *= 1000
    raw[0] = 0
    arrays = [public, raw]
    sources = {}
    for name, values in zip(["public", "raw"], arrays):
        path = tmp_path / f"{name}.npy"
        np.save(path, values)
        sources[name] = {"path": str(path), "standardize": name == "raw"}
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [{}, {}, {}, {}]}))
    output = tmp_path / "benchmark"
    output.mkdir()
    monkeypatch.setattr(search, "candidate_benchmark", lambda *a: {"output": str(output)})
    recipe = {
        "name": "mixed",
        "kind": "blocks",
        "sources": ["public", "raw"],
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
    stored = np.load(directory / "annual.npy")
    assert stored.shape == (4, 64, 8, 8)
    assert np.count_nonzero(stored[0, 32:]) == 0
    projections = [search.load_projection(directory / f"projection{i}.npz") for i in range(2)]
    reconstructed = search.compose_tile(recipe, [x[1] for x in arrays], projections)
    assert np.array_equal(reconstructed.astype(np.float16), stored[1])
    spec["sources"]["raw"]["standardize"] = False
    with pytest.raises(ValueError, match="identity changed"):
        search.prepare_recipe(spec, recipe)


def test_positive_mean_ranking_matches_joint_improvement_goal_without_sacrificing_a_head():
    from xuannv_embedding.training.representation_search import validation_rank

    reference = {"mlp": 0.60, "conv3x3": 0.59}
    a = {"heads": {"mlp": {"macro_f1": 0.606}, "conv3x3": {"macro_f1": 0.596}}}
    b = {"heads": {"mlp": {"macro_f1": 0.604}, "conv3x3": {"macro_f1": 0.605}}}
    c = {"heads": {"mlp": {"macro_f1": 0.599}, "conv3x3": {"macro_f1": 0.650}}}
    assert validation_rank(a, reference, "balanced") > validation_rank(b, reference, "balanced")
    assert validation_rank(b, reference, "positive_mean") > validation_rank(
        a, reference, "positive_mean"
    )
    assert validation_rank(b, reference, "positive_mean") > validation_rank(
        c, reference, "positive_mean"
    )
