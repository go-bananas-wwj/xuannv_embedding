import json

import numpy as np
import pytest
import torch

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.weak_semantic_context import export


def fixture_context(tmp_path):
    index = tmp_path / "index.json"
    dump(index, {"months": ["2024-01"], "records": [{"patch_id": "a"}, {"patch_id": "b"}]})
    training = tmp_path / "training.json"
    dump(
        training,
        {
            "steps": 10,
            "index": {"path": str(index), "sha256": sha(index)},
            "inputs": {"semantic": None},
            "weak_targets": {"path": str(tmp_path / "must-not-read-map"), "sha256": "0" * 64},
        },
    )
    identity = {"spec_sha256": sha(training), "implementation": {"training": "archived"}}
    weights = torch.randn(12, 64, 1, 1, generator=torch.Generator().manual_seed(41))
    bias = torch.arange(12, dtype=torch.float32)
    bias[0] = 1000  # Ignored label zero must not suppress foreground probabilities.
    model = tmp_path / "model.pt"
    torch.save(
        {
            "identity": identity,
            "step": 10,
            "model": {"worldcover_head.weight": weights, "worldcover_head.bias": bias},
            "training_summary": {"state": "complete", "high_resolution_used": False},
        },
        model,
    )
    embedding = tmp_path / "embedding.npy"
    values = np.random.default_rng(42).normal(size=(2, 64, 3, 3)).astype(np.float32)
    values[0, :, 0, 0] = 0
    np.save(embedding, values)
    header, status = tmp_path / "embedding-identity.json", tmp_path / "embedding-status.json"
    dump(header, {"training": identity, "model_sha256": sha(model)})
    dump(
        status,
        {
            "state": "complete",
            "sha256": sha(embedding),
            "shape": list(values.shape),
            "index_sha256": sha(index),
            "evaluation_maps_used": False,
        },
    )
    refs = dict(
        training=training,
        model=model,
        embedding=embedding,
        embedding_identity=header,
        embedding_status=status,
        index=index,
    )
    spec = {
        "schema": "weak-semantic-context-v1",
        **{k: {"path": str(v), "sha256": sha(v)} for k, v in refs.items()},
        "output": str(tmp_path / "context"),
    }
    path = tmp_path / "spec.json"
    dump(path, spec)
    return path, spec, values, weights, bias


def test_context_keeps_foreground_probabilities_without_opening_reference_maps(tmp_path):
    path, spec, values, weight, bias = fixture_context(tmp_path)
    result = export(path)
    actual = np.load(result)
    expected = torch.nn.functional.conv2d(torch.from_numpy(values), weight, bias)[:, 1:]
    expected = expected.softmax(1).numpy()
    expected[0, :, 0, 0] = 0
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert actual.shape == (2, 11, 3, 3)
    metadata = json.loads((result.parent / "metadata.json").read_text())
    assert metadata["reference_maps_opened"] is False
    assert metadata["high_resolution_used"] is False
    assert metadata["index"] == spec["index"]
    assert export(path) == result


@pytest.mark.parametrize("change", ["model", "index", "role"])
def test_context_rejects_crossed_checkpoint_index_and_source_roles(tmp_path, change):
    path, spec, _, _, _ = fixture_context(tmp_path)
    status_path = tmp_path / "embedding-status.json"
    header_path = tmp_path / "embedding-identity.json"
    if change == "model":
        value = json.loads(header_path.read_text())
        value["model_sha256"] = "0" * 64
        dump(header_path, value)
        spec["embedding_identity"]["sha256"] = sha(header_path)
    elif change == "index":
        value = json.loads(status_path.read_text())
        value["index_sha256"] = "0" * 64
        dump(status_path, value)
        spec["embedding_status"]["sha256"] = sha(status_path)
    else:
        saved = torch.load(spec["model"]["path"], weights_only=True)
        saved["training_summary"]["high_resolution_used"] = True
        torch.save(saved, spec["model"]["path"])
        spec["model"]["sha256"] = sha(spec["model"]["path"])
        value = json.loads(header_path.read_text())
        value["model_sha256"] = spec["model"]["sha256"]
        dump(header_path, value)
        spec["embedding_identity"]["sha256"] = sha(header_path)
    dump(path, spec)
    with pytest.raises(ValueError, match="identity|role"):
        export(path)


def test_context_rejects_changed_completed_features(tmp_path):
    path, _, _, _, _ = fixture_context(tmp_path)
    result = export(path)
    changed = np.load(result)
    changed[1, 0, 0, 0] = -1
    np.save(result, changed)
    with pytest.raises(ValueError, match="changed"):
        export(path)
