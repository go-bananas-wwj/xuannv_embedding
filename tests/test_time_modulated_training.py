import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_semantic_refinement import fixture_spec as base_fixture
from test_semantic_refinement import ref

from xuannv_embedding.export.context import dump
from xuannv_embedding.training import time_modulated as training


def fixture_spec(tmp_path):
    path, base = base_fixture(tmp_path)
    index = json.loads(Path(base["index"]["path"]).read_text())
    records = [{**r, "crs": r.get("crs", index.get("crs"))} for r in index["records"]]
    geo = tmp_path / "geometry.json"
    months = [202401 + i for i in range(12)]
    dump(geo, {"months": months, "records": records})
    rng = np.random.default_rng(52)
    refs = {}
    for name, channels in [("queries", 64), ("coverage", 1)]:
        p = tmp_path / (name + ".npy")
        np.save(p, rng.uniform(0.2, 1, (2, 12, channels, 1, 1)).astype(np.float32))
        refs[name] = ref(p)
    manifest = tmp_path / "query-manifest.json"
    dump(
        manifest,
        {
            "state": "complete",
            "schema": "temporal-public-queries-v1",
            "index": base["index"],
            "records": records,
            "months": months,
            "shape": [2, 12, 64, 1, 1],
            **refs,
            "map_targets_loaded_for_inference": False,
            "local_images_used": False,
        },
    )
    weights = tmp_path / "support.npy"
    np.save(weights, np.full((1, 12), 1 / 12, np.float32))
    contract = tmp_path / "time-contract.json"
    dump(
        contract,
        {
            "state": "complete",
            "months": months,
            "weights": ref(weights),
            "views": [
                {
                    "slot": 0,
                    "source_time": {
                        "start": "2024-01-01",
                        "end": "2025-01-01",
                        "precision": "year",
                        "analysis_period": "2024",
                    },
                }
            ],
            "physical_time_used_for_queries": True,
            "one_row_per_actual_view_not_month_copy": True,
        },
    )
    spec = {
        "schema": "time-modulated-refinement-v1",
        "base": base,
        "temporal": {
            "public_manifest": ref(manifest),
            "time_contract": ref(contract),
            "views": [
                {
                    "slot": 0,
                    "features": base["inputs"]["context"],
                    "coverage": base["inputs"]["context_mask"],
                    "geometry": ref(geo),
                }
            ],
        },
        "diversity": {"variance": 0.2, "covariance": 0.05},
    }
    dump(path, spec)
    return path, spec


def test_dated_training_resume_is_exact_and_export_retains_frozen_sources(tmp_path):
    path, spec = fixture_spec(tmp_path)
    assert training.train(path, max_updates=2)["state"] == "checkpointed"
    assert training.train(path)["state"] == "complete"
    root = Path(spec["base"]["output"])
    saved = torch.load(str(root / "model.weights.pt"), weights_only=True)
    other = copy.deepcopy(spec)
    other["base"]["output"] = str(tmp_path / "full")
    full = tmp_path / "full.json"
    dump(full, other)
    training.train(full)
    check = torch.load(str(tmp_path / "full/model.weights.pt"), weights_only=True)
    assert all(torch.equal(saved["model"][k], check["model"][k]) for k in saved["model"])
    training.export(path)
    result = np.load(root / "embedding/annual.npy")
    assert result.shape == (2, 64, 4, 4) and np.isfinite(result).all()
    assert training.train(path)["state"] == "complete"
    training.export(path)


def test_time_contract_rejects_forged_support_or_geography(tmp_path):
    path, spec = fixture_spec(tmp_path)
    contract_path = Path(spec["temporal"]["time_contract"]["path"])
    contract = json.loads(contract_path.read_text())
    np.save(contract["weights"]["path"], np.eye(1, 12, dtype=np.float32))
    contract["weights"] = ref(Path(contract["weights"]["path"]))
    dump(contract_path, contract)
    spec["temporal"]["time_contract"] = ref(contract_path)
    dump(path, spec)
    with pytest.raises(ValueError, match="physical"):
        training.train(path)


def test_new_schema_rejects_unknown_fields_and_negative_regularization(tmp_path):
    path, spec = fixture_spec(tmp_path)
    spec["unregistered"] = 1
    dump(path, spec)
    with pytest.raises(ValueError, match="specification"):
        training.specification(path)
    spec.pop("unregistered")
    spec["diversity"]["variance"] = -1
    dump(path, spec)
    with pytest.raises(ValueError, match="specification"):
        training.specification(path)


def test_auxiliary_teacher_missing_pixels_do_not_become_zero_reconstruction_labels(tmp_path):
    path, spec = fixture_spec(tmp_path)
    parsed, index = training.specification(path)
    data = training.TrainingData(parsed, index)
    batch = list(data.batch([0, 1]))
    batch[5][:, :, 0, 0] = 0
    model = training.TimeModulatedFusion(**parsed["model"])
    out = training.forward(model, tuple(batch), data.context_shape)
    assert not out.context_mask[:, :, 0, 0].any()
    assert out.context_mask[:, :, 1, 1].all()


def test_public_query_statistics_are_independent_of_local_observation_coverage(tmp_path):
    path, _ = fixture_spec(tmp_path)
    spec, index = training.specification(path)
    data = training.TrainingData(spec, index)
    observed = data.moments(seed=41)
    data.view_masks = [np.zeros_like(m) for m in data.view_masks]
    data.extra["context_mask"] = np.zeros_like(data.extra["context_mask"])
    missing = data.moments(seed=41)
    for name in ["public", "spectral", "query"]:
        assert all(np.array_equal(a, b) for a, b in zip(observed[name], missing[name]))
