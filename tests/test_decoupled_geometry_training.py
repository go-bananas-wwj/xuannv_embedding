import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from test_time_modulated_training import fixture_spec as parent_fixture

from xuannv_embedding.export.context import dump
from xuannv_embedding.training import decoupled_geometry as training


def fixture_spec(tmp_path):
    path, old = parent_fixture(tmp_path)
    old.pop("diversity")
    old.update(
        schema="decoupled-geometry-refinement-v1",
        relation={"weight": 4.0, "samples": 16, "same_class_weight": 2.0, "seed": 41},
    )
    dump(path, old)
    return path, old


def test_geometry_training_resume_exactly_matches_full_and_cold_export(tmp_path):
    path, spec = fixture_spec(tmp_path)
    training.train(path, max_updates=2)
    training.train(path)
    root = Path(spec["base"]["output"])
    a = torch.load(str(root / "model.weights.pt"), weights_only=True)
    full = copy.deepcopy(spec)
    full["base"]["output"] = str(tmp_path / "full")
    p = tmp_path / "full.json"
    dump(p, full)
    training.train(p)
    b = torch.load(str(tmp_path / "full/model.weights.pt"), weights_only=True)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert not any("local_mixer" in k for k in a["model"])
    training.export(path)
    assert np.isfinite(np.load(root / "embedding/annual.npy")).all()


def test_geometry_specification_rejects_unknown_or_invalid_relation_controls(tmp_path):
    path, spec = fixture_spec(tmp_path)
    for change in [{"weight": -1.0}, {"samples": True}, {"same_class_weight": 0.5}, {"unknown": 1}]:
        value = copy.deepcopy(spec)
        value["relation"].update(change)
        dump(path, value)
        with pytest.raises(ValueError, match="relation"):
            training.specification(path)


def test_geometry_zero_weight_does_not_add_a_training_gradient(tmp_path):
    path, spec = fixture_spec(tmp_path)
    spec["relation"]["weight"] = 0.0
    dump(path, spec)
    parsed, index = training.specification(path)
    data = training.temporal.TrainingData(parsed, index)
    model = training.PublicSemanticGeometryFusion(**parsed["model"])
    batch = data.batch([0, 1])
    out = training.temporal.forward(model, batch, data.context_shape)
    loss = training.batch_loss(out, batch, 201, parsed)
    assert loss["loss"].item() == 0 and not loss["loss"].requires_grad
