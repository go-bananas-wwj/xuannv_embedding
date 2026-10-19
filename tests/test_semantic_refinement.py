import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_frozen_fusion_training import fixture_spec as base_fixture
from test_frozen_fusion_training import ref

from xuannv_embedding.data_process.osm_weak_targets import AREA_CLASSES, GEOMETRY_CLASSES
from xuannv_embedding.export.context import dump
from xuannv_embedding.training import semantic_refinement as refinement


def fixture_spec(tmp_path):
    path, old = base_fixture(tmp_path)
    inputs = old["inputs"]
    inputs["spectral"] = inputs["spectral"][:1]
    inputs["detail"] = inputs.pop("semantic")
    rng = np.random.default_rng(45)
    for key, channels in [("context", 1024), ("context_mask", 1), ("context_target", 64)]:
        p = tmp_path / (key + ".npy")
        shape = (2, channels, 1, 1) if key != "context_target" else (2, channels, 4, 4)
        np.save(p, rng.uniform(0.2, 1, shape).astype(np.float32))
        inputs[key] = {**ref(p), "layout": "nchw", "index_sha256": old["index"]["sha256"]}
    weak = {
        "schema": "osm-weak-targets-v1",
        "state": "complete",
        "dataset": "OpenStreetMap",
        "index": old["index"],
        "area_legend": list(AREA_CLASSES),
        "geometry_legend": list(GEOMETRY_CLASSES),
        "unknown_is_background": False,
        "primary_reference_maps_read": False,
        "forest_wood_are_tag_classes": True,
    }
    records = json.loads(Path(old["index"]["path"]).read_text())["records"]
    weak["records"] = [{**r, "crs": "EPSG:32650"} for r in records]
    for key, count in [("area", 7), ("geometry", 4)]:
        p = tmp_path / (key + ".npy")
        y = rng.integers(0, count, (2, 4, 4), dtype=np.uint8)
        np.save(p, y)
        weak[key] = ref(p)
        weak[key + "_counts"] = np.bincount(y.ravel(), minlength=count).tolist()
    wp = tmp_path / "osm.json"
    dump(wp, weak)
    spec = {
        "schema": "semantic-refinement-v1",
        "index": old["index"],
        "inputs": inputs,
        "worldcover": old["weak_targets"],
        "osm": ref(wp),
        "context_source_shape": [16, 16],
        "model": {"width": 8, "public_residual_scale": 0.25, "local_residual_scale": 0.5},
        "loss": {
            "worldcover": 1.0,
            "area": 1.0,
            "geometry": 0.5,
            "spectral": 0.2,
            "detail": 0.05,
            "context": 0.05,
            "preservation": 0.05,
        },
        "steps": 4,
        "burnin_steps": 1,
        "warmup_steps": 1,
        "batch_size": 1,
        "lr": 0.001,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "seed": 41,
        "amp": False,
        "device": "cpu",
        "reserve_gib": 0,
        "output": old["output"],
    }
    dump(path, spec)
    return path, spec


def test_refinement_resume_exactly_matches_full_training_and_cold_export(tmp_path):
    path, spec = fixture_spec(tmp_path)
    assert refinement.train(path, max_updates=2)["state"] == "checkpointed"
    assert refinement.train(path)["state"] == "complete"
    first = torch.load(str(Path(spec["output"]) / "model.weights.pt"), weights_only=True)
    other = copy.deepcopy(spec)
    other["output"] = str(tmp_path / "full")
    full = tmp_path / "full.json"
    dump(full, other)
    refinement.train(full)
    second = torch.load(str(tmp_path / "full/model.weights.pt"), weights_only=True)
    assert all(torch.equal(first["model"][k], second["model"][k]) for k in first["model"])
    refinement.export(path)
    result = np.load(Path(spec["output"]) / "embedding/annual.npy")
    assert result.shape == (2, 64, 4, 4) and result.dtype == np.float16
    assert np.isfinite(result).all()
    refinement.export(path)
    root = Path(spec["output"])
    (root / "status.json").unlink()
    assert refinement.train(path)["state"] == "complete"
    with (root / "model.weights.pt").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        refinement.train(path)


def test_public_statistics_and_map_domain_ignore_local_availability(tmp_path):
    path, spec = fixture_spec(tmp_path)
    a = refinement.TrainingData(spec, json.loads(Path(spec["index"]["path"]).read_text()))
    ma = a.moments(seed=41)
    spec["inputs"]["context"] = None
    spec["inputs"]["context_mask"] = None
    spec["inputs"]["context_target"] = None
    spec["inputs"]["detail"] = None
    b = refinement.TrainingData(spec, json.loads(Path(spec["index"]["path"]).read_text()))
    mb = b.moments(seed=41)
    assert a.domain_sha256 == b.domain_sha256
    for name in ["public", "spectral"]:
        assert all(np.array_equal(x, y) for x, y in zip(ma[name], mb[name]))
    for key in a.class_weights:
        assert torch.equal(a.class_weights[key], b.class_weights[key])


def test_refinement_rejects_primary_reference_and_misaligned_osm(tmp_path):
    path, spec = fixture_spec(tmp_path)
    dump(path, {**spec, "evaluation": "never read"})
    with pytest.raises(ValueError, match="specification"):
        refinement.train(path)
    osm = Path(spec["osm"]["path"])
    data = json.loads(osm.read_text())
    data["records"][0]["bounds"][0] = -100
    dump(osm, data)
    spec["osm"] = ref(osm)
    dump(path, spec)
    with pytest.raises(ValueError, match="geographic"):
        refinement.train(path)


def test_padded_alignment_matches_existing_frozen_export_geometry():
    from xuannv_embedding.export.semantic_features import align_patch_features as old_align
    from xuannv_embedding.models.semantic_residual import align_patch_features

    x = torch.randn(2, 3, 5, 7, requires_grad=True)
    actual = align_patch_features(x, (73, 101), (16, 16), patch=16)
    assert torch.equal(actual, old_align(x, (73, 101), (16, 16), patch=16))
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
