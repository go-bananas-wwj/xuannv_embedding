import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_semantic_refinement import fixture_spec as weak_fixture

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.models.semantic_residual import SemanticResidualFusion
from xuannv_embedding.training import content_adaptive_residual as training
from xuannv_embedding.training import semantic_refinement as legacy


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def fixture_spec(tmp_path):
    _, old = weak_fixture(tmp_path)
    index = json.loads(Path(old["index"]["path"]).read_text())
    data = legacy.TrainingData(old, index)
    moments = data.moments(seed=41)
    sources, teachers = {}, {}
    for i, name in enumerate(["semantic", "geometry"]):
        source_spec = copy.deepcopy(old)
        source_spec["output"] = str(tmp_path / (name + "_source"))
        spec_path = tmp_path / (name + "_source.json")
        dump(spec_path, source_spec)
        torch.manual_seed(80 + i)
        net = SemanticResidualFusion(**old["model"])
        for key, pair in moments.items():
            net.set_statistics(key, *pair)
        with torch.no_grad():
            for layer in [net.public_delta, net.detail_delta, net.context_delta]:
                layer.weight.normal_(0, 0.04)
            batch = data.batch([0, 1])
            values = legacy._forward(net, batch, data.context_shape).embedding.numpy()
        weights = tmp_path / (name + "_source.pt")
        torch.save(
            {
                "identity": {"spec_sha256": sha(spec_path)},
                "step": old["steps"],
                "model": net.state_dict(),
                "summary": {
                    "weak_domain_sha256": data.domain_sha256,
                    "class_weights": {k: v.tolist() for k, v in data.class_weights.items()},
                },
            },
            weights,
        )
        array = tmp_path / (name + "_source.npy")
        np.save(array, values.astype(np.float16))
        status = tmp_path / (name + "_source_status.json")
        dump(
            status,
            {
                "state": "complete",
                "shape": list(values.shape),
                "sha256": sha(array),
                "index_sha256": old["index"]["sha256"],
                "model_sha256": sha(weights),
                "evaluation_maps_used": False,
            },
        )
        sources[name] = {**ref(array), "layout": "nchw", "index_sha256": old["index"]["sha256"]}
        teachers.update(
            {
                name + "_checkpoint": ref(weights),
                name + "_spec": ref(spec_path),
                name + "_export": ref(status),
            }
        )
    spec = {
        "schema": "content-adaptive-residual-v1",
        "index": old["index"],
        "inputs": {
            "public": old["inputs"]["public"],
            "spectral": old["inputs"]["spectral"],
            **sources,
        },
        "worldcover": old["worldcover"],
        "osm": old["osm"],
        "teacher": teachers,
        "model": {"width": 8, "initial_alpha": 0.15},
        "loss": {"preservation_weight": 0.05},
        **{
            key: old[key]
            for key in [
                "steps",
                "warmup_steps",
                "batch_size",
                "lr",
                "weight_decay",
                "grad_clip",
                "seed",
                "amp",
                "device",
                "reserve_gib",
            ]
        },
        "output": str(tmp_path / "adaptive"),
    }
    path = tmp_path / "adaptive.json"
    dump(path, spec)
    return path, spec


def test_adaptive_recovery_is_exact_and_terminal_export_is_cold_loadable(tmp_path):
    path, spec = fixture_spec(tmp_path)
    assert training.train(path, max_updates=2)["state"] == "checkpointed"
    assert training.train(path)["state"] == "complete"
    root = Path(spec["output"])
    first = torch.load(str(root / "model.weights.pt"), map_location="cpu", weights_only=True)
    other = copy.deepcopy(spec)
    other["output"] = str(tmp_path / "full")
    full = tmp_path / "full.json"
    dump(full, other)
    training.train(full)
    second = torch.load(
        str(tmp_path / "full/model.weights.pt"), map_location="cpu", weights_only=True
    )
    assert all(torch.equal(first["model"][k], second["model"][k]) for k in first["model"])
    assert first["summary"]["unique_training_tiles"] == 2
    assert first["summary"]["only_gate_parameters_optimized"] is True
    training.export(path)
    array = np.load(root / "embedding/annual.npy")
    assert array.shape == (2, 64, 4, 4) and array.dtype == np.float16
    assert np.isfinite(array).all()
    with (root / "model.weights.pt").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        training.train(path)


def test_adaptive_export_does_not_read_weak_label_payloads(tmp_path):
    path, spec = fixture_spec(tmp_path)
    training.train(path)
    Path(spec["worldcover"]["path"]).unlink()
    Path(spec["osm"]["path"]).unlink()
    training.export(path)
    assert np.isfinite(np.load(Path(spec["output"]) / "embedding/annual.npy")).all()


def test_adaptive_specification_rejects_unknown_duplicate_and_invalid_fields(tmp_path):
    path, spec = fixture_spec(tmp_path)
    for key, value in [("evaluation", "never read"), ("batch_size", True), ("lr", 0), ("seed", -1)]:
        dump(path, {**spec, key: value})
        with pytest.raises(ValueError):
            training.specification(path)
    path.write_text(json.dumps(spec)[:-1] + ', "steps": 8}')
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        training.specification(path)


def test_adaptive_rejects_incompatible_coordinates_and_wrong_teacher_export_binding(tmp_path):
    path, spec = fixture_spec(tmp_path)
    parsed, index = training.specification(path)
    old = copy.deepcopy(spec)
    spec["inputs"]["geometry"]["index_sha256"] = "0" * 64
    dump(path, spec)
    with pytest.raises(ValueError):
        training.TrainingData(*training.specification(path))
    dump(path, old)
    status = Path(old["teacher"]["geometry_export"]["path"])
    value = json.loads(status.read_text())
    value["model_sha256"] = "0" * 64
    dump(status, value)
    old["teacher"]["geometry_export"] = ref(status)
    dump(path, old)
    with pytest.raises(ValueError, match="[Ee]xport|[Tt]eacher"):
        training.TrainingData(*training.specification(path))
    assert parsed["steps"] == 4 and len(index["records"]) == 2


def test_adaptive_cli_dispatches_the_registered_command(capsys):
    from xuannv_embedding.training.experiment import main

    with pytest.raises(SystemExit) as error:
        main(["content-adaptive-residual", "--help"])
    assert error.value.code == 0
    assert "--phase" in capsys.readouterr().out


def test_export_rejects_changed_terminal_payload_before_first_export(tmp_path):
    path, spec = fixture_spec(tmp_path)
    training.train(path)
    with (Path(spec["output"]) / "model.weights.pt").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        training.export(path)


def test_inconsistent_frozen_export_shape_is_rejected(tmp_path):
    path, spec = fixture_spec(tmp_path)
    status = Path(spec["teacher"]["semantic_export"]["path"])
    value = json.loads(status.read_text())
    value["shape"] = [2, 64, 8, 8]
    dump(status, value)
    spec["teacher"]["semantic_export"] = ref(status)
    dump(path, spec)
    with pytest.raises(ValueError, match="shape|geometry"):
        training.TrainingData(*training.specification(path))


def test_previous_batch_and_forward_output_are_released_before_next_gather(tmp_path, monkeypatch):
    import weakref

    path, _ = fixture_spec(tmp_path)
    real_resident = training.ResidentFeatures
    real_forward = training.ContentAdaptiveResidual.forward
    previous = {"inputs": [], "output": None}

    class CheckedResident:
        def __init__(self, data, device):
            self.inner = real_resident(data, device)
            self.resident_bytes = self.inner.resident_bytes

        def batch(self, ids):
            assert all(ref() is None for ref in previous["inputs"])
            assert previous["output"] is None or previous["output"]() is None
            values = self.inner.batch(ids)
            previous["inputs"] = [
                weakref.ref(value) for value in values if isinstance(value, torch.Tensor)
            ]
            return values

    def checked_forward(model, *args, **kwargs):
        value = real_forward(model, *args, **kwargs)
        previous["output"] = weakref.ref(value)
        return value

    monkeypatch.setattr(training, "ResidentFeatures", CheckedResident)
    monkeypatch.setattr(training.ContentAdaptiveResidual, "forward", checked_forward)
    training.train(path)
