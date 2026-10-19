import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_content_adaptive_training import fixture_spec as parent_fixture
from test_content_adaptive_training import ref

from xuannv_embedding.export.context import dump
from xuannv_embedding.training import public_fallback_gate as training


def fixture_spec(tmp_path):
    path, old = parent_fixture(tmp_path)
    spec = copy.deepcopy(old)
    spec["schema"] = "public-fallback-gate-v1"
    spec["inputs"]["foundation"] = copy.deepcopy(old["inputs"]["semantic"])
    provenance = {
        "checkpoint": old["teacher"]["semantic_checkpoint"],
        "spec": old["teacher"]["semantic_spec"],
        "export": old["teacher"]["semantic_export"],
    }
    proof = tmp_path / "public-lineage.json"
    dump(
        proof,
        {
            "state": "passed",
            "only_public_observations": True,
            "primary_reference_maps_loaded": False,
            "checkpoint": provenance["checkpoint"],
            "specification": provenance["spec"],
            "embedding": {k: spec["inputs"]["foundation"][k] for k in ["path", "sha256"]},
            "index_sha256": old["index"]["sha256"],
        },
    )
    spec["fallback_teacher"] = {**provenance, "public_lineage": ref(proof)}
    spec["head_warmup_steps"] = 2
    spec["model"] = {"width": 8, "initial_weights": [0.05, 0.8075, 0.1425]}
    spec["loss"]["preservation_weight"] = 0.2
    spec["output"] = str(tmp_path / "fallback")
    dump(path, spec)
    return path, spec


def test_staged_training_recovery_matches_full_and_head_only_state_is_auditable(tmp_path):
    path, spec = fixture_spec(tmp_path)
    assert training.train(path, max_updates=2)["state"] == "checkpointed"
    root = Path(spec["output"])
    warm = torch.load(str(root / "latest.pt"), map_location="cpu", weights_only=True)
    assert len(warm["optimizer"]["state"]) == 6
    assert not warm["model"]["project.weight"].any()
    training.train(path)
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
    assert first["model"]["project.weight"].abs().sum() > 0
    assert first["summary"]["only_gate_parameters_optimized"] is False
    assert first["summary"]["auxiliary_heads_optimized"] is True
    training.export(path)
    values = np.load(root / "embedding/annual.npy")
    assert values.shape == (2, 64, 4, 4) and np.isfinite(values).all()


def test_fallback_export_does_not_load_weak_label_files(tmp_path):
    path, spec = fixture_spec(tmp_path)
    training.train(path)
    Path(spec["worldcover"]["path"]).unlink()
    Path(spec["osm"]["path"]).unlink()
    assert training.export(path)["weak_labels_loaded"] is False


def test_private_fallback_marker_and_incompatible_index_are_rejected(tmp_path):
    path, spec = fixture_spec(tmp_path)
    proof = Path(spec["fallback_teacher"]["public_lineage"]["path"])
    record = json.loads(proof.read_text())
    record["only_public_observations"] = False
    dump(proof, record)
    spec["fallback_teacher"]["public_lineage"] = ref(proof)
    dump(path, spec)
    with pytest.raises(ValueError, match="[Pp]ublic|lineage"):
        training.TrainingData(*training.specification(path))
    record["only_public_observations"] = True
    dump(proof, record)
    spec["fallback_teacher"]["public_lineage"] = ref(proof)
    spec["inputs"]["foundation"]["index_sha256"] = "0" * 64
    dump(path, spec)
    with pytest.raises(ValueError, match="index"):
        training.TrainingData(*training.specification(path))


def test_fallback_schema_rejects_unknown_fields_and_invalid_warmup(tmp_path):
    path, spec = fixture_spec(tmp_path)
    for key, value in [
        ("head_warmup_steps", True),
        ("head_warmup_steps", 4),
        ("evaluation", "never read"),
    ]:
        dump(path, {**spec, key: value})
        with pytest.raises(ValueError):
            training.specification(path)


def test_public_fallback_cli_dispatches_help(capsys):
    from xuannv_embedding.training.experiment import main

    with pytest.raises(SystemExit) as error:
        main(["public-fallback-gate", "--help"])
    assert error.value.code == 0
    assert "--phase" in capsys.readouterr().out
