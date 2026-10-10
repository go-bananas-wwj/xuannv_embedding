import copy

import pytest
import torch

from xuannv_embedding.training.multitask_followup import audit_checkpoint, process_identity


def checkpoint_fixture():
    base = {"model": {"a": torch.ones(2)}, "criterion": {"b": torch.zeros(2)}}
    state = {
        "epoch": 1,
        "git_sha": "a" * 40,
        "config_sha256": "b" * 64,
        "model": {"base.a": torch.ones(2), "new": torch.ones(2)},
        "criterion": {"b": torch.zeros(2)},
        "metrics": {"rank_random_states": [{}] * 6, "optimizer_steps": 8},
        "optimizer": {"state": {0: {"step": torch.tensor(8.0), "exp_avg": torch.ones(2)}}},
        "scheduler": {"last_epoch": 2},
    }
    registration = {"git_sha": "a" * 40, "config_sha256": "b" * 64, "world_size": 6}
    return state, base, registration


def test_followup_rejects_stale_checkpoint_and_skipped_updates():
    state, base, registration = checkpoint_fixture()
    with pytest.raises(ValueError, match="epoch"):
        audit_checkpoint(state, base, registration, epochs=3, steps=12)
    state["optimizer"]["state"][0]["step"] = torch.tensor(7.0)
    with pytest.raises(ValueError, match="actual optimizer"):
        audit_checkpoint(state, base, registration, epochs=2, steps=8)


def test_followup_rejects_changed_frozen_tensor_or_nonfinite_new_weight():
    state, base, registration = checkpoint_fixture()
    changed = copy.deepcopy(state)
    changed["model"]["base.a"][0] = 2
    with pytest.raises(ValueError, match="frozen"):
        audit_checkpoint(changed, base, registration, epochs=2, steps=8)
    state["model"]["new"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        audit_checkpoint(state, base, registration, epochs=2, steps=8)


def test_followup_accepts_complete_checkpoint_and_reports_real_steps():
    state, base, registration = checkpoint_fixture()
    result = audit_checkpoint(state, base, registration, epochs=2, steps=8)
    assert result["actual_optimizer_steps"] == [8]
    assert result["frozen_model_tensors"] == 1


def test_process_identity_distinguishes_live_process_from_missing_handle():
    import os

    identity = process_identity(os.getpid())
    assert identity["pid"] == os.getpid()
    assert identity == process_identity(os.getpid())
    assert process_identity(999999999) is None


def test_pairing_rejects_support_changes_and_inconsistent_selection_error(tmp_path):
    import json

    from xuannv_embedding.training.multitask_followup import paired_report

    directories = [tmp_path / "base", tmp_path / "candidate"]
    rows = []
    for family, source in [("C", "osm"), ("C", "esri"), ("R", "esri"), ("Q", "osm")]:
        rows.append(
            {
                "key": family + source,
                "family": family,
                "source": source,
                "task": "task",
                "seed": 1,
                "budget": 5,
                "error": 0.5,
                "metrics": {
                    "ap": 0.5,
                    "rmse": 0.5,
                    "support_tiles": [0],
                    "support_positions_sha256": "original",
                    "validation_pixels": 20,
                    "support_blocks": 1,
                    "validation_blocks": 2,
                    "queries": [0],
                },
            }
        )
    identity = {
        "protocol": "multitask-v3",
        "cache_sha256": "cache",
        "active_indices": [0, 1],
        "tasks": {},
        "test_scored": False,
    }
    for directory in directories:
        directory.mkdir()
        (directory / "identity.json").write_text(json.dumps(identity))
        (directory / "results.json").write_text(json.dumps(rows))
    assert paired_report(*directories)["paired_conditions"] == 4
    candidate = copy.deepcopy(rows)
    candidate[0]["metrics"]["support_positions_sha256"] = "changed"
    (directories[1] / "results.json").write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="support mismatch"):
        paired_report(*directories)
    candidate = copy.deepcopy(rows)
    candidate[0]["error"] = 0.1
    (directories[1] / "results.json").write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="selection error"):
        paired_report(*directories)


def test_followup_allows_only_registered_semantic_head_updates():
    state, base, registration = checkpoint_fixture()
    base["criterion"]["semantic_probe.probes.task.weight"] = torch.ones(1)
    state["criterion"]["semantic_probe.probes.task.weight"] = torch.zeros(1)
    registration["adaptation"] = {"freeze_base": True, "train_semantic_head": True}
    result = audit_checkpoint(state, base, registration, epochs=2, steps=8)
    assert result["updated_semantic_tensors"] == 1
    assert result["frozen_criterion_tensors"] == 1
    changed = copy.deepcopy(state)
    changed["criterion"]["b"][0] += 1
    with pytest.raises(ValueError, match="frozen criterion"):
        audit_checkpoint(changed, base, registration, epochs=2, steps=8)
    registration["adaptation"]["train_semantic_head"] = False
    with pytest.raises(ValueError, match="frozen criterion"):
        audit_checkpoint(state, base, registration, epochs=2, steps=8)
