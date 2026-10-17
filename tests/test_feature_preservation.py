import json

import numpy as np
import pytest
import torch
from torch import nn

from xuannv_embedding.export.context import sha
from xuannv_embedding.models.model import AEFOutput
from xuannv_embedding.training.feature_preservation import (
    attach_teacher,
    audit_frozen_checkpoint,
    freeze_public_parameters,
    load_teacher,
    preservation_loss,
)


def test_freeze_public_excludes_backbone_but_retains_fusion_and_auxiliary_gradients():
    system = nn.Module()
    system.model = nn.Module()
    system.model.base = nn.Linear(2, 2)
    system.model.encoders = nn.Linear(2, 2)
    system.criterion = nn.Linear(2, 1)
    freeze_public_parameters(system)
    assert not any(p.requires_grad for p in system.model.base.parameters())
    assert all(p.requires_grad for p in system.model.encoders.parameters())
    assert all(p.requires_grad for p in system.criterion.parameters())


def test_frozen_checkpoint_audit_rejects_any_changed_backbone_tensor(tmp_path):
    original = tmp_path / "initial.pt"
    checkpoint = tmp_path / "after.pt"
    model = {"base.weight": torch.ones(2, 2), "encoders.weight": torch.ones(2, 2)}
    torch.save({"model": model}, original)
    model["encoders.weight"] = torch.zeros(2, 2)
    torch.save({"model": model}, checkpoint)
    assert audit_frozen_checkpoint(checkpoint, original)["public_tensors_unchanged"]
    model["base.weight"][0, 0] = 0
    torch.save({"model": model}, checkpoint)
    with pytest.raises(ValueError, match="backbone"):
        audit_frozen_checkpoint(checkpoint, original)


def test_anchor_is_zero_for_matching_direction_and_has_student_only_gradients():
    values = torch.tensor([[[[[1.0]], [[0.0]]]]], requires_grad=True)
    output = AEFOutput(values, values.mean((3, 4)), {})
    teacher = torch.tensor([[[[1.0]], [[0.0]]]])
    loss = preservation_loss(output, teacher, torch.ones(1, 1, 1))
    assert loss.item() == 0
    different = torch.tensor([[[[0.0]], [[1.0]]]])
    loss = preservation_loss(output, different, torch.ones(1, 1, 1))
    loss.backward()
    assert loss.item() == 1
    assert values.grad.abs().sum() > 0
    assert different.grad is None


def test_anchor_ignores_invalid_pixels_and_keeps_finite_gradients_when_all_missing():
    values = torch.randn(1, 2, 3, 2, 2, requires_grad=True)
    output = AEFOutput(values, values.mean((3, 4)), {})
    target = torch.zeros(1, 3, 2, 2)
    target[:, :, 0, 0] = float("nan")
    loss = preservation_loss(output, target, torch.zeros(1, 2, 2))
    loss.backward()
    assert loss.item() == 0
    assert torch.isfinite(values.grad).all()


def test_teacher_attachment_uses_public_observation_validity_and_preserves_sources():
    sample = {
        "targets": {"s2_recon": torch.ones(2, 3, 2, 2)},
        "target_masks": {"s2_recon": torch.zeros(2, 2, 2)},
    }
    sample["target_masks"]["s2_recon"][1, 0, 0] = 1
    teacher = np.ones((1, 3, 2, 2), np.float16)
    result = attach_teacher(sample, 0, teacher)
    assert "public_teacher_embedding" not in sample["targets"]
    assert result["target_masks"]["public_teacher_embedding"].sum() == 1
    result["targets"]["public_teacher_embedding"].zero_()
    assert teacher.sum() == teacher.size


def test_teacher_rejects_changed_index_geometry_or_missing_tiles(tmp_path):
    records = [{"patch_id": "a", "bounds": [0, 0, 2, 2], "crs": "EPSG:2193"}]
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": records, "months": ["2024-01"]}))
    path = tmp_path / "teacher.npy"
    np.save(path, np.ones((1, 3, 2, 2), np.float16))
    reference = {
        "path": str(path),
        "sha256": sha(path),
        "index": {"path": str(index), "sha256": sha(index)},
        "checkpoint_sha256": "a" * 64,
    }
    cache = {"records": records, "months": ["2024-01"]}
    assert load_teacher(reference, cache, channels=3, size=2).shape == (1, 3, 2, 2)
    cache["records"] = [{**records[0], "bounds": [2, 2, 4, 4]}]
    with pytest.raises(ValueError, match="geograph"):
        load_teacher(reference, cache, channels=3, size=2)
    cache["records"] = []
    with pytest.raises(ValueError):
        load_teacher(reference, cache, channels=3, size=2)


def test_regional_objective_adds_registered_anchor_without_changing_other_losses():
    from xuannv_embedding.training.regional_base import RegionalObjective

    class Loss(nn.Module):
        def forward(self, output, *args):
            return {"total": output.embedding_map.square().mean()}

    values = torch.randn(1, 2, 3, 2, 2, requires_grad=True)
    output = AEFOutput(values, values.mean((3, 4)), {})
    criterion = RegionalObjective(Loss(), 3, 2, {})
    targets = {
        "esa_worldcover": torch.ones(1, 2, 2, dtype=torch.long),
        "public_teacher_embedding": torch.randn(1, 3, 2, 2),
    }
    masks = {"esa_worldcover": torch.ones(1, 2, 2), "public_teacher_embedding": torch.ones(1, 2, 2)}
    previous = criterion(output, targets, masks)
    criterion.teacher_weight = 2.0
    current = criterion(output, targets, masks)
    assert torch.allclose(current["total"] - previous["total"], 2 * current["public_anchor"])
    current["total"].backward()
    assert torch.isfinite(values.grad).all()
