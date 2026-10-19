import copy

import pytest
import torch
from test_time_modulated import inputs

from xuannv_embedding.models.gated_geometry import (
    GatedGeometryFusion,
    GatedSpatialContext,
    relational_geometry_loss,
)
from xuannv_embedding.models.time_modulated import TimeModulatedFusion


def test_zero_projection_recovers_unmixed_features_and_checkpoint_gradients_are_exact():
    torch.manual_seed(41)
    mixed = GatedGeometryFusion(width=8)
    baseline = TimeModulatedFusion(width=8)
    shared = {k: v for k, v in mixed.state_dict().items() if "local_mixer" not in k}
    baseline.load_state_dict(shared, strict=True)
    with torch.no_grad():
        mixed.base.context_delta.weight.normal_(0, 0.02)
        baseline.base.context_delta.weight.copy_(mixed.base.context_delta.weight)
    x = inputs(3)
    torch.testing.assert_close(mixed(**x).embedding, baseline(**x).embedding, rtol=0, atol=0)
    a = GatedSpatialContext(8)
    b = copy.deepcopy(a)
    a.train()
    b.eval()
    with torch.no_grad():
        a.project.weight.normal_(0, 0.03)
        b.load_state_dict(a.state_dict())
    y = torch.randn(2, 26, 5, 5, requires_grad=True)
    z = y.detach().clone().requires_grad_()
    a(y).square().mean().backward()
    b(z).square().mean().backward()
    torch.testing.assert_close(y.grad, z.grad, rtol=0, atol=0)
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(a.parameters(), b.parameters()))


def test_local_teacher_is_pre_affine_and_missing_sources_recover_learned_public():
    model = GatedGeometryFusion(width=8)
    x = inputs(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    first = model(**x)
    with torch.no_grad():
        model.affine.bias.add_(3)
    second = model(**x)
    assert torch.equal(first.local_teacher, second.local_teacher)
    x["detail"] = None
    x["views"].fill_(float("nan"))
    x["view_mask"].zero_()
    out = model(**x)
    assert torch.equal(out.embedding, out.public_embedding)
    assert not out.local_teacher_mask.any()
    assert torch.isfinite(out.local_teacher).all()


def test_geometry_matches_reference_and_uses_fp32_under_bf16_autocast():
    torch.manual_seed(21)
    student = torch.randn(1, 64, 2, 2, requires_grad=True)
    teacher = torch.randn(1, 1024, 2, 2, requires_grad=True)
    domain = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    labels = torch.tensor([[[1, 1], [2, 0]]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = relational_geometry_loss(
            student, teacher, domain, labels, step=0, samples=4, same_class_weight=2.0, seed=41
        )
    assert loss.dtype == torch.float32
    s = torch.nn.functional.normalize(student.flatten(2).transpose(1, 2).reshape(-1, 64), dim=1)
    t = torch.nn.functional.normalize(
        teacher.detach().flatten(2).transpose(1, 2).reshape(-1, 1024), dim=1
    )
    y = labels.flatten()
    w = torch.where((y[:, None] == y[None, :]) & (y[:, None] > 0), 2.0, 1.0)
    w.fill_diagonal_(0)
    reference = ((s @ s.T - t @ t.T).square() * w).sum() / w.sum()
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert student.grad.abs().sum() > 0 and teacher.grad is None


def test_absent_geometry_has_zero_finite_loss_and_invalid_targets_are_rejected():
    s = torch.randn(2, 64, 2, 2, requires_grad=True)
    t = torch.zeros(2, 1024, 2, 2)
    m = torch.zeros(2, 1, 2, 2, dtype=torch.bool)
    y = torch.zeros(2, 2, 2, dtype=torch.long)
    z = relational_geometry_loss(s, t, m, y, step=4, samples=4, same_class_weight=2.0, seed=41)
    assert z.item() == 0 and torch.isfinite(z)
    z.backward()
    assert torch.equal(s.grad, torch.zeros_like(s.grad))
    with pytest.raises(ValueError):
        relational_geometry_loss(
            s, t, m, y.float(), step=4, samples=4, same_class_weight=2.0, seed=41
        )
