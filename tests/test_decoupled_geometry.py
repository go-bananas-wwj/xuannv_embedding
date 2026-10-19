import copy

import torch
from test_time_modulated import inputs

from xuannv_embedding.models.decoupled_geometry import PublicSemanticGeometryFusion


def test_weak_semantics_cannot_directly_train_local_paths():
    torch.manual_seed(41)
    model = PublicSemanticGeometryFusion(width=8)
    with torch.no_grad():
        model.base.public_delta.weight.normal_(0, 0.03)
        model.base.context_delta.weight.normal_(0, 0.03)
    out = model(**inputs(2))
    loss = (
        out.worldcover_logits.square().mean()
        + out.area_logits.square().mean()
        + out.geometry_logits.square().mean()
    )
    loss.backward()
    assert model.base.public_delta.weight.grad.abs().sum() > 0
    for module in [
        model.affine,
        model.attention,
        model.base.context_delta,
        model.base.detail_delta,
    ]:
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in module.parameters())


def test_fused_reconstruction_still_trains_local_paths_and_missing_sources_fall_back():
    model = PublicSemanticGeometryFusion(width=8)
    with torch.no_grad():
        model.base.context_delta.weight.normal_(0, 0.03)
    x = inputs(2)
    out = model(**x)
    out.context_reconstruction.square().mean().backward()
    assert model.affine.weight.grad.abs().sum() > 0
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    x["detail"] = None
    x["view_mask"].zero_()
    x["views"].fill_(float("nan"))
    y = model(**x)
    assert torch.equal(y.embedding, y.public_embedding) and torch.isfinite(y.embedding).all()


def test_semantic_heads_do_not_depend_on_observed_local_image_values():
    model = PublicSemanticGeometryFusion(width=8)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    a = inputs(1)
    b = copy.deepcopy(a)
    b["views"].mul_(4)
    b["detail"].mul_(2)
    x, y = model(**a), model(**b)
    assert torch.equal(x.worldcover_logits, y.worldcover_logits)
    assert torch.equal(x.area_logits, y.area_logits)
    assert not torch.equal(x.embedding, y.embedding)
