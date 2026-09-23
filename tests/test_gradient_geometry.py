import pytest
import torch

from xuannv_embedding.training.gradient_geometry import embedding_geometry, gradient_geometry


def test_gradient_geometry_detects_opposition_and_keeps_zero_gradient_undefined():
    weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    unused = torch.nn.Parameter(torch.ones(2))
    loss = weight.sum()
    report = gradient_geometry(
        {"semantic": loss, "reconstruction": -2 * loss, "zero": loss * 0},
        {"fusion": {"weight": weight}, "unused": {"unused": unused}},
    )
    assert report["fusion"]["cosines"]["semantic|reconstruction"] == pytest.approx(-1)
    assert report["fusion"]["cosines"]["semantic|zero"] is None
    assert report["fusion"]["norms"]["reconstruction"] == pytest.approx(2 * 2**0.5)
    assert report["unused"]["norms"]["semantic"] == 0
    assert report["unused"]["cosines"]["semantic|reconstruction"] is None
    assert weight.grad is None and unused.grad is None
    assert torch.equal(weight, torch.tensor([1.0, 2.0]))


def test_embedding_geometry_measures_identity_and_rank_without_pixel_independence_claim():
    embedding = torch.zeros(1, 2, 3, 2, 2)
    embedding[:, :, 0] = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    report = embedding_geometry(embedding, embedding, stride=1)
    assert report["relative_l2"] == 0
    assert report["effective_rank"] == pytest.approx(1)
    assert report["sampled_vectors"] == 8
    with pytest.raises(ValueError, match="finite"):
        embedding_geometry(embedding * float("nan"), embedding, stride=1)
