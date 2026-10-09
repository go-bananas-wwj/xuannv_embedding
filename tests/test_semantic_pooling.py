import pytest
import torch

from xuannv_embedding.training.losses import SemanticProbeLoss


def test_mean_semantic_supervision_reaches_every_month():
    probe = SemanticProbeLoss(embed_dim=2, tasks=["field"], hidden_dim=0, pooling="mean")
    x = torch.randn(1, 6, 2, 3, 3, requires_grad=True)
    loss, _ = probe(x, {"field": torch.ones(1, 3, 3)}, None)
    loss.backward()
    assert x.grad is not None
    assert all(x.grad[:, month].abs().sum() > 0 for month in range(6))
    assert torch.equal(x.grad[:, 0], x.grad[:, 5])


def test_mean_probe_matches_explicit_average_with_shared_weights():
    mean = SemanticProbeLoss(embed_dim=2, tasks=["field"], hidden_dim=0, pooling="mean")
    monthly = SemanticProbeLoss(embed_dim=2, tasks=["field"], hidden_dim=0, month_index=0)
    monthly.load_state_dict(mean.state_dict(), strict=True)
    x = torch.randn(1, 6, 2, 3, 3)
    labels = {"field": torch.ones(1, 3, 3)}
    a, _ = mean(x, labels, None)
    b, _ = monthly(x.mean(1, keepdim=True), labels, None)
    assert torch.equal(a, b)


def test_probe_rejects_unknown_pooling():
    with pytest.raises(ValueError, match="pooling"):
        SemanticProbeLoss(embed_dim=2, tasks=["field"], pooling="median")
