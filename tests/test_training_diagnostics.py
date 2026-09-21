import pytest
import torch

from xuannv_embedding.training.diagnostics import gradient_report
from xuannv_embedding.training.experiment import validate_cached_targets


def test_gradient_report_measures_weighted_objectives_without_accumulating_gradients():
    model = torch.nn.Linear(2, 1, bias=False)
    x = torch.tensor([[1.0, 2.0]])
    loss = model(x).sum()
    rows = gradient_report({"raw": loss, "weighted": 3 * loss}, dict(model.named_parameters()))
    assert rows["weighted"]["groups"]["weight"]["norm"] == pytest.approx(
        3 * rows["raw"]["groups"]["weight"]["norm"]
    )
    assert model.weight.grad is None


def test_cache_allows_loss_weight_search_but_rejects_target_data_changes():
    a = {"optical": {"channels": 3, "source": "extra", "weight": 0.9}}
    b = {"optical": {"channels": 3, "source": "extra", "weight": 0.45}}
    validate_cached_targets(a, b)
    b["optical"]["channels"] = 4
    with pytest.raises(ValueError, match="target schema"):
        validate_cached_targets(a, b)
    assert a["optical"]["weight"] == 0.9
