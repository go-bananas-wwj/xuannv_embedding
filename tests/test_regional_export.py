import pytest
import torch

from xuannv_embedding.export.regional_trial import representations


def test_annual_mean_is_normalized_without_concatenating_months():
    values = torch.zeros(1, 12, 2, 2, 2)
    values[:, :6, 0] = 1
    values[:, 6:, 1] = 1
    result = representations(values)
    assert result["annual"].shape == (1, 2, 2, 2)
    torch.testing.assert_close(result["annual"].norm(dim=1), torch.ones(1, 2, 2))
    torch.testing.assert_close(result["april"], values[:, 3])
    torch.testing.assert_close(result["mar_may"], values[:, 3])
    with pytest.raises(ValueError):
        representations(values[:, :6])
