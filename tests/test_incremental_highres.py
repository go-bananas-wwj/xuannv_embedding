import pytest
import torch
from torch import nn

from xuannv_embedding.models.incremental_highres import HighResResidual, IncrementalHighResModel
from xuannv_embedding.models.model import AEFOutput


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(2, 4, 1)
        self.embed_dim = 4
        self.decoders = nn.ModuleDict({"public": nn.Conv2d(4, 2, 1)})

    def forward(self, frames, masks, timestamps):
        z = torch.nn.functional.normalize(self.projection(frames["s"][:, 0]), dim=1)[:, None]
        return AEFOutput(z, z.mean((-2, -1)), {})


@pytest.mark.parametrize("native", [False, True])
def test_missing_highres_is_exact_base_and_frozen_weights_do_not_update(native):
    torch.set_num_threads(1)
    base = Base()
    model = IncrementalHighResModel(
        base, {"extra": 3}, {"extra_recon": 3}, native=native, freeze_base=True
    )
    x = {"s": torch.randn(2, 1, 2, 8, 8)}
    mask, t = {"s": torch.ones(2, 1)}, torch.ones(2, 1)
    high = {"extra": torch.randn(2, 3, 24, 24)}
    missing = {"extra": torch.zeros(2, 1, 24, 24)}
    expected = base(x, mask, t).embedding_map.detach()
    before = {k: v.clone() for k, v in base.state_dict().items()}
    model.train()
    assert not base.training
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    output = model(x, mask, t, high, {"extra": torch.ones_like(missing["extra"])})
    torch.testing.assert_close(output.embedding_map, expected, rtol=0, atol=0)
    sum(v.square().mean() for v in output.reconstructions.values()).backward()
    opt.step()
    actual = model(x, mask, t, high, missing).embedding_map
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for k, v in base.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert all(p.grad is None for p in base.parameters())


def test_invalid_highres_values_cannot_contaminate_valid_features():
    branch = HighResResidual(3, 4, native=True)
    nn.init.normal_(branch.correction.weight)
    z = torch.nn.functional.normalize(torch.randn(1, 2, 4, 8, 8), dim=2)
    x = torch.randn(1, 3, 24, 24)
    mask = torch.ones(1, 1, 24, 24)
    mask[:, :, :12] = 0
    dirty = torch.where(mask.bool(), x, torch.full_like(x, float("nan")))
    a = branch(z, x, mask)
    b = branch(z, dirty, mask)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a[:, :, :, :4], z[:, :, :, :4], rtol=0, atol=0)
