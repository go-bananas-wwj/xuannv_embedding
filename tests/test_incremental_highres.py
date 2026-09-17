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


@pytest.mark.parametrize("freeze", [True, False])
def test_extending_a_learned_source_preserves_initial_output_and_freezes_only_old_modules(freeze):
    model = IncrementalHighResModel(
        Base(), {"first": 3}, {"first_recon": 3}, native=True, freeze_base=True
    )
    nn.init.normal_(model.branches["first"].correction.weight, std=0.01)
    frames = {"s": torch.randn(2, 1, 2, 8, 8)}
    masks, times = {"s": torch.ones(2, 1)}, torch.ones(2, 1)
    high = {"first": torch.randn(2, 3, 24, 24)}
    valid = {"first": torch.ones(2, 1, 24, 24)}
    expected = model(frames, masks, times, high, valid).embedding_map.detach()
    old = {k: v.clone() for k, v in model.state_dict().items()}
    model.extend({"second": 1}, {"second_recon": 1}, native=True, freeze_existing=freeze)
    assert list(model.branches) == ["first", "second"]
    model.train()
    assert all(p.requires_grad != freeze for p in model.base.parameters())
    assert all(p.requires_grad != freeze for p in model.branches["first"].parameters())
    assert all(p.requires_grad for p in model.branches["second"].parameters())
    high["second"] = torch.randn(2, 1, 16, 16)
    valid["second"] = torch.ones(2, 1, 16, 16)
    output = model(frames, masks, times, high, valid)
    torch.testing.assert_close(output.embedding_map, expected, rtol=0, atol=0)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    sum(v.square().mean() for v in output.reconstructions.values()).backward()
    opt.step()
    if freeze:
        for k, v in old.items():
            torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
        valid["second"].zero_()
        actual = model(frames, masks, times, high, valid).embedding_map
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert any(not torch.equal(v, model.state_dict()[k]) for k, v in old.items())
