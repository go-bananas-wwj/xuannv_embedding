from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xuannv_embedding.config import DataConfig
from xuannv_embedding.data.raster_dataset import RegionRasterDataset, _Observation
from xuannv_embedding.models.incremental_highres import HighResResidual
from xuannv_embedding.training.masking import apply_input_masking


def test_monthly_highres_preserves_time_and_missing_months_and_excludes_outside_window():
    data = DataConfig(["2026-01", "2026-02", "2026-03"], [], monthly_highres=True)
    ds = RegionRasterDataset.__new__(RegionRasterDataset)
    ds.config = SimpleNamespace(data=data)
    ds.months = [202601, 202602, 202603]
    ds.highres_sizes = {"extra": (2, 2)}
    obs = [
        _Observation(m, torch.full((1, 2, 2), v), torch.ones(2, 2))
        for m, v in [(202601, 2.0), (202603, 8.0), (202604, 100.0)]
    ]
    frames, valid = ds._highres_input("extra", obs, 1)
    assert frames.shape == (3, 1, 2, 2)
    assert frames.mean((1, 2, 3)).tolist() == [2.0, 0.0, 8.0]
    assert valid[:, 0, 0, 0].tolist() == [1.0, 0.0, 1.0]
    ds.config = SimpleNamespace(data=replace(data, monthly_highres=False))
    assert ds._highres_input("extra", obs, 1)[0].shape == (1, 2, 2)


def test_normalized_monthly_resampling_does_not_dilute_valid_constant():
    ds = RegionRasterDataset.__new__(RegionRasterDataset)
    ds.months, ds.output_size = [202601], (2, 2)
    valid = torch.tensor([[1.0, 0.0, 0.0]]).expand(3, -1)
    obs = [_Observation(202601, valid[None] * 10, valid)]
    frames, _, masks = ds._monthly_continuous(obs, 1, normalized=True)
    torch.testing.assert_close(frames[:, 0][masks.bool()], torch.full((2,), 10.0))


@pytest.mark.parametrize("native", [False, True])
def test_monthly_branch_keeps_months_separate_and_missing_month_is_exact_base(native):
    torch.manual_seed(7)
    branch = HighResResidual(1, 4, native=native)
    nn.init.normal_(branch.correction.weight, std=0.1)
    z = torch.nn.functional.normalize(torch.randn(1, 3, 4, 4, 4), dim=2)
    image, mask = torch.randn(1, 3, 1, 8, 8), torch.ones(1, 3, 1, 8, 8)
    mask[:, 1] = 0
    a = branch(z, image, mask)
    changed = image.clone()
    changed[:, 0] *= -3
    changed[:, 1] = float("nan")
    b = branch(z, changed, mask)
    assert not torch.equal(a[:, 0], b[:, 0])
    torch.testing.assert_close(a[:, 1:], b[:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(a[:, 1], z[:, 1], rtol=0, atol=0)
    a.sum().backward()
    assert branch.correction.weight.grad is not None
    with pytest.raises(ValueError, match="month|time"):
        branch(z, image[:, :2], mask[:, :2])


def test_month_dropout_hides_monthly_highres_without_changing_quality_or_targets():
    b = {
        "source_frames": {"s": torch.ones(1, 1, 1, 4, 4)},
        "source_masks": {"s": torch.ones(1, 1)},
        "highres_frames": {"hr": torch.ones(1, 1, 1, 8, 8)},
        "highres_masks": {"hr": torch.ones(1, 1, 1, 8, 8)},
        "highres_quality_masks": {"hr": torch.ones(1, 1, 1, 8, 8)},
        "targets": {"hr": torch.ones(1, 1, 1, 4, 4)},
    }
    apply_input_masking(
        b,
        {
            "enabled": True,
            "month_dropout_prob": 1.0,
            "max_months_per_sample": 1,
            "drop_availability_masks": False,
        },
    )
    assert b["highres_frames"]["hr"].count_nonzero() == 0
    assert b["highres_masks"]["hr"].count_nonzero() == 0
    assert b["source_frames"]["s"].count_nonzero() == 0
    assert b["highres_quality_masks"]["hr"].min() == 1
    assert b["targets"]["hr"].min() == 1


def test_monthly_spatial_mask_uses_same_ground_locations_across_resolutions():
    torch.manual_seed(9)
    b = {
        "source_frames": {"s": torch.ones(1, 2, 1, 4, 4)},
        "source_masks": {"s": torch.ones(1, 2)},
        "highres_frames": {"hr": torch.ones(1, 2, 1, 8, 8)},
        "highres_masks": {"hr": torch.ones(1, 2, 1, 8, 8)},
    }
    apply_input_masking(
        b,
        {
            "enabled": True,
            "spatial_block_prob": 1.0,
            "spatial_block_ratio": 0.5,
            "spatial_block_size": 1,
        },
    )
    expected = b["source_frames"]["s"].repeat_interleave(2, -1).repeat_interleave(2, -2)
    torch.testing.assert_close(b["highres_frames"]["hr"], expected, rtol=0, atol=0)
    torch.testing.assert_close(b["highres_masks"]["hr"], expected, rtol=0, atol=0)
