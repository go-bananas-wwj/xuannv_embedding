from types import SimpleNamespace

import numpy as np
import pytest
import torch

from xuannv_embedding.downstream.common_reconstruction import masked_embedding


class Echo(torch.nn.Module):
    def forward(self, frames, masks, timestamps, highres, highres_masks):
        value = frames["optical"]
        return SimpleNamespace(embedding_map=torch.cat([value, torch.full_like(value, 2)], 2))


def sample():
    return {
        "region": "fixture",
        "patch_id": "p0",
        "timestamps": torch.tensor([202512, 202601, 202602]),
        "source_frames": {
            "optical": torch.tensor([1.0, 1000.0, 3.0])[:, None, None, None]
            .expand(3, 1, 2, 2)
            .clone()
        },
        "source_masks": {"optical": torch.ones(3)},
        "highres_frames": {},
        "highres_masks": {},
        "targets": {},
        "target_masks": {},
        "supervised_labels": {},
        "supervised_label_masks": {},
    }


def test_static_mean_is_computed_after_target_masking_and_does_not_modify_source():
    data = sample()
    model = Echo().eval()
    monthly, a = masked_embedding(
        model, data, ["optical"], 1, prefix=False, device=torch.device("cpu")
    )
    mean, b = masked_embedding(
        model, data, ["optical"], 1, prefix=False, device=torch.device("cpu"), representation="mean"
    )
    np.testing.assert_allclose(monthly[:, 0, 0], [0, 2])
    np.testing.assert_allclose(mean[:, 0, 0], [4 / 3, 2])
    assert a == b and data["source_frames"]["optical"][1].eq(1000).all()
    data["source_frames"]["optical"][1] = float("nan")
    changed, _ = masked_embedding(
        model, data, ["optical"], 1, prefix=False, device=torch.device("cpu"), representation="mean"
    )
    np.testing.assert_array_equal(changed, mean)


def test_prefix_mean_cannot_include_future_observations():
    data = sample()
    data["source_frames"]["optical"][2] = float("nan")
    mean, _ = masked_embedding(
        Echo().eval(),
        data,
        ["optical"],
        1,
        prefix=True,
        device=torch.device("cpu"),
        representation="mean",
    )
    np.testing.assert_allclose(mean[:, 0, 0], [1 / 3, 2])


@pytest.mark.parametrize("representation", ["monthly", "mean"])
def test_same_l2_normalization_is_available_for_both_representations(representation):
    value, _ = masked_embedding(
        Echo().eval(),
        sample(),
        ["optical"],
        1,
        prefix=False,
        device=torch.device("cpu"),
        representation=representation,
        normalize_embedding=True,
    )
    np.testing.assert_allclose(np.linalg.norm(value, axis=0), 1, atol=1e-6)


def test_unknown_temporal_representation_is_rejected():
    with pytest.raises(ValueError, match="representation"):
        masked_embedding(
            Echo().eval(),
            sample(),
            ["optical"],
            1,
            prefix=False,
            device=torch.device("cpu"),
            representation="future",
        )
