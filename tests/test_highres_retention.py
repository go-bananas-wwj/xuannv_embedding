import copy
from types import SimpleNamespace

import pytest
import torch
from test_observation_ablation import batch_fixture

from xuannv_embedding.data.observation_ablation import retain_highres_inputs


def data(size=8):
    batch = batch_fixture()
    batch["highres_frames"]["detail"] = torch.ones(1, 3, 2, size, size)
    batch["highres_masks"]["detail"] = torch.ones(1, 3, 1, size, size)
    return batch


def test_retention_is_nested_exact_on_valid_pixels_and_does_not_change_targets():
    batch = data()
    batch["highres_masks"]["detail"][0, 1, 0, :2, :3] = 0
    before = copy.deepcopy(batch)
    last = torch.zeros_like(batch["highres_masks"]["detail"], dtype=torch.bool)
    rng = torch.random.get_rng_state()
    for fraction in [0, 0.25, 0.5, 0.75, 1]:
        current = retain_highres_inputs(batch, ["detail"], fraction=fraction, seed=41)
        mask = current["highres_masks"]["detail"] > 0
        assert not (last & ~mask).any()
        for month in range(3):
            assert mask[0, month].sum() == int(
                fraction * (batch["highres_masks"]["detail"][0, month] > 0).sum()
            )
        assert current["targets"] is batch["targets"]
        assert current["source_frames"]["optical"] is batch["source_frames"]["optical"]
        last = mask
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(before["highres_frames"]["detail"], batch["highres_frames"]["detail"])
    assert torch.equal(before["highres_masks"]["detail"], batch["highres_masks"]["detail"])


def test_mask_is_resolution_consistent_for_complete_grids_and_repeated_calls():
    small = retain_highres_inputs(data(4), ["detail"], fraction=0.5, seed=12)
    large = retain_highres_inputs(data(8), ["detail"], fraction=0.5, seed=12)
    expected = small["highres_masks"]["detail"].repeat_interleave(2, -1).repeat_interleave(2, -2)
    assert torch.equal(expected, large["highres_masks"]["detail"])
    other = retain_highres_inputs(data(8), ["detail"], fraction=0.5, seed=12)
    assert torch.equal(other["highres_masks"]["detail"], large["highres_masks"]["detail"])


def test_missing_nan_source_is_zeroed_and_static_inputs_are_supported():
    batch = data()
    batch["highres_masks"]["detail"].zero_()
    batch["highres_frames"]["detail"].fill_(float("nan"))
    result = retain_highres_inputs(batch, ["detail", "static"], fraction=0.5, seed=1)
    assert torch.isfinite(result["highres_frames"]["detail"]).all()
    assert result["highres_masks"]["static"].sum() == 2


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), True])
def test_invalid_retention_rejected(fraction):
    with pytest.raises(ValueError):
        retain_highres_inputs(data(), ["detail"], fraction=fraction, seed=41)


def test_unregistered_sources_nonbinary_masks_and_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        retain_highres_inputs(data(), ["missing"], fraction=0.5, seed=41)
    with pytest.raises(ValueError):
        retain_highres_inputs(data(), ["detail", "detail"], fraction=0.5, seed=41)
    batch = data()
    batch["highres_masks"]["detail"][0, 0, 0, 0, 0] = 0.2
    with pytest.raises(ValueError):
        retain_highres_inputs(batch, ["detail"], fraction=0.5, seed=41)


@pytest.mark.parametrize(
    "change", ["missing_fraction", "missing_source", "public", "overlap", "seed", "probe"]
)
def test_export_retention_contract_is_checked_before_source_reads(tmp_path, monkeypatch, change):
    from xuannv_embedding.training import experiment_export

    config = SimpleNamespace(
        model=SimpleNamespace(
            input_sources={
                "detail": SimpleNamespace(role="highres"),
                "optical": SimpleNamespace(role="temporal"),
            }
        ),
        data=SimpleNamespace(months=["2026-05"]),
    )
    monkeypatch.setattr(experiment_export.Config, "from_yaml", lambda p: config)
    args = SimpleNamespace(
        batch_size=1,
        config=tmp_path / "unused",
        drop_source=[],
        prefix_month=None,
        retain_highres_source=["detail"],
        highres_retention=0.5,
        retention_seed=41,
        probe_output=None,
        cache=tmp_path / "must_not_read",
    )
    if change == "missing_fraction":
        args.highres_retention = None
    if change == "missing_source":
        args.retain_highres_source = []
    if change == "public":
        args.retain_highres_source = ["optical"]
    if change == "overlap":
        args.drop_source = ["detail"]
    if change == "seed":
        args.retention_seed = True
    if change == "probe":
        args.probe_output = tmp_path / "forbidden_probe"
    with pytest.raises(ValueError):
        experiment_export.run(args)
