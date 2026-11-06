import numpy as np
import pytest
from affine import Affine


def test_cloud_resampling_keeps_physical_extent_and_does_not_fill_missing_pixels():
    from xuannv_embedding.data_process.v5_resolution import downsample_for_cloud

    values = np.arange(16, dtype="f4").reshape(1, 4, 4)
    valid = np.ones((4, 4), bool)
    valid[0, 0] = False
    values[0, 0, 0] = 999999
    transform = Affine(5, 0, 300000, 0, -5, 4000000)
    out, mask, coarse_transform = downsample_for_cloud(values, valid, transform, "EPSG:32650")
    assert out.shape == (1, 2, 2)
    assert coarse_transform * (2, 2) == transform * (4, 4)
    assert not mask[0, 0] and mask.sum() == 3
    assert out[0, 0, 0] == 0
    np.testing.assert_allclose(out[0, 1, 1], 12.5)


def test_resolution_comparison_reports_changes_without_claiming_accuracy():
    from xuannv_embedding.data_process.v5_resolution import mask_difference

    valid = np.array([[1, 1], [1, 0]], bool)
    baseline = np.array([[1, 0], [1, 0]], bool)
    candidate = np.array([[1, 1], [0, 0]], bool)
    result = mask_difference(baseline, candidate, valid)
    assert result["newly_valid_pixels"] == 1 and result["newly_invalid_pixels"] == 1
    assert result["disagreement_fraction_of_data"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="NoData"):
        mask_difference(baseline, np.ones((2, 2), bool), valid)


def test_resolution_cli_requires_explicit_pilot_and_keeps_data_only_route(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_resolution

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda _: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *args: None)
    monkeypatch.setattr(
        v5_resolution,
        "compare_jilin_resolution",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    argv = ["--stage", "cloud-resolution-audit", "--device-id", "7"]
    for key in ("source-root", "dataset-root", "report-root", "base-root", "model-dir"):
        argv += ["--" + key, str(tmp_path / key)]
    with pytest.raises(SystemExit):
        v5_cli.main(argv)
    argv += ["--quality-root", str(tmp_path / "pilot")]
    assert v5_cli.main(argv) == 0
    assert calls == [
        (
            (
                tmp_path / "dataset-root",
                tmp_path / "report-root",
                tmp_path / "pilot",
                tmp_path / "model-dir",
            ),
            {"device_id": 7},
        )
    ]
