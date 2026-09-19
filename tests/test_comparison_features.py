import numpy as np
import pytest
import torch

from xuannv_embedding.downstream.comparison_features import raw_features, validate_grid


def test_raw_features_preserve_channels_and_zero_invalid_observations():
    sample = {
        "source_frames": {"s2": torch.ones(2, 3, 4, 4)},
        "source_masks": {"s2": torch.ones(2, 1, 4, 4)},
    }
    sample["source_frames"]["s2"][-1, :, 0, 0] = float("nan")
    sample["source_masks"]["s2"][-1, :, 0, 0] = 0
    result = raw_features(sample, ["s2"])
    assert result.shape == (4, 4, 4)
    assert np.isfinite(result).all()
    assert (result[:, 0, 0] == 0).all()


def test_external_grid_rejects_geographic_offset():
    reference = {"bounds": [0, 0, 1280, 1280], "shape": [128, 128]}
    validate_grid(reference, [0, 0, 1280, 1280])
    with pytest.raises(ValueError):
        validate_grid(reference, [10, 0, 1290, 1280])


def test_static_and_temporal_feature_files_use_same_spatial_tensor(tmp_path):
    from xuannv_embedding.downstream.comparison_features import read_feature

    feature = torch.arange(48).reshape(3, 4, 4).float()
    torch.save(feature, tmp_path / "static.pt")
    np.savez(tmp_path / "temporal.npz", embedding=feature.numpy()[None])
    torch.testing.assert_close(read_feature(tmp_path / "static.pt"), feature)
    torch.testing.assert_close(read_feature(tmp_path / "temporal.npz"), feature)


@pytest.mark.parametrize("value", [torch.zeros(2, 3), torch.full((2, 3, 3), float("nan")), {}])
def test_static_feature_rejects_invalid_tensors(tmp_path, value):
    from xuannv_embedding.downstream.comparison_features import read_feature

    path = tmp_path / "invalid.pt"
    torch.save(value, path)
    with pytest.raises(ValueError):
        read_feature(path)


def test_dinov3_requires_completed_reproduction(tmp_path):
    import json
    from types import SimpleNamespace

    from xuannv_embedding.downstream.comparison_features import export_dinov3

    (tmp_path / "feature_manifest.json").write_text(json.dumps({"state": "audited"}))
    (tmp_path / "reproduction.json").write_text(json.dumps({"state": "mismatch"}))
    args = SimpleNamespace(source=tmp_path, output=tmp_path / "output")
    with pytest.raises(ValueError, match="audits must pass"):
        export_dinov3(args, {}, tmp_path / "cache.json")
    assert not args.output.exists()
