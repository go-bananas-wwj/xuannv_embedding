import numpy as np
import pytest

from xuannv_embedding.downstream.common_reconstruction import fit_ridge, grid_support, predict_ridge


def test_ridge_matches_augmented_least_squares_and_preserves_constant_dimension():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(37, 4)) + 17
    x[:, 3] = 0.3
    y = rng.normal(size=(37, 2)) + 9
    fitted = fit_ridge(x, y, alpha=10)
    scale = x.std(0)
    scale[scale <= 1e-12] = 1
    z = (x - x.mean(0)) / scale
    expected = np.linalg.lstsq(
        np.concatenate([z, np.sqrt(10) * np.eye(4)]),
        np.concatenate([y - y.mean(0), np.zeros((4, 2))]),
        rcond=None,
    )[0]
    np.testing.assert_allclose(fitted["weights"], expected, atol=1e-12)
    query = x[:3] + 2
    np.testing.assert_allclose(
        predict_ridge(fitted, query),
        ((query - x.mean(0)) / scale) @ expected + y.mean(0),
        atol=1e-12,
    )
    assert fitted["scale"][3] == 1


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf")])
def test_ridge_rejects_invalid_penalty(alpha):
    with pytest.raises(ValueError):
        fit_ridge(np.ones((3, 2)), np.ones((3, 1)), alpha=alpha)


def test_ridge_rejects_empty_or_nonfinite_training_support():
    with pytest.raises(ValueError):
        fit_ridge(np.ones((0, 2)), np.ones((0, 1)), alpha=10)
    with pytest.raises(ValueError):
        fit_ridge(np.array([[np.nan, 1]]), np.ones((1, 1)), alpha=10)
    with pytest.raises(ValueError):
        fit_ridge(np.ones((2, 2)), np.ones((3, 1)), alpha=10)


def test_fixed_support_grid_filters_original_target_mask_only():
    x = np.arange(2 * 4 * 4).reshape(2, 4, 4)
    y = np.arange(3 * 4 * 4).reshape(3, 4, 4)
    mask = np.ones((4, 4), dtype=bool)
    mask[1, 3] = False
    features, targets, positions = grid_support(x, y, mask, stride=2)
    np.testing.assert_array_equal(positions, [[1, 1], [3, 1], [3, 3]])
    np.testing.assert_array_equal(features, x[:, positions[:, 0], positions[:, 1]].T)
    np.testing.assert_array_equal(targets, y[:, positions[:, 0], positions[:, 1]].T)
    with pytest.raises(ValueError):
        grid_support(x, y, mask, stride=0)


def test_runner_fits_only_training_embeddings_and_never_reads_test_record(tmp_path, monkeypatch):
    import argparse
    import json
    from types import SimpleNamespace

    import torch

    import xuannv_embedding.downstream.common_reconstruction as audit

    cache = tmp_path / "cache"
    cache.mkdir()
    records = []
    for index, value in enumerate((2.0, 9.0)):
        sample = {
            "region": "fixture",
            "patch_id": str(index),
            "timestamps": torch.tensor([202512, 202601, 202602]),
            "source_frames": {"optical": torch.full((3, 1, 2, 2), value)},
            "source_masks": {"optical": torch.ones(3)},
            "highres_frames": {},
            "highres_masks": {},
            "targets": {"recon": torch.full((3, 1, 2, 2), value)},
            "target_masks": {"recon": torch.ones(3, 2, 2)},
            "supervised_labels": {},
            "supervised_label_masks": {},
        }
        path = cache / f"{index}.pt"
        torch.save(sample, path)
        records.append({"path": str(path), "sha256": audit._sha(path)})
    records.append({"path": str(cache / "unread_test.pt"), "sha256": "unread"})
    audit._json(
        cache / "cache.json",
        {"records": records, "split": {"train": [0], "validation": [1], "test": [2], "buffer": []}},
    )
    config_path, checkpoint = tmp_path / "config.yaml", tmp_path / "model.pt"
    config_path.write_text("fixture")
    checkpoint.write_text("fixture")
    audit._json(
        tmp_path / "run.json",
        {
            "config_sha256": audit._sha(config_path),
            "cache_sha256": audit._sha(cache / "cache.json"),
        },
    )
    config = SimpleNamespace(
        model=SimpleNamespace(
            target_heads={
                "recon": SimpleNamespace(source="optical", channels=1, loss_type="continuous")
            },
            input_sources={"optical": None},
        ),
        data=SimpleNamespace(months=["2025-12", "2026-01", "2026-02"]),
    )
    monkeypatch.setattr(audit.Config, "from_yaml", lambda _: config)
    calls = []

    class CheckingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, frames, masks, timestamps, highres, highres_masks):
            assert not self.training and not torch.is_grad_enabled()
            assert not self.weight.requires_grad
            assert not frames["optical"][:, 1:].count_nonzero()
            assert not masks["optical"][:, 1:].count_nonzero()
            calls.append(float(frames["optical"][0, 0, 0, 0, 0]))
            return SimpleNamespace(
                embedding_map=torch.ones(1, 3, 2, 2, 2),
                reconstructions={"recon": torch.full_like(frames["optical"], 999.0)},
            )

    monkeypatch.setattr(audit, "_load_model", lambda *a: (CheckingModel(), 1))
    monkeypatch.setattr(audit, "_setup_device", lambda _: (torch.device("cpu"), False, 0))
    args = argparse.Namespace(
        config=config_path,
        cache=cache,
        checkpoint=checkpoint,
        output=tmp_path / "out",
        device="cpu",
        target="recon",
        months=[1],
        aliases=[],
        context="prefix",
        alpha=10.0,
        sample_stride=1,
    )
    audit.run(args)
    assert calls == [2.0, 9.0]
    identity = json.loads((args.output / "identity.json").read_text())
    assert identity["support"]["1"]["count"] == 4
    assert identity["support"]["1"]["coefficients"] == 3
    assert identity["training_mean"][1] == [2]
    assert identity["test_records_read"] is False
    with np.load(args.output / "support_1.npz") as support:
        np.testing.assert_array_equal(support["positions"][:, 0], 0)
        np.testing.assert_array_equal(support["targets"], 2)
        np.testing.assert_array_equal(support["intercept"], 2)
    result = json.loads((args.output / "results.json").read_text())
    assert result["1:model_all"][0]["rmse"] == 7
    assert result["1:model_all"][0]["count"] == 4
    assert result["1:temporal_common"][0]["rmse"] == 0
    with pytest.raises(FileExistsError):
        audit.run(args)
    args.output = tmp_path / "empty_support"
    args.sample_stride = 8
    with pytest.raises(ValueError, match="nonempty paired support"):
        audit.run(args)
    assert json.loads((args.output / "status.json").read_text())["state"] == "failed"
