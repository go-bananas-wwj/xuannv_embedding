import numpy as np
import pytest
import torch

from xuannv_embedding.downstream.reconstruction import (
    hidden_month_inputs,
    metric_sums,
    summarize_sums,
    temporal_baseline,
)


def batch_fixture():
    image = torch.arange(3.0).reshape(1, 3, 1, 1, 1).expand(1, 3, 1, 2, 2).clone()
    return {
        "source_frames": {"optical": image, "radar": image.clone()},
        "source_masks": {"optical": torch.ones(1, 3), "radar": torch.ones(1, 3)},
        "highres_frames": {"duplicate": image.clone()},
        "highres_masks": {"duplicate": torch.ones(1, 3, 1, 2, 2)},
        "timestamps": torch.tensor([[202512, 202601, 202602]]),
        "targets": {"optical_recon": image},
    }


def test_hidden_month_removes_aliases_and_preserves_shared_target_tensor():
    batch = batch_fixture()
    masked = hidden_month_inputs(batch, ["optical", "duplicate"], 1, prefix=False)
    assert masked["source_frames"]["optical"][:, 1].count_nonzero() == 0
    assert masked["source_masks"]["optical"][:, 1].count_nonzero() == 0
    assert masked["highres_frames"]["duplicate"][:, 1].count_nonzero() == 0
    assert masked["highres_masks"]["duplicate"][:, 1].count_nonzero() == 0
    assert batch["targets"]["optical_recon"][:, 1].eq(1).all()
    assert torch.equal(masked["source_frames"]["radar"], batch["source_frames"]["radar"])


def test_prefix_removes_future_pixels_in_all_modalities_and_undated_static_inputs():
    batch = batch_fixture()
    batch["highres_frames"]["static"] = torch.ones(1, 1, 2, 2)
    batch["highres_masks"]["static"] = torch.ones(1, 1, 2, 2)
    masked = hidden_month_inputs(batch, ["optical"], 1, prefix=True)
    assert masked["source_frames"]["radar"][:, 2].count_nonzero() == 0
    assert masked["highres_frames"]["duplicate"][:, 2].count_nonzero() == 0
    assert masked["highres_frames"]["static"].count_nonzero() == 0
    assert masked["source_frames"]["radar"][:, 1].eq(1).all()


def test_temporal_baseline_uses_only_visible_months_and_reports_unavailable_pixels():
    values = np.array([0.0, 999.0, 4.0]).reshape(3, 1, 1, 1)
    valid = np.ones((3, 1, 1), bool)
    prediction, domain = temporal_baseline(values, valid, 1, prefix=False)
    assert prediction.item() == 2
    assert domain.item()
    prediction, domain = temporal_baseline(values, valid, 1, prefix=True)
    assert prediction.item() == 0
    assert domain.item()
    _, domain = temporal_baseline(values, valid, 0, prefix=True)
    assert not domain.item()


def test_metrics_ignore_invalid_nan_and_keep_unequal_band_counts():
    truth = np.zeros((2, 1, 2))
    prediction = np.array([[[2.0, np.nan]], [[1.0, 3.0]]])
    valid = np.array([[[True, False]], [[True, True]]])
    sums = metric_sums(prediction, truth, valid)
    result = summarize_sums(sums)
    assert result[0]["rmse"] == 2
    assert result[1]["rmse"] == pytest.approx(np.sqrt(5))
    assert result[0]["count"] == 1
    with pytest.raises(ValueError, match="nonfinite"):
        metric_sums(prediction, truth, np.ones_like(valid))


def test_empty_domain_is_not_reported_as_perfect_reconstruction():
    values = np.zeros((1, 2, 2))
    result = summarize_sums(metric_sums(values, values, np.zeros((2, 2), bool)))
    assert result == [{"count": 0, "rmse": None, "mae": None, "bias": None}]


def test_hidden_month_rejects_unknown_alias_and_out_of_range_month():
    with pytest.raises(ValueError):
        hidden_month_inputs(batch_fixture(), ["missing"], 1, prefix=False)
    with pytest.raises(ValueError):
        hidden_month_inputs(batch_fixture(), ["optical"], 3, prefix=False)


def test_runner_scores_validation_only_and_fits_mean_on_training(tmp_path, monkeypatch):
    import argparse
    import json
    from types import SimpleNamespace

    import xuannv_embedding.downstream.reconstruction as audit

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
    # A nonexistent held-out record must never be read, even for checksums.
    records.append({"path": str(cache / "unread_test.pt"), "sha256": "unread"})
    audit._json(
        cache / "cache.json",
        {"records": records, "split": {"train": [0], "validation": [1], "test": [2]}},
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

    class CheckingModel(torch.nn.Module):
        def forward(self, frames, masks, timestamps, highres, highres_masks):
            assert not frames["optical"][:, 1:].count_nonzero()
            assert not masks["optical"][:, 1:].count_nonzero()
            return SimpleNamespace(reconstructions={"recon": torch.zeros_like(frames["optical"])})

    monkeypatch.setattr(audit, "_load_model", lambda *a: (CheckingModel(), 1))
    monkeypatch.setattr(audit, "_setup_device", lambda _: (torch.device("cpu"), False, 0))
    output = tmp_path / "out"
    audit.run(
        argparse.Namespace(
            config=config_path,
            cache=cache,
            checkpoint=checkpoint,
            output=output,
            device="cpu",
            target="recon",
            months=[1],
            aliases=[],
            context="prefix",
        )
    )
    results = json.loads((output / "results.json").read_text())
    assert results["1:model_all"][0]["rmse"] == 9
    assert results["1:mean_all"][0]["rmse"] == 7
    assert results["1:temporal_common"][0]["rmse"] == 0
    assert json.loads((output / "identity.json").read_text())["training_mean"][1] == [2]
    with pytest.raises(FileExistsError):
        audit.run(
            argparse.Namespace(
                config=config_path,
                cache=cache,
                checkpoint=checkpoint,
                output=output,
                device="cpu",
                target="recon",
                months=[1],
                aliases=[],
                context="prefix",
            )
        )
