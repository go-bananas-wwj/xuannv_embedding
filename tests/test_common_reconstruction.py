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


@pytest.mark.parametrize("external", [False, True])
def test_runner_fits_only_training_embeddings_and_never_reads_test_record(
    tmp_path, monkeypatch, external
):
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
    document = {
        "records": records,
        "split": {"train": [0], "validation": [1], "test": [2], "buffer": []},
        "data": {"months": ["2025-12", "2026-01", "2026-02"], "patch_size": 2},
        "manifest_sha256": "registered-grid",
    }
    for index, record in enumerate(records):
        record.update(index=index, patch_id=str(index), bounds=[index, 0, index + 1, 1])
    audit._json(cache / "cache.json", document)
    target_cache = None
    if external:
        import copy

        target_cache = tmp_path / "target_cache"
        target_cache.mkdir()
        target_document = copy.deepcopy(document)
        target_document["model_targets"] = {
            "external_recon": {
                "source": "radar",
                "channels": 1,
                "loss_type": "continuous",
                "weight": 1.0,
            }
        }
        target_document["model_inputs"] = {"radar": {"channels": 1, "role": "highres"}}
        for index in (0, 1):
            sample = torch.load(records[index]["path"], weights_only=True)
            sample["targets"] = {"external_recon": sample["targets"]["recon"]}
            sample["target_masks"] = {"external_recon": sample["target_masks"]["recon"]}
            # External observations must never enter the model's input dictionaries.
            sample["source_frames"]["optical"].fill_(1234)
            sample["highres_frames"]["radar"] = torch.full((3, 1, 2, 2), 999.0)
            sample["highres_masks"]["radar"] = torch.ones(3)
            path = target_cache / f"{index}.pt"
            torch.save(sample, path)
            target_document["records"][index].update(path=str(path), sha256=audit._sha(path))
        target_document["records"][2]["path"] = str(target_cache / "unread_target_test.pt")
        audit._json(target_cache / "cache.json", target_document)
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
            target_heads=(
                {}
                if external
                else {
                    "recon": SimpleNamespace(source="optical", channels=1, loss_type="continuous")
                }
            ),
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
            assert not highres and not highres_masks
            first_hidden = 2 if external else 1
            assert not frames["optical"][:, first_hidden:].count_nonzero()
            assert not masks["optical"][:, first_hidden:].count_nonzero()
            if external:
                assert frames["optical"][:, 1].count_nonzero()
                assert masks["optical"][:, 1].count_nonzero()
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
        target="external_recon" if external else "recon",
        target_cache=target_cache,
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
    if external:
        assert identity["target_cache_sha256"] == audit._sha(target_cache / "cache.json")
        assert identity["absent_hidden_sources"] == ["radar"]
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
    if external:
        before = len(calls)
        with (target_cache / "0.pt").open("ab") as stream:
            stream.write(b"changed")
        args.output = tmp_path / "corrupt_target"
        with pytest.raises(ValueError, match="sample checksum changed"):
            audit.run(args)
        assert len(calls) == before
        assert not args.output.exists()


@pytest.mark.parametrize(
    "mismatch", ["months", "patch_size", "manifest", "bounds", "order", "split", "schema"]
)
def test_external_targets_reject_misaligned_reference_metadata(mismatch):
    import copy

    from xuannv_embedding.downstream.reconstruction_targets import paired_target_schema

    document = {
        "data": {"months": ["2025-12"], "patch_size": 2},
        "manifest_sha256": "same-grid",
        "split": {"train": [0], "validation": [1], "test": [2], "buffer": []},
        "records": [dict(index=i, patch_id=str(i), bounds=[i, 0, i + 1, 1]) for i in range(3)],
    }
    target = copy.deepcopy(document)
    target.update(
        model_targets={"recon": dict(source="radar", loss_type="continuous", channels=1, weight=1)},
        model_inputs={"radar": dict(channels=1, role="highres")},
    )
    assert paired_target_schema(document, target, "recon").channels == 1
    if mismatch in ["months", "patch_size"]:
        target["data"][mismatch] = ["2026-01"] if mismatch == "months" else 4
    elif mismatch == "manifest":
        target["manifest_sha256"] = "different-grid"
    elif mismatch == "bounds":
        target["records"][1]["bounds"][0] += 0.5
    elif mismatch == "order":
        target["records"].reverse()
    elif mismatch == "split":
        target["split"]["train"], target["split"]["test"] = [2], [0]
    else:
        target["model_targets"]["recon"]["loss_type"] = "categorical"
    with pytest.raises(ValueError, match="external target"):
        paired_target_schema(document, target, "recon")


@pytest.mark.parametrize("mismatch", ["region", "patch_id", "timestamps", "shape"])
def test_external_targets_reject_sample_identity_or_geometry_mismatch(mismatch, monkeypatch):
    import copy

    import torch

    import xuannv_embedding.downstream.reconstruction_targets as targets

    sample = dict(
        region="fixture",
        patch_id="0",
        timestamps=torch.tensor([202512]),
        targets={"recon": torch.ones(1, 1, 2, 2)},
        target_masks={"recon": torch.ones(1, 2, 2)},
    )
    reference = copy.deepcopy(sample)
    if mismatch in ["region", "patch_id"]:
        reference[mismatch] = "other"
    elif mismatch == "timestamps":
        reference["timestamps"][0] = 202601
    else:
        reference["targets"]["recon"] = torch.ones(1, 1, 4, 4)
    document = dict(data=dict(months=["2025-12"], patch_size=2), records=[dict(patch_id="0")])
    external = copy.deepcopy(document)
    monkeypatch.setattr(
        targets, "CachedSamples", lambda d, _: [sample if d is document else reference]
    )
    with pytest.raises(ValueError, match="external target sample"):
        list(targets.paired_samples(document, external, [0], name="recon", channels=1))


def test_absent_source_requires_explicit_opt_in_and_prefix_still_masks_future():
    import torch

    from xuannv_embedding.downstream.reconstruction import hidden_month_inputs

    batch = dict(
        timestamps=torch.tensor([[202512, 202601, 202602]]),
        source_frames={"optical": torch.ones(1, 3, 1, 2, 2), "static": torch.ones(1, 1, 2, 2)},
        source_masks={"optical": torch.ones(1, 3), "static": torch.ones(1)},
    )
    with pytest.raises(ValueError):
        hidden_month_inputs(batch, ["radar"], 1, prefix=True)
    result = hidden_month_inputs(batch, ["radar"], 1, prefix=True, allow_missing_sources=True)
    assert torch.equal(
        result["source_frames"]["optical"][:, :2], batch["source_frames"]["optical"][:, :2]
    )
    assert not result["source_frames"]["optical"][:, 2].count_nonzero()
    assert not result["source_masks"]["optical"][:, 2].count_nonzero()
    assert not result["source_frames"]["static"].count_nonzero()
    assert not result["source_masks"]["static"].count_nonzero()
    assert batch["source_frames"]["optical"].all()  # cached input was not mutated
    offline = hidden_month_inputs(batch, ["radar"], 1, prefix=False, allow_missing_sources=True)
    for key in ("source_frames", "source_masks"):
        for source in batch[key]:
            assert torch.equal(offline[key][source], batch[key][source])
