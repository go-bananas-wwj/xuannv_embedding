import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xuannv_embedding.training import experiment_export
from xuannv_embedding.training.experiment_export import export_indices, validate_export_identity


def test_export_rejects_another_spatial_folds_cache():
    run = {"config_sha256": "a", "cache_sha256": "b"}
    with pytest.raises(ValueError, match="cache"):
        validate_export_identity(run, config_sha="a", cache_sha="c")


def test_export_requires_matching_configuration_and_cache():
    run = {"config_sha256": "a", "cache_sha256": "b"}
    validate_export_identity(run, config_sha="a", cache_sha="b")
    with pytest.raises(ValueError, match="config"):
        validate_export_identity(run, config_sha="c", cache_sha="b")


def test_partial_export_keeps_original_record_order_and_full_default():
    document = {
        "records": [{"patch_id": str(i)} for i in range(4)],
        "split": {"train": [0, 2], "validation": [3], "test": [1], "buffer": []},
    }
    assert export_indices(document, None) == [0, 1, 2, 3]
    assert export_indices(document, ["validation"]) == [3]
    assert export_indices(document, ["validation", "test"]) == [1, 3]
    for splits in [[], ["validation", "validation"], ["unknown"], ["buffer"]]:
        with pytest.raises(ValueError):
            export_indices(document, splits)
    document["split"]["validation"] = [True]
    with pytest.raises(ValueError):
        export_indices(document, ["validation"])


@pytest.mark.parametrize("splits", [None, ["validation"]])
def test_export_materializes_only_requested_cache_records(tmp_path, monkeypatch, splits):
    cache = tmp_path / "cache"
    cache.mkdir()
    configuration = tmp_path / "config.yaml"
    configuration.write_text("test fixture")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    records = []
    for i in range(4):
        sample = cache / f"sample{i}.pt"
        if splits is None or i == 3:
            sample.write_bytes(b"cached fixture")
        records.append(
            {
                "patch_id": f"p{i}",
                "bounds": [i, 0, i + 1, 1],
                "path": str(sample),
                "sha256": experiment_export._sha(cache / "sample3.pt") if i == 3 else "unused",
            }
        )
    for record in records:
        sample = Path(record["path"])
        if sample.exists():
            record["sha256"] = experiment_export._sha(sample)
    document = {
        "records": records,
        "split": {"train": [0, 2], "test": [1], "validation": [3], "buffer": []},
    }
    (cache / "cache.json").write_text(json.dumps(document))
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "config_sha256": experiment_export._sha(configuration),
                "cache_sha256": experiment_export._sha(cache / "cache.json"),
                "git_sha": "a",
            }
        )
    )
    config = SimpleNamespace(
        model=SimpleNamespace(input_sources={}),
        data=SimpleNamespace(datasets=[], months=["2026-05"]),
    )
    monkeypatch.setattr(experiment_export.Config, "from_yaml", lambda p: config)
    monkeypatch.setattr(
        experiment_export, "build_training_system", lambda c: SimpleNamespace(model=None)
    )
    monkeypatch.setattr(
        experiment_export, "load_training_checkpoint", lambda *a, **k: {"git_sha": "a", "epoch": 0}
    )
    monkeypatch.setattr(experiment_export, "_setup_device", lambda d: ("cpu", False, None))
    monkeypatch.setattr(
        experiment_export,
        "CachedSamples",
        lambda doc, ix: [{"patch_ids": [doc["records"][i]["patch_id"]]} for i in ix],
    )
    monkeypatch.setattr(experiment_export, "DataLoader", lambda ds, **kw: ds)

    def export(model, batches, output, **kwargs):
        output.mkdir(exist_ok=True)
        result = []
        for batch in batches:
            for patch in batch["patch_ids"]:
                p = output / (patch + ".npz")
                np.savez(p, embedding=np.zeros((1, 1, 1, 1)), timestamps=[202605])
                result.append(p)
        return result

    monkeypatch.setattr(experiment_export, "export_embedding_batches", export)
    args = argparse.Namespace(
        config=configuration,
        cache=cache,
        checkpoint=checkpoint,
        output=tmp_path / "export",
        device="cpu",
        batch_size=1,
        export_split=splits,
    )
    experiment_export.run(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert len(manifest["records"]) == 4 and manifest["split"] == document["split"]
    expected = 4 if splits is None else 1
    assert len(list((args.output / "embeddings").glob("*.npz"))) == expected
    assert json.loads((args.output / "status.json").read_text())["total"] == expected
    if splits is None:
        assert "exported_indices" not in manifest
    else:
        assert manifest["exported_indices"] == [3] and manifest["exported_splits"] == splits
