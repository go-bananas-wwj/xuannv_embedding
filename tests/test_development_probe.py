import torch

from xuannv_embedding.downstream.development import select_support, validate_development_split


def test_support_is_nested_and_ignores_absent_or_all_negative_labels():
    labels = torch.zeros(12, 4, 4)
    labels[:10, 0, 0] = 1
    labels[11] = -1
    ids = [f"patch{i}" for i in range(12)]
    small = select_support(labels, ids, budget=5, seed=42)
    big = select_support(labels, ids, budget=10, seed=42)
    assert small == big[:5]
    assert 10 not in big and 11 not in big


def test_development_split_cannot_overlap_test():
    import pytest

    with pytest.raises(ValueError, match="overlap"):
        validate_development_split({"train": [0], "validation": [1], "test": [1]})


def test_probe_runs_all_heads_without_reading_test_features_or_labels(tmp_path):
    import argparse
    import json

    import numpy as np

    from xuannv_embedding.downstream.development import TASKS, run
    from xuannv_embedding.training.experiment import _sha

    cache, export = tmp_path / "cache", tmp_path / "export"
    cache.mkdir()
    export.mkdir()
    records, exports = [], []
    for i in range(13):
        sample = cache / f"p{i}.pt"
        features = export / f"p{i}.npz"
        if i < 12:
            labels = torch.zeros(4, 4)
            labels[:2] = 1
            keys = {k for names in TASKS.values() for k in names}
            torch.save(
                {
                    "supervised_labels": {k: labels for k in keys},
                    "supervised_label_masks": {k: torch.tensor(1.0) for k in keys},
                },
                sample,
            )
            np.savez(features, embedding=np.random.default_rng(i).normal(size=(1, 4, 4, 4)))
        records.append(
            {
                "path": str(sample),
                "patch_id": f"p{i}",
                "sha256": _sha(sample) if i < 12 else "test must not be opened",
            }
        )
        exports.append({"path": str(features)})
    (cache / "cache.json").write_text(
        json.dumps(
            {
                "records": records,
                "split": {"train": list(range(10)), "validation": [10, 11], "test": [12]},
            }
        )
    )
    (export / "manifest.json").write_text(
        json.dumps(
            {"records": exports, "months": ["2026-05"], "cache_sha256": _sha(cache / "cache.json")}
        )
    )
    output = tmp_path / "result"
    run(argparse.Namespace(cache=cache, embeddings=export, output=output, device="cpu"))
    report = json.loads((output / "results.json").read_text())
    assert len(report["rows"]) == 30
    assert report["metadata"]["test_scored"] is False
    assert all(set(r["support_patch_ids"]) <= {f"p{i}" for i in range(10)} for r in report["rows"])
    subset = tmp_path / "four_tasks"
    run(
        argparse.Namespace(
            cache=cache,
            embeddings=export,
            output=subset,
            device="cpu",
            tasks=["building", "road", "water", "green"],
            heads=["mlp"],
        )
    )
    selected = json.loads((subset / "results.json").read_text())
    assert len(selected["rows"]) == 8
    assert {r["task"] for r in selected["rows"]} == {"building", "road", "water", "green"}
    status = json.loads((subset / "status.json").read_text())
    assert status["completed"] == status["total"] == 8
