import numpy as np
import pytest

from xuannv_embedding.downstream.final_evaluation import boundary_counts, boundary_f1


def test_boundary_counts_ignore_tile_exterior_and_match_small_shift():
    truth = np.zeros((1, 32, 32), dtype=bool)
    truth[:, 8:20, 8:20] = True
    shifted = np.roll(truth, 1, axis=2)
    valid = np.ones_like(truth)
    counts = boundary_counts(shifted, truth, valid, radius=1)
    assert boundary_f1(counts) == pytest.approx(1.0)
    farther = np.roll(truth, 6, axis=2)
    assert boundary_f1(boundary_counts(farther, truth, valid, radius=1)) < 1


def test_boundary_empty_prediction_is_zero_for_present_reference():
    truth = np.zeros((1, 16, 16), dtype=bool)
    truth[:, 4:12, 4:12] = True
    assert boundary_f1(boundary_counts(np.zeros_like(truth), truth, np.ones_like(truth), 1)) == 0


def test_held_out_evaluation_uses_saved_threshold_and_head(tmp_path):
    import argparse
    import hashlib
    import json

    import torch

    from xuannv_embedding.downstream.development import TASKS
    from xuannv_embedding.downstream.final_evaluation import run
    from xuannv_embedding.downstream.heads import build_head
    from xuannv_embedding.training.experiment import _sha

    cache, export, probe = [tmp_path / name for name in ("cache", "export", "probe")]
    for directory in (cache, export, probe):
        directory.mkdir()
    records, features = [], []
    names = {n for ns in TASKS.values() for n in ns}
    for i in range(7):
        label = torch.zeros(8, 8)
        label[2:6, 2:6] = 1
        path = cache / f"{i}.pt"
        torch.save(
            {
                "supervised_labels": {n: label for n in names},
                "supervised_label_masks": {n: torch.ones(8, 8) for n in names},
            },
            path,
        )
        records.append({"patch_id": str(i), "path": str(path), "sha256": _sha(path)})
        path = export / f"{i}.npz"
        np.savez(path, embedding=np.ones((1, 4, 8, 8), dtype=np.float32))
        features.append({"patch_id": str(i), "path": str(path)})
    (cache / "cache.json").write_text(
        json.dumps(
            {"records": records, "split": {"train": list(range(5)), "validation": [5], "test": [6]}}
        )
    )
    (export / "manifest.json").write_text(
        json.dumps({"records": features, "cache_sha256": _sha(cache / "cache.json")})
    )
    model = build_head("mlp", embed_dim=4, num_classes=1)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(model.state_dict(), probe / "building_5_mlp.pt")
    row = {
        "task": "building",
        "budget": 5,
        "head": "mlp",
        "metrics": {"threshold": 0.99},
        "support_patch_ids": list(map(str, range(5))),
        "support_label_sha256": hashlib.sha256(
            torch.stack([label] * 5).numpy().tobytes()
        ).hexdigest(),
    }
    (probe / "results.json").write_text(
        json.dumps(
            {"rows": [row], "metadata": {"export_manifest_sha256": _sha(export / "manifest.json")}}
        )
    )
    (probe / "status.json").write_text(json.dumps({"state": "complete"}))
    output = tmp_path / "held_out"
    run(argparse.Namespace(cache=cache, embeddings=export, probe=probe, output=output))
    result = json.loads((output / "results.json").read_text())
    assert result["rows"][0]["metrics"]["threshold"] == 0.99
    assert result["rows"][0]["metrics"]["tp"] == 0
    assert result["metadata"]["test_patch_ids"] == ["6"]
