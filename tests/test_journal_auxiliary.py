import numpy as np

from xuannv_embedding.downstream.journal_auxiliary import run
from xuannv_embedding.export.context import dump
from xuannv_embedding.training.shared_fusion_search import ref


def test_same_features_support_classification_retrieval_and_coverage(tmp_path):
    rng = np.random.default_rng(41)
    values = rng.normal(size=(23, 16, 16, 64)).astype("float32")
    np.save(tmp_path / "features.npy", values)
    np.save(tmp_path / "valid.npy", np.ones((23, 16, 16), bool))
    np.savez(tmp_path / "labels.npz", target=(values[..., 0] > 0).astype("int8"))
    records = [{"patch_id": str(i), "bounds": [160 * i, 0, 160 * (i + 1), 160]} for i in range(23)]
    dump(tmp_path / "index.json", {"crs": "EPSG:32650", "records": records})
    data = {
        "state": "complete",
        "protocol": "full-region-data-v1",
        "indices": list(range(23)),
        "records": records,
        "train_positions": list(range(20)),
        "calibration_positions": [20, 21],
        "features": {"original": ref(tmp_path / "features.npy")},
        "valid": ref(tmp_path / "valid.npy"),
        "labels": ref(tmp_path / "labels.npz"),
        "size": 16,
        "crs": "EPSG:32650",
        "tasks": {f: {"reference": ["target"]} for f in ["C", "R", "Q"]},
    }
    dump(tmp_path / "data.json", data)
    for head, budget in [("mlp", 5), ("regression", 5), ("retrieval", 3)]:
        path = tmp_path / (head + ".json")
        dump(
            path,
            {
                "schema": "journal-auxiliary-job-v1",
                "reference": ref(tmp_path / "data.json"),
                "index": ref(tmp_path / "index.json"),
                "model": {"name": "baseline", "legacy_feature": "original"},
                "head": head,
                "budget": budget,
                "seed": 41,
                "preparation": str(tmp_path / "prepared"),
                "output": str(tmp_path / head),
            },
        )
        result = run(path, "cpu")
        assert result["state"] == "complete"
        assert result["same_representation_for_all_tasks"] is True
        assert result["new_encoder_fit"] is False
        assert len(result["rows"]) == 1
        assert run(path, "cpu") == result
