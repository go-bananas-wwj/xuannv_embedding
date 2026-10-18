import json

import pytest

from xuannv_embedding.training.fusion_transfer import main, read_spec, verified


def test_fixed_transfer_rejects_unknown_configuration_keys(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"schema": "fixed-regional-fusion-transfer-v1", "unknown": True}))
    with pytest.raises(ValueError, match="specification"):
        read_spec(path)


def test_fixed_transfer_rejects_changed_source_identity(tmp_path):
    path = tmp_path / "source"
    path.write_text("input")
    with pytest.raises(ValueError, match="identity"):
        verified({"path": str(path), "sha256": "0" * 64})


def test_fixed_transfer_help_has_no_training_or_search_phase(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "{prepare,semantic,features,report}" in help_text
    assert "train" not in help_text


def test_fixed_transfer_creates_readout_job_directory_on_first_registration(tmp_path):
    import numpy as np

    from xuannv_embedding.export.context import dump
    from xuannv_embedding.training.fusion_transfer import reference, register_datasets

    records = [
        {"patch_id": f"tile_{i}", "bounds": [i * 1280, 0, (i + 1) * 1280, 1280]} for i in range(21)
    ]
    index = tmp_path / "index.json"
    dump(index, {"crs": "EPSG:32650", "records": records})
    path = tmp_path / "base.npy"
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(21, 128, 128, 64))
    values[:] = 0
    values.flush()
    valid = tmp_path / "valid.npy"
    np.save(valid, np.ones((21, 128, 128), bool))
    labels = tmp_path / "labels.npz"
    np.savez_compressed(labels, target=np.zeros((21, 128, 128), np.int8))
    data = tmp_path / "original-data.json"
    dump(
        data,
        {
            "state": "complete",
            "protocol": "full-region-data-v1",
            "records": records,
            "indices": list(range(21)),
            "train_positions": list(range(20)),
            "calibration_positions": [20],
            "valid": reference(valid),
            "labels": reference(labels),
            "size": 128,
            "crs": "EPSG:32650",
            "features": {},
            "tasks": {"C": {"reference": ["target"]}, "R": {}, "Q": {}},
        },
    )
    dump(tmp_path / "features.json", {"base": reference(path)})
    spec = {
        "output": str(tmp_path),
        "index": reference(index),
        "support_seed": 41,
        "datasets": [{"name": "primary", "data": reference(data)}],
    }
    register_datasets(spec)
    jobs = json.loads((tmp_path / "jobs.json").read_text())
    assert {job["head"] for job in jobs} == {"mlp", "conv3x3"}
    assert len(list((tmp_path / "jobs").glob("*.json"))) == 2
