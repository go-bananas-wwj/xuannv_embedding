import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xuannv_embedding.downstream.worldcover_reference import CLASSES
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS, summarize_public
from xuannv_embedding.training import frozen_fusion


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def fixture_spec(tmp_path, output="run"):
    rng = np.random.default_rng(41)
    records = [{"patch_id": f"tile_{i}", "bounds": [i * 40, 0, i * 40 + 40, 40]} for i in range(2)]
    index = tmp_path / "index.json"
    dump(index, {"crs": "EPSG:32650", "records": records})
    inputs = {}
    for name, channels in [
        ("public", 64),
        ("spectral_old", 95),
        ("spectral_annual", 95),
        ("semantic", 64),
    ]:
        p = tmp_path / (name + ".npy")
        np.save(p, rng.uniform(0.2, 1, (2, channels, 4, 4)).astype(np.float32))
        inputs[name] = {**ref(p), "layout": "nchw", "index_sha256": sha(index)}
    frames = {k: np.ones((1, len(v), 1, 1), np.float32) for k, v in PUBLIC_BANDS.items()}
    masks = {k: np.ones((1, 1, 1), bool) for k in PUBLIC_BANDS}
    _, names = summarize_public(frames, masks, PUBLIC_BANDS)
    columns = tmp_path / "columns.json"
    dump(columns, {"columns": names})
    inputs["spectral"] = [
        {"array": inputs.pop(k), "columns": ref(columns)}
        for k in ["spectral_old", "spectral_annual"]
    ]
    targets, counts = [], np.zeros(12, np.int64)
    for i, record in enumerate(records):
        labels = rng.integers(1, 3, (4, 4), dtype=np.uint8)
        p = tmp_path / f"labels_{i}.npz"
        np.savez_compressed(p, labels=labels, valid=labels > 0)
        counts += np.bincount(labels.ravel(), minlength=12)
        targets.append({**record, **ref(p)})
    weak = tmp_path / "weak.json"
    dump(
        weak,
        {
            "state": "complete",
            "target": "esa_worldcover",
            "classes": 12,
            "ignore_index": 0,
            "source": {"dataset": "ESA WorldCover", "year": 2021, "version": "v200"},
            "legend": {
                str(i): {"official_code": code, "name": name}
                for i, (code, name) in enumerate(CLASSES.items(), 1)
            },
            "records": targets,
            "class_counts": counts.tolist(),
        },
    )
    spec = {
        "schema": "frozen-feature-fusion-training-v1",
        "index": ref(index),
        "inputs": inputs,
        "weak_targets": ref(weak),
        "model": {"width": 8, "residual_scale": 0.25, "spectral_channels": 190},
        "loss": {"spectral_weight": 0.2, "semantic_weight": 0.1, "preservation_weight": 0.02},
        "steps": 4,
        "batch_size": 1,
        "lr": 0.001,
        "weight_decay": 0.01,
        "seed": 41,
        "device": "cpu",
        "output": str(tmp_path / output),
    }
    path = tmp_path / (output + ".json")
    dump(path, spec)
    return path, spec


def test_frozen_fusion_resume_matches_uninterrupted_training_and_exports_64_dimensions(tmp_path):
    path, spec = fixture_spec(tmp_path)
    interrupted = frozen_fusion.train(path, max_updates=2)
    assert interrupted["state"] == "checkpointed"
    assert not (Path(spec["output"]) / "model.weights.pt").exists()
    finished = frozen_fusion.train(path)
    assert finished["state"] == "complete" and finished["unique_training_tiles"] == 2
    state = torch.load(str(Path(spec["output"]) / "model.weights.pt"), weights_only=True)["model"]
    other = copy.deepcopy(spec)
    other["output"] = str(tmp_path / "uninterrupted")
    p = tmp_path / "uninterrupted.json"
    dump(p, other)
    frozen_fusion.train(p)
    expected = torch.load(str(tmp_path / "uninterrupted/model.weights.pt"), weights_only=True)[
        "model"
    ]
    assert state.keys() == expected.keys()
    assert all(torch.equal(state[k], expected[k]) for k in state)
    frozen_fusion.export(path)
    result = np.load(Path(spec["output"]) / "embedding/annual.npy")
    assert result.shape == (2, 64, 4, 4) and np.isfinite(result).all()


def test_frozen_fusion_no_imagery_fit_never_reads_local_imagery_and_keeps_public_statistics(
    tmp_path,
):
    path, spec = fixture_spec(tmp_path)
    frozen_fusion.train(path)
    full = torch.load(str(Path(spec["output"]) / "model.weights.pt"), weights_only=True)["model"]
    spec["inputs"]["semantic"] = None
    Path(tmp_path / "semantic.npy").unlink()
    spec["output"] = str(tmp_path / "no_imagery")
    dump(path, spec)
    frozen_fusion.train(path)
    public = torch.load(str(tmp_path / "no_imagery/model.weights.pt"), weights_only=True)["model"]
    for k in ["public_mean", "public_std", "spectral_mean", "spectral_std"]:
        assert torch.equal(full[k], public[k])
    assert public["semantic_mean"].count_nonzero() == 0
    assert torch.equal(public["semantic_std"], torch.ones_like(public["semantic_std"]))


def test_frozen_fusion_rejects_evaluation_fields_and_legacy_osm_alias(tmp_path):
    path, spec = fixture_spec(tmp_path)
    dump(path, {**spec, "evaluation": "must-not-be-read"})
    with pytest.raises(ValueError, match="specification"):
        frozen_fusion.train(path)
    weak = tmp_path / "weak.json"
    meta = json.loads(weak.read_text())
    meta["source"]["dataset"] = "OSM"
    dump(weak, meta)
    spec["weak_targets"] = ref(weak)
    dump(path, spec)
    with pytest.raises(ValueError, match="WorldCover"):
        frozen_fusion.train(path)


def test_frozen_fusion_recovers_a_final_model_before_status_write_and_rejects_changed_bytes(
    tmp_path,
):
    path, spec = fixture_spec(tmp_path)
    frozen_fusion.train(path)
    root = Path(spec["output"])
    digest = sha(root / "model.weights.pt")
    (root / "status.json").unlink()
    recovered = frozen_fusion.train(path)
    assert recovered["state"] == "complete"
    assert sha(root / "model.weights.pt") == digest
    with (root / "model.weights.pt").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="final model changed"):
        frozen_fusion.train(path)
