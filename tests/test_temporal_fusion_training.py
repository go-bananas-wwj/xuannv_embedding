import json

import numpy as np
import pytest
import torch

from xuannv_embedding.downstream.worldcover_reference import CLASSES
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS
from xuannv_embedding.training.temporal_fusion import Data, export, masked_inputs, train


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def fixture_spec(tmp_path):
    rng = np.random.default_rng(41)
    records = [{"patch_id": f"p{i}", "bounds": [i * 80, 0, i * 80 + 80, 80]} for i in range(2)]
    index = tmp_path / "index.json"
    dump(index, {"crs": "EPSG:32650", "records": records})
    frames, masks = tmp_path / "frames.npy", tmp_path / "masks.npy"
    np.save(frames, rng.normal(size=(2, 12, 19, 8, 8)).astype(np.float32))
    np.save(masks, np.ones((2, 12, 3, 8, 8), bool))
    monthly = tmp_path / "monthly.json"
    dump(
        monthly,
        {
            "schema": "public-monthly-tensor-v1",
            "index": ref(index),
            "frames": ref(frames),
            "masks": ref(masks),
            "shape": [2, 12, 19, 8, 8],
            "months": list(range(202401, 202413)),
            "bands": PUBLIC_BANDS,
            "band_slices": {"s2": [0, 11], "s1": [11, 13], "landsat": [13, 19]},
            "map_or_local_image_values_used": False,
        },
    )
    static = {}
    for s, channels in [("public", 64), ("local", 64), ("terrain", 5)]:
        p = tmp_path / f"{s}.npy"
        np.save(p, rng.normal(size=(2, channels, 8, 8)).astype(np.float32))
        static[s] = {**ref(p), "layout": "nchw", "index_sha256": sha(index)}
    targets, count = [], np.zeros(12, np.int64)
    for i, record in enumerate(records):
        y = rng.integers(1, 12, size=(8, 8), dtype=np.uint8)
        p = tmp_path / f"weak_{i}.npz"
        np.savez_compressed(p, labels=y, valid=y > 0)
        targets.append({**record, **ref(p)})
        count += np.bincount(y.ravel(), minlength=12)
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
                str(i): {"official_code": k, "name": v}
                for i, (k, v) in enumerate(CLASSES.items(), 1)
            },
            "records": targets,
            "class_counts": count.tolist(),
        },
    )
    spec = tmp_path / "spec.json"
    dump(
        spec,
        {
            "schema": "regional-temporal-fusion-v1",
            "index": ref(index),
            "monthly": ref(monthly),
            "static": static,
            "weak_targets": ref(weak),
            "model": {"width": 8, "residual_scale": 0.5},
            "loss": {
                "worldcover_weight": 0.45,
                "observation_weight": 1.0,
                "visual_weight": 0.1,
                "preservation_weight": 0.02,
            },
            "steps": 2,
            "batch_size": 2,
            "lr": 0.0003,
            "weight_decay": 0.01,
            "seed": 41,
            "device": "cpu",
            "amp": False,
            "mask_warmup_fraction": 0.3,
            "output": str(tmp_path / "run"),
        },
    )
    return spec


def test_temporal_training_resumes_and_exports_without_access_to_map_targets(tmp_path):
    spec = fixture_spec(tmp_path)
    first = train(spec, max_updates=1)
    assert first["state"] == "checkpointed" and first["step"] == 1
    final = train(spec)
    root = tmp_path / "run"
    assert final["state"] == "complete" and final["unique_training_tiles"] == 2
    assert not (root / "latest.pt").exists()
    (root / "status.json").unlink()
    (root / "latest.pt").write_bytes(b"obsolete optimizer recovery")
    recovered = train(spec)
    assert recovered["state"] == "complete" and not (root / "latest.pt").exists()
    assert json.loads((root / "status.json").read_text())["state"] == "complete"
    # The inference-only loader must not need the training map or label arrays.
    (tmp_path / "weak.json").unlink()
    for p in tmp_path.glob("weak_*.npz"):
        p.unlink()
    result = export(spec)
    values = np.load(result)
    assert values.shape == (2, 64, 8, 8) and np.isfinite(values).all()
    metadata = json.loads((result.parent / "status.json").read_text())
    assert metadata["map_targets_loaded_for_inference"] is False
    assert export(spec) == result


def test_temporal_training_public_label_domain_is_independent_of_local_input(tmp_path):
    spec = fixture_spec(tmp_path)
    full = Data(spec)
    content = json.loads(spec.read_text())
    content["static"]["local"] = None
    content["output"] = str(tmp_path / "public")
    other = tmp_path / "public.json"
    dump(other, content)
    public = Data(other)
    assert public.domain_sha256 == full.domain_sha256
    torch.testing.assert_close(public.class_weights, full.class_weights)


def test_temporal_data_rejects_invalid_static_index(tmp_path):
    spec = fixture_spec(tmp_path)
    content = json.loads(spec.read_text())
    content["static"]["local"]["index_sha256"] = "0" * 64
    dump(spec, content)
    with pytest.raises(ValueError, match="index"):
        Data(spec)


def test_temporal_mask_schedule_is_identical_between_local_and_public_roles(tmp_path):
    spec = fixture_spec(tmp_path)
    data = Data(spec)
    batch = data.batch([0, 1])
    full = masked_inputs(batch, seed=41, active=True)
    without = masked_inputs((*batch[:5], None, batch[6]), seed=41, active=True)
    assert torch.equal(full[1], without[1])
    assert torch.equal(full[3], without[3])
