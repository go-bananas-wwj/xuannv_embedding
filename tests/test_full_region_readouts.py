import json

import numpy as np
import pytest

from xuannv_embedding.downstream import full_region_readouts as full
from xuannv_embedding.export.context import sha


def ref(p):
    return {"path": str(p), "sha256": sha(p)}


def setup(tmp_path):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(3, 16, 16, 3)).astype(np.float32)
    y = (x[..., 0] > 0).astype(np.int8)
    valid = np.ones(y.shape, bool)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "valid.npy", valid)
    np.savez(tmp_path / "labels.npz", target_a=y)
    data = {
        "state": "complete",
        "protocol": "full-region-data-v1",
        "indices": [10, 20, 30],
        "records": [{"patch_id": str(i)} for i in [10, 20, 30]],
        "train_positions": [0],
        "calibration_positions": [1],
        "features": {"m": ref(tmp_path / "x.npy")},
        "valid": ref(tmp_path / "valid.npy"),
        "labels": ref(tmp_path / "labels.npz"),
        "tasks": {"C": {"target": ["target_a"]}, "R": {}, "Q": {}},
    }
    p = tmp_path / "data.json"
    p.write_text(json.dumps(data))
    spec = {
        "protocol": full.PROTOCOL,
        "data": ref(p),
        "model": "m",
        "head": "linear",
        "budget": 1,
        "seed": 41,
        "device": "cpu",
        "archives": {},
        "output": str(tmp_path / "job"),
    }
    p = tmp_path / "job.json"
    p.write_text(json.dumps(spec))
    return p, spec


def test_full_region_scoring_includes_support_and_calibration_without_refitting(
    tmp_path, monkeypatch
):
    p, spec = setup(tmp_path)
    with pytest.raises(ValueError, match="calibration"):
        full.run(p, "full_region")
    full.run(p, "calibration")
    monkeypatch.setattr(full.primary, "_fit", lambda *a, **k: pytest.fail("query refitting"))
    full.run(p, "full_region")
    identity = json.loads((tmp_path / "job/full_region/identity.json").read_text())
    assert identity["full_region_tiles"] == 3
    assert identity["includes_downstream_support"] is True
    assert identity["parameters_refitted"] is False
    with np.load(tmp_path / "job/full_region/predictions/m/C_target_a_41_1.npz") as z:
        assert set(z["tiles"]) == {10, 20, 30}
        assert len(z["scores"]) == 3 * 16 * 16


def test_matching_saved_head_is_reused_and_changed_support_is_rejected(tmp_path, monkeypatch):
    p, spec = setup(tmp_path)
    full.run(p, "calibration")
    record = json.loads((tmp_path / "job/calibration/records/C_target_a_41_1.json").read_text())
    spec["archives"] = {
        "C_target_a_41_1": {
            **record,
            "prediction": ref(tmp_path / "job/calibration/predictions/m/C_target_a_41_1.npz"),
        }
    }
    spec["output"] = str(tmp_path / "reused")
    p = tmp_path / "reuse.json"
    p.write_text(json.dumps(spec))
    monkeypatch.setattr(full.primary, "_fit", lambda *a, **k: pytest.fail("archive refitting"))
    full.run(p, "calibration")
    identity = json.loads((tmp_path / "reused/calibration/identity.json").read_text())
    assert identity["reused_conditions"] == 1
    x = np.load(tmp_path / "x.npy")
    x[0, 0, 0, 0] += 100
    np.save(tmp_path / "x.npy", x)
    data_path = tmp_path / "data.json"
    data = json.loads(data_path.read_text())
    data["features"]["m"] = ref(tmp_path / "x.npy")
    data_path.write_text(json.dumps(data))
    spec["data"] = ref(data_path)
    spec["output"] = str(tmp_path / "changed")
    p = tmp_path / "changed.json"
    p.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="observations changed"):
        full.run(p, "calibration")


@pytest.mark.parametrize("head", ["mlp", "conv3x3", "rf", "unet"])
def test_neural_and_forest_archives_replay_all_region_without_retraining(
    tmp_path, monkeypatch, head
):
    p, spec = setup(tmp_path)
    spec["head"] = head
    p.write_text(json.dumps(spec))
    if head == "unet":
        fit = full.unet_readout.fit
        monkeypatch.setattr(
            full.unet_readout, "fit", lambda *a, **k: fit(*a, **k, checkpoints=(1, 2))
        )
    full.run(p, "calibration")
    key = f"C_target_a_41_1_{head}"
    record = json.loads((tmp_path / f"job/calibration/records/{key}.json").read_text())
    spec["archives"] = {
        key: {**record, "prediction": ref(tmp_path / f"job/calibration/predictions/m/{key}.npz")}
    }
    spec["output"] = str(tmp_path / "reused")
    p = tmp_path / "reuse.json"
    p.write_text(json.dumps(spec))
    monkeypatch.setattr(full, "_fit", lambda *a, **k: pytest.fail("unexpected refitting"))
    full.run(p, "calibration")
    full.run(p, "full_region")
    with np.load(tmp_path / f"reused/full_region/predictions/m/{key}.npz") as z:
        assert set(z["tiles"]) == {10, 20, 30}


@pytest.mark.parametrize("head,family", [("regression", "R"), ("retrieval", "Q")])
def test_auxiliary_readouts_reuse_frozen_parameters_on_full_domain(
    tmp_path, monkeypatch, head, family
):
    p, spec = setup(tmp_path)
    data_path = tmp_path / "data.json"
    data = json.loads(data_path.read_text())
    data["tasks"][family] = {"target": ["target_a"]}
    data_path.write_text(json.dumps(data))
    spec.update(data=ref(data_path), head=head)
    p.write_text(json.dumps(spec))
    full.run(p, "calibration")
    key = f"{family}_target_a_41_1"
    record = json.loads((tmp_path / f"job/calibration/records/{key}.json").read_text())
    spec["archives"] = {
        key: {**record, "prediction": ref(tmp_path / f"job/calibration/predictions/m/{key}.npz")}
    }
    spec["output"] = str(tmp_path / "reuse_aux")
    p = tmp_path / "reuse_aux.json"
    p.write_text(json.dumps(spec))
    monkeypatch.setattr(full, "_fit", lambda *a, **k: pytest.fail("unexpected fitting"))
    full.run(p, "calibration")
    full.run(p, "full_region")
    with np.load(tmp_path / f"reuse_aux/full_region/predictions/m/{key}.npz") as z:
        assert set(z["tiles"]) == {10, 20, 30}
