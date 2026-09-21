import json
from pathlib import Path

import numpy as np
import pytest

from xuannv_embedding.downstream import paired_multitask as workflow
from xuannv_embedding.export.context import sha


def write_json(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": sha(path)}


def fixture_spec(tmp_path):
    size = 16
    records = [{"patch_id": f"p{i}", "bounds": [i * 160, 0, i * 160 + 160, 160]} for i in range(3)]
    split = {"train": [0], "validation": [1], "test": [2], "buffer": []}
    cache = {
        "records": records,
        "split": split,
        "data": {"months": ["2026-05"], "patch_size": size},
    }
    cache_ref = write_json(tmp_path / "cache.json", cache)
    rng = np.random.default_rng(8)
    semantic = (np.indices((size, size))[1] % 6).astype(np.int8)
    binary = (np.indices((size, size))[1] >= 8).astype(np.int8)
    labels, models, test_paths = {}, {}, []
    for partition, indices in split.items():
        if partition == "buffer":
            continue
        path = tmp_path / f"labels_{partition}.npz"
        data = {"indices": indices, "cache_sha256": cache_ref["sha256"], "esri": semantic[None]}
        data.update({task: binary[None] for task in workflow.OSM_TASKS})
        np.savez(path, **data)
        labels[partition] = {"path": str(path), "sha256": sha(path)}
        if partition == "test":
            test_paths.append(path)
    for name, channels, kind in [("base", 3, "annual"), ("candidate", 5, "monthly")]:
        exported, digests = [], {}
        for i, record in enumerate(records):
            path = tmp_path / f"{name}_{i}.npz"
            data = {"embedding": rng.normal(size=(1, channels, size, size)).astype(np.float32)}
            if kind == "monthly":
                data["timestamps"] = [202605]
            valid = np.ones((size, size), bool)
            if name == "candidate":
                valid[0, 0] = False
            data["valid_mask"] = valid
            np.savez(path, **data)
            exported.append({**record, "path": str(path)})
            digests[record["patch_id"]] = sha(path)
            if i == 2:
                test_paths.append(path)
        manifest = {
            "records": exported,
            "split": split,
            "cache_sha256": cache_ref["sha256"],
            "months": ["annual_2025" if kind == "annual" else "2026-05"],
        }
        registered = write_json(tmp_path / f"{name}.json", manifest)
        models[name] = {
            "manifest_path": registered["path"],
            "manifest_sha256": registered["sha256"],
            "cache_path": cache_ref["path"],
            "cache_sha256": cache_ref["sha256"],
            "tile_sha256": digests,
            "selection": {
                "kind": kind,
                "period": "2025" if kind == "annual" else "2026-05",
                "evaluation_month": "2026-05",
                "channels": channels,
            },
        }
    geographic = write_json(
        tmp_path / "geography.json",
        {
            "state": "verified",
            "reference_cache_sha256": cache_ref["sha256"],
            "model_manifest_sha256": {name: m["manifest_sha256"] for name, m in models.items()},
            "evidence": [cache_ref],
            "scope": "synthetic common projected grid",
        },
    )
    spec = {
        "protocol": workflow.PROTOCOL,
        "output": str(tmp_path / "output"),
        "reference_cache": cache_ref,
        "labels": labels,
        "models": models,
        "month": "2026-05",
        "support_seeds": [7],
        "budgets": [1],
        "retrieval_budgets": [1],
        "method_groups": {name: [name] for name in models},
        "geographic_audit": geographic,
    }
    lock = {"state": "locked", "contract_sha256": workflow.contract_sha256(spec)}
    spec["lock"] = write_json(tmp_path / "lock.json", lock)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path, spec, test_paths


def test_two_phase_pipeline_pairs_domains_and_scores_without_refitting(tmp_path, monkeypatch):
    path, spec, test_paths = fixture_spec(tmp_path)
    for p in test_paths:
        p.rename(p.with_suffix(".withheld"))
    workflow.calibrate(path)
    for p in test_paths:
        p.with_suffix(".withheld").rename(p)
    root = Path(spec["output"])
    calibration = root / "calibration/identity.json"
    identity = json.loads(calibration.read_text())
    assert identity["test_scored"] is False
    assert len(identity["conditions"]) == 20
    assert np.load(root / "calibration/common_valid.npy").sum() == 2 * 255
    before = {
        str(p.relative_to(root)): sha(p) for p in (root / "calibration").rglob("*") if p.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("test scoring must never fit")

    monkeypatch.setattr(workflow, "fit_classification", forbidden)
    monkeypatch.setattr(workflow, "fit_regression", forbidden)
    monkeypatch.setattr(workflow, "freeze_retrieval", forbidden)
    workflow.score(path, sha(calibration))
    assert before == {
        str(p.relative_to(root)): sha(p) for p in (root / "calibration").rglob("*") if p.is_file()
    }
    final = json.loads((root / "test/identity.json").read_text())
    assert final["test_scored"] and final["parameters_refitted"] is False
    rows = json.loads((root / "test/results.json").read_text())
    assert set(rows) == {"base", "candidate"}
    assert all(len(v) == 20 for v in rows.values())
    assert {r["family"] for r in rows["base"]} == {"C", "R", "Q"}
    for row in rows["base"]:
        with (
            np.load(root / "test/predictions/base" / (row["key"] + ".npz")) as a,
            np.load(root / "test/predictions/candidate" / (row["key"] + ".npz")) as b,
        ):
            np.testing.assert_array_equal(a["truth"], b["truth"])
            np.testing.assert_array_equal(a["tiles"], b["tiles"])
    with pytest.raises(FileExistsError):
        workflow.score(path, sha(calibration))


@pytest.mark.parametrize("change", ["lock", "contract", "geography"])
def test_preflight_rejects_unlocked_or_changed_contract_before_outputs(tmp_path, change):
    path, spec, _ = fixture_spec(tmp_path)
    if change == "lock":
        record = json.loads(Path(spec["lock"]["path"]).read_text())
        record["state"] = "pending"
        spec["lock"] = write_json(Path(spec["lock"]["path"]), record)
    elif change == "contract":
        spec["budgets"] = [2]
    else:
        Path(spec["geographic_audit"]["path"]).write_text("{}")
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        workflow.calibrate(path)
    assert not Path(spec["output"]).exists()


def test_empty_test_label_domain_keeps_all_registered_conditions(tmp_path):
    path, spec, _ = fixture_spec(tmp_path)
    label_path = Path(spec["labels"]["test"]["path"])
    with np.load(label_path) as data:
        arrays = {k: data[k] for k in data.files}
    for task in (*workflow.OSM_TASKS, "esri"):
        arrays[task][:] = -1
    np.savez(label_path, **arrays)
    spec["labels"]["test"]["sha256"] = sha(label_path)
    spec["lock"] = write_json(
        Path(spec["lock"]["path"]),
        {"state": "locked", "contract_sha256": workflow.contract_sha256(spec)},
    )
    path.write_text(json.dumps(spec))
    workflow.calibrate(path)
    root = Path(spec["output"])
    workflow.score(path, sha(root / "calibration/identity.json"))
    results = json.loads((root / "test/results.json").read_text())
    for rows in results.values():
        assert len(rows) == 20
        assert all(r["metrics"]["observations"] == 0 for r in rows)
        assert all(r["metrics"].get("rmse", r["metrics"].get("ap")) is None for r in rows)


def test_cli_dispatches_separate_phases(tmp_path, monkeypatch):
    from xuannv_embedding.training.experiment import main

    calls = []
    monkeypatch.setattr(workflow, "calibrate", lambda path: calls.append(("fit", path)))
    monkeypatch.setattr(
        workflow, "score", lambda path, digest: calls.append(("score", path, digest))
    )
    spec = tmp_path / "spec.json"
    assert main(["calibrate-primary", "--spec", str(spec)]) == 0
    assert (
        main(["score-primary", "--spec", str(spec), "--calibration-identity-sha256", "a" * 64]) == 0
    )
    assert calls == [("fit", spec), ("score", spec, "a" * 64)]


def test_changed_frozen_readout_stops_before_test_feature_loading(tmp_path, monkeypatch):
    path, spec, _ = fixture_spec(tmp_path)
    workflow.calibrate(path)
    root = Path(spec["output"])
    identity = root / "calibration/identity.json"
    parameters = next((root / "calibration/readouts").rglob("parameters.npz"))
    parameters.write_bytes(parameters.read_bytes() + b"changed")
    monkeypatch.setattr(
        workflow, "read_features", lambda **kwargs: pytest.fail("test features opened")
    )
    with pytest.raises(ValueError, match="parameters"):
        workflow.score(path, sha(identity))


def test_label_bundle_partition_binding_is_checked(tmp_path):
    path, spec, _ = fixture_spec(tmp_path)
    target = Path(spec["labels"]["validation"]["path"])
    with np.load(target) as archive:
        data = {k: archive[k] for k in archive.files}
    data["indices"] = [2]
    np.savez(target, **data)
    spec["labels"]["validation"]["sha256"] = sha(target)
    spec["lock"] = write_json(
        Path(spec["lock"]["path"]),
        {
            "state": "locked",
            "contract_sha256": workflow.contract_sha256(spec),
        },
    )
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="label.*partition"):
        workflow.calibrate(path)
