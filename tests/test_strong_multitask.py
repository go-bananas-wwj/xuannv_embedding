import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from test_paired_multitask import fixture_spec, write_json

from xuannv_embedding.downstream import strong_multitask as workflow
from xuannv_embedding.export.context import sha


def strong_spec(tmp_path):
    path, primary, test_paths = fixture_spec(tmp_path)
    # One task has no valid held-out truth: preserve it as undefined, never invent zero AP.
    target = Path(primary["labels"]["test"]["path"])
    with np.load(target, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["osm_green"] = np.full_like(arrays["osm_green"], -1)
    np.savez(target, **arrays)
    primary["labels"]["test"]["sha256"] = sha(target)
    primary["lock"] = write_json(
        tmp_path / "lock.json",
        {"state": "locked", "contract_sha256": workflow.primary.contract_sha256(primary)},
    )
    path.write_text(json.dumps(primary))
    spec = {
        "protocol": workflow.PROTOCOL,
        "primary_spec": {"path": str(path), "sha256": sha(path)},
        "heads": list(workflow.HEADS),
        "neural_device": "cpu",
        "output": str(tmp_path / "strong"),
    }
    spec["lock"] = write_json(
        tmp_path / "strong-lock.json",
        {"state": "locked", "contract_sha256": workflow.contract_sha256(spec)},
    )
    target = tmp_path / "strong-spec.json"
    target.write_text(json.dumps(spec))
    return target, spec, primary, test_paths


@pytest.fixture(scope="module")
def calibration(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("strong_calibration")
    path, spec, primary, test_paths = strong_spec(tmp)
    # Metadata can be checked, but actual held-out files must not be opened in calibration.
    for p in test_paths:
        p.rename(p.with_suffix(".withheld"))
    original_fit = workflow.neural_readouts.fit_neural
    observed = []

    def inspect_context(*args, **kwargs):
        # The annual baseline itself is valid here, but the paired candidate is not.
        assert not args[3][:, 0, 0].any()
        assert not args[6][:, 0, 0].any()
        observed.append((args[0], args[1].shape[1]))
        return original_fit(*args, **kwargs)

    workflow.neural_readouts.fit_neural = inspect_context
    try:
        workflow.calibrate(path)
    finally:
        workflow.neural_readouts.fit_neural = original_fit
    assert len(observed) == 40
    assert set(observed) == {(head, channels) for head in ("mlp", "conv3x3") for channels in (3, 5)}
    for p in test_paths:
        p.with_suffix(".withheld").rename(p)
    return path, spec, primary


@pytest.fixture
def copied(calibration, tmp_path):
    _, spec, primary = calibration
    spec = dict(spec, output=str(tmp_path / "output"))
    shutil.copytree(
        Path(calibration[1]["output"]) / "calibration", Path(spec["output"]) / "calibration"
    )
    path = tmp_path / "strong.json"
    path.write_text(json.dumps(spec))
    return path, spec, primary


def test_all_five_heads_share_support_and_common_context_and_freeze_parameters(copied, monkeypatch):
    path, spec, primary = copied
    root = Path(spec["output"])
    cal = root / "calibration"
    identity = json.loads((cal / "identity.json").read_text())
    assert len(identity["conditions"]) == 50
    assert identity["test_scored"] is False
    assert set(identity["heads"]) == {"rf", "svm", "knn", "mlp", "conv3x3"}
    context = np.load(cal / "common_valid.npy")
    assert not context[:, 0, 0].any()
    for condition in identity["conditions"]:
        key = condition["key"]
        support = json.loads((cal / "readouts" / key / "support.json").read_text())
        assert support["support_tiles"] == [0]
        assert 0 not in np.load(cal / "readouts" / key / "positions.npy")
        if condition["head"] in ("mlp", "conv3x3"):
            for name in primary["models"]:
                record = json.loads((cal / "readouts" / key / name / "identity.json").read_text())
                assert record["metadata"]["fitted_pixels"] == 255
                assert record["metadata"]["invalid_context"] == "zero_after_standardization"
    # Remove source training/validation arrays; test scoring can use only frozen parameters.
    withheld = []
    for partition in ["train", "validation"]:
        withheld.append(Path(primary["labels"][partition]["path"]))
    for name in primary["models"]:
        manifest = json.loads(Path(primary["models"][name]["manifest_path"]).read_text())
        withheld.extend(Path(r["path"]) for r in manifest["records"][:2])
    for p in withheld:
        p.rename(p.with_suffix(".withheld"))

    def forbidden(*a, **k):
        raise AssertionError("test must not fit")

    monkeypatch.setattr(workflow.strong_classifiers, "fit_classifier", forbidden)
    monkeypatch.setattr(workflow.neural_readouts, "fit_neural", forbidden)
    before = {str(p.relative_to(cal)): sha(p) for p in cal.rglob("*") if p.is_file()}
    try:
        result = workflow.score(path, sha(cal / "identity.json"))
    finally:
        for p in withheld:
            p.with_suffix(".withheld").rename(p)
    assert result["parameters_refitted"] is False and result["test_scored"] is True
    assert before == {str(p.relative_to(cal)): sha(p) for p in cal.rglob("*") if p.is_file()}
    rows = json.loads((root / "test/results.json").read_text())
    domains = {}
    for name, records in rows.items():
        assert len(records) == 50
        for row in records:
            with np.load(root / "test/predictions" / name / (row["key"] + ".npz")) as data:
                assert data["truth"].shape == ((0,) if row["task"] == "osm_green" else (255,))
                assert np.isfinite(data["scores"]).all()
                if row["task"] == "osm_green":
                    assert row["metrics"]["ap"] is None
                    assert row["metrics"]["f1"] is None
                    assert row["metrics"]["observations"] == 0
                else:
                    assert row["metrics"]["ap"] == pytest.approx(
                        average_precision_score(data["truth"], data["scores"]), abs=1e-12
                    )
                domain = tuple(data["valid_indices"])
                assert domains.setdefault(row["task"], domain) == domain


@pytest.mark.parametrize(
    "which", ["support", "positions", "readout", "prediction", "common", "identity"]
)
def test_changed_calibration_is_rejected_before_test_access(copied, monkeypatch, which):
    path, spec, _ = copied
    cal = Path(spec["output"]) / "calibration"
    digest = sha(cal / "identity.json")
    files = {
        "support": "readouts/C_osm_building_7_1_rf/support.json",
        "positions": "readouts/C_osm_building_7_1_rf/positions.npy",
        "readout": "readouts/C_osm_building_7_1_rf/base/parameters.npz",
        "prediction": "predictions/base/C_osm_building_7_1_rf.npz",
        "common": "common_valid.npy",
        "identity": "identity.json",
    }
    p = cal / files[which]
    p.write_bytes(p.read_bytes() + b"changed")
    monkeypatch.setattr(
        workflow.primary, "_common", lambda *a, **k: pytest.fail("early test data access")
    )
    with pytest.raises(ValueError):
        workflow.score(path, digest)
    assert not (Path(spec["output"]) / "test").exists()


@pytest.mark.parametrize("change", ["heads", "device", "lock", "primary"])
def test_contract_changes_are_rejected_before_creating_output(tmp_path, change):
    path, spec, _, _ = strong_spec(tmp_path)
    if change == "heads":
        spec["heads"] = ["mlp"]
    if change == "device":
        spec["neural_device"] = "npu"
    if change == "lock":
        spec["lock"]["sha256"] = "0" * 64
    if change == "primary":
        spec["primary_spec"]["sha256"] = "0" * 64
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        workflow.calibrate(path)
    assert not Path(spec["output"]).exists()


def test_existing_calibration_is_not_overwritten(copied):
    path, _, _ = copied
    with pytest.raises(FileExistsError):
        workflow.calibrate(path)


def test_full_tile_support_is_shared_with_explicit_nested_internal_pixel_caps():
    pattern = (np.indices((128, 128)).sum(0) % 2).astype(np.int8)
    labels = np.stack([pattern, pattern, pattern])
    ids, indices = ["a", "b", "c"], [10, 20, 30]
    results = {}
    for head in workflow.HEADS:
        c = {"head": head, "budget": 2, "seed": 7}
        results[head] = workflow._support(labels, ids, [0, 1, 2], indices, c)
    assert len({tuple(v[0]) for v in results.values()}) == 1
    assert len({tuple(v[3]["support_tiles"]) for v in results.values()}) == 1
    rf, svm, knn, mlp, conv = (results[h] for h in workflow.HEADS)
    np.testing.assert_array_equal(rf[2], svm[2])
    np.testing.assert_array_equal(rf[1], knn[1])
    np.testing.assert_array_equal(knn[2], np.r_[rf[2][:1024], rf[2][4096:5120]])
    assert [len(v[2]) for v in (rf, svm, knn, mlp, conv)] == [8192, 8192, 2048, 32768, 32768]
