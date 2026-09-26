import copy
import json
from pathlib import Path

import numpy as np
import pytest
from test_paired_multitask import fixture_spec, write_json

from xuannv_embedding.downstream import fixed_stress, paired_multitask
from xuannv_embedding.downstream.multitask_features import FeatureSelection, read_features
from xuannv_embedding.export.context import sha


def fixture_stress(tmp_path, split="validation"):
    path, spec, test_paths = fixture_spec(tmp_path)
    model = spec["models"]["candidate"]
    manifest_path = Path(model["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    manifest.update(checkpoint_sha256="a" * 64, config_sha256="b" * 64, training_git_sha="c" * 40)
    manifest_path.write_text(json.dumps(manifest))
    model["manifest_sha256"] = sha(manifest_path)
    geo_path = Path(spec["geographic_audit"]["path"])
    geo = json.loads(geo_path.read_text())
    geo["model_manifest_sha256"]["candidate"] = sha(manifest_path)
    spec["geographic_audit"] = write_json(geo_path, geo)
    spec["lock"] = write_json(
        Path(spec["lock"]["path"]),
        {"state": "locked", "contract_sha256": paired_multitask.contract_sha256(spec)},
    )
    path.write_text(json.dumps(spec))
    paired_multitask.calibrate(path)
    root = Path(spec["output"])
    calibration = root / "calibration/identity.json"
    if split == "test":
        paired_multitask.score(path, sha(calibration))
    variants = {}
    for fraction in (1, 0.5, 0):
        name = "retain" + str(int(fraction * 100))
        partial = copy.deepcopy(manifest)
        partial["input_ablation"] = {
            "dropped_sources": [],
            "last_visible_month_index": None,
            "last_visible_month": None,
            "undated_static_inputs": "unchanged unless dropped",
            "training_time_causality_claim": False,
            "highres_retention": {
                "sources": ["highres_optical"],
                "fraction": fraction,
                "seed": 41,
                "rule": fixed_stress.RETENTION_RULE,
            },
        }
        feature = copy.deepcopy(model)
        for i, record in enumerate(partial["records"]):
            with np.load(record["path"]) as archive:
                data = {k: archive[k] for k in archive.files}
            data["embedding"] *= fraction
            p = tmp_path / f"{name}_{i}.npz"
            np.savez(p, **data)
            record["path"] = str(p)
            feature["tile_sha256"][record["patch_id"]] = sha(p)
        record = write_json(tmp_path / (name + ".json"), partial)
        feature.update(manifest_path=record["path"], manifest_sha256=record["sha256"])
        variants[name] = {"model": "candidate", "fraction": fraction, "features": feature}
    stress = {
        "protocol": fixed_stress.PROTOCOL,
        "base_spec": {"path": str(path), "sha256": sha(path)},
        "calibration_identity_sha256": sha(calibration),
        "baseline_identity_sha256": sha(
            root / ("test" if split == "test" else "calibration") / "identity.json"
        ),
        "split": split,
        "sources": ["highres_optical"],
        "fractions": [1, 0.5, 0],
        "mask_seed": 41,
        "variants": variants,
        "output": str(tmp_path / "stress"),
    }
    stress_path = tmp_path / "stress_spec.json"
    stress_path.write_text(json.dumps(stress))
    return stress_path, stress, test_paths


@pytest.mark.parametrize("split", ["validation", "test"])
def test_stress_reuses_frozen_heads_and_exact_baseline_domain(tmp_path, monkeypatch, split):
    path, spec, test_paths = fixture_stress(tmp_path, split)
    if split == "validation":
        for p in test_paths:
            p.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("stress scoring cannot refit")

    for name in ("_fit", "fit_classification", "fit_regression", "freeze_retrieval"):
        monkeypatch.setattr(paired_multitask, name, forbidden)
    result = fixed_stress.run(path)
    assert result["parameters_refitted"] is False
    assert result["test_scored"] == (split == "test")
    assert all(len(rows) == 20 for rows in result["results"].values())
    assert result["identity_control_verified"] == ["candidate"]
    root = Path(spec["output"])
    for row in result["results"]["retain100"]:
        assert row["maximum_absolute_score_change"] == 0
        with (
            np.load(root / "predictions/retain100" / (row["key"] + ".npz")) as a,
            np.load(root / "predictions/retain0" / (row["key"] + ".npz")) as b,
        ):
            for key in ("truth", "tiles", "valid_indices"):
                np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(FileExistsError):
        fixed_stress.run(path)


@pytest.mark.parametrize("change", ["checkpoint", "cache", "mask", "validity", "identity", "curve"])
def test_stress_rejects_changed_model_mask_domain_or_missing_control(tmp_path, change):
    path, spec, _ = fixture_stress(tmp_path)
    feature = spec["variants"]["retain50"]["features"]
    manifest_path = Path(feature["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    if change in ("checkpoint", "cache"):
        manifest[change + "_sha256"] = "d" * 64
    elif change == "mask":
        manifest["input_ablation"]["highres_retention"]["seed"] += 1
    elif change == "curve":
        del spec["variants"]["retain100"]
    else:
        if change == "identity":
            feature = spec["variants"]["retain100"]["features"]
            manifest_path = Path(feature["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
        tile = Path(manifest["records"][1]["path"])
        with np.load(tile) as archive:
            data = {k: archive[k] for k in archive.files}
        if change == "validity":
            data["valid_mask"][2, 2] = False
        else:
            data["embedding"] *= 2
        np.savez(tile, **data)
        feature["tile_sha256"]["p1"] = sha(tile)
    manifest_path.write_text(json.dumps(manifest))
    feature["manifest_sha256"] = sha(manifest_path)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        fixed_stress.run(path)


def test_partial_stress_exports_never_require_training_or_test_features(tmp_path):
    path, spec, _ = fixture_stress(tmp_path)
    for variant in spec["variants"].values():
        feature = variant["features"]
        manifest_path = Path(feature["manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        manifest.update(exported_indices=[1], exported_splits=["validation"])
        for i in [0, 2]:
            Path(manifest["records"][i]["path"]).unlink()
        feature["tile_sha256"] = {"p1": feature["tile_sha256"]["p1"]}
        manifest_path.write_text(json.dumps(manifest))
        feature["manifest_sha256"] = sha(manifest_path)
    path.write_text(json.dumps(spec))
    assert fixed_stress.run(path)["state"] == "complete"
    feature = spec["variants"]["retain100"]["features"]
    with pytest.raises(ValueError, match="not materialized"):
        read_features(
            **dict(feature, selection=FeatureSelection(**feature["selection"])), splits=("test",)
        )
