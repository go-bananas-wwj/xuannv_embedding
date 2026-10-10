import copy
import json

import numpy as np
import pytest
import torch

from xuannv_embedding.downstream.multitask_features import FeatureSelection, read_features
from xuannv_embedding.downstream.raw_monthly import export, features
from xuannv_embedding.export.context import sha


def fixture():
    sample = {
        "patch_id": "p0",
        "timestamps": torch.tensor([202604, 202605]),
        "source_frames": {"opt": torch.ones(2, 2, 2, 2)},
        "source_masks": {"opt": torch.tensor([1.0, 0.0])},
        "highres_frames": {"hr": torch.arange(18).reshape(2, 1, 3, 3).float()},
        "highres_masks": {"hr": torch.ones(2, 1, 3, 3)},
    }
    inputs = {"opt": {"channels": 2, "role": "temporal"}, "hr": {"channels": 1, "role": "highres"}}
    return sample, inputs


def test_noninteger_area_overlap_and_channel_order():
    sample, inputs = fixture()
    x, valid, channels = features(sample, inputs, ["opt", "hr"], ["2026-04", "2026-05"], 2)
    assert x.shape == (10, 2, 2)
    assert valid.all()
    np.testing.assert_array_equal(x[:3], 1)
    np.testing.assert_array_equal(x[3:6], 0)
    # Independent exact overlaps: each destination has 2/3 and 1/3 source weights.
    np.testing.assert_allclose(x[6], [[4 / 3, 8 / 3], [16 / 3, 20 / 3]], atol=1e-6)
    np.testing.assert_array_equal(x[7], 1)
    assert channels[6] == {"source": "hr", "month": "2026-04", "band": 0}
    assert channels[7]["band"] == "availability"


def test_masked_area_mean_excludes_missing_nan_and_retains_fraction():
    sample, inputs = fixture()
    sample["highres_masks"]["hr"][0, 0, 0, 0] = 0
    sample["highres_frames"]["hr"][0, 0, 0, 0] = float("nan")
    sample["source_frames"]["opt"][1] = float("nan")
    x, _, _ = features(sample, inputs, ["opt", "hr"], ["2026-04", "2026-05"], 2)
    assert np.isfinite(x).all()
    assert x[7, 0, 0] == pytest.approx(5 / 9)
    assert x[6, 0, 0] == pytest.approx(12 / 5)


def test_target_and_quality_labels_are_not_inputs():
    sample, inputs = fixture()
    expected = features(sample, inputs, ["opt", "hr"], ["2026-04", "2026-05"], 2)
    for key in ["targets", "target_masks", "supervised_labels", "highres_quality_masks"]:
        sample[key] = {"anything": object()}
    actual = features(sample, inputs, ["opt", "hr"], ["2026-04", "2026-05"], 2)
    np.testing.assert_array_equal(actual[0], expected[0])


def test_all_modalities_missing_is_explicit_invalid_zero():
    sample, inputs = fixture()
    sample["source_masks"]["opt"].zero_()
    sample["highres_masks"]["hr"].zero_()
    x, valid, _ = features(sample, inputs, ["opt", "hr"], ["2026-04", "2026-05"], 2)
    assert not x.any() and not valid.any()


@pytest.mark.parametrize("bad", ["time", "channels", "mask", "nonfinite", "extra_source"])
def test_rejects_input_contract_errors(bad):
    sample, inputs = fixture()
    if bad == "time":
        sample["timestamps"][0] = 202603
    elif bad == "channels":
        sample["source_frames"]["opt"] = torch.ones(2, 3, 2, 2)
    elif bad == "mask":
        sample["highres_masks"]["hr"][0, 0, 0, 0] = 0.5
    elif bad == "nonfinite":
        sample["source_frames"]["opt"][0, 0, 0, 0] = float("nan")
    else:
        sample["source_frames"]["other"] = torch.ones(2, 1, 2, 2)
    with pytest.raises(ValueError):
        features(sample, inputs, ["opt", "hr"], ["2026-04", "2026-05"], 2)


def make_spec(tmp_path):
    sample, inputs = fixture()
    records = []
    for i in range(4):
        s = copy.deepcopy(sample)
        s["patch_id"] = f"p{i}"
        path = tmp_path / f"sample{i}.pt"
        if i < 2:
            torch.save(s, path)
        records.append(
            {
                "patch_id": s["patch_id"],
                "path": str(path),
                "sha256": sha(path) if path.exists() else "0" * 64,
                "bounds": [i * 2, 0, i * 2 + 2, 2],
            }
        )
    cache = {
        "data": {"months": ["2026-04", "2026-05"], "patch_size": 2},
        "model_inputs": inputs,
        "records": records,
        "split": {"train": [0], "validation": [1], "test": [2], "buffer": [3]},
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(cache))
    spec = {
        "protocol": "monthly-input-raw-v1",
        "cache": {"path": str(path), "sha256": sha(path)},
        "sources": ["opt", "hr"],
        "period": "2026-05",
        "splits": ["train", "validation"],
        "output": str(tmp_path / "output"),
    }
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    return sp, spec


def test_export_selected_only_roundtrips_feature_contract(tmp_path):
    sp, spec = make_spec(tmp_path)
    result = export(sp)
    assert result["test_records_read"] is False
    out = tmp_path / "output"
    manifest = out / "manifest.json"
    loaded = read_features(
        manifest,
        spec["cache"]["path"],
        manifest_sha256=sha(manifest),
        cache_sha256=spec["cache"]["sha256"],
        tile_sha256=result["tile_sha256"],
        selection=FeatureSelection("raw", "2026-05", "2026-05", 10),
    )
    assert loaded.values.shape == (2, 2, 2, 10)
    assert not (out / "tile_000002.npz").exists()
    with pytest.raises(FileExistsError):
        export(sp)


def test_changed_sample_fails_without_complete_manifest(tmp_path):
    sp, _ = make_spec(tmp_path)
    with (tmp_path / "sample0.pt").open("ab") as f:
        f.write(b"change")
    with pytest.raises(ValueError, match="digest"):
        export(sp)
    assert not (tmp_path / "output/manifest.json").exists()
    assert json.loads((tmp_path / "output/status.json").read_text())["state"] == "failed"


def test_rejects_unregistered_sources_and_future_period(tmp_path):
    sp, spec = make_spec(tmp_path)
    for key, value in [("sources", ["hr"]), ("period", "2026-04"), ("splits", ["pilot"])]:
        changed = {**spec, key: value}
        sp.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            export(sp)
