import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_raw_monthly import make_spec

from xuannv_embedding.downstream.late_fusion import export, highres_statistics
from xuannv_embedding.downstream.multitask_features import FeatureSelection, read_features
from xuannv_embedding.export.context import sha


def sample():
    return {
        "timestamps": torch.tensor([202604, 202605]),
        "highres_frames": {"hr": torch.arange(18).reshape(2, 1, 3, 3).float()},
        "highres_masks": {"hr": torch.ones(2, 1, 3, 3)},
    }


def statistics(s):
    return highres_statistics(
        s, {"hr": {"channels": 1, "role": "highres"}}, ["hr"], ["2026-04", "2026-05"], "2026-05", 2
    )


def test_area_moments_match_independent_noninteger_overlap():
    s = sample()
    s["highres_masks"]["hr"][1, 0, 0, 0] = 0
    s["highres_frames"]["hr"][1, 0, 0, 0] = float("nan")
    actual, channels = statistics(s)
    for y in range(2):
        for x in range(2):
            values, weights = [], []
            for sy in range(3):
                for sx in range(3):
                    weight = max(0, min((y + 1) * 1.5, sy + 1) - max(y * 1.5, sy))
                    weight *= max(0, min((x + 1) * 1.5, sx + 1) - max(x * 1.5, sx))
                    if weight and s["highres_masks"]["hr"][1, 0, sy, sx]:
                        values.append(float(s["highres_frames"]["hr"][1, 0, sy, sx]))
                        weights.append(weight)
            mean = np.average(values, weights=weights)
            std = np.sqrt(np.average((np.asarray(values) - mean) ** 2, weights=weights))
            np.testing.assert_allclose(actual[:, y, x], [mean, std, sum(weights) / 2.25], atol=1e-6)
    assert [c["statistic"] for c in channels] == ["mean", "std", "availability"]


def test_statistics_ignore_labels_other_months_and_missing_values():
    s = sample()
    expected = statistics(s)[0]
    s["targets"] = object()
    s["target_masks"] = object()
    s["supervised_labels"] = object()
    s["highres_frames"]["hr"][0] = float("nan")
    np.testing.assert_array_equal(statistics(s)[0], expected)
    s["highres_masks"]["hr"].zero_()
    s["highres_frames"]["hr"][:] = float("nan")
    np.testing.assert_array_equal(statistics(s)[0], 0)


def test_centered_variance_retains_small_variation_on_large_offset():
    s = sample()
    s["highres_frames"]["hr"] = s["highres_frames"]["hr"].double()
    original = statistics(s)[0][1]
    s["highres_frames"]["hr"] += 1e10
    np.testing.assert_allclose(statistics(s)[0][1], original, atol=2e-6)


@pytest.mark.parametrize("invalid", ["timestamps", "mask", "channels", "nonfinite"])
def test_invalid_statistics_inputs_fail(invalid):
    s = sample()
    if invalid == "timestamps":
        s["timestamps"][1] = 202603
    elif invalid == "mask":
        s["highres_masks"]["hr"][1, 0, 0, 0] = 0.5
    elif invalid == "channels":
        s["highres_frames"]["hr"] = torch.ones(2, 2, 3, 3)
    else:
        s["highres_frames"]["hr"][1, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        statistics(s)


def setup_export(tmp_path):
    _, raw_spec = make_spec(tmp_path)
    cache = json.loads((tmp_path / "cache.json").read_text())
    manifest = {
        "months": ["2026-04", "2026-05"],
        "cache_sha256": raw_spec["cache"]["sha256"],
        "split": cache["split"],
        "records": [],
    }
    digests = {}
    for i, record in enumerate(cache["records"]):
        path = tmp_path / f"base{i}.npz"
        row = {"patch_id": record["patch_id"], "bounds": record["bounds"], "path": str(path)}
        if i < 2:
            np.savez(
                path,
                embedding=np.ones((2, 3, 2, 2), np.float32),
                timestamps=np.array([202604, 202605]),
                valid_mask=np.ones((2, 2), bool),
            )
            row["sha256"] = digests[record["patch_id"]] = sha(path)
        manifest["records"].append(row)
    mp = tmp_path / "base_manifest.json"
    mp.write_text(json.dumps(manifest))
    spec = {
        "protocol": "monthly-late-fusion-v1",
        "cache": raw_spec["cache"],
        "base": {
            "manifest_path": str(mp),
            "manifest_sha256": sha(mp),
            "cache_path": raw_spec["cache"]["path"],
            "cache_sha256": raw_spec["cache"]["sha256"],
            "tile_sha256": digests,
            "channels": 3,
        },
        "sources": ["hr"],
        "month": "2026-05",
        "splits": ["train", "validation"],
        "output": str(tmp_path / "late"),
    }
    sp = tmp_path / "late_spec.json"
    sp.write_text(json.dumps(spec))
    return sp, spec


def test_export_roundtrip_reads_only_selected_splits_and_keeps_base_validity(tmp_path):
    path, spec = setup_export(tmp_path)
    report = export(path)
    assert report["channels"] == 6 and not report["test_records_read"]
    manifest = tmp_path / "late/manifest.json"
    loaded = read_features(
        manifest,
        spec["cache"]["path"],
        manifest_sha256=sha(manifest),
        cache_sha256=spec["cache"]["sha256"],
        tile_sha256=report["tile_sha256"],
        selection=FeatureSelection("monthly", "2026-05", "2026-05", 6),
    )
    np.testing.assert_array_equal(loaded.values[..., :3], 1)
    assert loaded.valid.all() and not (tmp_path / "late/tile_000002.npz").exists()
    with pytest.raises(FileExistsError):
        export(path)


def test_export_rejects_misaligned_base_geometry_and_changed_samples(tmp_path):
    path, spec = setup_export(tmp_path)
    mp = Path(spec["base"]["manifest_path"])
    original = json.loads(mp.read_text())
    changed = copy.deepcopy(original)
    changed["records"][0]["bounds"][0] += 1
    mp.write_text(json.dumps(changed))
    spec["base"]["manifest_sha256"] = sha(mp)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        export(path)
    mp.write_text(json.dumps(original))
    spec["base"]["manifest_sha256"] = sha(mp)
    path.write_text(json.dumps(spec))
    with (tmp_path / "sample0.pt").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="digest"):
        export(path)
