import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_calendar import calendar_coefficients, prepare
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS


def fixture_calendar(tmp_path):
    frames = np.ones((2, 12, 19, 2, 2), np.float32)
    frames += np.sin(np.arange(12) * 2 * np.pi / 12)[None, :, None, None, None]
    masks = np.ones((2, 12, 3, 2, 2), bool)
    masks[1, :, 1] = False
    fp, mp, ip, npth = [
        tmp_path / n for n in ["frames.npy", "masks.npy", "index.json", "norm.json"]
    ]
    np.save(fp, frames)
    np.save(mp, masks)
    dump(ip, {"records": [{"patch_id": str(i)} for i in range(2)]})
    dump(npth, {"source": "already normalized monthly cache"})

    def ref(p):
        return {"path": str(p), "sha256": sha(p)}

    metadata = tmp_path / "monthly.json"
    dump(
        metadata,
        {
            "schema": "public-monthly-tensor-v1",
            "months": list(range(202401, 202413)),
            "index": ref(ip),
            "frames": ref(fp),
            "masks": ref(mp),
            "normalization": ref(npth),
            "shape": list(frames.shape),
            "normalization_clip": [-6, 6],
            "band_slices": {"s2": [0, 11], "s1": [11, 13], "landsat": [13, 19]},
            "bands": PUBLIC_BANDS,
            "map_or_local_image_values_used": False,
            "source_observations": {"path": "must-not-open-ancestors", "sha256": "0" * 64},
            "period_provenance": {"path": "must-not-open-ancestors", "sha256": "0" * 64},
        },
    )
    spec = tmp_path / "spec.json"
    dump(
        spec,
        {
            "schema": "public-calendar-statistics-v1",
            "monthly": ref(metadata),
            "index": ref(ip),
            "ridge": 0.1,
            "reserve_gib": 0,
            "output": str(tmp_path / "features"),
        },
    )
    return spec


def test_calendar_coefficients_preserve_dates_and_recover_known_annual_signal():
    months = list(range(202401, 202413))
    signal = 2 + np.sin(np.arange(12) * 2 * np.pi / 12)
    x = np.broadcast_to(signal[:, None, None, None], (12, 19, 2, 2)).copy()
    mask = np.ones((12, 3, 2, 2), bool)
    actual = calendar_coefficients(x, mask, months, ridge=0.1).reshape(5, 19, 2, 2)
    np.testing.assert_allclose(actual[0], 2, atol=1e-6)
    np.testing.assert_allclose(actual[1], 6 / 6.1, atol=1e-6)
    np.testing.assert_allclose(actual[2:], 0, atol=1e-6)
    order = np.random.default_rng(41).permutation(12)
    np.testing.assert_allclose(
        calendar_coefficients(x[order], mask[order], [months[i] for i in order], ridge=0.1),
        actual.reshape(95, 2, 2),
        atol=1e-6,
    )
    mismatched = calendar_coefficients(x, mask, months[1:] + months[:1], ridge=0.1)
    assert not np.allclose(mismatched, actual.reshape(95, 2, 2))


def test_calendar_coefficients_ignore_masked_values_and_handle_one_or_no_observation():
    x = np.full((12, 19, 2, 2), np.nan)
    mask = np.zeros((12, 3, 2, 2), bool)
    mask[4, 0] = True
    x[4, :11] = 3
    actual = calendar_coefficients(x, mask, list(range(202401, 202413)), ridge=0.1)
    coefs = actual.reshape(5, 19, 2, 2)
    np.testing.assert_allclose(coefs[0, :11], 3, atol=1e-5)
    np.testing.assert_allclose(coefs[1:, :11], 0, atol=1e-5)
    np.testing.assert_array_equal(coefs[:, 11:], 0)
    np.testing.assert_allclose(
        calendar_coefficients(
            np.nan_to_num(x, nan=999), mask, list(range(202401, 202413)), ridge=0.1
        ),
        actual,
    )


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate_month",
        "mask_dtype",
        "valid_nan",
        "wrong_shape",
        "zero_ridge",
        "nan_ridge",
        "bool_ridge",
    ],
)
def test_calendar_coefficients_reject_invalid_inputs(failure):
    x, mask = np.ones((12, 19, 2, 2)), np.ones((12, 3, 2, 2), bool)
    months, ridge = list(range(202401, 202413)), 0.1
    if failure == "duplicate_month":
        months[-1] = months[0]
    elif failure == "mask_dtype":
        mask = mask.astype(float)
    elif failure == "valid_nan":
        x[0, 0, 0, 0] = np.nan
    elif failure == "wrong_shape":
        x = x[:-1]
    else:
        ridge = {"zero_ridge": 0, "nan_ridge": np.nan, "bool_ridge": True}[failure]
    with pytest.raises(ValueError):
        calendar_coefficients(x, mask, months, ridge=ridge)


def test_calendar_preparation_is_map_free_repeatable_and_detects_changed_outputs(tmp_path):
    spec = fixture_calendar(tmp_path)
    metadata = prepare(spec)
    d = json.loads(metadata.read_text())
    x = np.load(d["features"]["path"])
    assert x.shape == (2, 95, 2, 2) and np.isfinite(x).all()
    assert d["map_labels_used"] is False and d["feature_order"] == "basis_then_band"
    assert d["observation_count_histograms"]["s1"][0] == 4
    assert d["observation_count_histograms"]["s1"][12] == 4
    assert prepare(spec) == metadata
    x[0, 0, 0, 0] += 1
    np.save(d["features"]["path"], x)
    with pytest.raises(ValueError, match="changed"):
        prepare(spec)


@pytest.mark.parametrize("failure", ["unknown_field", "wrong_index", "disk_reserve"])
def test_calendar_preparation_rejects_unsafe_specifications(tmp_path, monkeypatch, failure):
    spec = fixture_calendar(tmp_path)
    data = json.loads(spec.read_text())
    if failure == "unknown_field":
        data["labels"] = "must not be an input"
    elif failure == "wrong_index":
        p = tmp_path / "another-index.json"
        dump(p, {"records": [{"patch_id": "other"}]})
        data["index"] = {"path": str(p), "sha256": sha(p)}
    else:
        monkeypatch.setattr(
            "xuannv_embedding.export.public_calendar.shutil.disk_usage",
            lambda p: SimpleNamespace(free=0),
        )
    dump(spec, data)
    with pytest.raises((ValueError, RuntimeError)):
        prepare(spec)


@pytest.mark.parametrize("failure", ["out_of_order", "wrong_bands", "missing_index"])
def test_calendar_preparation_rejects_incompatible_monthly_metadata(tmp_path, failure):
    spec = fixture_calendar(tmp_path)
    data = json.loads(spec.read_text())
    path = data["monthly"]["path"]
    monthly = json.loads(Path(path).read_text())
    if failure == "out_of_order":
        monthly["months"] = monthly["months"][::-1]
    elif failure == "wrong_bands":
        monthly["bands"]["s2"] = monthly["bands"]["s2"][::-1]
    else:
        monthly.pop("index")
    dump(path, monthly)
    data["monthly"]["sha256"] = sha(path)
    dump(spec, data)
    with pytest.raises(ValueError):
        prepare(spec)
