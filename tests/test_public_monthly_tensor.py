import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xuannv_embedding.data_process.public_monthly_tensor import prepare
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS


def fixture_monthly(tmp_path, normalized=False, missing=None):
    records = []
    canonical = {}
    stats = {s: {"mean": [100] * len(b), "std": [2] * len(b)} for s, b in PUBLIC_BANDS.items()}
    bands = {s: list(reversed(b)) for s, b in PUBLIC_BANDS.items()}
    for i in range(2):
        frames, masks = {}, {}
        for s, names in PUBLIC_BANDS.items():
            value = np.arange(12 * len(names) * 4, dtype=np.float32).reshape(12, len(names), 2, 2)
            value = value / 100 + i
            valid = np.ones((12, 2, 2), bool)
            valid[-1] = False
            if s == "landsat" and i == 1:
                valid[:] = False
            if s == missing:
                valid[:] = False
            value.transpose(0, 2, 3, 1)[~valid] = np.nan
            canonical[(i, s)] = (value.copy(), valid.copy())
            ordered = value[:, [names.index(b) for b in bands[s]]]
            frames[s] = (ordered - 100) / 2 if normalized else ordered
            if normalized and s in ["s2", "landsat"]:
                frames[s] = np.concatenate([frames[s], np.full((12, 1, 2, 2), 999)], 1)
            masks[s + "_recon"] = torch.from_numpy(valid.astype(np.float32))
        path = tmp_path / f"input_{i}.{'pt' if normalized else 'npz'}"
        if normalized:
            torch.save(
                {
                    "source_frames": {s: torch.from_numpy(v) for s, v in frames.items()},
                    "target_masks": masks,
                    "timestamps": torch.arange(202401, 202413),
                    "targets": "must never be read",
                    "highres_frames": "must never be read",
                },
                path,
            )
        else:
            np.savez_compressed(
                path,
                **frames,
                **{s + "_valid": masks[s + "_recon"].numpy() > 0 for s in frames},
                labels=np.array([{"must_not_read": True}], dtype=object),
            )
        records.append(
            {
                "patch_id": str(i),
                "bounds": [i * 20, 0, i * 20 + 20, 20],
                "crs": "EPSG:32650",
                "input": {"path": str(path), "sha256": sha(path)},
            }
        )
    inputs, statpath, index = (tmp_path / n for n in ["inputs.json", "stats.json", "index.json"])
    dump(inputs, {"records": records})
    dump(statpath, stats)
    dump(index, {"records": [{k: r[k] for k in ["patch_id", "bounds", "crs"]} for r in records]})

    def ref(p):
        return {"path": str(p), "sha256": sha(p)}

    observations = tmp_path / "observations.json"
    dump(
        observations,
        {
            "schema": "named-public-statistics-v2",
            "kind": "normalized_tensor" if normalized else "physical_npz",
            "bands": bands,
            "inputs": ref(inputs),
            "statistics": ref(statpath),
            "radiometry": {
                s: {b: {"scale": 1, "offset": 0} for b in bs} for s, bs in bands.items()
            },
            "output": str(tmp_path / "unused"),
        },
    )
    period = tmp_path / "period.json"
    dump(
        period,
        {
            "input_months": [f"2024-{m:02}" for m in range(1, 13)],
            "observations": ref(observations),
            "index": ref(index),
        },
    )
    spec = tmp_path / "spec.json"
    dump(
        spec,
        {
            "schema": "public-monthly-tensor-v1",
            "observations": ref(observations),
            "period": ref(period),
            "index": ref(index),
            "reserve_gib": 0,
            "output": str(tmp_path / "monthly"),
        },
    )
    return spec, canonical


@pytest.mark.parametrize("normalized", [False, True])
def test_monthly_cache_keeps_calendar_band_order_and_ignores_nonpublic_values(tmp_path, normalized):
    spec, canonical = fixture_monthly(tmp_path, normalized)
    output = prepare(spec)
    metadata = json.loads(output.read_text())
    frames = np.load(metadata["frames"]["path"])
    masks = np.load(metadata["masks"]["path"])
    normalization = json.loads(Path(metadata["normalization"]["path"]).read_text())
    assert frames.shape == (2, 12, 19, 2, 2)
    assert masks.shape == (2, 12, 3, 2, 2)
    offset = 0
    for j, (s, bands) in enumerate(PUBLIC_BANDS.items()):
        mean = np.asarray(normalization[s]["mean"])[None, :, None, None]
        scale = np.asarray(normalization[s]["std"])[None, :, None, None]
        for i in range(2):
            value, valid = canonical[(i, s)]
            actual = frames[i, :, offset : offset + len(bands)]
            np.testing.assert_array_equal(masks[i, :, j], valid)
            np.testing.assert_array_equal(actual.transpose(0, 2, 3, 1)[~valid], 0)
            decoded = actual * scale + mean
            np.testing.assert_allclose(
                decoded.transpose(0, 2, 3, 1)[valid], value.transpose(0, 2, 3, 1)[valid], atol=2e-5
            )
        offset += len(bands)
    assert metadata["months"] == list(range(202401, 202413))
    assert metadata["map_or_local_image_values_used"] is False
    assert prepare(spec) == output


def test_monthly_cache_rejects_changed_completed_output(tmp_path):
    spec, _ = fixture_monthly(tmp_path)
    output = prepare(spec)
    metadata = json.loads(output.read_text())
    value = np.load(metadata["frames"]["path"])
    value[0, 0, 0, 0, 0] += 1
    np.save(metadata["frames"]["path"], value)
    with pytest.raises(ValueError, match="changed"):
        prepare(spec)


def test_monthly_cache_rejects_nonconsecutive_period(tmp_path):
    spec, _ = fixture_monthly(tmp_path, True)
    content = json.loads(spec.read_text())
    period = Path(content["period"]["path"])
    data = json.loads(period.read_text())
    data["input_months"][-1] = "2025-01"
    dump(period, data)
    content["period"]["sha256"] = sha(period)
    dump(spec, content)
    with pytest.raises(ValueError, match="consecutive"):
        prepare(spec)


def test_monthly_cache_rejects_intrinsic_tensor_timestamp_mismatch(tmp_path):
    spec, _ = fixture_monthly(tmp_path, True)
    content = json.loads(spec.read_text())
    period = Path(content["period"]["path"])
    data = json.loads(period.read_text())
    data["input_months"] = [f"2025-{m:02}" for m in range(1, 13)]
    dump(period, data)
    content["period"]["sha256"] = sha(period)
    dump(spec, content)
    with pytest.raises(ValueError, match="timestamp"):
        prepare(spec)


def test_monthly_cache_handles_an_entirely_missing_public_modality(tmp_path):
    spec, _ = fixture_monthly(tmp_path, missing="s1")
    metadata = json.loads(prepare(spec).read_text())
    a = np.load(metadata["frames"]["path"])
    mask = np.load(metadata["masks"]["path"])
    normal = json.loads(Path(metadata["normalization"]["path"]).read_text())
    np.testing.assert_array_equal(a[:, :, 11:13], 0)
    assert not mask[:, :, 1].any()
    assert normal["s1"] == {"mean": [0, 0], "std": [1, 1], "observations": 0}


def test_monthly_cache_rejects_a_crossed_geographic_index(tmp_path):
    spec, _ = fixture_monthly(tmp_path)
    content = json.loads(spec.read_text())
    index = Path(content["index"]["path"])
    value = json.loads(index.read_text())
    value["records"].reverse()
    dump(index, value)
    content["index"]["sha256"] = sha(index)
    period = Path(content["period"]["path"])
    value = json.loads(period.read_text())
    value["index"] = content["index"]
    dump(period, value)
    content["period"]["sha256"] = sha(period)
    dump(spec, content)
    with pytest.raises(ValueError, match="geographic"):
        prepare(spec)


def test_monthly_cache_binds_the_imported_radiometry_implementation(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import public_monthly_tensor as module

    spec, _ = fixture_monthly(tmp_path)
    prepare(spec)
    original = module.sha

    def changed(path):
        return "0" * 64 if Path(path) == Path(module.public_statistics.__file__) else original(path)

    monkeypatch.setattr(module, "sha", changed)
    with pytest.raises(ValueError, match="identity changed"):
        prepare(spec)
