import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.prithvi_features import (
    normalize_reflectance,
    pool_temporal,
    read_public_s2,
    validate_ranges,
)
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS

BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
MEAN = [1087, 1342, 1433, 2734, 1958, 1363]
STD = [2248, 2179, 2178, 1850, 1242, 1049]


def test_normalizer_uses_physical_wavelengths_and_fills_missing_with_model_mean():
    bands = list(reversed(PUBLIC_BANDS["s2"]))
    x = np.empty((12, 11, 2, 2), np.float32)
    for j, name in enumerate(bands):
        x[:, j] = 0.1 + PUBLIC_BANDS["s2"].index(name) / 100
    valid = np.ones((12, 2, 2), bool)
    valid[-1] = False
    x[-1] = np.nan
    values, mask = normalize_reflectance(x, valid, bands, MEAN, STD)
    for j, name in enumerate(BANDS):
        physical = 0.1 + PUBLIC_BANDS["s2"].index(name) / 100
        assert values[j, 0, 0, 0] == pytest.approx((physical * 10000 - MEAN[j]) / STD[j])
    assert torch.count_nonzero(values[:, -1]) == 0
    assert mask.dtype == torch.bool
    assert values.shape == (6, 12, 2, 2)


def test_record_reader_applies_offset_once_and_uses_canonical_return_order(tmp_path):
    declared = {s: list(reversed(names)) for s, names in PUBLIC_BANDS.items()}
    arrays = {}
    for s, names in declared.items():
        a = np.zeros((12, len(names), 2, 2), np.float32)
        for j, name in enumerate(names):
            a[:, j] = 2000 + 10 * PUBLIC_BANDS[s].index(name)
        arrays[s] = a
        arrays[s + "_valid"] = np.ones((12, 2, 2), bool)
    path = tmp_path / "input.npz"
    np.savez(path, **arrays, labels=np.array([{"must_not_read": True}], dtype=object))
    obs = {
        "kind": "physical_npz",
        "bands": declared,
        "radiometry": {
            s: {name: {"scale": 0.0001, "offset": -0.1} for name in names}
            for s, names in declared.items()
        },
    }
    record = {"input": {"path": str(path), "sha256": sha(path)}}
    values, _ = read_public_s2(record, obs, {}, list(range(202401, 202413)), MEAN, STD)
    for j, name in enumerate(BANDS):
        physical = (2000 + 10 * PUBLIC_BANDS["s2"].index(name)) * 0.0001 - 0.1
        assert values[j, 0, 0, 0] == pytest.approx((physical * 10000 - MEAN[j]) / STD[j], abs=2e-6)
    # A second offset would make the blue reflectance approximately -0.1.
    assert values[0, 0, 0, 0] == pytest.approx((1000 - MEAN[0]) / STD[0], abs=2e-6)


@pytest.mark.parametrize("failure", ["missing_band", "duplicate_band", "bad_mask", "valid_nan"])
def test_normalizer_rejects_ambiguous_or_invalid_inputs(failure):
    bands = list(BANDS)
    values = np.zeros((12, 6, 2, 2), np.float32)
    valid = np.ones((12, 2, 2), bool)
    if failure == "missing_band":
        bands[-1] = "B01"
    elif failure == "duplicate_band":
        bands[-1] = bands[0]
    elif failure == "bad_mask":
        valid = valid.astype(np.float32)
    else:
        values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        normalize_reflectance(values, valid, bands, MEAN, STD)


def test_temporal_pool_uses_observed_area_and_zeroes_all_missing_locations():
    values = torch.full((1, 2, 12, 2, 2), 1e6)
    values[:, :, 0] = 2
    values[:, :, 1] = 6
    valid = torch.zeros(1, 12, 32, 32, dtype=torch.bool)
    valid[:, 0] = True
    valid[:, 1, :, ::2] = True
    valid[:, :, 16:, 16:] = False
    pooled, count = pool_temporal(values, valid)
    assert pooled[0, 0, 0, 0] == pytest.approx(10 / 3)
    assert count[0, 0, 0, 0] == 1.5
    assert torch.count_nonzero(pooled[:, :, 1, 1]) == 0
    assert count[0, 0, 1, 1] == 0


def test_temporal_pool_rejects_silent_border_crop_or_grid_mismatch():
    with pytest.raises(ValueError):
        pool_temporal(torch.ones(1, 2, 12, 2, 2), torch.ones(1, 12, 31, 32, dtype=torch.bool))
    with pytest.raises(ValueError):
        pool_temporal(torch.ones(1, 2, 11, 2, 2), torch.ones(1, 12, 32, 32, dtype=torch.bool))


def test_shard_ranges_cover_every_record_once_before_a_regional_fit():
    assert validate_ranges([(2, 4), (0, 2)], 4) == [(0, 2), (2, 4)]
    for ranges in [[(0, 2), (3, 4)], [(0, 3), (2, 4)], [(0, 2)], [(0, 2), (0, 2)]]:
        with pytest.raises(ValueError, match="range|cover|overlap"):
            validate_ranges(ranges, 4)


def fixture_export(tmp_path, *, count=2, missing=None):
    def ref(p):
        return {"path": str(p), "sha256": sha(p)}

    declared = {s: list(reversed(bs)) for s, bs in PUBLIC_BANDS.items()}
    records = []
    for i in range(count):
        data = {}
        for s, names in declared.items():
            data[s] = np.full((12, len(names), 128, 128), 2000 + i * 50, np.float32)
            valid = np.ones((12, 128, 128), bool)
            if s == "s2" and i == missing:
                valid[:] = False
                data[s][:] = np.nan
            data[s + "_valid"] = valid
        raw = tmp_path / f"source_{i}.npz"
        np.savez_compressed(raw, **data, labels=np.array([{"must_not_read": True}], dtype=object))
        records.append(
            {
                "patch_id": str(i),
                "bounds": [i * 1280, 0, (i + 1) * 1280, 1280],
                "crs": "EPSG:32650",
                "input": ref(raw),
            }
        )
    inp, index, stats = [tmp_path / n for n in ["records.json", "index.json", "stats.json"]]
    dump(inp, {"records": records})
    dump(index, {"records": [{k: r[k] for k in ["patch_id", "bounds", "crs"]} for r in records]})
    dump(stats, {})
    obs = tmp_path / "observations.json"
    dump(
        obs,
        {
            "schema": "named-public-statistics-v2",
            "kind": "physical_npz",
            "inputs": ref(inp),
            "statistics": ref(stats),
            "bands": declared,
            "radiometry": {
                s: {b: {"scale": 0.0001, "offset": -0.1} for b in bs} for s, bs in declared.items()
            },
        },
    )
    period = tmp_path / "period.json"
    dump(
        period,
        {
            "input_months": [f"2024-{i:02}" for i in range(1, 13)],
            "observations": ref(obs),
            "index": ref(index),
        },
    )
    monthly = tmp_path / "monthly.json"
    dump(
        monthly,
        {
            "schema": "public-monthly-tensor-v1",
            "observations": ref(obs),
            "period": ref(period),
            "index": ref(index),
            "reserve_gib": 0,
            "output": str(tmp_path / "unused"),
        },
    )
    weights = tmp_path / "weights.pt"
    torch.save({"fixture": torch.zeros(1)}, weights)
    cfg = tmp_path / "model.json"
    dump(
        cfg,
        {
            "architecture": "prithvi_eo_v2_300",
            "num_features": 64,
            "pretrained_cfg": {
                "img_size": 32,
                "num_frames": 4,
                "patch_size": [1, 16, 16],
                "in_chans": 6,
                "embed_dim": 64,
                "depth": 1,
                "num_heads": 4,
                "decoder_embed_dim": 32,
                "decoder_depth": 1,
                "decoder_num_heads": 4,
                "mlp_ratio": 4,
                "coords_encoding": [],
                "coords_scale_learn": False,
                "mask_ratio": 0.75,
                "norm_pix_loss": False,
                "bands": ["B02", "B03", "B04", "B05", "B06", "B07"],
                "mean": MEAN,
                "std": STD,
                "origin_url": "fixture",
                "paper_ids": "fixture",
            },
        },
    )
    spec = tmp_path / "export.json"
    dump(
        spec,
        {
            "schema": "prithvi-public-features-v1",
            "monthly_spec": ref(monthly),
            "weights": ref(weights),
            "model_config": ref(cfg),
            "output": str(tmp_path / "out"),
            "seed": 41,
            "fit_samples": 64,
            "batch_size": 1,
            "reserve_gib": 0,
            "inference_dtype": "float32",
            "output_shape": [32, 32],
            "fit_min_frames": 0.9,
            "output_min_coverage": 0.05,
        },
    )
    return spec


class FakeEncoder(torch.nn.Module):
    def __init__(self, calls, fail_at=None):
        super().__init__()
        self.calls, self.fail_at = calls, fail_at
        self.local_calls = 0

    def forward(self, x):
        self.calls.extend(x[:, 0, 0, 0, 0].tolist())
        self.local_calls += 1
        if self.local_calls == self.fail_at:
            raise RuntimeError("simulated interruption")
        h, w = x.shape[-2] // 16, x.shape[-1] // 16
        grid = torch.arange(h * w).reshape(1, 1, 1, h, w).float()
        channel = torch.arange(1, 65).reshape(1, 64, 1, 1, 1).float()
        base = x[:, 0, 0, 0, 0].reshape(-1, 1, 1, 1, 1)
        return torch.sin((grid + 1) * channel / 10 + base).expand(-1, -1, 12, -1, -1)


def fake_loader(monkeypatch, calls, fail_at=None):
    from xuannv_embedding.export import prithvi_features as pf

    monkeypatch.setattr(
        pf,
        "load_model",
        lambda *a: (FakeEncoder(calls, fail_at), {"encoder_keys": 0, "ignored_decoder_keys": 0}),
    )


def test_projection_waits_for_all_shards_and_preserves_missing_tiles(tmp_path, monkeypatch):
    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path, missing=1)
    calls = []
    fake_loader(monkeypatch, calls)
    pf.extract(spec, device_name="cpu", start=0, stop=1)
    with pytest.raises(ValueError, match="cover"):
        pf.project(spec)
    pf.extract(spec, device_name="cpu", start=1, stop=2)
    output = pf.project(spec)
    value = np.load(output)
    assert value.shape == (2, 64, 32, 32)
    assert np.isfinite(value).all()
    assert np.count_nonzero(value[1]) == 0
    assert not np.allclose(np.linalg.norm(value[0], axis=0), 1)
    state = json.loads((output.parent / "status.json").read_text())
    assert state["fit_samples"] == 64
    assert state["radiometry_application"] == "monthly reader exactly once"
    assert state["per_region_projection"] is True
    assert state["map_labels_used"] is False
    import shutil

    shutil.rmtree(output.parent / "shards")
    assert pf.project(spec) == output
    value[0, 0, 0, 0] += 1
    np.save(output, value)
    with pytest.raises(ValueError, match="changed"):
        pf.project(spec)


def test_extraction_resumes_committed_prefix_without_repeating_it(tmp_path, monkeypatch):
    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path, count=3)
    calls = []
    fake_loader(monkeypatch, calls, fail_at=2)
    with pytest.raises(RuntimeError, match="interruption"):
        pf.extract(spec, device_name="cpu", start=0, stop=3)
    root = tmp_path / "out/shards/000000-000003"
    before = np.load(root / "native.partial.npy")[0].copy()
    fake_loader(monkeypatch, calls)
    pf.extract(spec, device_name="cpu", start=0, stop=3)
    np.testing.assert_array_equal(np.load(root / "native.npy")[0], before)
    assert len(calls) == 4
    assert calls.count(calls[0]) == 1
    assert len(set(calls)) == 3
    assert json.loads((root / "status.json").read_text())["completed"] == 3
    prior = len(calls)
    pf.extract(spec, device_name="cpu", start=0, stop=3)
    assert len(calls) == prior


def test_extraction_rejects_corrupted_committed_prefix(tmp_path, monkeypatch):
    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path)
    calls = []
    fake_loader(monkeypatch, calls, fail_at=2)
    with pytest.raises(RuntimeError):
        pf.extract(spec, device_name="cpu", start=0, stop=2)
    path = tmp_path / "out/shards/000000-000002/native.partial.npy"
    value = np.load(path, mmap_mode="r+")
    value[0, 0, 0, 0] += 1
    value.flush()
    fake_loader(monkeypatch, calls)
    with pytest.raises(ValueError, match="Corrupt committed"):
        pf.extract(spec, device_name="cpu", start=0, stop=2)


@pytest.mark.parametrize("failure", ["unknown", "boolean_batch", "duplicate", "changed_weight"])
def test_feature_spec_rejects_ambiguous_or_changed_inputs(tmp_path, failure):
    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path)
    value = json.loads(spec.read_text())
    if failure == "unknown":
        value["unexpected"] = True
    elif failure == "boolean_batch":
        value["batch_size"] = True
    elif failure == "duplicate":
        spec.write_text(spec.read_text().rstrip()[:-1] + ',"seed":42}')
    else:
        Path(value["weights"]["path"]).write_bytes(b"changed")
    if failure != "duplicate":
        dump(spec, value)
    with pytest.raises(ValueError):
        pf.read_spec(spec)


def test_unique_cli_exposes_explicit_prithvi_phases(capsys):
    from xuannv_embedding.training.experiment import main

    with pytest.raises(SystemExit) as done:
        main(["prithvi-features", "--help"])
    assert done.value.code == 0
    assert "extract,project" in capsys.readouterr().out


def test_shard_finalization_recovers_after_only_first_array_was_renamed(tmp_path, monkeypatch):
    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path)
    calls = []
    fake_loader(monkeypatch, calls)
    original = Path.replace

    def interrupt_counts(path, destination):
        if path.name == "counts.partial.npy":
            raise RuntimeError("interrupted finalization")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", interrupt_counts)
    with pytest.raises(RuntimeError, match="finalization"):
        pf.extract(spec, device_name="cpu", start=0, stop=2)
    monkeypatch.setattr(Path, "replace", original)
    monkeypatch.setattr(pf, "load_model", lambda *a: pytest.fail("completed prefix inferred again"))
    shard = pf.extract(spec, device_name="cpu", start=0, stop=2)
    assert json.loads((shard / "status.json").read_text())["stage"] == "complete"
    assert (shard / "native.npy").exists() and (shard / "counts.npy").exists()


def test_regional_projection_fits_once_from_all_shards(tmp_path, monkeypatch):
    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path, count=3)
    fake_loader(monkeypatch, [])
    shards = [pf.extract(spec, device_name="cpu", start=i, stop=i + 1) for i in range(3)]
    original = pf.PCA
    fits = []

    class RegionalPCA(original):
        def fit(self, x, y=None):
            fits.append(x.copy())
            return super().fit(x, y)

    monkeypatch.setattr(pf, "PCA", RegionalPCA)
    monkeypatch.setattr(pf, "_setup_device", lambda *a: pytest.fail("projection initialized NPU"))
    pf.project(spec)
    assert len(fits) == 1 and len(fits[0]) == 64
    sampled = {row.tobytes() for row in fits[0]}
    for shard in shards:
        values = np.load(shard / "native.npy").astype(np.float32)[0].reshape(64, -1).T
        assert sampled.intersection(row.tobytes() for row in values)


def test_completed_shard_and_source_identity_cannot_be_silently_overwritten(tmp_path, monkeypatch):
    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path)
    fake_loader(monkeypatch, [])
    shard = pf.extract(spec, device_name="cpu", start=0, stop=1)
    data = np.load(shard / "native.npy", mmap_mode="r+")
    data[0, 0, 0, 0] += 1
    data.flush()
    with pytest.raises(ValueError, match="changed"):
        pf.extract(spec, device_name="cpu", start=0, stop=1)
    content = json.loads(spec.read_text())
    content["seed"] += 1
    dump(spec, content)
    with pytest.raises(ValueError, match="identity"):
        pf.extract(spec, device_name="cpu", start=0, stop=1)


def test_projection_refuses_overlapping_complete_shards(tmp_path, monkeypatch):
    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path)
    fake_loader(monkeypatch, [])
    pf.extract(spec, device_name="cpu", start=0, stop=2)
    pf.extract(spec, device_name="cpu", start=0, stop=1)
    with pytest.raises(ValueError, match="overlap"):
        pf.project(spec)


def test_extraction_checks_reserve_before_model_or_array_allocation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path)
    monkeypatch.setattr(pf.shutil, "disk_usage", lambda *a: SimpleNamespace(free=0))
    monkeypatch.setattr(pf, "load_model", lambda *a: pytest.fail("model allocated below reserve"))
    with pytest.raises(RuntimeError, match="reserve"):
        pf.extract(spec, device_name="cpu", start=0, stop=1)
    assert not list((tmp_path / "out").glob("**/*.npy"))


def test_resume_rejects_a_change_of_numerical_backend(tmp_path, monkeypatch):
    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path)
    fake_loader(monkeypatch, [], fail_at=2)
    with pytest.raises(RuntimeError):
        pf.extract(spec, device_name="cpu", start=0, stop=2)
    monkeypatch.setattr(pf, "_setup_device", lambda *a: (torch.device("cuda:0"), None, None))
    monkeypatch.setattr(pf, "load_model", lambda *a: pytest.fail("new backend was initialized"))
    with pytest.raises(ValueError, match="numerical mode"):
        pf.extract(spec, device_name="cuda:0", start=0, stop=2)


def test_projection_rejects_mixed_shard_precision(tmp_path, monkeypatch):
    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path)
    fake_loader(monkeypatch, [])
    pf.extract(spec, device_name="cpu", start=0, stop=1)
    second = pf.extract(spec, device_name="cpu", start=1, stop=2)
    state = json.loads((second / "status.json").read_text())
    state["actual_inference_dtype"] = "bfloat16"
    dump(second / "status.json", state)
    with pytest.raises(ValueError, match="numerical modes"):
        pf.project(spec)


def test_unowned_output_directory_is_rejected_without_writing_a_lock(tmp_path):
    from xuannv_embedding.export import prithvi_features as pf

    spec = fixture_export(tmp_path)
    root = tmp_path / "out"
    root.mkdir()
    (root / "existing.txt").write_text("keep")
    with pytest.raises(ValueError, match="Unowned"):
        pf.extract(spec, device_name="cpu", start=0, stop=1)
    assert [p.name for p in root.iterdir()] == ["existing.txt"]
