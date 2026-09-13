from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from xuannv_embedding.data_process.v5_rasters import NativeRaster


def frame():
    texture = gaussian_filter(np.random.default_rng(42).normal(size=(256, 256)), 1).astype("f4")
    values = np.stack([texture, texture, texture])
    return NativeRaster(
        values,
        np.ones(values.shape, bool),
        ("B3", "B4", "B5"),
        (5, 0, 300000, 0, -5, 4000000),
        "EPSG:32650",
    )


def test_clear_alignment_retains_low_scene_fraction_and_ignores_cloud_values():
    from xuannv_embedding.data_process.v5_clear_intraband import apply_clear_mask
    from xuannv_embedding.data_process.v5_intraband import inspect_intraband

    source = frame()
    mask = np.zeros(source.values.shape, bool)
    for y in [0, 160]:
        for x in [0, 160]:
            mask[:, y : y + 96, x : x + 96] = True
    assert mask.mean() < 0.6
    masked = apply_clear_mask(source, mask, source.band_ids)
    expected = inspect_intraband(masked, reference_band="B4", gsd=5)
    assert expected["status"] == "passed"
    changed = source.values.copy()
    changed[~mask] = np.random.default_rng(3).normal(size=(~mask).sum()) * 1e8
    result = inspect_intraband(
        apply_clear_mask(replace(source, values=changed), mask, source.band_ids),
        reference_band="B4",
        gsd=5,
    )
    assert result == expected and not result["pixel_fusion_authorized"]
    assert source.valid.all()


def test_clear_alignment_rejects_invalid_contract_and_preserves_missing_channels():
    from xuannv_embedding.data_process.v5_clear_intraband import apply_clear_mask
    from xuannv_embedding.data_process.v5_intraband import inspect_intraband

    source = frame()
    with pytest.raises(ValueError, match="band identities"):
        apply_clear_mask(source, source.valid, source.band_ids[::-1])
    with pytest.raises(ValueError, match="boolean"):
        apply_clear_mask(source, source.valid.astype(float), source.band_ids)
    with pytest.raises(ValueError, match="shape"):
        apply_clear_mask(source, source.valid[0], source.band_ids)
    source.valid[-1] = False
    masked = apply_clear_mask(source, np.ones_like(source.valid), source.band_ids)
    assert not masked.valid[-1].any()
    assert inspect_intraband(masked, reference_band="B4", gsd=5)["status"] == "uncertain"
    cloudy = apply_clear_mask(source, np.zeros_like(source.valid), source.band_ids)
    result = inspect_intraband(cloudy, reference_band="B4", gsd=5)
    assert result["status"] == "uncertain"
    assert all(pair["valid_windows"] == 0 for pair in result["pairs"])


def test_clear_calibration_cli_requires_quality_snapshot_and_has_separate_writer_lock(
    tmp_path, monkeypatch
):
    import fcntl

    from xuannv_embedding.data_process import v5_clear_intraband, v5_cli

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(v5_clear_intraband, "calibrate_clear", lambda *a, **k: calls.append((a, k)))
    args = ["--stage", "clear-alignment-calibration", "--alignment-version", "v5"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + key, str(tmp_path / key)]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--sensor-family", "gaofen"]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--quality-root", str(tmp_path / "qa")]
    source = tmp_path / "source-root"
    source.mkdir(exist_ok=True)
    with (source / ".clear-alignment-calibration.gaofen.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another V5"):
            v5_cli.main(args)
    assert v5_cli.main(args) == 0
    assert calls[0][1] == {"version": "v5"}


def gaofen_qa(tmp_path, count=2):
    import pandas as pd
    import zarr
    from test_v5_parallel_intraband import _row

    from xuannv_embedding.data_process.v5_intraband import _read_row
    from xuannv_embedding.data_process.v5_quality import quality_masks
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    data, qa = tmp_path / "data", tmp_path / "qa"
    registry = data / "registry/national_62000.parquet"
    registry.parent.mkdir(parents=True)
    rows = pd.DataFrame([_row(tmp_path, f"obs{i}") for i in range(count)])
    rows[["patch_id", "split"]].to_parquet(registry, index=False)
    qa.mkdir()
    config = {
        "limit": None,
        "registry_sha256": sha256(registry),
        "input_bands": ["red", "green", "nir"],
        "buffer_m": 30,
        "code_sha256": {
            "v5_quality.py": sha256(
                Path(
                    __import__(
                        "xuannv_embedding.data_process.v5_quality", fromlist=["__file__"]
                    ).__file__
                )
            )
        },
    }
    write_json(qa / "source.lock.json", config)
    masks = zarr.open_group(str(qa / "valid_masks.zarr"), mode="w")
    classes = zarr.open_group(str(qa / "classes.zarr"), mode="w")
    masks.attrs.update({**config, "bitorder": "little", "packed_axis": -1})
    classes.attrs.update(config)
    masks.create_dataset("completed", data=np.ones(count, bool))
    labels = np.zeros((count, 160, 160), "u1")
    labels[:, :20] = 1
    classes.create_dataset("classes", data=labels)
    table = []
    packed = {name: [] for name in ["data_valid_packed", "ms_valid_packed", "before_buffer_packed"]}
    for i, row in enumerate(rows.itertuples()):
        native, _, _ = _read_row(row, "gaofen")
        valid = native.valid.all(axis=0)
        result = quality_masks(labels[i], valid, gsd=8)
        for name, value in [
            ("data_valid_packed", valid),
            ("ms_valid_packed", result["valid"]),
            ("before_buffer_packed", result["before_buffer"]),
        ]:
            packed[name].append(np.packbits(value, axis=-1, bitorder="little"))
        table.append(
            {
                "pair_id": row.observation_id,
                "patch_id": row.patch_id,
                "sensor": row.sensor,
                "split": row.split,
                "year": row.year,
                "ms_path": row.path,
                "ms_sha256": row.file_sha256,
                "ms_valid_pixels": int(result["valid"].sum()),
                "quality_status": "recomputed_needs_visual_review",
            }
        )
    for name, values in packed.items():
        masks.create_dataset(name, data=np.stack(values))
    table = pd.DataFrame(table)
    table.to_parquet(qa / "observation_quality.parquet", index=False)
    table[["pair_id"]].to_parquet(qa / "observation_order.parquet", index=False)
    return data, qa, rows


def test_gaofen_clear_reader_verifies_raw_pixels_masks_and_metadata(tmp_path):
    import zarr

    from xuannv_embedding.data_process.v5_clear_intraband import NativeQualityReader

    data, qa, rows = gaofen_qa(tmp_path)
    reader = NativeQualityReader(data, "gaofen", qa)
    row = next(rows.itertuples())
    raw, clear, _, _, provenance = reader.read(row)
    assert raw.valid.all() and not clear.valid[:, :24].any()
    assert clear.valid[:, 24:].all() and provenance["quality_mask_sha256"]
    masks = zarr.open_group(str(qa / "valid_masks.zarr"), mode="a")
    masks["ms_valid_packed"][0, 80, 0] = 0
    with pytest.raises(ValueError, match="mask differs"):
        reader.read(row)
    (qa / "observation_order.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="QA inputs changed"):
        reader.verify_unchanged()


def test_gaofen_clear_reader_rejects_permuted_order_even_when_counts_match(tmp_path):
    import pandas as pd

    from xuannv_embedding.data_process.v5_clear_intraband import NativeQualityReader

    data, qa, _ = gaofen_qa(tmp_path)
    order = pd.read_parquet(qa / "observation_order.parquet")
    order.iloc[::-1].to_parquet(qa / "observation_order.parquet", index=False)
    with pytest.raises(ValueError, match="order"):
        NativeQualityReader(data, "gaofen", qa)


def test_clear_calibration_rejects_insufficient_original_calibration(tmp_path):
    from xuannv_embedding.data_process.v5_clear_intraband import calibrate_clear
    from xuannv_embedding.data_process.v5_intraband import calibrate_family
    from xuannv_embedding.data_process.v5_sources import sha256

    data, qa, rows = gaofen_qa(tmp_path)
    report = tmp_path / "report"
    # The original calibration needs four successful positions per variant. Two is insufficient.
    target = data / "quality/cloud/gaofen"
    target.mkdir(parents=True)
    import shutil

    shutil.copy(qa / "observation_quality.parquet", target / "observation_quality.parquet")
    assert calibrate_family(data, report, "gaofen", version="v5")["status"] == "failed"
    before = sha256(target / "observation_quality.parquet")
    with pytest.raises(ValueError, match="passed original"):
        calibrate_clear(data, report, "gaofen", qa)
    assert sha256(target / "observation_quality.parquet") == before


def test_clear_calibration_replays_frozen_masks_and_detects_source_corruption(tmp_path):
    import shutil

    from xuannv_embedding.data_process.v5_clear_intraband import calibrate_clear
    from xuannv_embedding.data_process.v5_intraband import calibrate_family
    from xuannv_embedding.data_process.v5_sources import sha256

    data, qa, rows = gaofen_qa(tmp_path, count=4)
    target = data / "quality/cloud/gaofen"
    target.mkdir(parents=True)
    shutil.copy(qa / "observation_quality.parquet", target / "observation_quality.parquet")
    report = tmp_path / "report"
    assert calibrate_family(data, report, "gaofen", version="v5")["status"] == "passed"
    first = calibrate_clear(data, report, "gaofen", qa)
    assert first["status"] == "passed" and first["processed"] == 4
    assert not first["pixel_fusion_authorized"]
    locked = Path(first["output"])
    before, mtime = sha256(locked), locked.stat().st_mtime_ns
    second = calibrate_clear(data, report, "gaofen", qa)
    assert second["sha256"] == before and locked.stat().st_mtime_ns == mtime
    Path(rows.iloc[0].path).write_bytes(b"corrupt source")
    with pytest.raises(ValueError, match="source pixels changed"):
        calibrate_clear(data, report, "gaofen", qa)
    assert sha256(locked) == before


def test_jilin_clear_reader_verifies_frozen_quality_receipts_and_source_catalog(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import pandas as pd
    import zarr
    from test_v5_jilin_quality import Predictor, record

    from xuannv_embedding.data_process import v5_followup
    from xuannv_embedding.data_process.v5_clear_intraband import NativeQualityReader
    from xuannv_embedding.data_process.v5_jilin_quality import process_jilin_quality
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    data, source, report, models = [tmp_path / p for p in ["data", "source", "report", "models"]]
    registry = data / "registry/national_62000.parquet"
    registry.parent.mkdir(parents=True)
    pd.DataFrame([{"patch_id": "national", "split": "train"}]).to_parquet(registry, index=False)
    row = record(tmp_path, "jilin1_ms_5m")
    catalog = data / "observations/highres/jilin1/partial_bands/fixed"
    catalog.mkdir(parents=True)
    pd.DataFrame([row]).to_parquet(catalog / "files_with_partial_bands.parquet", index=False)
    write_json(
        catalog / "catalog.lock.json", {"fingerprint": {"registry_sha256": sha256(registry)}}
    )
    write_json(
        catalog.parent / "current.json",
        {
            "version": "fixed",
            "lock_path": str(catalog / "catalog.lock.json"),
            "lock_sha256": sha256(catalog / "catalog.lock.json"),
        },
    )
    monkeypatch.setattr(v5_followup, "partial_catalog_finished", lambda *a: True)
    models.mkdir()
    for i in (0, 1):
        (models / f"ocm_v4_model_{i}_96_910b4.om").write_bytes(b"fixture")

    class Factory(Predictor):
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    result = process_jilin_quality(source, data, report, models, predictor_factory=Factory)
    output = Path(result["output"])
    reader = NativeQualityReader(data, "jilin1", output)
    raw, masked, _, _, provenance = reader.read(SimpleNamespace(**row))
    assert raw.valid.all() and not masked.valid[:, :176].any()
    assert masked.valid[:, 176:].all() and provenance["clear_fraction_by_band"][0] < 0.6
    masks = zarr.open_group(result["quality_root"] + "/valid_masks.zarr", mode="a")
    key = row["observation_id"] + "/valid"
    masks[key][0, 0, 0] = True
    with pytest.raises(ValueError, match="QA mask changed"):
        reader.read(SimpleNamespace(**row))
    (catalog / "files_with_partial_bands.parquet").write_bytes(b"changed catalog")
    with pytest.raises(ValueError, match="catalog"):
        NativeQualityReader(data, "jilin1", output)
