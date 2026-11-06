from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def record(tmp_path, product, bands=None, *, scene="scene", fill=None):
    from xuannv_embedding.data_process.v5_partial_bands import inspect_partial_jilin
    from xuannv_embedding.data_process.v5_rasters import inspect_jilin

    gsd, count = {
        "jilin1_ms_5m": (5, 6),
        "jilin1_extra_10m": (10, 12),
        "jilin1_extra_20m": (20, 19),
        "jilin1_b0_5m": (5, 1),
    }[product]
    full = ["B0"] if product == "jilin1_b0_5m" else [f"B{i}" for i in range(1, count + 1)]
    bands = full if bands is None else bands
    path = tmp_path / scene / "JL1GP01" / (product + ".tif")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 1280 // gsd
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=n,
        height=n,
        count=len(bands),
        dtype="int16",
        nodata=-28672,
        crs="EPSG:32643",
        transform=from_origin(0, 1280, gsd, gsd),
    ) as ds:
        pixels = np.stack([np.full((n, n), int(b[1:]) * 100, dtype="i2") for b in bands])
        if fill is not None:
            fill(pixels, bands)
        ds.write(pixels)
        ds.descriptions = tuple(f"{b}(0.5)" for b in bands)
        ds.scales = (0.0001,) * len(bands)
        ds.update_tags(
            units="reflectance",
            acquisition_time="2020-03-31T12:00:00",
            source_product=scene,
            source_signature=scene,
            patch_id="patch",
        )
    r = inspect_jilin(path) if bands == full else inspect_partial_jilin(path)
    r.update(
        observation_id=scene + ":" + product,
        scene_group_id=scene,
        patch_id="national",
        split="train",
        sensor="JL1GP01",
    )
    return r


class Predictor:
    calls = 0

    def predict_batch(self, arrays):
        self.calls += 1
        np.testing.assert_allclose(arrays[0][:, 10, 10], [0.05, 0.04, 0.06])
        labels = np.zeros((1, 256, 256), "u1")
        labels[:, :170] = 1
        return labels, np.ones(labels.shape, "f4")


def test_partial_spectral_channels_keep_individual_validity_after_cloud(tmp_path):
    from xuannv_embedding.data_process.v5_jilin_quality import infer_scene

    ms = record(tmp_path, "jilin1_ms_5m", ["B2", "B3", "B4", "B5", "B6"])
    extra = record(tmp_path, "jilin1_extra_20m", [f"B{i}" for i in range(1, 20) if i != 14])
    predictor = Predictor()
    result = infer_scene([ms, extra], predictor)
    assert predictor.calls == 1
    a = result["branches"][ms["observation_id"]]
    assert not a["valid"][0].any() and a["valid"][1:].any()
    assert not a["row"]["strict_scene_qualified"] and a["row"]["available"]
    assert a["row"]["valid_pixels_by_band"][0] == 0
    b = result["branches"][extra["observation_id"]]
    assert b["valid"].shape == (7, 64, 64) and not b["valid"][1].any()
    assert b["valid"][0].any() and not b["valid"][:, :44].any()
    assert result["cloud"]["cloud_buffered"][175].all()
    assert not result["cloud"]["cloud_buffered"][176].any()


def test_unrelated_band_nodata_does_not_erase_cloud_input_evidence(tmp_path):
    from xuannv_embedding.data_process.v5_jilin_quality import infer_scene

    def fill(pixels, bands):
        pixels[bands.index("B1")] = -28672

    ms = record(tmp_path, "jilin1_ms_5m", fill=fill)
    result = infer_scene([ms], Predictor())
    assert result["cloud"]["data_valid"].all()
    branch = result["branches"][ms["observation_id"]]
    assert not branch["valid"][0].any() and branch["valid"][1:].any()


@pytest.mark.parametrize("missing_ms", [False, True])
def test_missing_cloud_inputs_are_explicit_and_never_become_clear(tmp_path, missing_ms):
    from xuannv_embedding.data_process.v5_jilin_quality import infer_scene

    rows = [record(tmp_path, "jilin1_b0_5m")]
    if not missing_ms:
        rows.append(record(tmp_path, "jilin1_ms_5m", ["B1", "B2", "B3", "B4", "B5"]))
    predictor = Predictor()
    result = infer_scene(rows, predictor)
    assert predictor.calls == 0 and result["quality_status"] == "qa_missing"
    assert result["reason"] == ("no_same_scene_5m_ms" if missing_ms else "missing_B5_B4_B6")
    for branch in result["branches"].values():
        assert branch["data_valid"].any() and not branch["valid"].any()
        assert not branch["row"]["available"]
    assert (result["cloud"]["classes"] == 255).all()


def test_cloud_rejects_changed_source_and_invalid_prediction(tmp_path):
    from xuannv_embedding.data_process.v5_jilin_quality import infer_scene

    ms = record(tmp_path, "jilin1_ms_5m")

    class Bad(Predictor):
        def predict_batch(self, arrays):
            return np.full((1, 256, 256), 4), np.ones((1, 256, 256))

    with pytest.raises(ValueError, match="prediction"):
        infer_scene([ms], Bad())
    Path(ms["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        infer_scene([ms], Predictor())


def test_quality_snapshots_reuse_verified_arrays_and_reject_corrupt_masks(tmp_path, monkeypatch):
    import json

    import pandas as pd
    import zarr

    import xuannv_embedding.data_process.v5_followup as followup
    from xuannv_embedding.data_process.v5_jilin_quality import process_jilin_quality
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    data, source, report, models = [tmp_path / k for k in ["data", "source", "report", "models"]]
    records = [
        record(tmp_path, "jilin1_ms_5m", ["B2", "B3", "B4", "B5", "B6"]),
        record(tmp_path, "jilin1_extra_20m", [f"B{i}" for i in range(1, 20) if i != 14]),
        record(tmp_path, "jilin1_b0_5m", scene="orphan"),
    ]
    root = data / "observations/highres/jilin1/partial_bands/fixture"
    root.mkdir(parents=True)
    pd.DataFrame(records).to_parquet(root / "files_with_partial_bands.parquet", index=False)
    write_json(root / "catalog.lock.json", {"fingerprint": {"source_packages": {"archive": {}}}})
    write_json(
        root.parent / "current.json",
        {
            "version": "fixture",
            "lock_path": str(root / "catalog.lock.json"),
            "lock_sha256": sha256(root / "catalog.lock.json"),
        },
    )
    monkeypatch.setattr(followup, "partial_catalog_finished", lambda *args: True)
    models.mkdir()
    for i in (0, 1):
        (models / f"ocm_v4_model_{i}_96_910b4.om").write_bytes(b"fake")
    calls = []

    class Factory(Predictor):
        def __init__(self, **kwargs):
            calls.append("init")

        def predict_batch(self, arrays):
            calls.append("infer")
            return super().predict_batch(arrays)

        def close(self):
            calls.append("close")

    result = process_jilin_quality(source, data, report, models, predictor_factory=Factory)
    assert result["processed_scenes"] == 2 and result["branches"] == 3
    assert result["counts"]["qa_missing"] == 1
    assert calls.count("infer") == 1
    output = Path(result["output"])
    lock = output / "quality.lock.json"
    locked_bytes = lock.read_bytes()
    table = pd.read_parquet(output / "observation_quality.parquet")
    assert table.loc[table.scene_group_id == "orphan", "qa_missing_reason"].tolist() == [
        "no_same_scene_5m_ms"
    ]
    calls.clear()
    replay = process_jilin_quality(source, data, report, models, predictor_factory=Factory)
    assert replay["reused_scenes"] == 2 and not calls
    assert lock.read_bytes() == locked_bytes
    pilot = process_jilin_quality(source, data, report, models, predictor_factory=Factory, limit=1)
    pilot_rows = pd.read_parquet(Path(pilot["output"]) / "observation_quality.parquet")
    assert set(pilot_rows.scene_group_id) == {"scene"}
    masks = zarr.open_group(result["quality_root"] + "/valid_masks.zarr", mode="a")
    path = records[0]["observation_id"] + "/valid"
    masks[path][0, 0, 0] = True
    with pytest.raises(ValueError, match="cached per-band quality mask changed"):
        process_jilin_quality(source, data, report, models, predictor_factory=Factory)
    assert json.loads(lock.read_text())["training_authorized"] is False


def test_native_jilin_quality_is_reachable_through_the_single_cli(tmp_path, monkeypatch):
    import xuannv_embedding.data_process.v5_cli as cli
    import xuannv_embedding.data_process.v5_jilin_quality as module
    from xuannv_embedding.data_process.v5_cli import main

    monkeypatch.setattr(cli, "lock_source", lambda *args: {"manifest_sha256": {}})
    monkeypatch.setattr(cli, "input_lock", lambda *args: None)
    called = []
    monkeypatch.setattr(
        module, "process_jilin_quality", lambda *args, **kwargs: called.append((args, kwargs)) or {}
    )
    arguments = ["--stage", "jilin-quality", "--device-id", "7", "--max-scenes", "32"]
    for name in ["source-root", "dataset-root", "report-root", "base-root", "model-dir"]:
        arguments += ["--" + name, str(tmp_path / name)]
    assert main(arguments) == 0
    assert called[0][1] == {"device_id": 7, "limit": 32}
