import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_statistics_inventory_excludes_other_splits_and_deduplicates_content():
    from xuannv_embedding.data_process.v5_highres_statistics import training_rows

    rows = pd.DataFrame(
        [
            dict(
                observation_id="a",
                product_id="ms",
                sensor="sensor",
                file_sha256="x",
                split="train",
                year=2020,
            ),
            dict(
                observation_id="b",
                product_id="ms",
                sensor="sensor",
                file_sha256="x",
                split="train",
                year=2020,
            ),
            dict(
                observation_id="c",
                product_id="ms",
                sensor="sensor",
                file_sha256="z",
                split="test",
                year=2021,
            ),
        ]
    )
    selected, excluded = training_rows(rows)
    assert selected.observation_id.tolist() == ["a"]
    assert set(excluded.reason) == {"non_training_split", "duplicate_content"}
    with pytest.raises(ValueError, match="duplicate observation"):
        training_rows(pd.concat([rows, rows.iloc[:1]]))
    changed = rows.copy()
    changed.loc[1, "split"] = "val"
    with pytest.raises(ValueError, match="cross-split"):
        training_rows(changed)


def test_highres_shared_reader_keeps_all_jilin_resolutions_and_missing_masks(tmp_path, monkeypatch):
    from test_v5_clear_audit import partial_quality

    from xuannv_embedding.data_process.v5_highres_reader import HighresQualityReader

    data, qa, _ = partial_quality(tmp_path, monkeypatch)
    reader = HighresQualityReader(data, "jilin1", qa)
    assert len(reader.inventory) == 2
    for row in reader.inventory.itertuples():
        frame, proof = reader.read(row)
        assert frame.values.shape == (6, 256, 256)
        assert not frame.values[~frame.valid].any()
        assert proof["quality_mask_sha256"]
        if row.observation_id.startswith("no_ref"):
            assert not frame.valid.any()


def test_statistics_stream_matches_direct_and_replays_without_recalculating(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_highres_statistics as module
    from xuannv_embedding.data_process.v5_rasters import NativeRaster
    from xuannv_embedding.data_process.v5_sources import sha256

    values = np.arange(24, dtype="f4").reshape(2, 3, 4) - 8
    mask = np.ones(values.shape, bool)
    mask[:, 0, 0] = False
    source = tmp_path / "source"
    source.write_bytes(b"fixed")
    source_hash = sha256(source)

    class Reader:
        def __init__(self, *args):
            self.files = {str(source): source_hash}
            self.configuration = {"unit": "reflectance"}
            self.inventory = pd.DataFrame(
                [
                    dict(
                        observation_id=f"o{i}",
                        product_id="p",
                        sensor="s",
                        file_sha256=f"h{i}",
                        split=split,
                        year=2020 + i,
                        patch_id=f"patch{i}",
                        path=str(source),
                    )
                    for i, split in enumerate(["train", "test"])
                ]
            )
            self.registry = pd.DataFrame(
                [
                    dict(patch_id=f"patch{i}", longitude=100 + i, latitude=30, split=split)
                    for i, split in enumerate(["train", "test"])
                ]
            ).set_index("patch_id")

        def read(self, row):
            assert row.split == "train"
            if sha256(source) != source_hash:
                raise ValueError("source changed")
            return NativeRaster(values, mask, ("a", "b"), (1, 0, 0, 0, -1, 3), "EPSG:32650"), {
                "quality_mask_sha256": "mask",
                "file_sha256": source_hash,
            }

        def verify_unchanged(self):
            if sha256(source) != source_hash:
                raise ValueError("source changed")

    monkeypatch.setattr(module, "HighresQualityReader", Reader)
    result = module.run_highres_statistics(
        tmp_path / "data", tmp_path / "report", "jilin1", tmp_path / "qa", reservoir_size=64
    )
    assert result["processed"] == 1 and result["excluded_observations"] == 1
    output = Path(result["output"])
    table = json.loads((output / "statistics.json").read_text())
    stats = table["products"]["p/s"]["statistics"]
    for i in range(2):
        assert stats["count"][i] == int(mask[i].sum())
        assert stats["mean"][i] == pytest.approx(values[i][mask[i]].mean())
        assert stats["std"][i] == pytest.approx(values[i][mask[i]].std())
    assert table["products"]["p/s"]["status"] == "passed"
    contributions = pd.read_parquet(output / "contributions.parquet")
    assert len(contributions) == 2
    assert contributions.valid_pixel_fraction_of_product_band.eq(1).all()
    assert json.loads((output / "products/p__s.json").read_text())["status"] == "passed"
    before = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in output.iterdir() if p.is_file()}
    monkeypatch.setattr(
        module.StreamingBandStatistics,
        "update",
        lambda *a: pytest.fail("recalculated validated cache"),
    )
    replay = module.run_highres_statistics(
        tmp_path / "data", tmp_path / "report", "jilin1", tmp_path / "qa", reservoir_size=64
    )
    assert replay["reused"]
    assert before == {
        p.name: (sha256(p), p.stat().st_mtime_ns) for p in output.iterdir() if p.is_file()
    }
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        module.run_highres_statistics(
            tmp_path / "data", tmp_path / "report", "jilin1", tmp_path / "qa", reservoir_size=64
        )


def test_statistics_empty_channel_is_failure_not_fake_normalization():
    from xuannv_embedding.data_process.v5_highres_statistics import finish_product
    from xuannv_embedding.data_process.v5_statistics import StreamingBandStatistics

    stats = StreamingBandStatistics(2)
    x = np.arange(12).reshape(2, 2, 3)
    valid = np.ones_like(x, dtype=bool)
    valid[1] = False
    stats.update(x, valid)
    result = finish_product(stats, ["a", "b"])
    assert result["status"] == "failed" and result["failed_bands"] == ["b"]
    assert result["statistics"] is None


def test_highres_statistics_cli_requires_frozen_quality_and_separate_lock(tmp_path, monkeypatch):
    import fcntl

    from xuannv_embedding.data_process import v5_cli, v5_highres_statistics

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(
        v5_highres_statistics, "run_highres_statistics", lambda *a, **k: calls.append((a, k))
    )
    args = ["--stage", "highres-statistics"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + key, str(tmp_path / key)]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--sensor-family", "jilin1", "--quality-root", str(tmp_path / "qa")]
    source = tmp_path / "source-root"
    source.mkdir(exist_ok=True)
    with (source / ".highres-statistics.jilin1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another V5"):
            v5_cli.main(args)
    assert v5_cli.main(args) == 0 and len(calls) == 1


def test_gaofen_pan_statistics_reader_verifies_geographic_cloud_transfer(tmp_path):
    import rasterio
    import zarr
    from rasterio.transform import from_origin
    from test_v5_clear_intraband import gaofen_qa

    from xuannv_embedding.data_process.v5_highres_reader import HighresQualityReader
    from xuannv_embedding.data_process.v5_sources import sha256

    data, qa, _ = gaofen_qa(tmp_path, count=1)
    table = pd.read_parquet(qa / "observation_quality.parquet")
    path = tmp_path / "pan.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=640,
        width=640,
        count=1,
        dtype="uint16",
        nodata=0,
        crs="EPSG:32650",
        transform=from_origin(300000, 4000000, 2, 2),
    ) as ds:
        ds.write(np.full((1, 640, 640), 12, dtype="uint16"))
    table["pan_path"] = str(path)
    table["pan_sha256"] = sha256(path)
    mask = np.ones((640, 640), bool)
    mask[:96] = False
    table["pan_valid_pixels"] = int(mask.sum())
    table.to_parquet(qa / "observation_quality.parquet", index=False)
    masks = zarr.open_group(str(qa / "valid_masks.zarr"), mode="a")
    masks.create_dataset(
        "pan_valid_packed", data=np.packbits(mask[None], axis=-1, bitorder="little")
    )
    reader = HighresQualityReader(data, "gaofen", qa)
    assert set(reader.inventory.product_id) == {"gaofen_ms", "gaofen_pan"}
    row = next(reader.inventory.loc[reader.inventory.product_id.eq("gaofen_pan")].itertuples())
    frame, proof = reader.read(row)
    assert frame.values.shape == (1, 640, 640) and np.array_equal(frame.valid[0], mask)
    assert not frame.values[~frame.valid].any() and proof["unit"] == "native_stored_dn"
    masks["pan_valid_packed"][0, 300, 0] = 0
    with pytest.raises(ValueError, match="geographic transfer"):
        reader.read(row)


def test_jilin_statistics_reader_handles_b0_and_extra_spectra_at_native_resolution(
    tmp_path, monkeypatch
):
    from test_v5_jilin_quality import Predictor, record

    from xuannv_embedding.data_process import v5_followup
    from xuannv_embedding.data_process.v5_highres_reader import HighresQualityReader
    from xuannv_embedding.data_process.v5_jilin_quality import process_jilin_quality
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    data = tmp_path / "data"
    registry = data / "registry/national_62000.parquet"
    registry.parent.mkdir(parents=True)
    pd.DataFrame([{"patch_id": "national", "split": "train"}]).to_parquet(registry, index=False)
    products = ["jilin1_ms_5m", "jilin1_b0_5m", "jilin1_extra_10m", "jilin1_extra_20m"]
    rows = [record(tmp_path, p) for p in products]
    catalog = data / "observations/highres/jilin1/partial_bands/fixed"
    catalog.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(catalog / "files_with_partial_bands.parquet", index=False)
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
    models = tmp_path / "models"
    models.mkdir()
    for i in [0, 1]:
        (models / f"ocm_v4_model_{i}_96_910b4.om").write_bytes(b"fixture")

    class Factory(Predictor):
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    output = process_jilin_quality(
        tmp_path / "source", data, tmp_path / "report", models, predictor_factory=Factory
    )
    reader = HighresQualityReader(data, "jilin1", Path(output["output"]))
    expected = {
        "jilin1_ms_5m": (6, 256, 256),
        "jilin1_b0_5m": (1, 256, 256),
        "jilin1_extra_10m": (6, 128, 128),
        "jilin1_extra_20m": (7, 64, 64),
    }
    for row in reader.inventory.itertuples():
        frame, proof = reader.read(row)
        assert frame.values.shape == expected[row.product_id]
        assert not frame.values[~frame.valid].any() and proof["unit"] == "reflectance"
        if row.product_id == "jilin1_extra_20m":
            assert frame.band_ids[0] == "B13"
