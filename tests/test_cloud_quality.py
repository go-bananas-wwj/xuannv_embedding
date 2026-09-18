import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process.cloud_quality import load_s2, screened_pixels
from xuannv_embedding.data_process.observation_raster import sha256_file


def test_cloud_and_invalid_pixels_cannot_enter_statistics():
    values = np.array([[[2, 1000], [6, 99]]] * 10, dtype=np.float32)
    numeric = np.array([[True, True], [True, False]])
    scores = np.zeros((4, 2, 2), dtype=np.float32)
    scores[0] = 1
    scores[:, 0, 1] = [0, 1, 0, 0]
    qa, moments = screened_pixels(values, numeric, scores)
    assert qa[0].tolist() == [[1, 0], [1, 0]]
    assert qa[1, 1, 1] == 255
    assert moments["band_mean"] == [4] * 10
    assert moments["band_variance"] == [4] * 10
    assert moments["band_counts"] == [2] * 10
    assert moments["cloud_class_counts"] == [2, 1, 0, 0]
    json.dumps(moments, allow_nan=False)


def test_all_cloud_has_no_fake_zero_statistics():
    values = np.ones((10, 2, 2))
    scores = np.zeros((4, 2, 2))
    scores[1] = 1
    _, moments = screened_pixels(values, np.ones((2, 2), dtype=bool), scores)
    assert moments["band_counts"] == [0] * 10
    assert moments["band_mean"] == []
    scores[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="output"):
        screened_pixels(values, np.ones((2, 2), dtype=bool), scores)


def test_source_checksum_and_mask_alignment_are_verified(tmp_path):
    path = tmp_path / "s2.tif"
    mask = tmp_path / "numeric.tif"
    profile = dict(
        driver="GTiff",
        width=128,
        height=128,
        count=10,
        dtype="uint16",
        crs="EPSG:32650",
        transform=from_origin(512000, 3585280, 10, 10),
    )
    values = np.ones((10, 128, 128), dtype=np.uint16)
    values[:, 0, 0] = 65535
    with rasterio.open(path, "w", **profile) as raster:
        raster.write(values)
    with rasterio.open(mask, "w", **{**profile, "count": 1, "dtype": "uint8"}) as raster:
        raster.write(np.ones((1, 128, 128), dtype=np.uint8))
    observation = dict(
        path="s2.tif", mask="numeric.tif", sha256=sha256_file(path), parent_key="32650:400:2800"
    )
    _, valid, _ = load_s2(tmp_path, observation)
    assert not valid[0, 0]
    observation["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        load_s2(tmp_path, observation)
    observation["sha256"] = sha256_file(path)
    with rasterio.open(mask, "r+") as raster:
        raster.transform = from_origin(512010, 3585280, 10, 10)
    with pytest.raises(ValueError, match="mask geometry"):
        load_s2(tmp_path, observation)


def test_finalization_uses_train_only_and_drops_all_cloud_months(tmp_path):
    import sqlite3

    from xuannv_embedding.data_process.cloud_quality import finalize
    from xuannv_embedding.utils.manifest import ManifestRecord, load_manifest, write_manifest

    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "summary.json").write_text(json.dumps({"data_root": str(tmp_path)}))
    (source / "sources.json").write_text("{}")
    fingerprints = {}
    database = sqlite3.connect(output / "results-0.sqlite")
    database.execute("CREATE TABLE results(path TEXT PRIMARY KEY,payload TEXT)")
    for split, mean in [("train", 2.0), ("validation", 10000.0)]:
        observations = []
        for month in (1, 2, 4, 5, 7, 8, 10):
            o = dict(
                path=f"{split}/{month}.tif",
                source="s2",
                date=f"2020-{month:02}",
                valid_fraction=1.0,
                band_counts=[4] * 10,
                band_mean=[mean] * 10,
                band_variance=[1.0] * 10,
            )
            observations.append(o)
            database.execute(
                "INSERT INTO results VALUES (?,?)",
                (
                    o["path"],
                    json.dumps({**o, "status": "accepted" if month != 10 else "no_clear_pixels"}),
                ),
            )
        record = ManifestRecord(
            patch_id=split,
            region="annual_2020",
            sources={"s2": [o["path"] for o in observations]},
            quality={},
            provenance={"year": 2020, "split": split, "observations": {"s2": observations}},
        )
        path = source / f"2020.{split}.manifest.jsonl"
        write_manifest(path, [record], months=[f"2020-{m:02}" for m in range(1, 13)])
        fingerprints[path.name] = sha256_file(path)
    database.commit()
    database.close()
    (output / "progress-0.json").write_text(json.dumps({"status": "complete"}))
    (output / "run.json").write_text(
        json.dumps(
            {
                "limit": 0,
                "input": str(source),
                "workers": 1,
                "total": 14,
                "input_manifests": fingerprints,
            }
        )
    )
    summary = finalize(output)
    stats = json.loads((output / "statistics/s2_stats.json").read_text())
    assert stats["mean"] == [2.0] * 10
    assert stats["num_files"] == 6
    assert summary["cloud_screening_counts"]["no_clear_pixels"] == 2
    assert summary["training_ready"] is False
    doc = load_manifest(output / "2020.train.manifest.jsonl")
    assert len(doc.records[0].sources["s2"]) == 6
    assert doc.records[0].quality["s2_months_min_20pct"] == 6
    assert (
        sha256_file(source / "2020.train.manifest.jsonl")
        == fingerprints["2020.train.manifest.jsonl"]
    )
