from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process.prepare_observations import (
    acquisition_date,
    finalize,
    prepare_one,
    product_schema,
)
from xuannv_embedding.utils.manifest import ManifestRecord, load_manifest, write_manifest


def make_observation(tmp_path, *, date="20200102", split="train", offset=0, value=10, group="20m"):
    member = (
        f"ownerfix_epsg32650_c400_r2800/JL1GP01/"
        f"{date}_JL1GP01_PMS2_{date}120000_{group}_abcdef.tif"
    )
    path = tmp_path / "raw" / member
    path.parent.mkdir(parents=True, exist_ok=True)
    spacing = int(group.removesuffix("m"))
    size = 1280 // spacing
    values = np.full((2, size, size), value, dtype=np.int16)
    values[0, 0, 0] = -28672
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=2,
        dtype="int16",
        width=size,
        height=size,
        crs="EPSG:32650",
        transform=from_origin(512000 + offset, 3585280, spacing, spacing),
        nodata=-28672,
    ) as dataset:
        dataset.write(values)
        dataset.descriptions = ("first", "second")
    return {
        "archive_member": member,
        "parent_key": "32650:400:2800",
        "split": split,
        "archive": "example.tar.gz",
    }


def test_acquisition_date_uses_leading_date_not_incidental_hash_digits():
    assert acquisition_date("20200229_JL1_PMS1_20200229123456_20210303.tif") == "2020-02-29"
    with pytest.raises(ValueError):
        acquisition_date("20200230_JL1_PMS1.tif")
    with pytest.raises(ValueError, match="conflicting"):
        acquisition_date("20200101_JL1_PMS1_20200102120000_abcdef.tif")


def test_numeric_mask_preserves_pixels_and_separates_spectral_schemas(tmp_path):
    original = make_observation(tmp_path)
    source = tmp_path / "raw" / original["archive_member"]
    before = source.read_bytes()
    result = prepare_one(tmp_path / "raw", tmp_path / "out", original)
    assert result["status"] == "usable"
    assert result["band_mean"] == [10, 10]
    assert result["band_counts"] == [4095, 4095]
    assert source.read_bytes() == before
    with rasterio.open(tmp_path / "out" / result["mask_path"]) as dataset:
        assert dataset.read(1)[0, 0] == 0
    metadata = dict(result, band_names=["second", "first"])
    changed, _ = product_schema(original["archive_member"], metadata)
    assert changed != result["source_signature"]


def test_grid_and_date_rejections_do_not_become_training_paths(tmp_path):
    original = make_observation(tmp_path, offset=10)
    record = prepare_one(tmp_path / "raw", tmp_path / "out", original)
    assert record["status"] == "quarantined"
    assert record["issue"] == "grid_mismatch"
    assert "materialized_path" not in record
    original = make_observation(tmp_path, date="20200230")
    assert prepare_one(tmp_path / "raw", tmp_path / "out", original)["status"] == "quarantined"


def test_statistics_use_train_only_and_year_manifests_are_separate(tmp_path):
    output = tmp_path / "out"
    records = []
    for date, split, value in [("20200102", "train", 10), ("20210102", "train", 20)]:
        original = make_observation(tmp_path, date=date, split=split, value=value)
        records.append(prepare_one(tmp_path / "raw", output, original))
    validation = dict(
        records[0], parent_key="32650:401:2800", split="validation", band_mean=[10000, 10000]
    )
    records.append(validation)
    (output / "parts").mkdir()
    (output / "parts" / "example.tar.gz.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    summary = finalize(output, ["example.tar.gz"], 4)
    assert summary["observations"] == {"usable": 3}
    stats = json.loads(next((output / "statistics" / "2020").glob("*.json")).read_text())
    assert stats["mean"] == [10, 10]
    assert stats["num_files"] == 1
    document = load_manifest(output / "2020.train.manifest.jsonl")
    assert document.meta.months == [f"2020-{month:02d}" for month in range(1, 13)]
    assert len(document.records) == 1
    assert all(
        "20200102" in path for paths in document.records[0].sources.values() for path in paths
    )


def test_pairing_reads_real_batches_and_backpropagates(tmp_path):
    from xuannv_embedding.data_process.pair_observations import pair, smoke_backward

    highres = tmp_path / "highres"
    original = make_observation(tmp_path, group="5m")
    result = prepare_one(tmp_path / "raw", highres, original)
    (highres / "parts").mkdir()
    (highres / "parts" / "example.tar.gz.jsonl").write_text(json.dumps(result) + "\n")
    finalize(highres, ["example.tar.gz"], 4)
    lowres = tmp_path / "lowres.partial"
    path = lowres / "s2/2020/01/parent.tif"
    path.parent.mkdir(parents=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=10,
        dtype="uint16",
        width=128,
        height=128,
        crs="EPSG:32650",
        transform=from_origin(512000, 3585280, 10, 10),
        nodata=0,
    ) as dataset:
        dataset.write(np.full((10, 128, 128), 10, dtype=np.uint16))
    with rasterio.open(
        path.with_name("parent_mask.tif"),
        "w",
        driver="GTiff",
        count=1,
        dtype="uint8",
        width=128,
        height=128,
        crs="EPSG:32650",
        transform=from_origin(512000, 3585280, 10, 10),
        nodata=0,
    ) as dataset:
        dataset.write(np.ones((1, 128, 128), dtype=np.uint8))
    (lowres / "statistics").mkdir()
    (lowres / "statistics/s2_stats.json").write_text(
        json.dumps({"mean": [0] * 10, "std": [1] * 10})
    )
    (lowres / "selected_parents.jsonl").write_text("{}\n")
    record = ManifestRecord(
        patch_id="parent_32650:400:2800", region="test", sources={"s2": ["s2/2020/01/parent.tif"]}
    )
    for split in ("train", "validation", "test"):
        write_manifest(
            lowres / f"{split}.manifest.jsonl",
            [record] if split == "train" else [],
            months=[f"2020-{month:02d}" for month in range(1, 13)],
        )
    with pytest.raises(ValueError, match="completed lowres"):
        pair(highres, lowres, tmp_path / "forbidden")
    output = tmp_path / "paired"
    report = pair(highres, lowres, output, 1)
    assert report["data_interface_ready"]
    assert report["probe_only"]
    assert smoke_backward(output / "2020.train.loader-smoke.yaml")["finite_highres_gradients"]


def test_archive_catalog_hash_mismatch_blocks_pairing(tmp_path):
    from xuannv_embedding.data_process.pair_observations import verify_extracted_hashes

    catalog = tmp_path / "catalog"
    (catalog / "catalog_parts").mkdir(parents=True)
    highres = tmp_path / "highres"
    (highres / "parts").mkdir(parents=True)
    (highres / "fingerprint.json").write_text(
        json.dumps(
            {
                "catalog": str(catalog),
                "archives": [{"archive": "one.tar.gz"}],
            }
        )
    )
    (catalog / "catalog_parts/one.tar.gz.jsonl").write_text(
        json.dumps({"archive_member": "image.tif", "sha256": "original"}) + "\n"
    )
    (highres / "parts/one.tar.gz.jsonl").write_text(
        json.dumps({"archive_member": "image.tif", "sha256": "changed", "status": "usable"}) + "\n"
    )
    with pytest.raises(ValueError, match="differs from archive catalog"):
        verify_extracted_hashes(highres)
