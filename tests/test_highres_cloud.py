import json
import sqlite3

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process.highres_cloud import (
    finalize,
    read_highres_item,
    spectral_indices,
)
from xuannv_embedding.data_process.observation_raster import sha256_file
from xuannv_embedding.utils.manifest import ManifestRecord, load_manifest, write_manifest

NAMES = [f"B{i + 1}({w})" for i, w in enumerate((0.414, 0.444, 0.485, 0.562, 0.655, 0.834))]


def test_spectral_mapping_requires_documented_nir():
    assert spectral_indices({"band_names": NAMES}) == [4, 3, 5]
    with pytest.raises(ValueError, match="Unique"):
        spectral_indices({"band_names": NAMES[:5]})
    with pytest.raises(ValueError, match="wavelength"):
        spectral_indices({"band_names": [None] * 6})


def test_native_highres_reader_checks_wavelengths_scaling_and_masks(tmp_path):
    pytest.importorskip("omnicloudmask")
    profile = dict(
        driver="GTiff",
        count=6,
        width=256,
        height=256,
        dtype="int16",
        crs="EPSG:32650",
        transform=from_origin(512000, 3585280, 5, 5),
    )
    values = np.full((6, 256, 256), 1000, dtype="int16")
    values[:, 0, 0] = 32767
    with rasterio.open(tmp_path / "ms.tif", "w", **profile) as raster:
        raster.write(values)
        raster.descriptions = tuple(NAMES)
        raster.scales = (0.0001,) * 6
    with rasterio.open(
        tmp_path / "mask.tif", "w", **{**profile, "count": 1, "dtype": "uint8"}
    ) as raster:
        mask = np.ones((1, 256, 256), dtype="uint8")
        mask[:, 0, 1] = 0
        raster.write(mask)
    item = dict(
        path="ms.tif",
        mask="mask.tif",
        sha256=sha256_file(tmp_path / "ms.tif"),
        mask_sha256=sha256_file(tmp_path / "mask.tif"),
        parent_key="32650:400:2800",
        channels=6,
        cloud_rgbnir_indices=[4, 3, 5],
        stored_scales=[0.0001] * 6,
        stored_offsets=[0.0] * 6,
    )
    result = read_highres_item((item["path"], json.dumps(item)), tmp_path)
    assert result[3] is None
    assert result[2][3].shape == (3, 256, 256)
    assert not result[2][1][0, :2].any()
    item["cloud_rgbnir_indices"] = [0, 1, 2]
    assert "mapping changed" in read_highres_item((item["path"], json.dumps(item)), tmp_path)[3]
    item["sha256"] = "0" * 64
    assert "checksum" in read_highres_item((item["path"], json.dumps(item)), tmp_path)[3]


def test_final_merge_retains_pan_and_refits_clear_train_statistics(tmp_path):
    base, output = tmp_path / "base", tmp_path / "output"
    base.mkdir()
    output.mkdir()
    (base / "summary.json").write_text(
        json.dumps({"data_root": str(tmp_path), "annual_model_ready": False})
    )
    (base / "sources.json").write_text(json.dumps({"ms": {}, "pan": {}, "missing_nir": {}}))
    database = sqlite3.connect(output / "results-0.sqlite")
    database.execute("CREATE TABLE results(path TEXT PRIMARY KEY,payload TEXT)")
    fingerprints = {}
    for split, mean in (("train", 2.0), ("validation", 10000.0)):
        ms = dict(
            path=f"{split}/ms.tif",
            source="ms",
            sha256="a" * 64,
            parent_key=split,
            date="2020-01-02",
            band_counts=[4] * 6,
            band_mean=[mean] * 6,
            band_variance=[1.0] * 6,
        )
        pan = {
            **ms,
            "source": "pan",
            "path": f"{split}/pan.tif",
            "band_counts": [4],
            "band_mean": [5.0],
            "band_variance": [2.0],
        }
        obs = {"ms": [ms], "pan": [pan], "missing_nir": [ms]}
        record = ManifestRecord(
            patch_id=split,
            region="annual_2020",
            sources={s: [o["path"] for o in items] for s, items in obs.items()},
            quality={"pan2m_supported": True},
            provenance={"split": split, "year": 2020, "observations": obs},
        )
        path = base / f"2020.{split}.manifest.jsonl"
        write_manifest(path, [record], months=[f"2020-{m:02}" for m in range(1, 13)])
        fingerprints[path.name] = sha256_file(path)
        database.execute(
            "INSERT INTO results VALUES (?,?)",
            (ms["path"], json.dumps({**ms, "status": "accepted"})),
        )
    database.commit()
    database.close()
    (output / "progress-0.json").write_text(json.dumps({"status": "complete"}))
    (output / "run.json").write_text(
        json.dumps(
            dict(
                workers=1,
                total=2,
                input=str(base),
                input_manifests=fingerprints,
                supported_sources={"ms": [4, 3, 5]},
                rejected_sources={"missing_nir": "NIR unavailable"},
            )
        )
    )
    summary = finalize(output, base)
    assert summary["training_ready"] is False
    assert summary["annual_model_ready"] is False
    stats = json.loads((output / "statistics/ms_stats.json").read_text())
    assert stats["mean"] == [2.0] * 6
    doc = load_manifest(output / "2020.train.manifest.jsonl")
    assert doc.records[0].sources["pan"] == ["train/pan.tif"]
    assert "missing_nir" not in doc.records[0].sources
    assert doc.records[0].quality["ms5m_supported"]
