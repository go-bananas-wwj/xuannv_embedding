import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process.v5_partial_bands import inspect_partial_jilin, read_jilin_branch


def _raster(tmp_path, names, gsd=20):
    path = tmp_path / "JL1GP01" / "partial.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 1280 // gsd
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=len(names),
        dtype="int16",
        crs="EPSG:32650",
        transform=from_origin(500000, 4001280, gsd, gsd),
        nodata=-28672,
    ) as out:
        for i, name in enumerate(names, 1):
            a = np.full((size, size), i * 100, "i2")
            a[0, 0] = -28672
            a[0, 1] = 0
            out.write(a, i)
            out.set_band_description(i, f"{name}(0.834)")
        out.scales = (0.0001,) * len(names)
        out.offsets = (0.0,) * len(names)
        out.update_tags(
            units="reflectance",
            acquisition_time="2020-06-01 12:00:00",
            source_product="scene",
            source_signature="signature",
            patch_id="source_patch",
        )
    return path


def test_missing_native_redundant_band_does_not_discard_complete_supplement(tmp_path):
    names = [f"B{i}" for i in range(1, 20) if i != 6]
    path = _raster(tmp_path, names)
    record = inspect_partial_jilin(path)
    assert record["missing_native_band_ids"] == ["B6"]
    assert record["missing_selected_band_ids"] == []
    native = read_jilin_branch(record)
    assert native.values.shape == (7, 64, 64) and native.band_ids == tuple(
        f"B{i}" for i in range(13, 20)
    )
    assert np.isclose(native.values[0, 1, 1], 0.12)
    assert not native.valid[:, 0, 0].any() and native.valid[:, 0, 1].all()


def test_absent_spectral_channel_stays_invalid_after_normalization(tmp_path):
    names = [f"B{i}" for i in range(1, 20) if i not in [6, 14, 19]]
    path = _raster(tmp_path, names)
    record = inspect_partial_jilin(path)
    native = read_jilin_branch(record, mean=np.ones(7), std=np.ones(7) * 2)
    assert record["missing_selected_band_ids"] == ["B14", "B19"]
    assert not native.valid[[1, 6]].any() and not native.values[[1, 6]].any()
    assert native.valid[0, 1, 1] and native.values[0, 1, 1] < 0
    quality = np.ones((7, 64, 64), bool)
    quality[0] = False
    assert not read_jilin_branch(record, quality=quality).valid[0].any()
    with pytest.raises(ValueError):
        read_jilin_branch(record, mean=np.ones(6), std=np.ones(6))


def test_partial_contract_rejects_unknown_ids_and_changed_radiometry(tmp_path):
    path = _raster(tmp_path, ["B1", "B99"])
    with pytest.raises(ValueError):
        inspect_partial_jilin(path)
    path.unlink()
    path = _raster(tmp_path, [f"B{i}" for i in range(1, 19)])
    record = inspect_partial_jilin(path)
    with rasterio.open(path, "r+") as dst:
        dst.scales = (1.0,) * dst.count
    with pytest.raises(ValueError):
        inspect_partial_jilin(path)
    with pytest.raises(ValueError, match="changed"):
        read_jilin_branch(record)


def test_missing_cloud_input_is_explicit_and_cannot_be_invented(tmp_path):
    path = _raster(tmp_path, ["B1", "B2", "B3", "B4", "B5"], gsd=5)
    record = inspect_partial_jilin(path)
    assert record["cloud_input_band_ids_available"] is False
    assert record["missing_selected_band_ids"] == ["B6"]
    assert not read_jilin_branch(record).valid[5].any()


def test_partial_catalog_keeps_full_inventory_and_spatial_split_and_requires_verified_archive(
    tmp_path,
):
    import json
    import tarfile
    from pathlib import Path

    import pandas as pd

    from xuannv_embedding.data_process.v5_partial_bands import catalog_partial_bands
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    source = tmp_path / "source"
    data = tmp_path / "data"
    report = tmp_path / "report"
    (source / "manifests/extracted").mkdir(parents=True)
    (source / "packages").mkdir()
    (data / "registry").mkdir(parents=True)
    (data / "observations/highres/jilin1").mkdir(parents=True)
    report.mkdir()
    path = _raster(source / "extracted/source_patch", [f"B{i}" for i in range(1, 19)])
    registry = pd.DataFrame(
        {
            "patch_id": ["national"],
            "split": ["val"],
            "grid_epsg": [32650],
            "utm_bounds": [[500000, 4000000, 501280, 4001280]],
        }
    )
    registry.to_parquet(data / "registry/national_62000.parquet", index=False)
    rejected = pd.DataFrame({"path": [str(path)], "reason": ["unverified Jilin band contract"]})
    rejected.to_parquet(report / "rejected_files.parquet", index=False)
    full = data / "observations/highres/jilin1/files.parquet"
    pd.DataFrame(
        [
            {
                "observation_id": "prior_complete",
                "path": "prior_complete.tif",
                "file_sha256": "prior_receipt",
                "selected_band_ids": ["B0"],
                "band_ids": ["B0"],
                "year": 2020,
                "metadata_status": "verified",
            }
        ]
    ).to_parquet(full, index=False)
    full_hash = sha256(full)
    raw_hash = sha256(path)
    archive = "test.tar.gz"
    with tarfile.open(source / "packages" / archive, "w:gz") as tar:
        tar.add(path, arcname="source_patch/JL1GP01/partial.tif")
    spec = {
        "archive": archive,
        "bytes": (source / "packages" / archive).stat().st_size,
        "sha256": sha256(source / "packages" / archive),
        "tiff_count": 1,
    }
    pd.DataFrame({"archive": [archive], "patchid": ["source_patch"]}).to_csv(
        source / "manifests/ARCHIVE_INDEX.tsv", sep="\t", index=False
    )
    write_json(
        source / "manifests/source.lock.json",
        {
            "revision": "fixed",
            "archives": [spec],
            "manifest_sha256": {
                "ARCHIVE_INDEX.tsv": sha256(source / "manifests/ARCHIVE_INDEX.tsv")
            },
        },
    )
    pd.DataFrame([{**spec, "status": "complete", "actual_bytes": spec["bytes"]}]).to_parquet(
        source / "manifests/download_status.parquet", index=False
    )
    write_json(
        source / "manifests/extracted" / (archive + ".json"),
        {"status": "complete", "sha256": spec["sha256"]},
    )
    integrity = report / "integrity_shards" / (archive + ".json")
    write_json(
        integrity,
        {"status": "complete", "sha256": spec["sha256"], "decoded_tiffs": 1, "failures": []},
    )
    result = catalog_partial_bands(source, data, report)
    assert result["verified_partial_files"] == 1 and result["selected_bands_missing_files"] == 1
    records = pd.read_parquet(Path(result["output"]) / "files.parquet")
    assert records.iloc[0].split == "val"
    assert records.iloc[0].patch_id == "national" and records.iloc[
        0
    ].missing_selected_band_ids.tolist() == ["B19"]
    merged = pd.read_parquet(Path(result["output"]) / "files_with_partial_bands.parquet")
    prior = merged.loc[merged.observation_id == "prior_complete"].iloc[0]
    assert bool(prior.selected_bands_complete) is True
    assert prior.missing_selected_band_ids.tolist() == []
    assert bool(prior.cloud_input_band_ids_available) is False
    assert sha256(full) == full_hash and sha256(path) == raw_hash
    lock = Path(result["output"]) / "catalog.lock.json"
    locked_bytes = lock.read_bytes()
    again = catalog_partial_bands(source, data, report)
    assert result["output"] == again["output"] and result["output_sha256"] == again["output_sha256"]
    assert lock.read_bytes() == locked_bytes
    bad = json.loads(integrity.read_text())
    bad["status"] = "failed"
    write_json(integrity, bad)
    with pytest.raises(ValueError, match="decoded, verified"):
        catalog_partial_bands(source, data, report)


def test_partial_catalog_cli_routes_to_source_and_frozen_registry(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_partial_bands

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda _: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    monkeypatch.setattr(v5_partial_bands, "catalog_partial_bands", lambda *a: calls.append(a))
    argv = ["--stage", "catalog-partial-bands"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        argv += ["--" + key, str(tmp_path / key)]
    assert v5_cli.main(argv) == 0
    assert calls == [
        (tmp_path / "source-root", tmp_path / "dataset-root", tmp_path / "report-root")
    ]
