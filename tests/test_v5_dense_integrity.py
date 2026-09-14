from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from xuannv_embedding.data_process.v5_dense_integrity import inspect_dense_archive


def make_registry():
    return pd.DataFrame(
        [
            {
                "patch_id": "location",
                "split": "train",
                "grid_epsg": 32650,
                "utm_bounds": [500000.0, 3000000.0, 501280.0, 3001280.0],
            }
        ]
    )


def test_dense_integrity_checks_pixels_and_grid_without_guessing_radiometry(tmp_path):
    archive = tmp_path / "monthly.zip"
    with MemoryFile() as file:
        with file.open(
            driver="GTiff",
            width=128,
            height=128,
            count=2,
            dtype="float32",
            crs="EPSG:32650",
            transform=from_origin(500000, 3001280, 10 + 6e-12, 10 + 7e-12),
        ) as dst:
            dst.write(np.ones((2, 128, 128), dtype="f4"))
        blob = file.read()
    with ZipFile(archive, "w") as output:
        output.writestr("source/a.tif", blob)
        output.writestr("README.md", "original metadata")
    result = inspect_dense_archive(
        archive, make_registry(), tmp_path / "audit", product="s1_local", year=2020, month=1
    )
    assert result["status"] == "integrity_checked_contract_pending"
    assert result["decoded_tiffs"] == 1
    assert result["missing_patches"] == 0
    assert result["auxiliary_files"] == 1
    repeated = inspect_dense_archive(
        archive, make_registry(), tmp_path / "audit", product="s1_local", year=2020, month=1
    )
    assert repeated["source_sha256"] == result["source_sha256"]
    assert Path(result["inventory_path"]).is_file()


def test_corrupt_tiff_is_reported_and_missing_location_remains_missing(tmp_path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("bad.tif", b"not a raster")
    result = inspect_dense_archive(
        archive, make_registry(), tmp_path / "audit", product="s1_local", year=2020, month=1
    )
    assert result["status"] == "failed"
    assert result["failed_tiffs"] == 1
    assert result["missing_patches"] == 1


def test_landsat_keeps_its_43_pixel_grid_and_does_not_invent_band_identity(tmp_path):
    import json

    archive = tmp_path / "landsat.zip"
    with MemoryFile() as file:
        with file.open(
            driver="GTiff",
            width=43,
            height=43,
            count=6,
            dtype="uint16",
            crs="EPSG:32650",
            transform=from_origin(500000, 3001280, 1280 / 43 + 6e-12, 1280 / 43 + 7e-12),
        ) as dst:
            dst.write(np.ones((6, 43, 43), dtype="u2"))
        blob = file.read()
    with ZipFile(archive, "w") as output:
        output.writestr("source/a.tif", blob)
    result = inspect_dense_archive(
        archive, make_registry(), tmp_path / "audit", product="landsat_local", year=2020, month=1
    )
    assert result["decoded_tiffs"] == 1 and result["failed_tiffs"] == 0
    assert result["stored_grid_contract"] == {
        "bands": 6,
        "shape": [43, 43],
        "pixel_size_m": 1280 / 43,
    }
    assert not result["physical_contract_verified"]
    frame = pd.read_parquet(result["inventory_path"])
    metadata = json.loads(frame.iloc[0].metadata_json)
    assert metadata["descriptions"] == [None] * 6
    assert metadata["radiometry_status"] == "unverified"
    assert metadata["shape"] == [43, 43]


def test_landsat_wrong_shape_or_band_count_is_still_rejected(tmp_path):
    for i, (size, count) in enumerate([(128, 6), (43, 3), (42, 6)]):
        archive = tmp_path / f"wrong-{i}.zip"
        with MemoryFile() as file:
            with file.open(
                driver="GTiff",
                width=size,
                height=size,
                count=count,
                dtype="uint16",
                crs="EPSG:32650",
                transform=from_origin(500000, 3001280, 1280 / size, 1280 / size),
            ) as dst:
                dst.write(np.ones((count, size, size), dtype="u2"))
            blob = file.read()
        with ZipFile(archive, "w") as output:
            output.writestr("wrong.tif", blob)
        result = inspect_dense_archive(
            archive,
            make_registry(),
            tmp_path / f"audit-{i}",
            product="landsat_local",
            year=2020,
            month=1,
        )
        assert result["failed_tiffs"] == 1 and result["decoded_tiffs"] == 0


def test_landsat_only_v2_audit_preserves_old_reports_and_scans_exactly_24_months(
    tmp_path, monkeypatch
):
    from xuannv_embedding.data_process import v5_dense_integrity as module

    data, report = tmp_path / "data", tmp_path / "report"
    (data / "registry").mkdir(parents=True)
    make_registry().to_parquet(data / "registry/national_62000.parquet")
    report.mkdir()
    old = report / "dense_integrity_summary.json"
    old.write_text('{"legacy":true}')
    calls = []

    def inspect(path, registry, output_root, **kwargs):
        calls.append((path, output_root, kwargs))
        return {"status": "integrity_checked_contract_pending", **kwargs}

    monkeypatch.setattr(module, "inspect_dense_archive", inspect)
    result = module.audit_dense_integrity(
        tmp_path / "dense", data, report, audit_version="v2", selected_product="landsat_local"
    )
    assert result["selected_archives"] == 24 and result["failed_archives"] == 0
    assert {(c[2]["year"], c[2]["month"]) for c in calls} == {
        (y, m) for y in [2020, 2021] for m in range(1, 13)
    }
    assert all(c[2]["product"] == "landsat_local" for c in calls)
    assert all(
        c[1] == report / "dense_integrity/v2/landsat_local/dense_integrity_shards" for c in calls
    )
    assert old.read_text() == '{"legacy":true}'
    assert (report / "dense_integrity/v2/landsat_local/dense_integrity_summary.json").exists()


def test_dense_v2_cli_uses_isolated_product_lock_and_passes_scope(tmp_path, monkeypatch):
    import fcntl

    import pytest

    from xuannv_embedding.data_process import v5_cli, v5_dense_integrity

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(
        v5_dense_integrity, "audit_dense_integrity", lambda *a, **k: calls.append((a, k))
    )
    args = [
        "--stage",
        "dense-integrity",
        "--dense-root",
        str(tmp_path / "dense"),
        "--dense-audit-version",
        "v2",
        "--dense-product",
        "landsat_local",
    ]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + key, str(tmp_path / key)]
    source = tmp_path / "source-root"
    source.mkdir()
    with (source / ".dense-integrity.v2.landsat_local.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another V5"):
            v5_cli.main(args)
    assert v5_cli.main(args) == 0
    assert calls[0][1] == {"audit_version": "v2", "selected_product": "landsat_local"}
