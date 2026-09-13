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
