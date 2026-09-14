import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process.v5_catalog import catalog_file, grid_lookup


def test_catalog_matches_geometry_not_directory_name(tmp_path):
    folder = tmp_path / "misleading_directory" / "JL1GP01"
    folder.mkdir(parents=True)
    path = folder / "image.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=256,
        height=256,
        count=6,
        dtype="int16",
        nodata=-28672,
        crs="EPSG:32643",
        transform=from_origin(0, 1280, 5, 5),
    ) as ds:
        ds.write(np.ones((6, 256, 256), dtype="int16"))
        ds.scales = (0.0001,) * 6
        ds.descriptions = tuple(f"B{i}(0.5)" for i in range(1, 7))
        ds.update_tags(
            units="reflectance",
            acquisition_time="2020-12-31 23:30:00",
            source_product="scene",
            source_signature="abc",
            patch_id="source_alias",
        )
    grid = pd.DataFrame(
        [
            {
                "patch_id": "grid_owner",
                "grid_epsg": 32643,
                "utm_bounds": [0, 0, 1280, 1280],
                "split": "test",
            }
        ]
    )
    row = catalog_file(path, grid_lookup(grid))
    assert row["patch_id"] == "grid_owner" and row["split"] == "test"
    assert row["time_zone"] == "unspecified"
    assert row["year"] == 2020 and row["source_patch_id"] == "source_alias"


def test_catalog_rejects_ambiguous_grid():
    row = {"patch_id": "p", "grid_epsg": 32643, "utm_bounds": [0, 0, 1280, 1280], "split": "train"}
    with pytest.raises(ValueError, match="ambiguous"):
        grid_lookup(pd.DataFrame([row, row]))
