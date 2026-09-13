import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process.v5_rasters import inspect_jilin, read_native, select_annual


def raster(tmp_path, names=("B2(0.444)", "B1(0.416)")):
    p = tmp_path / "image.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=2,
        dtype="int16",
        nodata=-28672,
        crs="EPSG:32643",
        transform=from_origin(0, 10, 5, 5),
    ) as ds:
        ds.write(np.array([[[0, -10], [1000, -28672]], [[2, 3], [4, -28672]]], dtype="int16"))
        ds.descriptions = names
        ds.scales = (0.0001, 0.0001)
        ds.offsets = (0, 0.2)
    return p


def test_native_reader_selects_named_bands_and_preserves_valid_zero_negative(tmp_path):
    p = raster(tmp_path)
    result = read_native(p, ["B1", "B2"])
    assert result.band_ids == ("B1", "B2")
    np.testing.assert_allclose(result.values[0, 0, 0], 0.2002)
    assert result.values[1, 0, 0] == 0 and result.valid[1, 0, 0]
    assert result.values[1, 0, 1] < 0
    assert not result.valid[:, 1, 1].any()
    assert np.isfinite(result.values).all()


def test_native_reader_rejects_unknown_and_duplicate_bands(tmp_path):
    p = raster(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        read_native(p, ["B7"])
    p = raster(tmp_path, ("B1(0.4)", "B1(0.5)"))
    with pytest.raises(ValueError, match="duplicate"):
        read_native(p, ["B1"])


def test_annual_selection_includes_late_year_and_excludes_other_year():
    rows = [
        {"scene_group_id": str(i), "year": year, "acquired_at": date, "clear_fraction": clear}
        for i, (year, date, clear) in enumerate(
            [
                (2020, "2020-01-01", 0.5),
                (2020, "2020-12-01", 0.9),
                (2021, "2021-01-01", 1),
                (2020, "2020-02-01", 0.8),
                (2020, "2020-06-01", 0.7),
            ]
        )
    ]
    result = select_annual(rows, year=2020, limit=3)
    assert {r["scene_group_id"] for r in result} == {"1", "3", "4"}
    assert result == select_annual(list(reversed(rows)), year=2020, limit=3)


def test_catalog_rejects_unverified_jilin_contract(tmp_path):
    p = raster(tmp_path)
    with pytest.raises(ValueError):
        inspect_jilin(p)
