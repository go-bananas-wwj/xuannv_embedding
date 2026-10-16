from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from xuannv_embedding.data_process.native_dem import JoinedReader, NativeDemArchive


def test_joined_reader_seeks_across_parts_and_preserves_originals(tmp_path: Path) -> None:
    paths = [tmp_path / "one", tmp_path / "two"]
    paths[0].write_bytes(b"abc")
    paths[1].write_bytes(b"defg")
    with JoinedReader(paths) as stream:
        assert stream.read(5) == b"abcde"
        assert stream.seek(-3, io.SEEK_END) == 4
        assert stream.read() == b"efg"
        stream.seek(30)
        assert stream.read(10) == b""
        with pytest.raises(ValueError, match="Negative"):
            stream.seek(-1)
    assert [path.read_bytes() for path in paths] == [b"abc", b"defg"]


def make_native_archive(tmp_path: Path, *, split_inside_tiff: bool) -> tuple[list[Path], dict]:
    pytest.importorskip("rasterio")
    from pyproj import Transformer
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    from xuannv_embedding.data_process.target_splits import geometry_record

    x, y = Transformer.from_crs(4326, 32646, always_xy=True).transform(91.5, 36.5)
    row = geometry_record(f"parent_32646:{int(x // 1280)}:{int(y // 1280)}")
    west, _, _, north = transform_bounds(32646, 4326, *row["utm_bounds"])
    data = np.broadcast_to(100 + np.arange(128, dtype="float32")[:, None], (128, 128)).copy()
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=128,
            height=128,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(west - 0.005, north + 0.005, 1 / 3600, 1 / 3600),
        ) as ds:
            ds.write(data, 1)
            ds.update_tags(AREA_OR_POINT="Point")
        payload = memory.read()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("static/Copernicus_DSM_COG_10_N36_00_E091_00_DEM.tif", payload)
    zipped = stream.getvalue()
    cutoff = len(zipped) // 2 if split_inside_tiff else len(zipped) - 32
    paths = [tmp_path / "dem.part1", tmp_path / "dem.part2"]
    paths[0].write_bytes(zipped[:cutoff])
    paths[1].write_bytes(zipped[cutoff:])
    return paths, row


@pytest.mark.parametrize("split_inside_tiff", [False, True])
def test_native_dsm_read_is_north_up_and_missing_coverage_is_masked(
    tmp_path: Path, split_inside_tiff: bool
) -> None:
    paths, row = make_native_archive(tmp_path, split_inside_tiff=split_inside_tiff)
    with NativeDemArchive(paths) as archive:
        patch = archive.read_patch(row)
        assert patch["values"].shape == (32, 32)
        assert patch["valid"].all()
        assert (np.diff(patch["values"][:, 16]) > 0).all()
        assert 100 < patch["values"].min() < patch["values"].max() < 228
        missing = archive.read_patch(
            {"grid_epsg": 32643, "utm_bounds": [400000, 4000000, 401280, 4001280]}
        )
        assert not missing["valid"].any()
        assert not missing["values"].any()


def test_native_cache_publication_checks_each_row_and_keeps_missing_mask(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    zarr = pytest.importorskip("zarr")
    from xuannv_embedding.data_process.target_prepare import native_cache

    paths, row = make_native_archive(tmp_path, split_inside_tiff=False)
    integrity = {
        "all_zip_member_crc_passed": True,
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": "fixture",
            }
            for path in paths
        ],
    }
    row["target_row"] = 0
    manifest = native_cache([row], integrity, tmp_path / "cache", workers=1)
    assert manifest["complete"] and manifest["native_dem_ready"]
    assert not manifest["full_training_acceptance"]
    receipts = pq.read_table(tmp_path / "cache/index.parquet").to_pylist()
    assert len(receipts[0]["row_sha256"]) == 64
    group = zarr.open_group(str(tmp_path / "cache/dem40m.zarr"), mode="r")
    assert group.attrs["complete"]
    assert group["valid"][0].all()
