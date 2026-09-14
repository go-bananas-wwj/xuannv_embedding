import io
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process.v5_dem_geometry import (
    ConcatenatedReader,
    legacy_slope,
    slope_support,
    stored_member_range,
    write_sparse_descriptor,
)


def test_concatenated_reader_crosses_parts_and_supports_zip_random_access(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("entry.txt", b"known content")
    raw = data.getvalue()
    parts = [tmp_path / "part1", tmp_path / "part2"]
    parts[0].write_bytes(raw[:53])
    parts[1].write_bytes(raw[53:])
    with ConcatenatedReader(parts) as stream:
        stream.seek(50)
        assert stream.read(9) == raw[50:59]
        stream.seek(-4, io.SEEK_END)
        assert stream.read(99) == raw[-4:]
        with pytest.raises(ValueError):
            stream.seek(-1)
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            assert archive.read("entry.txt") == b"known content"
    assert stream.closed


def test_sparse_zip_reads_raster_across_physical_split_without_copying_pixels(tmp_path):
    image = tmp_path / "raster.tif"
    values = np.arange(256, dtype="f4").reshape(16, 16)
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        count=1,
        height=16,
        width=16,
        dtype="float32",
        crs="EPSG:32650",
        transform=from_origin(300000, 4000000, 10, 10),
    ) as dst:
        dst.write(values, 1)
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.write(image, "folder/raster.tif")
    raw = data.getvalue()
    middle = len(raw) // 2
    parts = [tmp_path / "a & one.part1", tmp_path / "b.part2"]
    for part, value in zip(parts, [raw[:middle], raw[middle:]], strict=True):
        part.write_bytes(value)
    with ConcatenatedReader(parts) as stream, zipfile.ZipFile(stream) as archive:
        offset, size = stored_member_range(stream, archive.getinfo("folder/raster.tif"))
    uri = write_sparse_descriptor(parts, tmp_path / "source.xml", offset=offset, length=size)
    with rasterio.open(uri) as src:
        np.testing.assert_equal(src.read(1), values)
    assert parts[0].read_bytes() + parts[1].read_bytes() == raw


def test_slope_requires_observed_derivative_neighbors_without_clipping_negative_elevation():
    y, x = np.mgrid[:8, :8]
    elevation = (5 * x - 100).astype("f4")
    valid = np.ones((8, 8), bool)
    np.testing.assert_allclose(
        legacy_slope(elevation, valid), np.degrees(np.arctan(0.5)), atol=1e-5
    )
    assert slope_support(valid).all()
    valid[3, 3] = False
    supported = slope_support(valid)
    assert not supported[3, 3] and not supported[2, 3] and not supported[3, 4]
    assert supported[2, 2] and supported[0, 0]
    np.testing.assert_equal(legacy_slope(elevation, np.zeros_like(valid)), 0)


def test_dem_cli_keeps_static_geometry_separate_from_annual_labels(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_dem_geometry

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda _: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *args: None)
    monkeypatch.setattr(v5_dem_geometry, "audit_dem_geometry", lambda *a, **k: calls.append((a, k)))
    argv = ["--stage", "dem-geometry", "--max-patches", "32"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        argv += ["--" + key, str(tmp_path / key)]
    assert v5_cli.main(argv) == 0
    assert calls == [((tmp_path / "dataset-root", tmp_path / "report-root"), {"max_patches": 32})]


def test_full_dem_audit_reconstructs_static_targets_reuses_results_and_rejects_changed_pixels(
    tmp_path,
):
    import hashlib

    import pandas as pd
    import zarr

    from xuannv_embedding.data_process.v5_dem_geometry import audit_dem_geometry
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    data, report = tmp_path / "data", tmp_path / "report"
    (data / "registry").mkdir(parents=True)
    (data / "targets").mkdir()
    report.mkdir()
    pd.DataFrame(
        [
            {
                "patch_id": "p",
                "grid_epsg": 32650,
                "utm_bounds": [300000, 3998720, 301280, 4000000],
                "split": "train",
            }
        ]
    ).to_parquet(data / "registry/national_62000.parquet", index=False)
    image = tmp_path / "Copernicus_DSM_COG_10_N00_00_E000_00_DEM.tif"
    y, x = np.mgrid[:128, :128]
    elevation = (5 * x - 100).astype("f4")
    valid = np.ones_like(elevation, bool)
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        count=1,
        height=128,
        width=128,
        dtype="float32",
        crs="EPSG:32650",
        transform=from_origin(300000, 4000000, 10, 10),
        nodata=-32767,
    ) as dst:
        dst.write(elevation, 1)
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.write(image, image.name)
    payload = memory.getvalue()
    size = len(payload) // 2
    parts = [tmp_path / f"static_copernicus_dem_glo30.zip.part{i}" for i in [1, 2]]
    for part, blob in zip(parts, [payload[:size], payload[size:]], strict=True):
        part.write_bytes(blob)
    write_json(
        report / "target_source_audit.json",
        {
            "status": "source_audit_finished",
            "failed_sources": 0,
            "sources": [
                {"family": "static", "path": str(p), "actual_sha256": sha256(p)} for p in parts
            ],
        },
    )
    label_path = tmp_path / "targets.zarr"
    root = zarr.open_group(str(label_path), mode="w")
    root.attrs["patch_ids"] = ["p"]
    targets = root.create_group("targets")
    masks = root.create_group("valid_masks")
    metadata, audited = [], []
    for name, values in [
        ("dem_elevation", elevation),
        ("dem_slope", legacy_slope(elevation, valid)),
    ]:
        targets.array(name, values[None])
        masks.array(name, valid[None])
        metadata.append(
            {
                "family": "static",
                "array": f"targets/{name}",
                "path": str(label_path),
                "registry_order_verified": True,
                "temporal_mode": "static",
            }
        )
        for group, array in [("targets", values), ("valid_masks", valid)]:
            audited.append(
                {
                    "family": "static",
                    "array": f"{group}/{name}",
                    "decoded_values_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
            )
    pd.DataFrame(metadata).to_parquet(data / "targets/manifest.parquet", index=False)
    pd.DataFrame(audited).to_parquet(report / "target_value_audit.parquet", index=False)
    first = audit_dem_geometry(data, report)
    assert first["processed_targets"] == 2 and first["failed_targets"] == 0
    assert first["unsupported_slope_pixels"] == 0
    assert audit_dem_geometry(data, report)["reused_targets"] == 2
    root["targets/dem_slope"][0, 0, 0] = 89
    with pytest.raises(ValueError, match="changed after completed value audit"):
        audit_dem_geometry(data, report)
    parts[0].write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source changed"):
        audit_dem_geometry(data, report)
