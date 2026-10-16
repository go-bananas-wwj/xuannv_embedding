from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xuannv_embedding.data_process.target_audit import (
    audit_spatial_splits,
    local_path,
    read_memberships,
    run_audit,
    scan_static,
    summarize,
    verify_archive,
)


def test_spatial_block_overlap_detected_even_without_position_overlap() -> None:
    index = [
        {"target_row": 0, "patch_id": "a", "macro_id": "block", "spatial_component_id": "c"},
        {"target_row": 1, "patch_id": "b", "macro_id": "block", "spatial_component_id": "c"},
    ]
    memberships = [{"target_row": 0, "split": "train"}, {"target_row": 1, "split": "test"}]
    report = audit_spatial_splits(index, memberships)
    assert report["blocks"]["macro_id"]["train/test"]["shared_blocks"] == ["block"]
    assert report["blocks"]["macro_id"]["train/test"]["right_positions_in_shared_blocks"] == ["b"]


def static_archive(root: Path, name: str, values: np.ndarray, mask: np.ndarray) -> list[dict]:
    zarr = pytest.importorskip("zarr")
    path = root / "static" / f"{name}.zarr.zip"
    path.parent.mkdir(exist_ok=True)
    declarations = []
    with zarr.ZipStore(str(path), mode="w") as store:
        group = zarr.group(store)
        group.attrs["patch_ids"] = ["a", "b", "c"]
        for prefix, data in (("targets", values), ("valid_masks", mask)):
            array = group.create_dataset(f"{prefix}/{name}", data=data, chunks=(2, 128, 128))
            declarations.append(
                {
                    "array": array.path,
                    "shape": list(array.shape),
                    "chunks": list(array.chunks),
                    "dtype": str(array.dtype),
                }
            )
    return declarations


def test_full_scan_masks_missing_values_and_does_not_leak_other_rows(tmp_path: Path) -> None:
    values = np.full((3, 128, 128), 10, dtype=np.float32)
    values[1] = 10000  # held-out row must not affect train normalization
    values[2] = np.nan  # missing is not zero elevation
    mask = np.ones(values.shape, dtype=bool)
    mask[2] = False
    declarations = static_archive(tmp_path, "dem_elevation", values, mask)
    stats = scan_static(tmp_path, "static/dem_elevation.zarr.zip", declarations, ["a", "b", "c"])
    train = summarize(stats, [0, 2])
    assert train["positions_without_valid_pixels"] == 1
    assert train["mean"] == 10
    assert train["population_std"] == 0
    assert train["valid_fraction"] == 0.5
    assert train["nonfinite_valid_pixels"] == 0
    assert summarize(stats, [2])["mean"] is None
    with pytest.raises(ValueError, match="row identities"):
        scan_static(tmp_path, "static/dem_elevation.zarr.zip", declarations, ["b", "a", "c"])


def test_worldcover_rejects_valid_zero_and_counts_rare_codes(tmp_path: Path) -> None:
    values = np.full((3, 128, 128), 95, dtype=np.uint8)
    values[1] = 100
    values[2] = 0
    mask = np.ones(values.shape, dtype=bool)
    declarations = static_archive(tmp_path, "worldcover_2020", values, mask)
    stats = scan_static(tmp_path, "static/worldcover_2020.zarr.zip", declarations, ["a", "b", "c"])
    summary = summarize(stats, [0, 1, 2])
    assert summary["class_pixels"]["95"] == 128 * 128
    assert summary["class_pixels"]["100"] == 128 * 128
    assert summary["illegal_valid_pixels"] == 128 * 128


def test_archive_hash_failure_is_not_accepted(tmp_path: Path) -> None:
    (tmp_path / "data.zip").write_bytes(b"bad")
    receipt = verify_archive(tmp_path, {"path": "data.zip", "bytes": 3, "sha256": "0" * 64})
    assert not receipt["passed"]
    for path in ("../outside", "/tmp/outside"):
        with pytest.raises(ValueError, match="escapes"):
            local_path(tmp_path, path)


def test_manifest_checksum_failure_aborts_instead_of_counting_partial_rows(tmp_path: Path) -> None:
    path = tmp_path / "2020.train.manifest.jsonl"
    path.write_text("")
    Path(str(path) + ".meta.json").write_text(
        json.dumps(
            {
                "months": [f"2020-{month:02d}" for month in range(1, 13)],
                "sha256": "0" * 64,
                "record_count": 0,
            }
        )
    )
    with pytest.raises(ValueError, match="checksum/count"):
        read_memberships(tmp_path, {})


@pytest.mark.parametrize("cross_year_overlap", [False, True])
def test_acceptance_roundtrip_counts_static_train_once_and_detects_cross_year_leak(
    tmp_path: Path, cross_year_overlap: bool
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rasterio = pytest.importorskip("rasterio")
    zarr = pytest.importorskip("zarr")
    from rasterio.transform import from_origin

    from xuannv_embedding.data_process.target_audit import digest

    data = tmp_path / "data"
    targets, annual = data / "targets", data / "annual"
    targets.mkdir(parents=True)
    annual.mkdir()
    patches = [f"parent_32643:{300 + i}:3300" for i in range(3)]
    values = np.full((3, 128, 128), 10, dtype=np.float32)
    values[1:] = 10000
    specs = static_archive(targets, "dem_elevation", values, np.ones_like(values, dtype=bool))
    filename = "static/dem_elevation.zarr.zip"
    # Rewrite only metadata in this test fixture; production input archives stay read-only.
    with zarr.ZipStore(str(targets / filename), mode="a") as store:
        zarr.open_group(store, mode="a").attrs["patch_ids"] = patches
    index = []
    for i, patch in enumerate(patches):
        x = (300 + i) * 1280
        bounds = [x, 3300 * 1280, x + 1280, 3301 * 1280]
        index.append(
            {
                "target_row": i,
                "patch_id": patch,
                "grid_epsg": 32643,
                "utm_bounds": bounds,
                "macro_id": "one_block",
                "spatial_component_id": "one_component",
            }
        )
        with rasterio.open(
            data / f"{patch}.tif",
            "w",
            driver="GTiff",
            width=128,
            height=128,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
            transform=from_origin(x, bounds[3], 10, 10),
        ) as ds:
            ds.write(np.ones((1, 128, 128), dtype=np.uint16))
    # Nontrivial parquet row order proves target_row controls the join.
    pq.write_table(pa.Table.from_pylist(index[::-1]), targets / "index.parquet")
    for year in (2020, 2021):
        for i, split in enumerate(("train", "validation", "test")):
            selected = 1 - i if cross_year_overlap and year == 2021 and i < 2 else i
            patch = patches[selected]
            record = {
                "patch_id": patch,
                "grid": {"parent_key": patch.removeprefix("parent_")},
                "sources": {"s2": [f"{patch}.tif"]},
                "quality": {
                    "highres_supported": False,
                    "ms5m_supported": False,
                    "pan2m_supported": False,
                },
            }
            path = annual / f"{year}.{split}.manifest.jsonl"
            path.write_text(json.dumps(record) + "\n")
            Path(str(path) + ".meta.json").write_text(
                json.dumps(
                    {
                        "months": [f"{year}-{m:02d}" for m in range(1, 13)],
                        "record_count": 1,
                        "sha256": digest(path),
                    }
                )
            )
    (targets / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "xuannv.static-target-publication.v1",
                "version": "fixture",
                "image_version": "fixture",
                "patches": 3,
                "years": [2020, 2021],
                "index": "index.parquet",
                "files": [
                    {
                        "path": filename,
                        "bytes": (targets / filename).stat().st_size,
                        "sha256": digest(targets / filename),
                    }
                ],
                "arrays": [{**spec, "file": filename} for spec in specs],
            }
        )
    )
    (targets / "README.md").write_text("Fixture")
    (targets / "validation.json").write_text("{}")
    output = tmp_path / "audit"
    contract = run_audit(targets, annual, data, output, workers=1)
    assert contract["data_checks_passed"] is not cross_year_overlap
    assert not contract["training_ready"]
    stats = json.loads((output / "train_statistics.json").read_text())["targets"]["dem_elevation"]
    assert stats["positions"] == (0 if cross_year_overlap else 1)
    assert stats["mean"] == (None if cross_year_overlap else 10)
