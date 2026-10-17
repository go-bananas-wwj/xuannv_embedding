from __future__ import annotations

import hashlib
import json
import os
import pickle

import numpy as np
import pytest

from xuannv_embedding.data.targets import AnnualTargetReader, collate_targets
from xuannv_embedding.data_process.target_audit import digest, write_json


@pytest.fixture
def target_bundle(tmp_path, request):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    zarr = pytest.importorskip("zarr")
    from test_native_dem import make_native_archive

    from xuannv_embedding.data_process.target_prepare import native_cache
    from xuannv_embedding.data_process.target_splits import geometry_record

    paths, first = make_native_archive(tmp_path, split_inside_tiff=False)
    epsg, col, row = first["patch_id"].removeprefix("parent_").split(":")
    second = geometry_record(f"parent_{epsg}:{int(col) + 1}:{row}")
    count = getattr(request, "param", 2)
    positions = [first, second]
    if count == 3:
        positions.append(geometry_record(f"parent_{epsg}:{int(col) + 2}:{row}"))
    index = [dict(value, target_row=i) for i, value in enumerate(positions)]
    root, prepared = tmp_path / "targets", tmp_path / "prepared"
    (root / "static").mkdir(parents=True)
    prepared.mkdir()
    pq.write_table(pa.Table.from_pylist(index), root / "index.parquet")
    integrity = {
        "all_zip_member_crc_passed": True,
        "files": [
            dict(
                path=str(path),
                bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,
                sha256=digest(path),
            )
            for path in paths
        ],
    }
    native_cache(index, integrity, prepared / "native_dem", workers=1)
    entries = []
    for year, code in [(2020, 50), (2021, 80)]:
        name = f"worldcover_{year}"
        relative = f"static/{name}.zarr.zip"
        values = np.full((count, 128, 128), code, dtype=np.uint8)
        valid = np.ones_like(values, dtype=bool)
        values[:, 0, 0], valid[:, 0, 0] = 0, False
        with zarr.ZipStore(str(root / relative), mode="w") as store:
            group = zarr.group(store=store)
            group.create_dataset(f"targets/{name}", data=values)
            group.create_dataset(f"valid_masks/{name}", data=valid)
        entries.append(
            dict(
                path=relative,
                bytes=(root / relative).stat().st_size,
                sha256=digest(root / relative),
            )
        )
    write_json(root / "manifest.json", {"patches": count, "files": entries})
    write_json(
        prepared / "experiment_splits.json",
        {
            "roles": {"development": [positions[-1]], "validation": [], "confirmation": []},
            "screen_patch_ids": [first["patch_id"]],
            "expanded_patch_ids": [value["patch_id"] for value in positions[:-1]],
            "allowed_train_patch_ids": [value["patch_id"] for value in positions[:-1]],
        },
    )
    write_json(
        prepared / "train_statistics.screen.json",
        {
            "scope": "screen",
            "unique_positions": 1,
            "target_rows_sha256": hashlib.sha256(json.dumps([0]).encode()).hexdigest(),
            "targets": {"dem_elevation": {"mean": 150, "population_std": 20}},
        },
    )
    receipt = {
        "status": "engineering_data_prepared",
        "target_manifest_sha256": digest(root / "manifest.json"),
        "target_index_sha256": digest(root / "index.parquet"),
        "splits_sha256": digest(prepared / "experiment_splits.json"),
        "train_statistics_sha256": {"screen": digest(prepared / "train_statistics.screen.json")},
        "native_manifest_sha256": digest(prepared / "native_dem/manifest.json"),
        "formal_training_released": False,
    }
    write_json(prepared / "preparation.json", receipt)
    return root, prepared, digest(prepared / "preparation.json"), index


def test_target_year_mapping_missing_mask_and_sealed_roles(target_bundle):
    root, prepared, sha, index = target_bundle
    with AnnualTargetReader(root, prepared, preparation_sha256=sha) as reader:
        first = reader.read(index[0]["patch_id"], 2020)
        second = reader.read(index[0]["patch_id"], 2021)
        assert first["dem_elevation"]["values"].shape == (32, 32)
        assert first["worldcover"]["values"][1, 1] == 4
        assert second["worldcover"]["values"][1, 1] == 7
        assert first["worldcover"]["values"][0, 0] == -100
        assert not first["worldcover"]["valid"][0, 0]
        batch = collate_targets([first, second])
        assert batch["worldcover"]["values"].shape == (2, 128, 128)
        with pytest.raises(ValueError, match="sealed"):
            reader.read(index[1]["patch_id"], 2020, role="development")
        with pytest.raises(ValueError, match="outside"):
            reader.read(index[1]["patch_id"], 2020)
        with pytest.raises(ValueError, match="year"):
            reader.read(index[0]["patch_id"], True)
        clone = pickle.loads(pickle.dumps(reader))
        try:
            assert not clone.groups and not clone.stores
            np.testing.assert_array_equal(
                clone.read(index[0]["patch_id"], 2020)["dem_elevation"]["values"],
                first["dem_elevation"]["values"],
            )
        finally:
            clone.close()
    with AnnualTargetReader(
        root, prepared, preparation_sha256=sha, allow_evaluation=True
    ) as reader:
        assert reader.read(index[1]["patch_id"], 2020, role="development")


def test_target_pin_and_native_row_tampering_are_rejected(target_bundle):
    root, prepared, sha, index = target_bundle
    with pytest.raises(ValueError, match="preparation identity"):
        AnnualTargetReader(root, prepared, preparation_sha256="0" * 64)
    import zarr

    group = zarr.open_group(str(prepared / "native_dem/dem40m.zarr"), mode="r+")
    group["values"][0, 0, 0] += 1
    with AnnualTargetReader(root, prepared, preparation_sha256=sha) as reader:
        with pytest.raises(ValueError, match="row integrity"):
            reader.read(index[0]["patch_id"], 2020)


def test_open_worldcover_archive_changes_are_detected(target_bundle):
    root, prepared, sha, index = target_bundle
    with AnnualTargetReader(root, prepared, preparation_sha256=sha) as reader:
        reader.read(index[0]["patch_id"], 2020)
        path = root / "static/worldcover_2020.zarr.zip"
        status = path.stat()
        os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000))
        with pytest.raises(ValueError, match="changed after"):
            reader.read(index[0]["patch_id"], 2020)
