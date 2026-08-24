from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xuannv_embedding.data_process import partition as MODULE


def load_module() -> object:
    return MODULE


def test_joint_partition_assigns_every_shape_once_at_each_exact_capacity() -> None:
    """Catches a joint partition that loses, duplicates, or miscounts Shapes."""
    module = load_module()
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    assignment = module.partition_equal_capacity(
        x=x,
        y=y,
        shard_capacities=[2, 2, 2, 2, 2, 2],
    )

    assert assignment.min() == 1
    assert assignment.max() == 6
    assert sorted(np.bincount(assignment, minlength=7)[1:].tolist()) == [2, 2, 2, 2, 2, 2]


def test_region_bounds_feature_is_a_wgs84_bbox_with_machine_readable_metadata() -> None:
    """The lightweight range file must not be confused with exact Shape membership."""
    module = load_module()
    summary = {
        "shard_id": 1,
        "shape_count": 42,
        "centroid_wgs84": {"longitude": 101.5, "latitude": 30.5},
        "longitude_range": [100.0, 103.0],
        "latitude_range": [29.0, 32.0],
        "counts_by_grid_id": {"utm48n": 42},
    }

    feature = module.region_bounds_feature(summary)

    assert feature["type"] == "Feature"
    assert feature["properties"]["shard_id"] == "shard_01"
    assert feature["properties"]["geometry_role"] == "center_coordinate_bbox_index"
    assert feature["properties"]["shape_count"] == 42
    assert feature["geometry"] == {
        "type": "Polygon",
        "coordinates": [
            [[100.0, 29.0], [103.0, 29.0], [103.0, 32.0], [100.0, 32.0], [100.0, 29.0]]
        ],
    }


def test_write_region_indices_creates_root_and_per_shard_geojson(tmp_path: Path) -> None:
    """Existing deliveries can gain machine-readable range indexes without rewriting Parquet."""
    module = load_module()
    summary = {
        "shard_id": 1,
        "shape_count": 42,
        "centroid_wgs84": {"longitude": 101.5, "latitude": 30.5},
        "longitude_range": [100.0, 103.0],
        "latitude_range": [29.0, 32.0],
        "counts_by_grid_id": {"utm48n": 42},
    }
    (tmp_path / "shards" / "shard_01").mkdir(parents=True)
    (tmp_path / "tenfold_partition_manifest.json").write_text(
        json.dumps({"shards": [summary]}), encoding="utf-8"
    )

    module.write_region_indices(tmp_path)

    root_index = json.loads((tmp_path / "china_tenfold_shard_regions.geojson").read_text())
    refreshed_manifest = json.loads((tmp_path / "tenfold_partition_manifest.json").read_text())
    shard_index = json.loads(
        (tmp_path / "shards" / "shard_01" / "region_bounds.geojson").read_text()
    )
    assert root_index["features"] == [shard_index]
    assert shard_index["properties"]["exact_membership"].startswith("Use the GeoParquet")
    assert (
        refreshed_manifest["region_bounds_index"]["path"] == "china_tenfold_shard_regions.geojson"
    )


def test_region_index_notes_are_added_to_existing_readmes(tmp_path: Path) -> None:
    """A refreshed delivery explains how to use the new lightweight boundary indexes."""
    module = load_module()
    (tmp_path / "README.md").write_text("# Package\n", encoding="utf-8")
    shard_readme = tmp_path / "shards" / "shard_01" / "README.md"
    shard_readme.parent.mkdir(parents=True)
    shard_readme.write_text("# shard_01\n", encoding="utf-8")

    module.add_region_index_readme_notes(tmp_path, [1])

    assert "china_tenfold_shard_regions.geojson" in (tmp_path / "README.md").read_text()
    assert "region_bounds.geojson" in shard_readme.read_text()


def _write_tenfold_fixture(root: Path, parent_root: Path) -> None:
    shards = []
    parent_keys = []
    for shard_id in range(1, 11):
        key = f"326{shard_id:02d}:1:1"
        parent_keys.append(key)
        shard_root = root / "shards" / f"shard_{shard_id:02d}" / f"utm{shard_id:02d}n"
        shard_root.mkdir(parents=True)
        pq.write_table(
            pa.table({"parent_key": [key], "shard_id": [shard_id]}),
            shard_root / "part-00000.parquet",
        )
        shards.append({"shard_id": shard_id, "shape_count": 1})
    (root / "tenfold_partition_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "china_full_grid_tenfold_spatial_partition_v4_reordered",
                "source_parent_count": 10,
                "validation": {"shard_counts": {str(index): 1 for index in range(1, 11)}},
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )
    all_root = parent_root / "all" / "utm43n"
    all_root.mkdir(parents=True)
    pq.write_table(pa.table({"parent_key": parent_keys}), all_root / "part-00000.parquet")


def test_tenfold_read_only_audit_proves_exact_parent_membership(tmp_path: Path) -> None:
    delivery = tmp_path / "delivery"
    parent = tmp_path / "parent"
    _write_tenfold_fixture(delivery, parent)

    report = MODULE.audit_tenfold_delivery(delivery, parent_grid_root=parent, batch_size=3)

    assert report["passed"] is True
    assert report["actual_count"] == 10
    assert report["unique_parent_count"] == 10
    assert report["duplicate_parent_count"] == 0
    assert report["missing_parent_count"] == 0
    assert report["unknown_parent_count"] == 0


def test_tenfold_audit_rejects_duplicate_and_wrong_shard_id(tmp_path: Path) -> None:
    delivery = tmp_path / "delivery"
    parent = tmp_path / "parent"
    _write_tenfold_fixture(delivery, parent)
    bad_path = delivery / "shards" / "shard_02" / "utm02n" / "part-00000.parquet"
    pq.write_table(
        pa.table({"parent_key": ["32601:1:1"], "shard_id": [9]}),
        bad_path,
    )

    report = MODULE.audit_tenfold_delivery(delivery, parent_grid_root=parent, batch_size=3)

    assert report["passed"] is False
    assert report["duplicate_parent_count"] == 1
    assert report["wrong_shard_id_count"] == 1
    assert report["missing_parent_count"] == 1
