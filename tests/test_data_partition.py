from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
