from __future__ import annotations

from xuannv_embedding.data_process import reorder as MODULE


def load_module() -> object:
    return MODULE


def test_requested_order_moves_old_shard_eight_to_new_shard_one() -> None:
    """The user-facing shard sequence must match the agreed review order exactly."""
    module = load_module()

    remap = module.build_shard_remap([8, 1, 2, 3, 4, 5, 6, 7, 9, 10])

    assert remap == {8: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 9: 9, 10: 10}


def test_reordered_summary_has_new_id_and_preserves_old_source_id() -> None:
    """The manifest must make the renumbering auditable without changing any counts or bounds."""
    module = load_module()
    summary = {
        "shard_id": 8,
        "shape_count": 578_578,
        "centroid_wgs84": {"longitude": 112.3, "latitude": 35.8},
        "longitude_range": [104.3, 122.7],
        "latitude_range": [32.0, 39.2],
        "counts_by_grid_id": {"utm50n": 1},
    }

    reordered = module.reorder_summary(summary, new_shard_id=1)

    assert reordered["shard_id"] == 1
    assert reordered["source_shard_id"] == 8
    assert reordered["shape_count"] == summary["shape_count"]
    assert reordered["longitude_range"] == summary["longitude_range"]
