from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xuannv_embedding.data_process.local_inventory import (
    InventoryError,
    build_grouped_split,
    read_sampled_grid,
)
from xuannv_embedding.data_process.preflight import classify_source


def _write_grid(root: Path) -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "patch_id": f"parent_326{43 + index % 3}:{index}:1",
                "macro_id": f"macro-{index}",
                "grid_id": f"utm{43 + index % 3}n",
                "grid_epsg": 32643 + index % 3,
                "sampled": True,
            }
        )
    path = root / "sampled" / "utm43n" / "part-00000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_reads_only_sampled_frozen_grid_rows(tmp_path: Path) -> None:
    _write_grid(tmp_path)

    table = read_sampled_grid(tmp_path)

    assert table.num_rows == 20
    assert set(table.column_names) >= {"patch_id", "macro_id", "grid_id", "grid_epsg"}


def test_grouped_split_is_exact_deterministic_and_macro_disjoint(tmp_path: Path) -> None:
    _write_grid(tmp_path)
    candidates = read_sampled_grid(tmp_path)

    first = build_grouped_split(candidates, train_count=12, val_count=4, test_count=4, seed=42)
    second = build_grouped_split(candidates, train_count=12, val_count=4, test_count=4, seed=42)

    assert first.to_pylist() == second.to_pylist()
    counts = first.group_by("split").aggregate([("patch_id", "count")]).to_pylist()
    assert {row["split"]: row["patch_id_count"] for row in counts} == {
        "train": 12,
        "val": 4,
        "test": 4,
    }
    macro_splits: dict[str, set[str]] = {}
    for row in first.select(["macro_id", "split"]).to_pylist():
        macro_splits.setdefault(row["macro_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in macro_splits.values())


def test_grouped_split_refuses_impossible_exact_counts() -> None:
    candidates = pa.Table.from_pylist(
        [
            {"patch_id": f"p{index}", "macro_id": "same", "grid_id": "utm50n", "grid_epsg": 32650}
            for index in range(3)
        ]
    )

    with pytest.raises(InventoryError, match="无法.*精确"):
        build_grouped_split(candidates, train_count=1, val_count=1, test_count=1, seed=42)


@pytest.mark.parametrize(
    ("role", "time_precision", "already_resampled", "provenance", "expected"),
    [
        ("dense", "month", True, "verified", "monthly_patch_already_resampled"),
        ("dense", "month", False, "verified", "monthly_patch_native_grid"),
        ("highres", "exact", False, "verified", "raw_scene"),
        ("highres", "exact", False, "legacy_unverified", "legacy_unverified"),
    ],
)
def test_preflight_classifies_processing_state(
    role: str,
    time_precision: str,
    already_resampled: bool,
    provenance: str,
    expected: str,
) -> None:
    assert classify_source(role, time_precision, already_resampled, provenance) == expected
