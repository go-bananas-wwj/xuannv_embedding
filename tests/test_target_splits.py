from __future__ import annotations

import json
from pathlib import Path

import pytest

from xuannv_embedding.data_process.target_splits import (
    freeze_splits,
    geometry_record,
    read_seen_patches,
)


def test_seen_pool_extraction_includes_nested_ids_but_not_paths(tmp_path: Path) -> None:
    path = tmp_path / "seen.json"
    path.write_text(
        json.dumps(
            {
                "records": [{"patch_id": "parent_32646:400:3000"}],
                "elsewhere": [["city", "test", "parent_32645:300:2000"]],
                "path": "data/parent_32643:400:3000.tif",
            }
        )
    )
    assert read_seen_patches([path]) == {"parent_32646:400:3000", "parent_32645:300:2000"}


def test_new_holdout_excludes_cross_zone_geometry_and_keeps_both_years() -> None:
    pytest.importorskip("shapely")
    from pyproj import Transformer

    index, memberships = [], []

    def add(epsg, col, row, split):
        record = geometry_record(f"parent_{epsg}:{col}:{row}")
        record.update(
            target_row=len(index),
            spatial_component_id=f"component{len(index)}",
            dominant_landcover=95,
        )
        index.append(record)
        for year in (2020, 2021):
            memberships.append(
                {
                    "patch_id": record["patch_id"],
                    "year": year,
                    "split": split,
                    "highres_supported": bool(record["target_row"] % 2),
                }
            )
        return record["patch_id"]

    x, y = Transformer.from_crs(4326, 32645, always_xy=True).transform(90, 35)
    add(32645, int(x // 1280), int(y // 1280), "train")
    add(32645, 300, 2900, "train")
    add(32645, 350, 2800, "train")
    for i in range(6):
        add(32645, 200 + 50 * i, 3500, "validation")
    for i in range(4):
        add(32645, 200 + 50 * i, 4000, "test")
    x, y = Transformer.from_crs(4326, 32646, always_xy=True).transform(90, 35)
    foreign_overlap = add(32646, int(x // 1280), int(y // 1280), "test")
    result = freeze_splits(
        index,
        memberships,
        set(),
        development=2,
        validation=2,
        confirmation=2,
        screen_positions=1,
        expanded_positions=2,
    )
    selected = [row for rows in result["roles"].values() for row in rows]
    assert len(selected) == 6
    assert foreign_overlap not in {row["patch_id"] for row in selected}
    assert all(row["nearest_baseline_boundary_m"] >= 1280 for row in selected)
    assert all(row["nearest_other_role_boundary_m"] >= 1280 for row in selected)
    assert all("dominant_landcover" not in row for row in selected)
    assert set(result["screen_patch_ids"]) <= set(result["expanded_patch_ids"])
    assert result["annual_records"] == {"screen": 2, "expanded": 4}
