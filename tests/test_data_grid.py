from __future__ import annotations

import hashlib
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest
from pyproj import Transformer
from shapely import normalize, set_precision
from shapely.affinity import translate
from shapely.geometry import box

from xuannv_embedding.data_process import grid as MODULE


def _macro() -> dict[str, int | str]:
    return {
        "grid_epsg": 32650,
        "grid_id": "utm50n",
        "macro_col": 34,
        "macro_row": 344,
    }


def _boundary():
    return box(116.28, 39.83, 116.31, 39.86)


def synthetic_parent_records(count: int) -> list[dict[str, object]]:
    """Return canonical records spanning two macro cells for writer tests."""
    spec = MODULE.GridSpec(boundary_version="test")
    first_macro = _macro()
    second_macro = {**first_macro, "macro_col": first_macro["macro_col"] + 1}
    records: list[dict[str, object]] = []
    for index in range(count):
        macro = first_macro if index < count // 2 else second_macro
        grid_col = int(macro["macro_col"]) * spec.macro_side_patches + index % 10
        grid_row = int(macro["macro_row"]) * spec.macro_side_patches + index // 10
        records.append(
            MODULE.build_patch_record(
                macro,
                grid_col,
                grid_row,
                116.28 + index * 0.001,
                39.83 + index * 0.001,
                spec,
            )
        )
    return records


def test_writer_partitions_all_sampled_and_unsampled(tmp_path: Path) -> None:
    records = synthetic_parent_records(count=12)
    sampled = {str(records[1]["parent_key"]), str(records[7]["parent_key"])}

    summary = MODULE.write_zone_records(iter(records), sampled, tmp_path, batch_size=5)

    assert summary.all_count == 12
    assert summary.sampled_count == 2
    assert summary.unsampled_count == 10
    assert summary.sampled_count + summary.unsampled_count == summary.all_count

    def read_partition(name: str) -> gpd.GeoDataFrame:
        parts = sorted((tmp_path / name / "utm50n").glob("*.parquet"))
        frames = [gpd.read_parquet(part) for part in parts]
        return gpd.GeoDataFrame(pd.concat(frames), crs="EPSG:4326")

    all_records = read_partition("all")
    sampled_records = read_partition("sampled")
    unsampled_records = read_partition("unsampled")

    assert all_records.crs.to_epsg() == 4326
    assert set(sampled_records["parent_key"]).isdisjoint(unsampled_records["parent_key"])
    assert set(sampled_records["parent_key"]) == sampled
    assert len(all_records) == len(sampled_records) + len(unsampled_records)
    assert {"patch_id", "parent_key", "schema_version", "atlas_version", "boundary_version"} <= set(
        all_records.columns
    )
    assert all_records.geometry.geom_type.eq("Polygon").all()

    shapefile_parts = sorted((tmp_path / "all" / "utm50n").glob("*.shp"))
    shapefile_records = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(part) for part in shapefile_parts]), crs="EPSG:4326"
    )
    assert shapefile_records.crs.to_epsg() == 4326
    assert len(shapefile_records) == 12
    assert {"PATCH_ID", "UTM_EPSG", "GRID_COL", "GRID_ROW", "MACRO_ID", "SAMPLED"} <= set(
        shapefile_records.columns
    )


def test_writer_rejects_shapefile_larger_than_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "SHAPEFILE_MAX_BYTES", 1)

    with pytest.raises(ValueError, match="one Shapefile feature exceeds"):
        MODULE.write_zone_records(synthetic_parent_records(count=1), set(), tmp_path, batch_size=1)

    assert not list(tmp_path.rglob("*.shp"))
    assert not list(tmp_path.rglob("*.parquet"))


def test_writer_splits_shapefiles_at_component_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = synthetic_parent_records(count=12)
    probe = tmp_path / "probe.shp"
    MODULE._shapefile_records(records[:1], set()).to_file(probe, index=False)
    one_record_component = max(
        probe.with_suffix(suffix).stat().st_size for suffix in (".shp", ".shx", ".dbf")
    )
    component_cap = math.ceil(one_record_component / 0.9)
    monkeypatch.setattr(MODULE, "SHAPEFILE_MAX_BYTES", component_cap)

    MODULE.write_zone_records(records, set(), tmp_path / "output", batch_size=12)

    parts = sorted((tmp_path / "output" / "all" / "utm50n").glob("*.shp"))
    assert len(parts) > 1
    assert all("rowblock-" in part.name and "part-" in part.name for part in parts)
    for part in parts:
        assert all(
            part.with_suffix(suffix).stat().st_size < component_cap
            for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg")
            if part.with_suffix(suffix).exists()
        )


def test_geoparquet_has_canonical_fields_and_zstd_metadata(tmp_path: Path) -> None:
    record = synthetic_parent_records(count=1)[0]
    MODULE.write_zone_records([record], set(), tmp_path, batch_size=1)
    parquet_path = next((tmp_path / "all" / "utm50n").glob("*.parquet"))
    result = gpd.read_parquet(parquet_path)
    row = result.iloc[0]

    assert {"wgs84_bounds", "identity_hash", "footprint_hash"} <= set(result.columns)
    assert tuple(row["wgs84_bounds"]) == pytest.approx(row.geometry.bounds)
    expected_identity = (
        f"{row['atlas_version']}:{row['grid_epsg']}:{row['grid_col']}:{row['grid_row']}"
    )
    assert row["identity_hash"] == hashlib.sha256(expected_identity.encode("utf-8")).hexdigest()
    expected_footprint = normalize(set_precision(row.geometry, 1e-9)).wkb
    assert row["footprint_hash"] == hashlib.sha256(expected_footprint).hexdigest()

    metadata = pq.ParquetFile(parquet_path).metadata
    assert {
        metadata.row_group(0).column(column).compression
        for column in range(metadata.row_group(0).num_columns)
    } == {"ZSTD"}


def test_membership_audit_requires_every_sample_exactly_once() -> None:
    atlas = ["32650:1:1", "32650:1:2", "32650:1:3"]
    sampled = ["32650:1:1", "32650:1:3"]

    audit = MODULE.audit_sample_membership(atlas, sampled)

    assert audit["all_count"] == 3
    assert audit["sampled_count"] == 2
    assert audit["unsampled_count"] == 1
    assert audit["matched"] == 2
    assert audit["missing"] == []
    assert audit["duplicate_atlas_keys"] == []


def test_membership_audit_reports_missing_and_duplicated_sampled_cells() -> None:
    atlas = ["32650:1:1", "32650:1:1", "32650:1:2"]
    sampled = ["32650:1:1", "32650:1:3", "32650:1:3"]

    audit = MODULE.audit_sample_membership(atlas, sampled)

    assert audit["matched"] == 0
    assert audit["missing"] == ["32650:1:3"]
    assert audit["duplicate_atlas_keys"] == ["32650:1:1"]
    assert audit["duplicate_sampled_keys"] == ["32650:1:3"]


def test_partitioned_audit_detects_altered_footprints_and_cross_zone_overlap(
    tmp_path: Path,
) -> None:
    records = synthetic_parent_records(count=2)
    sampled_registry = []
    for record in records:
        geometry = MODULE._wgs84_geometry(record)
        sampled_registry.append(
            {
                "grid_epsg": record["grid_epsg"],
                "grid_col": record["grid_col"],
                "grid_row": record["grid_row"],
                "canonical_wgs84_footprint_hash": hashlib.sha256(
                    normalize(set_precision(geometry, 1e-9)).wkb
                ).hexdigest(),
            }
        )
    MODULE.write_zone_records(records, {str(records[0]["parent_key"])}, tmp_path, batch_size=1)

    parquet_path = next((tmp_path / "all" / "utm50n").glob("*.parquet"))
    frame = gpd.read_parquet(parquet_path)
    frame.loc[0, "footprint_hash"] = "0" * 64
    frame.loc[0, "geometry"] = frame.loc[0, "geometry"].buffer(0.0001)
    frame.to_parquet(parquet_path, index=False, compression="zstd")

    audit = MODULE.audit_grid_package(tmp_path, sampled_registry, batch_size=1)

    assert audit["all_count"] == 2
    assert audit["sampled_count"] == 1
    assert audit["unsampled_count"] == 1
    assert audit["hash_mismatches"]["footprint_hash"] == 1
    assert audit["hash_mismatches"]["sampled_registry_footprint_hash"] == 1
    assert audit["max_footprint_coordinate_difference"] > 0
    assert audit["invalid_geometry_count"] == 1


def test_partitioned_audit_reports_same_and_cross_zone_positive_area_overlaps(
    tmp_path: Path,
) -> None:
    first = synthetic_parent_records(count=1)[0]
    to_utm49 = Transformer.from_crs(4326, 32649, always_xy=True)
    to_wgs84 = Transformer.from_crs(32649, 4326, always_xy=True)
    easting, northing = to_utm49.transform(111.0, 39.84)
    grid_col = math.floor(easting / MODULE.PARENT_SIDE_METERS)
    grid_row = math.floor(northing / MODULE.PARENT_SIDE_METERS)
    longitude, latitude = to_wgs84.transform(
        (grid_col + 0.5) * MODULE.PARENT_SIDE_METERS,
        (grid_row + 0.5) * MODULE.PARENT_SIDE_METERS,
    )
    second = MODULE.build_patch_record(
        {
            "grid_epsg": 32649,
            "grid_id": "utm49n",
            "macro_col": grid_col // 10,
            "macro_row": grid_row // 10,
        },
        grid_col,
        grid_row,
        longitude,
        latitude,
        MODULE.GridSpec(boundary_version="test"),
    )
    records = [first, second]
    for record in records:
        MODULE.write_zone_records([record], set(), tmp_path, batch_size=1)

    first_path = next((tmp_path / "all" / "utm50n").glob("*.parquet"))
    second_path = next((tmp_path / "all" / "utm49n").glob("*.parquet"))
    first_frame = gpd.read_parquet(first_path)
    second_frame = gpd.read_parquet(second_path)
    second_frame.loc[0, "geometry"] = first_frame.loc[0, "geometry"]
    second_frame.to_parquet(second_path, index=False, compression="zstd")

    audit = MODULE.audit_grid_package(tmp_path, [], batch_size=1)

    assert audit["same_zone_positive_overlap_count"] == 0
    assert audit["cross_zone_overlap_violation_count"] == 1
    assert audit["max_cross_zone_overlap_fraction"] == pytest.approx(1.0)


def test_partitioned_audit_reports_same_zone_positive_area_overlap(tmp_path: Path) -> None:
    records = synthetic_parent_records(count=2)
    MODULE.write_zone_records(records, set(), tmp_path, batch_size=1)
    paths = sorted((tmp_path / "all" / "utm50n").glob("*.parquet"))
    frame = gpd.read_parquet(paths[0])
    frame.loc[1, "geometry"] = frame.loc[0, "geometry"]
    frame.to_parquet(paths[0], index=False, compression="zstd")

    audit = MODULE.audit_grid_package(tmp_path, [], batch_size=1)

    assert audit["same_zone_positive_overlap_count"] == 1


def test_utm_seam_policy_accepts_small_adjacent_boundary_overlap() -> None:
    seam = MODULE.assess_utm_seam_overlap_policy(
        overlap_pair_count=10_411,
        overlap_area_m2=2_842_239_086.0,
        total_parent_count=5_785_781,
        non_adjacent_pair_count=0,
        owner_order_mismatch_count=0,
        off_seam_pair_count=0,
        max_pair_overlap_fraction=0.7419,
    )

    assert seam["passed"] is True
    assert seam["global_duplicate_area_fraction"] == pytest.approx(0.00029983248)
    assert seam["max_pair_overlap_fraction"] == pytest.approx(0.7419)


def test_utm_seam_policy_rejects_non_adjacent_or_excessive_overlap() -> None:
    seam = MODULE.assess_utm_seam_overlap_policy(
        overlap_pair_count=2,
        overlap_area_m2=20_000_000.0,
        total_parent_count=100,
        non_adjacent_pair_count=1,
        owner_order_mismatch_count=0,
        off_seam_pair_count=0,
        max_pair_overlap_fraction=0.9,
    )

    assert seam["passed"] is False


def test_reconcile_utm_seam_audit_does_not_mask_other_failures() -> None:
    base = {
        "passed": False,
        "missing_sampled_count": 1,
        "cross_zone_overlap_violation_count": 7,
        "hash_mismatches": {
            "identity_hash": 0,
            "footprint_hash": 0,
            "sampled_registry_footprint_hash": 0,
        },
    }
    seam = MODULE.assess_utm_seam_overlap_policy(
        overlap_pair_count=7,
        overlap_area_m2=1.0,
        total_parent_count=100,
        non_adjacent_pair_count=0,
        owner_order_mismatch_count=0,
        off_seam_pair_count=0,
        max_pair_overlap_fraction=0.5,
    )

    with pytest.raises(ValueError, match="other blocking failures"):
        MODULE.reconcile_utm_seam_audit(base, seam)


def test_partitioned_audit_rejects_shifted_footprint_with_recomputed_hash(tmp_path: Path) -> None:
    record = synthetic_parent_records(count=1)[0]
    MODULE.write_zone_records([record], set(), tmp_path, batch_size=1)
    parquet_path = next((tmp_path / "all" / "utm50n").glob("*.parquet"))
    frame = gpd.read_parquet(parquet_path)
    shifted_geometry = translate(frame.loc[0, "geometry"], xoff=0.0001)
    frame.loc[0, "geometry"] = shifted_geometry
    frame.loc[0, "footprint_hash"] = hashlib.sha256(
        normalize(set_precision(shifted_geometry, 1e-9)).wkb
    ).hexdigest()
    frame.to_parquet(parquet_path, index=False, compression="zstd")

    audit = MODULE.audit_grid_package(tmp_path, [], batch_size=1)

    assert audit["hash_mismatches"]["footprint_hash"] == 1
    assert audit["footprint_coordinate_mismatch_count"] == 1
    assert audit["passed"] is False


def test_partitioned_audit_derives_all_footprint_from_integer_grid_coordinates(
    tmp_path: Path,
) -> None:
    record = synthetic_parent_records(count=1)[0]
    MODULE.write_zone_records([record], set(), tmp_path, batch_size=1)
    parquet_path = next((tmp_path / "all" / "utm50n").glob("*.parquet"))
    frame = gpd.read_parquet(parquet_path)
    shifted_bounds = [value + MODULE.PARENT_SIDE_METERS for value in frame.loc[0, "utm_bounds"]]
    shifted_geometry = MODULE._wgs84_geometry(
        {"grid_epsg": frame.loc[0, "grid_epsg"], "utm_bounds": shifted_bounds}
    )
    frame.at[0, "utm_bounds"] = shifted_bounds
    frame.at[0, "wgs84_bounds"] = list(shifted_geometry.bounds)
    frame.loc[0, "geometry"] = shifted_geometry
    frame.loc[0, "footprint_hash"] = hashlib.sha256(
        normalize(set_precision(shifted_geometry, 1e-9)).wkb
    ).hexdigest()
    frame.to_parquet(parquet_path, index=False, compression="zstd")

    audit = MODULE.audit_grid_package(tmp_path, [], batch_size=1)

    assert audit["stored_utm_bounds_mismatch_count"] == 1
    assert audit["hash_mismatches"]["footprint_hash"] == 1
    assert audit["passed"] is False


def test_partitioned_audit_rejects_swapped_sampled_and_unsampled_children(tmp_path: Path) -> None:
    records = synthetic_parent_records(count=2)
    sampled_key = str(records[0]["parent_key"])
    MODULE.write_zone_records(records, {sampled_key}, tmp_path, batch_size=2)
    sampled_path = next((tmp_path / "sampled" / "utm50n").glob("*.parquet"))
    unsampled_path = next((tmp_path / "unsampled" / "utm50n").glob("*.parquet"))
    sampled_payload = sampled_path.read_bytes()
    unsampled_payload = unsampled_path.read_bytes()
    sampled_path.write_bytes(unsampled_payload)
    unsampled_path.write_bytes(sampled_payload)

    audit = MODULE.audit_grid_package(
        tmp_path,
        [
            {
                "grid_epsg": records[0]["grid_epsg"],
                "grid_col": records[0]["grid_col"],
                "grid_row": records[0]["grid_row"],
            }
        ],
        batch_size=1,
    )

    assert audit["exact_partition_membership_mismatch_count"] == 2
    assert audit["passed"] is False


def test_partitioned_audit_rejects_rehashed_child_metadata_and_geometry(tmp_path: Path) -> None:
    record = synthetic_parent_records(count=1)[0]
    MODULE.write_zone_records([record], {str(record["parent_key"])}, tmp_path, batch_size=1)
    parquet_path = next((tmp_path / "sampled" / "utm50n").glob("*.parquet"))
    frame = gpd.read_parquet(parquet_path)
    shifted_geometry = translate(frame.loc[0, "geometry"], xoff=0.0001)
    frame.loc[0, "geometry"] = shifted_geometry
    frame.at[0, "wgs84_bounds"] = list(shifted_geometry.bounds)
    frame.loc[0, "footprint_hash"] = hashlib.sha256(
        normalize(set_precision(shifted_geometry, 1e-9)).wkb
    ).hexdigest()
    frame.to_parquet(parquet_path, index=False, compression="zstd")

    audit = MODULE.audit_grid_package(
        tmp_path,
        [
            {
                "grid_epsg": record["grid_epsg"],
                "grid_col": record["grid_col"],
                "grid_row": record["grid_row"],
            }
        ],
        batch_size=1,
    )

    assert audit["child_partition_metadata_mismatch_count"] == 1
    assert audit["child_partition_geometry_mismatch_count"] == 1
    assert audit["passed"] is False


def test_writer_keeps_output_empty_when_a_later_partition_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = synthetic_parent_records(count=2)
    sampled = {str(records[0]["parent_key"])}
    original_to_parquet = gpd.GeoDataFrame.to_parquet
    calls = 0

    def fail_second_parquet_write(self, path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second partition failure")
        return original_to_parquet(self, path, *args, **kwargs)

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail_second_parquet_write)

    with pytest.raises(OSError, match="synthetic second partition failure"):
        MODULE.write_zone_records(records, sampled, tmp_path, batch_size=2)

    assert not list(tmp_path.rglob("*.parquet"))
    assert not list(tmp_path.rglob("*.shp"))


def test_grid_spec_rejects_noncanonical_parent_cell_size() -> None:
    with pytest.raises(ValueError, match="side_m must be exactly 1280"):
        MODULE.GridSpec(side_m=640)


def test_enumerator_emits_exact_1280m_parent_cells() -> None:
    spec = MODULE.GridSpec(side_m=1280, macro_side_patches=10, boundary_version="test")
    records = list(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))

    assert records
    for record in records:
        minx, miny, maxx, maxy = record["utm_bounds"]
        assert maxx - minx == 1280
        assert maxy - miny == 1280
        assert minx == record["grid_col"] * 1280
        assert miny == record["grid_row"] * 1280
        assert record["parent_key"] == (
            f'{record["grid_epsg"]}:{record["grid_col"]}:{record["grid_row"]}'
        )


def test_enumerator_records_have_deterministic_ids() -> None:
    spec = MODULE.GridSpec(boundary_version="test")

    first = list(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))
    second = list(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))

    assert [record["patch_id"] for record in first] == [record["patch_id"] for record in second]
    assert all(record["patch_id"] == f"parent_{record['parent_key']}" for record in first)


def test_enumerator_gives_each_record_a_unique_macro_local_key() -> None:
    spec = MODULE.GridSpec(boundary_version="test")
    records = list(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))

    keys = {
        (record["macro_id"], record["macro_local_col"], record["macro_local_row"])
        for record in records
    }
    assert len(keys) == len(records)
    assert all(0 <= record["macro_local_col"] < 10 for record in records)
    assert all(0 <= record["macro_local_row"] < 10 for record in records)


def test_enumerator_filters_cells_owned_by_another_utm_zone() -> None:
    spec = MODULE.GridSpec(boundary_version="test")
    to_utm49 = Transformer.from_crs(4326, 32649, always_xy=True)
    easting, northing = to_utm49.transform(117.0, 39.84)
    macro = {
        "grid_epsg": 32649,
        "grid_id": "utm49n",
        "macro_col": math.floor(easting / 12800),
        "macro_row": math.floor(northing / 12800),
    }
    boundary = box(116.9, 39.7, 117.1, 40.0)

    records = list(MODULE.enumerate_macro_patch_records(macro, boundary, spec))

    assert records == []


def test_enumerator_rejects_epsg_outside_china_owner_range() -> None:
    macro = {**_macro(), "grid_epsg": 32642, "grid_id": "utm42n"}

    with pytest.raises(ValueError, match="32643 through 32653"):
        list(MODULE.enumerate_macro_patch_records(macro, _boundary(), MODULE.GridSpec()))


def test_enumerator_preserves_wgs84_longitude_latitude_axis_order() -> None:
    spec = MODULE.GridSpec(boundary_version="test")
    records = list(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))

    assert records
    for record in records:
        assert 116.28 <= record["longitude"] <= 116.31
        assert 39.83 <= record["latitude"] <= 39.86


def test_validate_patch_record_rejects_misaligned_bounds() -> None:
    spec = MODULE.GridSpec(boundary_version="test")
    record = next(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))
    record["utm_bounds"][0] += 1

    with pytest.raises(ValueError, match="utm_bounds"):
        MODULE.validate_patch_record(record, spec)


def test_validate_patch_record_rejects_epsg_outside_china_owner_range() -> None:
    spec = MODULE.GridSpec(boundary_version="test")
    to_utm42 = Transformer.from_crs(4326, 32642, always_xy=True)
    easting, northing = to_utm42.transform(69.0, 30.0)
    grid_col = math.floor(easting / 1280)
    grid_row = math.floor(northing / 1280)
    macro = {
        "grid_epsg": 32642,
        "grid_id": "utm42n",
        "macro_col": grid_col // 10,
        "macro_row": grid_row // 10,
    }
    record = MODULE.build_patch_record(macro, grid_col, grid_row, 69.0, 30.0, spec)

    with pytest.raises(ValueError, match="32643 through 32653"):
        MODULE.validate_patch_record(record, spec)


def test_wgs84_transformer_is_reused_for_same_grid_zone() -> None:
    MODULE._transformer_to_wgs84.cache_clear()
    record = {"grid_epsg": 32650, "utm_bounds": [500000, 4400000, 501280, 4401280]}

    first = MODULE._wgs84_geometry(record)
    second = MODULE._wgs84_geometry(record)

    assert first.equals_exact(second, tolerance=0.0)
    assert MODULE._transformer_to_wgs84.cache_info().misses == 1
    assert MODULE._transformer_to_wgs84.cache_info().hits == 1
