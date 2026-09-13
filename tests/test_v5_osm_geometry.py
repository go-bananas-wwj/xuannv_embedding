import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from xuannv_embedding.data_process.v5_osm_geometry import (
    CHANNELS,
    encode_coverage,
    rasterize_records,
)


def test_polygon_line_and_point_rasterization_have_explicit_physical_support():
    records = [
        {"geometry": box(101, 101, 199, 199), "channels": 1, "width_m": 0},
        {"geometry": LineString([(-2, 301), (-2, 399)]), "channels": 2, "width_m": 8},
        {"geometry": Point(640, 640), "channels": 1 << 15, "width_m": 0},
    ]
    a = rasterize_records(records, (0, 0, 1280, 1280), 128, 4, CHANNELS)
    assert a.shape == (30, 128, 128)
    assert a[0, 108:118, 10:20].min() == 1 and a[0].sum() == 100
    assert a[1, :, 0].sum() > 0 and not a[1, :, 2:].any()
    assert 0 < a[15].max() <= 1 and not a[15, :50].any()
    fine = rasterize_records(records, (0, 0, 1280, 1280), 512, 1, ("building", "road_all"))
    assert fine.shape == (2, 512, 512) and fine[0].sum() == 1600


def test_current_only_geometry_remains_unknown_and_coverage_is_quantized():
    old = np.zeros((2, 2, 2), "f4")
    current = old.copy()
    old[0, 0, 0] = 0.25
    current[1, 1, 1] = 1
    result = encode_coverage(old, current)
    assert result["targets"][0, 0, 0] == 64 and result["states"][0, 0, 0] == 1
    assert result["confidence"][0, 0, 0] == 255 and result["source_bits"][0, 0, 0] == 1
    assert result["states"][1, 1, 1] == 0 and result["targets"][1, 1, 1] == 0
    assert result["source_bits"][1, 1, 1] == 2
    with pytest.raises(ValueError):
        encode_coverage(old, current + 2)


def test_geometry_rejects_unknown_channels_invalid_width_and_grid():
    for records in [
        [{"geometry": Point(1, 1), "channels": 1 << 30, "width_m": 0}],
        [{"geometry": LineString([(0, 0), (1, 1)]), "channels": 2, "width_m": -1}],
    ]:
        with pytest.raises(ValueError):
            rasterize_records(records, (0, 0, 1280, 1280), 128, 4, CHANNELS)
    with pytest.raises(ValueError):
        rasterize_records([], (0, 0, 1000, 1280), 128, 4, CHANNELS)


def _spatial_index(path, year, include_edge=True):
    import json
    import sqlite3

    from pyproj import Transformer
    from shapely.ops import transform

    from xuannv_embedding.data_process.v5_sources import sha256

    raw = path.with_suffix(".raw")
    raw.write_bytes(b"fixed raw fixture source")
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT); "
        "CREATE TABLE features(feature_id INTEGER PRIMARY KEY,channels INTEGER,"
        "width_m REAL,wkb BLOB); "
        "CREATE VIRTUAL TABLE feature_bounds USING rtree(feature_id,minx,maxx,miny,maxy);"
    )
    projected = [(box(500101, 4000101, 500199, 4000199), 1, 0)]
    if include_edge:
        projected.append((LineString([(499995, 4000301), (499995, 4000399)]), 2, 16))
    transformer = Transformer.from_crs(32650, 4326, always_xy=True)
    for i, (geometry, channels, width) in enumerate(projected):
        g = transform(transformer.transform, geometry)
        b = g.bounds
        connection.execute("INSERT INTO features VALUES(?,?,?,?)", (i, channels, width, g.wkb))
        connection.execute(
            "INSERT INTO feature_bounds VALUES(?,?,?,?,?)", (i, b[0], b[2], b[1], b[3])
        )
    metadata = {
        "schema": "xuannv.osm30-spatial-index.v1",
        "channel_order": list(CHANNELS),
        "snapshot_date": f"{year}-01-01",
        "source_path": str(raw),
        "source_sha256": sha256(raw),
        "feature_counts": {"area": 1, "line": int(include_edge)},
    }
    connection.executemany(
        "INSERT INTO metadata VALUES(?,?)", [(k, json.dumps(v)) for k, v in metadata.items()]
    )
    connection.commit()
    connection.close()
    return {"path": str(path), "sha256": sha256(path)}


def test_spatial_index_checks_raw_sources_dates_and_features_outside_target(tmp_path):
    from xuannv_embedding.data_process.v5_osm_geometry import SpatialIndex

    reference = _spatial_index(tmp_path / "source.sqlite", 2020)
    source = SpatialIndex(reference, year=2020)
    old, new, counts = source.coverages(32650, (500000, 4000000, 501280, 4001280), 128, 4, CHANNELS)
    assert counts["legacy_features"] == 1 and counts["expanded_features"] == 2
    assert not old[1].any() and new[1, :, 0].any()
    source.close()
    with pytest.raises(ValueError, match="snapshot date"):
        SpatialIndex(reference, year=2021)
    (tmp_path / "source.raw").write_bytes(b"changed")
    with pytest.raises(ValueError, match="original OSM source"):
        SpatialIndex(reference, year=2020)


def test_osm_geometry_full_audit_compares_all_fields_and_preserves_boundary_candidates(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    import pandas as pd
    import zarr

    from xuannv_embedding.data_process.v5_osm_geometry import FIELDS, STRUCTURE, audit_osm_geometry
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    data = tmp_path / "data"
    report = tmp_path / "report"
    (data / "registry").mkdir(parents=True)
    (data / "targets").mkdir()
    report.mkdir()
    registry = pd.DataFrame(
        {
            "patch_id": ["p"],
            "split": ["train"],
            "grid_epsg": [32650],
            "utm_bounds": [[500000, 4000000, 501280, 4001280]],
        }
    )
    registry.to_parquet(data / "registry/national_62000.parquet", index=False)
    refs = {str(y): _spatial_index(tmp_path / f"{y}.sqlite", y) for y in [2020, 2021]}
    current = _spatial_index(tmp_path / "current.sqlite", 2026, include_edge=False)
    base = tmp_path / "osm.zarr"
    root = zarr.open_group(str(base), mode="w")
    root.attrs.update(
        {
            "patch_ids": ["p"],
            "target_names": list(CHANNELS),
            "structure_names": list(STRUCTURE),
            "external_evidence": None,
            "target_encoding": "uint8_coverage_or_soft_evidence_confidence",
            "years": [2020, 2021],
            "unlabeled_is_unknown": True,
            "indexes": refs,
            "current_index": current,
        }
    )
    manifest = []
    values = []
    for year in [2020, 2021]:
        for branch, channels, pixels in [("", CHANNELS, 128), ("structure_2p5m/", STRUCTURE, 512)]:
            for field in FIELDS:
                for channel in channels:
                    a = np.zeros((1, pixels, pixels), "u1")
                    if channel == "building":
                        sl = (
                            (slice(108, 118), slice(10, 20))
                            if pixels == 128
                            else (slice(432, 472), slice(40, 80))
                        )
                        a[(0, *sl)] = {
                            "states": 1,
                            "confidence": 255,
                            "targets": 255,
                            "source_bits": 3,
                        }[field]
                    name = f"{year}/{branch}{field}/{channel}"
                    root.create_dataset(name, data=a)
                    manifest.append(
                        {
                            "family": "osm",
                            "path": str(base),
                            "array": name,
                            "registry_order_verified": True,
                            "source_metadata_sha256": sha256(base / ".zattrs"),
                        }
                    )
                    values.append(
                        {
                            "family": "osm",
                            "path": str(base),
                            "array": name,
                            "decoded_values_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                            "status": "values_checked_provenance_pending",
                        }
                    )
    mp = data / "targets/manifest.parquet"
    pd.DataFrame(manifest).to_parquet(mp, index=False)
    vp = report / "target_value_audit.parquet"
    pd.DataFrame(values).to_parquet(vp, index=False)
    write_json(
        report / "osm_temporal_progress.json",
        {
            "status": "temporal_cross_checks_finished",
            "failed_groups": 0,
            "metadata_sha256": sha256(base / ".zattrs"),
            "manifest_sha256": sha256(mp),
            "value_audit_sha256": sha256(vp),
        },
    )
    first = audit_osm_geometry(data, report)
    assert first["compared_views"] == 4 and first["failed_views"] == 0
    assert first["boundary_changed_views"] == 4 and first["boundary_historical_added_pixels"] > 0
    assert first["boundary_candidates_authorized_as_labels"] is False
    assert not root["2020/states/road_all"][:].any()
    again = audit_osm_geometry(data, report)
    assert again["reused_views"] == 4
    table = pd.read_parquet(first["output"])
    candidate = Path(first["output"]).parent / table.iloc[0].candidate_file
    original_candidate = candidate.read_bytes()
    candidate.write_bytes(b"changed")
    with pytest.raises(ValueError, match="boundary candidate changed"):
        audit_osm_geometry(data, report)
    candidate.write_bytes(original_candidate)
    root["2020/targets/building"][0, 110, 11] = 0
    with pytest.raises(ValueError, match="labels changed"):
        audit_osm_geometry(data, report)
    assert (
        json.loads((report / "osm_geometry_full.json").read_text())["status"]
        != "osm_geometry_audit_finished"
    )


def test_osm_geometry_cli_preserves_explicit_pilot_scope(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_osm_geometry

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda _: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    monkeypatch.setattr(v5_osm_geometry, "audit_osm_geometry", lambda *a, **k: calls.append((a, k)))
    argv = ["--stage", "osm-geometry", "--max-patches", "32"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        argv += ["--" + key, str(tmp_path / key)]
    assert v5_cli.main(argv) == 0
    assert calls == [((tmp_path / "dataset-root", tmp_path / "report-root"), {"max_patches": 32})]
