import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import transform


def geometry_case():
    bounds = (500000, 4000000, 501280, 4001280)
    inside = box(499950, 3999950, 501330, 4001330)
    outside_bad = Polygon(
        [(502000, 4000000), (502100, 4000100), (502000, 4000100), (502100, 4000000)]
    )
    raw = transform(Transformer.from_crs(32650, 4326, always_xy=True).transform, inside)
    return raw, MultiPolygon([inside, outside_bad]), bounds


def test_local_projection_recovery_requires_independent_pixel_agreement():
    from xuannv_embedding.data_process.v5_osm_projection import recover_polygon

    raw, projected, bounds = geometry_case()
    assert raw.is_valid and not projected.is_valid
    local, proof = recover_polygon(raw, projected, 32650, bounds)
    assert local.is_valid and proof["comparison_pixels"] == 512 * 512
    assert proof["linework_different_pixels"] == proof["structure_different_pixels"] == 0
    assert proof["source_modified"] is False
    changed = MultiPolygon([box(500000, 4000000, 500640, 4001280), list(projected.geoms)[1]])
    with pytest.raises(ValueError, match="local repair methods disagree"):
        recover_polygon(raw, changed, 32650, bounds)


def test_invalid_source_or_nonmetric_grid_cannot_be_repaired():
    from xuannv_embedding.data_process.v5_osm_projection import recover_polygon

    raw, projected, bounds = geometry_case()
    with pytest.raises(ValueError, match="source polygon"):
        recover_polygon(Polygon([(0, 0), (1, 1), (0, 1), (1, 0)]), projected, 32650, bounds)
    with pytest.raises(ValueError, match="metric"):
        recover_polygon(raw, projected, 4326, bounds)


def test_corrections_reconstruct_actual_sources_and_preserve_original_labels(tmp_path):
    from test_v5_osm_geometry import _osm_fixture

    from xuannv_embedding.data_process.v5_osm_corrections import build_osm_corrections
    from xuannv_embedding.data_process.v5_sources import sha256

    data, report, original, _ = _osm_fixture(tmp_path)
    result = build_osm_corrections(data, report)
    assert result["corrected_views"] == 4 and result["failed_views_remaining"] == 0
    assert not original["2020/states/road_all"][:].any()
    root = Path(result["output"])
    rows = pd.read_parquet(root / "manifest.parquet")
    assert set(rows.year) == {2020, 2021}
    assert rows.split.eq("train").all() and rows.source_reconstruction_passed.all()
    with np.load(root / rows.iloc[0].correction_file) as arrays:
        assert arrays["states"].max() == 1
        assert not arrays["states"][arrays["targets"] == 0].any()
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    assert build_osm_corrections(data, report)["reused"]
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}
    (root / rows.iloc[0].correction_file).write_bytes(b"changed")
    with pytest.raises(ValueError, match="correction.*changed"):
        build_osm_corrections(data, report)


def test_changed_base_pixels_and_incomplete_scan_cannot_publish_corrections(tmp_path):
    from test_v5_osm_geometry import _osm_fixture

    from xuannv_embedding.data_process.v5_osm_corrections import build_osm_corrections

    data, report, original, _ = _osm_fixture(tmp_path)
    original["2020/targets/building"][0, 110, 11] = 0
    with pytest.raises(ValueError, match="original labels.*reconstruction"):
        build_osm_corrections(data, report)
    p = report / "osm_geometry_full.json"
    record = json.loads(p.read_text())
    record["status"] = "running"
    p.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="complete OSM"):
        build_osm_corrections(data, report)


def test_osm_correction_cli_requires_complete_scope(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_osm_corrections

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    monkeypatch.setattr(v5_osm_corrections, "build_osm_corrections", lambda *a: calls.append(a))
    args = ["--stage", "osm-corrections"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + key, str(tmp_path / key)]
    assert v5_cli.main(args) == 0 and len(calls) == 1
    with pytest.raises(ValueError, match="complete"):
        v5_cli.main(args + ["--max-patches", "1"])


def test_no_changed_or_failed_views_produces_explicit_empty_manifest(tmp_path):
    from test_v5_osm_geometry import _osm_fixture

    from xuannv_embedding.data_process.v5_osm_corrections import build_osm_corrections

    data, report, _, _ = _osm_fixture(tmp_path, include_edge=False)
    result = build_osm_corrections(data, report)
    assert result["corrected_views"] == result["verified_views"] == 0
    table = pd.read_parquet(Path(result["output"]) / "manifest.parquet")
    assert table.empty and {"patch_id", "year", "resolution"}.issubset(table.columns)
    assert build_osm_corrections(data, report)["reused"]
