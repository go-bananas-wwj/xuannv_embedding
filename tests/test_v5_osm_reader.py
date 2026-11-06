from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def fixture(tmp_path, include_edge=True):
    from test_v5_osm_geometry import _osm_fixture

    from xuannv_embedding.data_process.v5_osm_corrections import build_osm_corrections

    data, report, original, _ = _osm_fixture(tmp_path, include_edge=include_edge)
    result = build_osm_corrections(data, report)
    return data, report, original, Path(result["output"])


def test_reader_batches_native_views_and_keeps_original_labels_unchanged(tmp_path):
    from xuannv_embedding.data_process.v5_osm_reader import CorrectedOSMReader

    data, _, original, root = fixture(tmp_path)
    reader = CorrectedOSMReader(data, root)
    keys = [("p", year, resolution) for year in [2020, 2021] for resolution in ["10m", "2p5m"]]
    result = dict(reader.iter_views(keys))
    assert set(result) == set(keys) and reader.decoded_chunks == 1
    assert result[("p", 2020, "10m")]["states"].shape == (30, 128, 128)
    assert result[("p", 2021, "2p5m")]["states"].shape == (4, 512, 512)
    assert result[("p", 2020, "10m")]["states"][1].any()
    assert not original["2020/states/road_all"][:].any()
    for fields in result.values():
        assert set(np.unique(fields["states"])).issubset({0, 1})
        assert np.array_equal(fields["states"] == 1, fields["targets"] > 0)


def test_reader_verifies_unmodified_views_and_never_reuses_stale_pixels(tmp_path):
    from xuannv_embedding.data_process.v5_osm_reader import CorrectedOSMReader

    data, _, original, root = fixture(tmp_path, include_edge=False)
    reader = CorrectedOSMReader(data, root)
    assert reader.read("p", 2020, "10m")["targets"][0, 110, 11] == 255
    original["2020/targets/building"][0, 110, 11] = 0
    with pytest.raises(ValueError, match="source chunk"):
        reader.read("p", 2020, "10m")


def test_reader_detects_changed_overlay_and_publication_after_initialization(tmp_path):
    from xuannv_embedding.data_process.v5_osm_reader import CorrectedOSMReader

    data, _, _, root = fixture(tmp_path)
    reader = CorrectedOSMReader(data, root)
    row = pd.read_parquet(root / "manifest.parquet").iloc[0]
    path = root / row.correction_file
    content = path.read_bytes()
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="correction array"):
        reader.read(row.patch_id, int(row.year), row.resolution)
    path.write_bytes(content)
    (root / "output.lock.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="publication"):
        reader.read("p", 2020, "10m")


def test_reader_rejects_invalid_calendar_resolution_and_duplicate_requests(tmp_path):
    from xuannv_embedding.data_process.v5_osm_reader import CorrectedOSMReader

    data, _, _, root = fixture(tmp_path)
    reader = CorrectedOSMReader(data, root)
    for key in [("p", 2022, "10m"), ("unknown", 2020, "10m"), ("p", 2020, "5m")]:
        with pytest.raises(ValueError, match="OSM view key"):
            reader.read(*key)
    with pytest.raises(ValueError, match="duplicate"):
        list(reader.iter_views([("p", 2020, "10m")] * 2))


def test_negative_rule_change_prevents_verification_success():
    from xuannv_embedding.data_process.v5_negative_corrections import corrected_overlay
    from xuannv_embedding.data_process.v5_negative_rules import TASKS
    from xuannv_embedding.data_process.v5_osm_reader import verify_negative_view

    shape = (128, 128)
    states = {t: np.zeros(shape, "u1") for t in TASKS}
    params = dict(erosion_pixels=0, steep_slope_degrees=20, negative_confidence=224)
    old = dict(
        states=states,
        worldcover=np.full(shape, 80, "u1"),
        worldcover_valid=np.ones(shape, bool),
        slope=np.zeros(shape, "f4"),
        slope_valid=np.ones(shape, bool),
    )
    old["overlay"] = corrected_overlay(
        states, old["worldcover"], old["worldcover_valid"], old["slope"], old["slope_valid"], params
    )
    reader = SimpleNamespace(
        evidence=SimpleNamespace(read=lambda *a: old, parameters=params),
        rows=pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["patch_id", "year"])),
    )
    fields = {"states": np.zeros((30, *shape), "u1")}
    verify_negative_view(reader, "p", 0, 2020, fields)
    fields["states"][0, 64, 64] = 1
    with pytest.raises(ValueError, match="negative overlay requires regeneration"):
        verify_negative_view(reader, "p", 0, 2020, fields)


def test_reader_verification_cli_requires_explicit_correction_version(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_osm_reader

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    monkeypatch.setattr(v5_osm_reader, "verify_osm_reader", lambda *a: calls.append(a))
    args = ["--stage", "osm-correction-verify"]
    for k in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + k, str(tmp_path / k)]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--correction-root", str(tmp_path / "version")]
    assert v5_cli.main(args) == 0 and calls[0][-1] == tmp_path / "version"
    with pytest.raises(ValueError, match="complete"):
        v5_cli.main(args + ["--max-patches", "1"])


def test_reader_rejects_a_different_dataset_registry(tmp_path):
    import shutil

    from xuannv_embedding.data_process.v5_osm_reader import CorrectedOSMReader

    data, _, _, root = fixture(tmp_path)
    other = tmp_path / "other-data"
    shutil.copytree(data, other)
    path = other / "registry/national_62000.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "patch_id"] = "different"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="registry"):
        CorrectedOSMReader(other, root)


def test_reader_verifier_replays_without_rewriting_its_artifacts(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_osm_reader as module
    from xuannv_embedding.data_process.v5_sources import sha256

    data, report, _, root = fixture(tmp_path)
    negative = tmp_path / "negative"
    negative.mkdir()
    for name in ["audit.json", "corrections.lock.json", "dem.lock.json"]:
        (negative / name).write_text("{}")
    fake = SimpleNamespace(
        evidence=SimpleNamespace(
            audit_path=negative / "audit.json", audit_sha=sha256(negative / "audit.json")
        ),
        directory=negative,
        demlock=negative / "dem.lock.json",
    )
    monkeypatch.setattr(module, "CorrectedNegativeReader", lambda *a: fake)
    monkeypatch.setattr(module, "verify_negative_view", lambda *a: None)
    first = module.verify_osm_reader(data, report, root)
    assert first["verified_views"] == 4 and first["negative_views_verified"] == 2
    output = Path(first["output"])
    files = {p: (sha256(p), p.stat().st_mtime_ns) for p in output.iterdir()}
    module.verify_osm_reader(data, report, root)
    assert files == {p: (sha256(p), p.stat().st_mtime_ns) for p in files}
