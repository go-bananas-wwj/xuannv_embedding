import numpy as np
import pytest


def annual(tmp_path):
    from test_v5_target_geometry import annual_fixture

    from xuannv_embedding.data_process.v5_target_geometry import audit_target_geometry

    data, report, root = annual_fixture(tmp_path)
    audit_target_geometry(data, report, "clcd")
    return data, report, root


def test_annual_reader_keeps_years_distinct_and_checks_native_masks(tmp_path):
    from xuannv_embedding.data_process.v5_target_reader import AnnualTargetReader

    data, report, _ = annual(tmp_path)
    reader = AnnualTargetReader(data, report, audits={"clcd": "target_geometry_clcd_full.json"})
    frames = dict(reader.iter_views([("p", 2020), ("p", 2021)]))
    assert frames[("p", 2020)]["clcd"]["values"].shape == (128, 128)
    assert frames[("p", 2020)]["clcd"]["values"].min() == 1
    assert frames[("p", 2021)]["clcd"]["values"].min() == 2
    assert frames[("p", 2020)]["clcd"]["valid"].dtype == bool
    for key in [("p", 2022), ("missing", 2020)]:
        with pytest.raises(ValueError, match="annual target key"):
            list(reader.iter_views([key]))


def test_annual_reader_rechecks_actual_values_between_batches(tmp_path):
    from xuannv_embedding.data_process.v5_target_reader import AnnualTargetReader

    data, report, root = annual(tmp_path)
    reader = AnnualTargetReader(data, report, audits={"clcd": "target_geometry_clcd_full.json"})
    list(reader.iter_views([("p", 2020)]))
    root["targets/clcd_2020"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="target chunk"):
        list(reader.iter_views([("p", 2020)]))


def test_annual_reader_rejects_changed_source_or_audit(tmp_path):
    from xuannv_embedding.data_process.v5_target_reader import AnnualTargetReader

    data, report, _ = annual(tmp_path)
    reader = AnnualTargetReader(data, report, audits={"clcd": "target_geometry_clcd_full.json"})
    (tmp_path / "static_clcd_china_2020.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="annual target evidence"):
        reader.verify_unchanged()
    with pytest.raises(ValueError, match="annual target evidence"):
        AnnualTargetReader(data, report, audits={"clcd": "target_geometry_clcd_full.json"})


def test_merge_osm_keeps_unknown_and_only_uses_known_negative_overlay():
    from xuannv_embedding.data_process.v5_target_reader import merge_osm_fields

    positive = {
        k: np.zeros((30, 128, 128), "u1")
        for k in ["targets", "states", "confidence", "source_bits"]
    }
    positive["targets"][0, 10, 10] = 125
    positive["states"][0, 10, 10] = 1
    positive["confidence"][0, 10, 10] = 255
    positive["source_bits"][0, 10, 10] = 3
    negative = {k: np.zeros((4, 128, 128), "u1") for k in ["states", "confidence"]}
    negative["states"][0, 11, 11] = 3
    negative["confidence"][0, 11, 11] = 224
    merged = merge_osm_fields(positive, negative)
    assert merged["states"][0, 10, 10] == 1 and merged["targets"][0, 10, 10] == 125
    assert merged["states"][0, 11, 11] == 3 and merged["confidence"][0, 11, 11] == 224
    assert merged["states"][0, 12, 12] == 0
    assert positive["states"][0, 11, 11] == 0
    negative["states"][0, 10, 10] = 3
    negative["confidence"][0, 10, 10] = 224
    with pytest.raises(ValueError, match="positive and negative"):
        merge_osm_fields(positive, negative)


def composite_fixture(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import pandas as pd
    from test_v5_osm_reader import fixture

    from xuannv_embedding.data_process import v5_osm_reader, v5_target_reader
    from xuannv_embedding.data_process.v5_sources import sha256

    data, report, original, root = fixture(tmp_path)
    directory = tmp_path / "negative"
    demdir = tmp_path / "dem"
    directory.mkdir()
    demdir.mkdir()
    for path in [
        directory / "audit.json",
        directory / "corrections.lock.json",
        directory / "manifest.parquet",
        demdir / "corrections.lock.json",
        demdir / "manifest.parquet",
    ]:
        path.write_text("{}")
    registry = pd.read_parquet(data / "registry/national_62000.parquet").set_index("patch_id")
    zeros = {k: np.zeros((4, 128, 128), "u1") for k in ["states", "confidence"]}
    zeros["states"][0, 1, 1] = 3
    zeros["confidence"][0, 1, 1] = 224
    demfields = {
        k: np.zeros((128, 128), "f4" if "valid" not in k else bool)
        for k in ["elevation", "slope", "elevation_valid", "slope_valid"]
    }
    fake = SimpleNamespace(
        evidence=SimpleNamespace(
            audit_path=directory / "audit.json", audit_sha=sha256(directory / "audit.json")
        ),
        directory=directory,
        demlock=demdir / "corrections.lock.json",
        dem=SimpleNamespace(
            directory=demdir,
            rows=pd.DataFrame({"correction_file": [""]}, index=registry.index),
            read=lambda p: demfields,
        ),
        read=lambda p, y: {k: v.copy() for k, v in zeros.items()},
    )
    monkeypatch.setattr(v5_osm_reader, "CorrectedNegativeReader", lambda *a: fake)
    monkeypatch.setattr(v5_osm_reader, "verify_negative_view", lambda *a: None)
    proof = v5_osm_reader.verify_osm_reader(data, report, root)

    def annual_views(keys):
        for p, y in keys:
            yield (p, y), {
                "clcd": {
                    "values": np.full((128, 128), y - 2019, "u1"),
                    "valid": np.ones((128, 128), bool),
                }
            }

    annual_reader = SimpleNamespace(
        registry=registry,
        indices={"p": 0},
        files={},
        sources={},
        iter_views=annual_views,
        verify_unchanged=lambda **k: None,
        decoded_chunks=0,
    )
    monkeypatch.setattr(v5_target_reader, "AnnualTargetReader", lambda *a: annual_reader)
    monkeypatch.setattr(v5_target_reader, "CorrectedNegativeReader", lambda *a: fake)
    return data, report, root, Path(proof["output"]), original, json


def test_composite_reader_preserves_year_native_resolution_and_unknown(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_target_reader as module

    data, report, root, _, _, _ = composite_fixture(tmp_path, monkeypatch)
    reader = module.ValidatedTargetReader(data, report, root)
    views = dict(reader.iter_views([("p", 2020), ("p", 2021)]))
    assert views[("p", 2020)]["clcd"]["values"].min() == 1
    assert views[("p", 2021)]["clcd"]["values"].min() == 2
    for view in views.values():
        assert view["osm_10m"]["states"][0, 1, 1] == 3
        assert view["osm_10m"]["states"][0, 64, 64] == 0
        assert view["osm_2p5m"]["states"].shape == (4, 512, 512)
        assert 3 not in view["osm_2p5m"]["states"]
    assert reader.osm.decoded_chunks == 1


def test_composite_reader_requires_current_negative_reconciliation(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_target_reader as module

    data, report, root, proof, _, json = composite_fixture(tmp_path, monkeypatch)
    reader = module.ValidatedTargetReader(data, report, root)
    path = proof / "verification.lock.json"
    lock = json.loads(path.read_text())
    lock["fingerprint"]["negative_correction_lock_sha256"] = "changed"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="evidence changed"):
        list(reader.iter_views([("p", 2020)]))
    with pytest.raises(ValueError, match="reconciliation"):
        module.ValidatedTargetReader(data, report, root)


def test_composite_verifier_replays_actual_pixels_without_rewriting(tmp_path, monkeypatch):
    from pathlib import Path

    from xuannv_embedding.data_process import v5_target_reader as module
    from xuannv_embedding.data_process.v5_sources import sha256

    data, report, root, _, original, _ = composite_fixture(tmp_path, monkeypatch)
    first = module.verify_target_reader(data, report, root)
    assert first["verified_views"] == 2
    output = Path(first["output"])
    files = {p: (sha256(p), p.stat().st_mtime_ns) for p in output.iterdir()}
    module.verify_target_reader(data, report, root)
    assert files == {p: (sha256(p), p.stat().st_mtime_ns) for p in files}
    original["2020/targets/building"][0, 0, 0] = 5
    with pytest.raises(ValueError, match="source chunk"):
        module.verify_target_reader(data, report, root)


def test_composite_cli_requires_correction_and_full_verification(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_target_reader

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    monkeypatch.setattr(v5_target_reader, "verify_target_reader", lambda *a: calls.append(a))
    args = ["--stage", "target-reader-verify"]
    for k in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + k, str(tmp_path / k)]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--correction-root", str(tmp_path / "version")]
    assert v5_cli.main(args) == 0 and calls[0][-1] == tmp_path / "version"
    with pytest.raises(ValueError, match="complete"):
        v5_cli.main(args + ["--max-patches", "1"])
