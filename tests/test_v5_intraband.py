import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift

from xuannv_embedding.data_process.v5_rasters import NativeRaster


def _frame():
    ref = gaussian_filter(np.random.default_rng(42).normal(size=(160, 160)), 1).astype("f4")
    values = np.stack([ref, ref, ref, shift(ref, (2, -1), order=1, mode="reflect")])
    return NativeRaster(
        values,
        np.ones(values.shape, bool),
        ("blue", "green", "red", "nir"),
        (8, 0, 300000, 0, -8, 4000000),
        "EPSG:32650",
    )


def test_intraband_separates_relative_offsets_from_absolute_georegistration():
    from xuannv_embedding.data_process.v5_intraband import inspect_intraband

    frame = _frame()
    before = frame.values.copy()
    result = inspect_intraband(frame, reference_band="green", gsd=8)
    assert result["status"] == "over_limit"
    assert result["pixel_fusion_authorized"] is False
    nir = next(p for p in result["pairs"] if p["moving_band"] == "nir")
    np.testing.assert_allclose(nir["translation_yx_m"], [-16, 8], atol=2)
    np.testing.assert_array_equal(frame.values, before)


def test_intraband_missing_texture_is_never_reported_as_aligned():
    from xuannv_embedding.data_process.v5_intraband import inspect_intraband

    frame = _frame()
    frame.valid[-1] = False
    result = inspect_intraband(frame, reference_band="green", gsd=8)
    assert result["status"] == "uncertain"
    assert result["pairs"][-1]["valid_windows"] == 0
    with pytest.raises(ValueError, match="reference"):
        inspect_intraband(frame, reference_band="unknown", gsd=8)


def test_calibration_checks_known_real_texture_shifts_and_skips_textureless_input():
    from xuannv_embedding.data_process.v5_intraband import calibrate_texture

    frame = _frame()
    result = calibrate_texture(frame.values[1], frame.valid[1], gsd=8)
    assert result["status"] == "passed"
    assert len(result["cases"]) == 4
    assert max(c["error_pixels"] for c in result["cases"]) <= 0.35
    blank = calibrate_texture(np.ones((160, 160), "f4"), np.ones((160, 160), bool), gsd=8)
    assert blank["status"] == "insufficient_texture"


def test_band_audit_reuses_only_matching_pixels_and_requires_calibration(tmp_path, monkeypatch):
    import pandas as pd

    from xuannv_embedding.data_process import v5_intraband as module
    from xuannv_embedding.data_process.v5_sources import sha256

    data, report = tmp_path / "data", tmp_path / "report"
    rows = []
    for i in range(9):
        path = tmp_path / f"{i}.tif"
        path.write_bytes(str(i).encode())
        rows.append(
            {
                "observation_id": str(i),
                "patch_id": str(i),
                "path": str(path),
                "file_sha256": sha256(path),
                "sensor": "GF1",
                "split": "train",
                "year": 2020,
            }
        )
    monkeypatch.setattr(module, "_inventory", lambda *args: pd.DataFrame(rows))
    monkeypatch.setattr(module, "read_native", lambda *args, **kwargs: _frame())
    with pytest.raises(FileNotFoundError):
        module.run_intraband(data, report, "gaofen")
    assert module.calibrate_family(data, report, "gaofen")["status"] == "passed"
    first = module.run_intraband(data, report, "gaofen")
    assert first["counts"]["over_limit"] == 9 and first["counts"]["reused"] == 0
    assert first["pixel_fusion_authorized"] is False
    cached = module.run_intraband(data, report, "gaofen")
    assert cached["counts"]["reused"] == 9
    profile = data / "quality/alignment/intraband/gaofen/v1/calibration.json"
    profile_hash = sha256(profile)
    assert module.calibrate_family(data, report, "gaofen")["reused"] is True
    assert sha256(profile) == profile_hash
    runtime = module._runtime_versions()
    with monkeypatch.context() as runtime_change:
        runtime_change.setattr(module, "_runtime_versions", lambda: {**runtime, "numpy": "changed"})
        with pytest.raises(ValueError, match="runtime"):
            module.run_intraband(data, report, "gaofen")
    rows[8]["year"] = 2021
    metadata_changed = module.run_intraband(data, report, "gaofen")
    assert metadata_changed["counts"]["reused"] == 8
    table = pd.read_parquet(metadata_changed["output"])
    assert table.loc[table.observation_id == "8", "year"].iloc[0] == 2021
    (tmp_path / "8.tif").write_bytes(b"changed")
    changed = module.run_intraband(data, report, "gaofen")
    assert changed["counts"]["rejected"] == 1 and changed["counts"]["reused"] == 8
    (tmp_path / "0.tif").write_bytes(b"changed calibration input")
    with pytest.raises(ValueError, match="calibration source"):
        module.run_intraband(data, report, "gaofen")


def test_native_band_cli_requires_family_and_preserves_versioned_data_only_route(
    tmp_path, monkeypatch
):
    from xuannv_embedding.data_process import v5_cli, v5_intraband

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda _: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *args: None)
    monkeypatch.setattr(
        v5_intraband, "run_intraband", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    argv = ["--stage", "band-alignment", "--alignment-version", "v2", "--workers", "3"]
    for key in ("source-root", "dataset-root", "report-root", "base-root"):
        argv += ["--" + key, str(tmp_path / key)]
    with pytest.raises(SystemExit):
        v5_cli.main(argv)
    argv += ["--sensor-family", "gaofen"]
    assert v5_cli.main(argv) == 0
    assert calls == [
        (
            (tmp_path / "dataset-root", tmp_path / "report-root", "gaofen"),
            {"version": "v2", "workers": 3},
        )
    ]
