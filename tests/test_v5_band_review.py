import numpy as np
import pytest

from xuannv_embedding.data_process.v5_band_review import select_examples, stretch_band


def test_review_selection_is_deterministic_unique_and_covers_status_sensor_groups():
    rows = [
        {
            "observation_id": f"{i:03d}",
            "patch_id": f"p{i // 2}",
            "sensor": "A" if i < 12 else "B",
            "status": ["passed", "uncertain", "over_limit"][i % 3],
        }
        for i in range(24)
    ]
    selected = select_examples(rows, sample_size=9)
    assert selected == select_examples(rows[::-1], sample_size=9)
    assert len(selected) == len({r["patch_id"] for r in selected}) == 9
    assert {r["status"] for r in selected} == {"passed", "uncertain", "over_limit"}
    assert {r["sensor"] for r in selected} == {"A", "B"}
    assert len(select_examples(rows[:2], sample_size=9)) == 1
    with pytest.raises(ValueError):
        select_examples(rows, sample_size=0)


def test_display_stretch_preserves_source_and_never_uses_invalid_values():
    source = np.arange(100, dtype="f4").reshape(10, 10)
    valid = np.ones_like(source, dtype=bool)
    valid[0] = False
    changed = source.copy()
    changed[0] = np.nan
    before = changed.copy()
    np.testing.assert_equal(stretch_band(source, valid), stretch_band(changed, valid))
    np.testing.assert_equal(changed, before)
    assert not stretch_band(changed, np.zeros_like(valid)).any()


def test_review_renders_frozen_receipts_and_rejects_changed_source(tmp_path, monkeypatch):
    import json

    import pandas as pd
    import rasterio
    from rasterio.transform import from_origin

    from xuannv_embedding.data_process import v5_band_review as module
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    source = tmp_path / "source.tif"
    values = np.random.default_rng(9).uniform(100, 1000, (4, 160, 160)).astype("f4")
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        count=4,
        height=160,
        width=160,
        dtype="float32",
        crs="EPSG:32650",
        transform=from_origin(300000, 4000000, 8, 8),
    ) as dst:
        dst.write(values)
    source_hash = sha256(source)
    row = {
        "observation_id": "scene",
        "patch_id": "patch",
        "sensor": "GF1",
        "split": "train",
        "year": 2020,
        "path": str(source),
        "file_sha256": source_hash,
    }
    monkeypatch.setattr(module, "_inventory", lambda *args: pd.DataFrame([row]))
    data, report = tmp_path / "data", tmp_path / "report"
    audit = data / "quality/alignment/intraband/gaofen/v1"
    calibration = audit / "calibration.json"
    write_json(
        calibration,
        {"status": "passed", "fingerprint": {"code_sha256": module._code_fingerprint()}},
    )
    fingerprint = {**row, "calibration_sha256": sha256(calibration)}
    pairs = [
        {"moving_band": b, "status": "passed", "translation_yx_m": [0, 0], "windows": []}
        for b in ["blue", "red", "nir"]
    ]
    write_json(
        audit / "receipts/ab/sample.json",
        {"fingerprint": fingerprint, "result": {**row, "status": "passed", "pairs": pairs}},
    )
    result = module.review_native_bands(data, report, "gaofen", sample_size=3)
    assert result["rendered"] == 1 and result["required"] == 3
    assert result["pixel_fusion_authorized"] is False
    assert sha256(source) == source_hash
    gallery = report / "diagnostics/native_band/gaofen/v1"
    assert (gallery / "sample_00.png").stat().st_size > 1000
    before = sha256(gallery / "samples.json")
    assert module.review_native_bands(data, report, "gaofen", sample_size=3)["rendered"] == 1
    assert sha256(gallery / "samples.json") == before
    rows = json.loads((gallery / "samples.json").read_text())["samples"]
    assert len(rows) == 1
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source file changed"):
        module.review_native_bands(data, report, "gaofen", sample_size=3)


def test_band_review_cli_requires_family_and_uses_data_only_version_route(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_band_review, v5_cli

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda _: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *args: None)
    monkeypatch.setattr(v5_band_review, "review_native_bands", lambda *a, **k: calls.append((a, k)))
    argv = ["--stage", "band-review", "--alignment-version", "v5"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        argv += ["--" + key, str(tmp_path / key)]
    with pytest.raises(SystemExit):
        v5_cli.main(argv)
    assert v5_cli.main(argv + ["--sensor-family", "jilin1"]) == 0
    assert calls == [
        ((tmp_path / "dataset-root", tmp_path / "report-root", "jilin1"), {"version": "v5"})
    ]
