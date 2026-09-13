import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, shift


def _row(tmp_path, identity="observation"):
    from xuannv_embedding.data_process.v5_sources import sha256

    path = tmp_path / (identity + ".tif")
    texture = gaussian_filter(np.random.default_rng(37).normal(size=(160, 160)), 1).astype("f4")
    pixels = np.stack([texture, texture, texture, shift(texture, (2, -1), order=1, mode="reflect")])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=160,
        height=160,
        count=4,
        dtype="float32",
        crs="EPSG:32650",
        transform=from_origin(300000, 4000000, 8, 8),
    ) as ds:
        ds.write(pixels)
    return {
        "observation_id": identity,
        "patch_id": identity,
        "path": str(path),
        "file_sha256": sha256(path),
        "sensor": "GF1",
        "split": "train",
        "year": 2020,
    }


def test_process_worker_reuses_legacy_receipt_and_rejects_changed_source(tmp_path):
    from xuannv_embedding.data_process.v5_parallel_intraband import inspect_job
    from xuannv_embedding.data_process.v5_sources import write_json

    row = _row(tmp_path)
    fingerprint = {
        "calibration_sha256": "calibration",
        "code_sha256": {},
        "version": "v5",
        "family": "gaofen",
    }
    root = tmp_path / "audit"
    key = hashlib.sha256(row["observation_id"].encode()).hexdigest()
    receipt = root / "receipts" / key[:2] / (key + ".json")
    expected = {**fingerprint, **{k: v for k, v in row.items() if k != "path"}}
    legacy = {
        "status": "over_limit",
        "observation_id": row["observation_id"],
        "pairs": [],
        "pixel_fusion_authorized": False,
        "finished_at": "fixed-original-time",
    }
    write_json(receipt, {"fingerprint": expected, "result": legacy})
    before = receipt.read_bytes()
    result, reused = inspect_job((row, "gaofen", str(root), fingerprint))
    assert reused and result == legacy and receipt.read_bytes() == before
    Path(row["path"]).write_bytes(b"changed source")
    result, reused = inspect_job((row, "gaofen", str(root), fingerprint))
    assert not reused and result["status"] == "rejected" and not result["pixel_fusion_authorized"]


def test_process_execution_matches_core_and_replays_without_changing_calibration(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import pandas as pd

    from xuannv_embedding.data_process import v5_intraband as core
    from xuannv_embedding.data_process import v5_parallel_intraband as module
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    data, report = tmp_path / "data", tmp_path / "report"
    rows = pd.DataFrame([_row(tmp_path, "first"), _row(tmp_path, "second")])
    calibration = data / "quality/alignment/intraband/gaofen/v5/calibration.json"
    write_json(calibration, {"status": "passed"})
    monkeypatch.setattr(
        module, "calibrate_family", lambda *a, **kw: {"status": "passed", "sensors": {"GF1": {}}}
    )
    monkeypatch.setattr(module, "_inventory", lambda *a: rows)
    original_hash = sha256(calibration)
    result = module.run_parallel_intraband(data, report, "gaofen", version="v5", workers=2)
    table = pd.read_parquet(result["output"])
    assert result["processed_observations"] == 2
    for row in rows.itertuples():
        frame, gsd, reference = core._read_row(SimpleNamespace(**row._asdict()), "gaofen")
        expected = core.inspect_intraband(frame, reference_band=reference, gsd=gsd)
        actual = table.loc[table.observation_id == row.observation_id].iloc[0]
        assert json.loads(actual.pairs) == expected["pairs"] and actual.status == expected["status"]
    assert sha256(calibration) == original_hash
    replay = module.run_parallel_intraband(data, report, "gaofen", version="v5", workers=2)
    assert replay["counts"]["reused"] == 2 and not replay["pixel_fusion_authorized"]
    assert replay["execution"]["backend"] == "processes"
    with pytest.raises(ValueError, match="workers"):
        module.run_parallel_intraband(data, report, "gaofen", version="v5", workers=0)


def test_parallel_cli_requires_family_and_shares_existing_writer_lock(tmp_path, monkeypatch):
    import fcntl

    from xuannv_embedding.data_process import v5_cli, v5_parallel_intraband

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(
        v5_parallel_intraband, "run_parallel_intraband", lambda *a, **kw: calls.append((a, kw))
    )
    args = ["--stage", "band-alignment-parallel", "--alignment-version", "v5", "--workers", "8"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + key, str(tmp_path / key)]
    with pytest.raises(SystemExit):
        v5_cli.main(args)
    args += ["--sensor-family", "gaofen"]
    source = tmp_path / "source-root"
    source.mkdir(exist_ok=True)
    with (source / ".band-alignment.gaofen.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another V5"):
            v5_cli.main(args)
    assert v5_cli.main(args) == 0
    assert calls[0][1] == {"version": "v5", "workers": 8}
