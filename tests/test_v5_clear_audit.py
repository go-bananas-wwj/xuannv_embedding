import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def test_clear_audit_keeps_cloudy_and_all_splits_in_inventory(tmp_path):
    from test_v5_clear_intraband import gaofen_qa

    from xuannv_embedding.data_process.v5_clear_audit import AuditReader, audit_inventory

    data, qa, _ = gaofen_qa(tmp_path, count=3)
    table = pd.read_parquet(qa / "observation_quality.parquet")
    table["split"] = ["train", "val", "test"]
    table.to_parquet(qa / "observation_quality.parquet", index=False)
    table[["patch_id", "split"]].to_parquet(data / "registry/national_62000.parquet", index=False)
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    config = json.loads((qa / "source.lock.json").read_text())
    config["registry_sha256"] = sha256(data / "registry/national_62000.parquet")
    write_json(qa / "source.lock.json", config)
    import zarr

    for name in ["valid_masks.zarr", "classes.zarr"]:
        zarr.open_group(str(qa / name), mode="a").attrs.update(config)
    rows = audit_inventory(AuditReader(data, "gaofen", qa))
    assert len(rows) == 3 and set(rows.split) == {"train", "val", "test"}


def partial_quality(tmp_path, monkeypatch):
    from test_v5_jilin_quality import Predictor, record

    from xuannv_embedding.data_process import v5_followup
    from xuannv_embedding.data_process.v5_jilin_quality import process_jilin_quality
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    data, source, report, models = [tmp_path / p for p in ["data", "source", "report", "models"]]
    registry = data / "registry/national_62000.parquet"
    registry.parent.mkdir(parents=True)
    pd.DataFrame([{"patch_id": "national", "split": "train"}]).to_parquet(registry, index=False)
    rows = [
        record(tmp_path, "jilin1_ms_5m", ["B2", "B3", "B4", "B5", "B6"], scene="partial"),
        record(tmp_path, "jilin1_ms_5m", ["B1", "B2", "B3", "B5", "B6"], scene="no_ref"),
    ]
    catalog = data / "observations/highres/jilin1/partial_bands/fixed"
    catalog.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(catalog / "files_with_partial_bands.parquet", index=False)
    write_json(
        catalog / "catalog.lock.json", {"fingerprint": {"registry_sha256": sha256(registry)}}
    )
    write_json(
        catalog.parent / "current.json",
        {
            "version": "fixed",
            "lock_path": str(catalog / "catalog.lock.json"),
            "lock_sha256": sha256(catalog / "catalog.lock.json"),
        },
    )
    monkeypatch.setattr(v5_followup, "partial_catalog_finished", lambda *a: True)
    models.mkdir()
    for i in (0, 1):
        (models / f"ocm_v4_model_{i}_96_910b4.om").write_bytes(b"fixture")

    class Factory(Predictor):
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    result = process_jilin_quality(source, data, report, models, predictor_factory=Factory)
    return data, Path(result["output"]), rows


def test_partial_band_audit_keeps_pairs_and_missing_reference_uncertain(tmp_path, monkeypatch):
    from xuannv_embedding.data_process.v5_clear_audit import (
        AuditReader,
        audit_inventory,
        inspect_clear,
    )

    data, qa, rows = partial_quality(tmp_path, monkeypatch)
    reader = AuditReader(data, "jilin1", qa)
    inventory = audit_inventory(reader)
    assert len(inventory) == 2
    for row in inventory.itertuples():
        result = inspect_clear(reader, row)
        assert result["status"] == "uncertain" and len(result["pairs"]) == 5
        assert not result["pixel_fusion_authorized"]
        if row.observation_id.startswith("partial"):
            assert result["missing_bands"] == ["B1"]
            pair = next(p for p in result["pairs"] if p["moving_band"] == "B1")
            assert pair["reason"] == "missing_spectral_band"
        else:
            assert result["missing_bands"] == ["B4"]
            assert all(p["reason"] == "missing_reference_band" for p in result["pairs"])
    import zarr

    masks = zarr.open_group(str(reader.root / "valid_masks.zarr"), mode="a")
    key = "partial:jilin1_ms_5m/valid"
    masks[key][1, 0, 0] = True
    with pytest.raises(ValueError, match="QA mask changed"):
        reader.read(SimpleNamespace(**rows[0]))


def test_clear_receipts_recheck_source_masks_and_reject_result_tampering(tmp_path, monkeypatch):
    from test_v5_clear_intraband import gaofen_qa

    from xuannv_embedding.data_process import v5_clear_audit as module

    data, qa, rows = gaofen_qa(tmp_path)
    reader = module.AuditReader(data, "gaofen", qa)
    row = next(rows.itertuples())
    root = tmp_path / "audit"
    result, reused = module.audit_one(reader, row, root, {"algorithm": "fixed"})
    assert not reused and result["status"] == "over_limit"
    monkeypatch.setattr(module, "inspect_clear", lambda *a, **k: pytest.fail("recomputed cache"))
    cached, reused = module.audit_one(reader, row, root, {"algorithm": "fixed"})
    assert reused and cached == result
    path = next(root.glob("receipts/*/*.json"))
    receipt = json.loads(path.read_text())
    receipt["result"]["status"] = "passed"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt result changed"):
        module.audit_one(reader, row, root, {"algorithm": "fixed"})
    Path(row.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source file changed"):
        module.audit_one(reader, row, root, {"algorithm": "fixed"})


def test_clear_audit_requires_sealed_passed_calibration(tmp_path):
    from xuannv_embedding.data_process.v5_clear_audit import verify_calibration

    with pytest.raises(FileNotFoundError):
        verify_calibration(tmp_path, tmp_path, "gaofen", tmp_path / "calibration")


def test_clear_audit_cli_requires_roots_and_has_family_lock(tmp_path, monkeypatch):
    import fcntl

    from xuannv_embedding.data_process import v5_clear_audit, v5_cli

    monkeypatch.setattr(v5_cli, "lock_source", lambda *a: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *a: None)
    calls = []
    monkeypatch.setattr(v5_clear_audit, "run_clear_audit", lambda *a, **k: calls.append((a, k)))
    args = ["--stage", "clear-band-audit", "--workers", "2"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        args += ["--" + key, str(tmp_path / key)]
    for extra in [[], ["--sensor-family", "gaofen"], ["--quality-root", str(tmp_path / "qa")]]:
        args += extra
        with pytest.raises(SystemExit):
            v5_cli.main(args)
    args += ["--clear-calibration-root", str(tmp_path / "calibration")]
    source = tmp_path / "source-root"
    source.mkdir(exist_ok=True)
    with (source / ".clear-band-audit.gaofen.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another V5"):
            v5_cli.main(args)
    assert v5_cli.main(args) == 0
    assert calls[0][1] == {"workers": 2, "limit": None}


def test_clear_audit_process_replay_freezes_outputs_and_detects_tampering(tmp_path):
    from test_v5_clear_training import prepared_qa

    from xuannv_embedding.data_process.v5_clear_audit import run_clear_audit
    from xuannv_embedding.data_process.v5_clear_training import calibrate_training
    from xuannv_embedding.data_process.v5_sources import sha256

    data, qa, report = prepared_qa(tmp_path)
    calibration = calibrate_training(data, report, "gaofen", qa)
    root = Path(calibration["output"]).parent
    first = run_clear_audit(data, report, "gaofen", qa, root, workers=2)
    assert first["processed"] == 8 and first["counts"].get("rejected", 0) == 0
    assert first["execution_status"] == "finished" and not first["pixel_fusion_authorized"]
    output = Path(first["output"])
    files = [
        output / n
        for n in ["inputs.parquet", "inputs.lock.json", "observations.parquet", "output.lock.json"]
    ]
    before = [(sha256(p), p.stat().st_mtime_ns) for p in files]
    replay = run_clear_audit(data, report, "gaofen", qa, root, workers=1)
    assert replay["counts"]["reused"] == 8
    assert [(sha256(p), p.stat().st_mtime_ns) for p in files] == before
    table = pd.read_parquet(output / "observations.parquet")
    table.loc[0, "status"] = "fabricated"
    table.to_parquet(output / "observations.parquet", index=False)
    with pytest.raises(ValueError, match="output seal changed"):
        run_clear_audit(data, report, "gaofen", qa, root, workers=2)
