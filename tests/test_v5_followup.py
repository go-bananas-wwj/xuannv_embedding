from pathlib import Path

import pytest

from xuannv_embedding.data_process.v5_followup import next_source_action


def test_followup_catalogs_partial_sources_but_waits_before_full_quality():
    assert next_source_action(total=64, verified=2, cataloged=0, quality_status=None) == "catalog"
    assert next_source_action(total=64, verified=2, cataloged=2, quality_status=None) is None
    assert next_source_action(total=64, verified=64, cataloged=64, quality_status=None) == "quality"
    assert (
        next_source_action(
            total=64, verified=64, cataloged=64, quality_status="inferred_needs_visual_review"
        )
        is None
    )


def test_missing_source_manifest_never_starts_quality():
    assert next_source_action(total=0, verified=0, cataloged=0, quality_status=None) is None


@pytest.mark.parametrize("band_exit,partial_exit", [(0, 0), (1, 0), (0, 1)])
def test_finite_followup_ignores_pilot_completion_and_stops_before_training(
    tmp_path, monkeypatch, band_exit, partial_exit
):
    from types import SimpleNamespace

    import xuannv_embedding.data_process.v5_followup as module
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    args = SimpleNamespace(
        source_root=tmp_path / "source",
        dataset_root=tmp_path / "data",
        report_root=tmp_path / "report",
        base_root=tmp_path / "base",
        model_dir=tmp_path / "models",
        device_id=1,
        alignment_version="v5",
        workers=2,
    )
    write_json(
        args.source_root / "manifests/source.lock.json",
        {"archives": [{"archive": "a", "sha256": "abc"}]},
    )
    write_json(
        args.report_root / "integrity_shards/a.json", {"status": "complete", "sha256": "abc"}
    )
    write_json(args.report_root / "download_worker.json", {"pid": 99999999})
    write_json(
        args.report_root / "jilin_cloud_summary.json",
        {"status": "inferred_needs_visual_review", "selected_scenes": 1, "processed_scenes": 1},
    )
    monkeypatch.setattr(module, "process_identity", lambda pid: None)
    monkeypatch.setattr(module, "report_progress", lambda *args: None)
    actions = []
    import xuannv_embedding.data_process.v5_intraband as intraband

    monkeypatch.setattr(intraband, "current_inventory_fingerprint", lambda *args: "current")
    monkeypatch.setattr(
        module,
        "band_inventory_finished",
        lambda *args: band_exit == 0 and "band-alignment" in actions,
    )

    monkeypatch.setattr(
        module,
        "partial_catalog_finished",
        lambda *args: partial_exit == 0 and "catalog-partial-bands" in actions,
    )

    def run(command, **kwargs):
        action = command[command.index("--stage") + 1]
        actions.append(action)
        catalog = args.dataset_root / "observations/highres/jilin1/files.parquet"
        if action == "catalog":
            catalog.parent.mkdir(parents=True)
            catalog.write_bytes(b"catalog")
            write_json(args.report_root / "grid_match_report.json", {"processed_archives": 1})
        elif action == "catalog-partial-bands":
            return SimpleNamespace(returncode=partial_exit)
        elif action == "band-alignment":
            assert command[command.index("--alignment-version") + 1] == "v5"
            assert command[command.index("--sensor-family") + 1] == "jilin1"
            return SimpleNamespace(returncode=band_exit)
        elif action == "quality":
            write_json(
                args.dataset_root / "quality/cloud/jilin1/source.lock.json",
                {"limit": None, "catalog_sha256": sha256(catalog)},
            )
        else:
            raise AssertionError("unexpected stage")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.follow_started_jobs(args)
    expected = ["catalog", "catalog-partial-bands"]
    if not partial_exit:
        expected += ["band-alignment"] + ([] if band_exit else ["quality"])
    assert actions == expected
    assert len(result["failures"]) == int(bool(band_exit or partial_exit))
    assert result["status"] == "stopped_for_remaining_data_gates"
    assert result["training_authorized"] is False


def test_followup_audits_each_catalog_before_cloud_without_repeating_finished_audit():
    common = dict(total=64, verified=3, cataloged=3, quality_status=None)
    assert next_source_action(**common, band_ready=False) == "band-alignment"
    assert next_source_action(**common, band_ready=True) is None
    assert next_source_action(**{**common, "verified": 4}, band_ready=False) == "catalog"
    full = {**common, "verified": 64, "cataloged": 64}
    assert next_source_action(**full, band_ready=False) == "band-alignment"
    assert next_source_action(**full, band_ready=True) == "quality"
    assert next_source_action(**full, band_ready=False, band_running=True) is None


def test_band_completion_requires_current_version_sources_code_and_output(tmp_path, monkeypatch):
    import xuannv_embedding.data_process.v5_followup as module
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    root = tmp_path / "quality/alignment/intraband/jilin1/v5"
    root.mkdir(parents=True)
    calibration = root / "calibration.json"
    write_json(calibration, {"status": "passed"})
    (root / "observations.parquet").write_bytes(b"audited table")
    code = tmp_path / "algorithm.py"
    code.write_text("version one")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "followup.py"))
    summary = {
        "status": "intraband_audit_finished",
        "version": "v5",
        "family": "jilin1",
        "input_inventory_sha256": "current",
        "calibration_sha256": sha256(calibration),
        "code_sha256": {"algorithm.py": sha256(code)},
        "processed_observations": 100,
        "selected_observations": 100,
        "counts": {"passed": 50, "uncertain": 40, "over_limit": 10},
    }
    assert module.band_inventory_finished(tmp_path, summary, "current", "v5")
    assert not module.band_inventory_finished(tmp_path, summary, "new", "v5")
    assert not module.band_inventory_finished(tmp_path, summary, "current", "v6")
    assert not module.band_inventory_finished(
        tmp_path, {**summary, "processed_observations": 99}, "current", "v5"
    )
    code.write_text("version two")
    assert not module.band_inventory_finished(tmp_path, summary, "current", "v5")
    code.write_text("version one")
    write_json(calibration, {"status": "failed"})
    assert not module.band_inventory_finished(tmp_path, summary, "current", "v5")


def test_followup_refreshes_partial_bands_before_native_audit_or_cloud():
    common = dict(total=64, verified=6, cataloged=6, quality_status=None)
    assert next_source_action(**common, partial_ready=False) == "catalog-partial-bands"
    assert (
        next_source_action(**common, partial_ready=False, band_ready=False)
        == "catalog-partial-bands"
    )
    assert next_source_action(**{**common, "verified": 7}, partial_ready=False) == "catalog"
    assert (
        next_source_action(**{**common, "verified": 64, "cataloged": 64}, partial_ready=False)
        == "catalog-partial-bands"
    )
    assert next_source_action(**common, partial_ready=True, band_ready=True) is None


def test_partial_completion_requires_matching_inputs_runtime_code_and_output(tmp_path):
    import json
    from importlib.metadata import version

    import numpy as np
    import pandas as pd
    import rasterio

    import xuannv_embedding.data_process.v5_followup as module
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    source, data, report = [tmp_path / name for name in ["source", "data", "report"]]
    root = data / "observations/highres/jilin1"
    paths = {
        "registry_sha256": data / "registry/national_62000.parquet",
        "rejected_inventory_sha256": report / "rejected_files.parquet",
        "complete_catalog_sha256": root / "files.parquet",
        "source_lock_sha256": source / "manifests/source.lock.json",
    }
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key)
    fingerprint = {key: sha256(path) for key, path in paths.items()}
    for key, name in [
        ("code_sha256", "v5_partial_bands.py"),
        ("reader_sha256", "v5_rasters.py"),
        ("grid_matcher_sha256", "v5_catalog.py"),
    ]:
        fingerprint[key] = sha256(Path(module.__file__).with_name(name))
    fingerprint["runtime"] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rasterio": rasterio.__version__,
        "gdal": rasterio.__gdal_version__,
        "pyarrow": version("pyarrow"),
    }
    import hashlib

    catalog_version = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    output = root / "partial_bands" / catalog_version
    output.mkdir(parents=True)
    table = output / "files.parquet"
    merged = output / "files_with_partial_bands.parquet"
    table.write_bytes(b"partial table")
    merged.write_bytes(b"merged table")
    lock = output / "catalog.lock.json"
    summary = {
        "status": "partial_band_catalog_finished",
        "fingerprint": fingerprint,
        "output": str(output),
        "output_sha256": sha256(table),
        "augmented_catalog_sha256": sha256(merged),
    }
    pointer = root / "partial_bands/current.json"

    def publish():
        write_json(lock, summary)
        write_json(
            pointer,
            {"version": catalog_version, "lock_path": str(lock), "lock_sha256": sha256(lock)},
        )

    publish()
    assert module.partial_catalog_finished(source, data, report)
    merged.write_bytes(b"changed table")
    assert not module.partial_catalog_finished(source, data, report)
    merged.write_bytes(b"merged table")
    paths["rejected_inventory_sha256"].write_text("new rejected input")
    assert not module.partial_catalog_finished(source, data, report)
    paths["rejected_inventory_sha256"].write_text("rejected_inventory_sha256")
    summary["fingerprint"]["runtime"]["numpy"] = "unknown"
    publish()
    assert not module.partial_catalog_finished(source, data, report)
    pointer.write_text("{")
    assert not module.partial_catalog_finished(source, data, report)
