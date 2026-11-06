from pathlib import Path

import pytest

from xuannv_embedding.data_process.v5_followup import next_source_action


def test_followup_catalogs_each_source_batch_before_incremental_quality():
    assert next_source_action(total=64, verified=2, cataloged=0, quality_status=None) == "catalog"
    assert (
        next_source_action(total=64, verified=2, cataloged=2, quality_status=None)
        == "jilin-quality"
    )
    assert (
        next_source_action(total=64, verified=64, cataloged=64, quality_status=None)
        == "jilin-quality"
    )
    assert (
        next_source_action(
            total=64, verified=64, cataloged=64, quality_status="inferred_needs_visual_review"
        )
        is None
    )


def test_missing_source_manifest_never_starts_quality():
    assert next_source_action(total=0, verified=0, cataloged=0, quality_status=None) is None


@pytest.mark.parametrize("band_ready,partial_exit", [(True, 0), (False, 0), (False, 1)])
def test_finite_followup_ignores_pilot_completion_and_stops_before_training(
    tmp_path, monkeypatch, band_ready, partial_exit
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
        lambda *args: band_ready,
    )

    monkeypatch.setattr(
        module,
        "partial_catalog_finished",
        lambda *args: partial_exit == 0 and "catalog-partial-bands" in actions,
    )

    monkeypatch.setattr(module, "jilin_quality_finished", lambda *args: "jilin-quality" in actions)

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
        elif action == "jilin-quality":
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
        expected += ["jilin-quality"]
    assert actions == expected
    assert len(result["failures"]) == int(bool(partial_exit))
    assert result["status"] == "stopped_for_remaining_data_gates"
    assert result["training_authorized"] is False
    assert result["band_inventory_finished"] is band_ready
    assert result["alignment_required_for_quality"] is False


def test_followup_quality_does_not_wait_for_optional_native_alignment():
    common = dict(total=64, verified=3, cataloged=3, quality_status=None)
    assert next_source_action(**common, band_ready=False) == "jilin-quality"
    assert next_source_action(**common, band_ready=True) == "jilin-quality"
    assert next_source_action(**{**common, "verified": 4}, band_ready=False) == "catalog"
    full = {**common, "verified": 64, "cataloged": 64}
    assert next_source_action(**full, band_ready=False) == "jilin-quality"
    assert next_source_action(**full, band_ready=True) == "jilin-quality"
    assert next_source_action(**{**full, "quality_status": "finished"}, band_ready=False) is None
    assert next_source_action(**full, band_ready=False, band_running=True) == "jilin-quality"
    assert (
        next_source_action(**full, band_ready=False, band_running=True, quality_running=True)
        is None
    )


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


def test_followup_refreshes_partial_bands_before_cloud():
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
    assert next_source_action(**common, partial_ready=True, band_ready=True) == "jilin-quality"


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


def test_incremental_quality_never_shares_an_active_cloud_worker():
    assert (
        next_source_action(
            total=64, verified=6, cataloged=6, quality_status=None, quality_running=True
        )
        is None
    )
    assert (
        next_source_action(
            total=64, verified=6, cataloged=6, quality_status="finished", quality_running=False
        )
        is None
    )


def test_quality_completion_rejects_pilots_stale_catalogs_and_changed_outputs(tmp_path):
    import hashlib
    import json
    from importlib.metadata import version

    import numpy as np
    import pandas as pd
    import rasterio
    import zarr

    import xuannv_embedding.data_process.v5_followup as module
    from xuannv_embedding.data_process.v5_sources import sha256, write_json

    data, report, models = [tmp_path / name for name in ["data", "report", "models"]]
    models.mkdir()
    model = models / "weight.om"
    model.write_bytes(b"weight")
    pointer = data / "observations/highres/jilin1/partial_bands/current.json"
    catalog_lock = data / "observations/highres/jilin1/partial_bands/partial/catalog.lock.json"
    write_json(catalog_lock, {"catalog": "one"})
    write_json(
        pointer,
        {"version": "partial", "lock_path": str(catalog_lock), "lock_sha256": sha256(catalog_lock)},
    )
    code = Path(module.__file__).with_name("v5_jilin_quality.py")
    config = {
        "code_sha256": {code.name: sha256(code)},
        "models_sha256": {model.name: sha256(model)},
        "runtime": {
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "gdal": rasterio.__gdal_version__,
            "zarr": zarr.__version__,
            "pandas": pd.__version__,
            "pyarrow": version("pyarrow"),
        },
    }
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    root = data / "quality/cloud/jilin1/v2" / config_id
    output = root / "catalogs/snapshot"
    write_json(root / "configuration.lock.json", config)
    output.mkdir(parents=True)
    table = output / "observation_quality.parquet"
    table.write_bytes(b"table")
    lock = output / "quality.lock.json"
    locked = {
        "snapshot": {
            "limit": None,
            "catalog_lock_sha256": sha256(catalog_lock),
            "configuration_id": config_id,
        },
        "processed_scenes": 10,
        "quality_table_sha256": sha256(table),
    }
    summary = {
        "status": "inferred_needs_visual_review",
        "partial_catalog_version": "partial",
        "selected_scenes": 10,
        "processed_scenes": 10,
        "output": str(output),
        "quality_root": str(root),
    }

    def publish():
        write_json(lock, locked)
        summary["quality_lock_sha256"] = sha256(lock)
        write_json(report / "jilin_quality_v2_full.json", summary)

    publish()
    assert module.jilin_quality_finished(data, report, models)
    locked["snapshot"]["limit"] = 32
    publish()
    assert not module.jilin_quality_finished(data, report, models)
    locked["snapshot"]["limit"] = None
    publish()
    summary["partial_catalog_version"] = "older"
    publish()
    assert not module.jilin_quality_finished(data, report, models)
    summary["partial_catalog_version"] = "partial"
    publish()
    table.write_bytes(b"damaged")
    assert not module.jilin_quality_finished(data, report, models)
    table.write_bytes(b"table")
    model.write_bytes(b"changed")
    assert not module.jilin_quality_finished(data, report, models)
