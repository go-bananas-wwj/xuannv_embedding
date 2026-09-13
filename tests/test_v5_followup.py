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


def test_finite_followup_ignores_pilot_completion_and_stops_before_training(tmp_path, monkeypatch):
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

    def run(command, **kwargs):
        action = command[command.index("--stage") + 1]
        actions.append(action)
        catalog = args.dataset_root / "observations/highres/jilin1/files.parquet"
        if action == "catalog":
            catalog.parent.mkdir(parents=True)
            catalog.write_bytes(b"catalog")
            write_json(args.report_root / "grid_match_report.json", {"processed_archives": 1})
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
    assert actions == ["catalog", "quality"]
    assert result["status"] == "stopped_for_remaining_data_gates"
    assert result["training_authorized"] is False
