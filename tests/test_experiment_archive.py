from __future__ import annotations

import json
from pathlib import Path

from xuannv_embedding.data_process.experiments import archive_runs


def test_archive_records_dataset_epochs_and_heartbeat(tmp_path: Path) -> None:
    source = tmp_path / "source"
    run = source / "design-5e"
    (run / "heartbeats").mkdir(parents=True)
    (run / "heartbeats/rank.jsonl").write_text(
        '{"updated_at_unix": 10, "epoch": 1}\n{"updated_at_unix": 20, "epoch": 5}\n',
        encoding="utf-8",
    )
    (run / "run.json").write_text(
        json.dumps(
            {
                "configured_epochs": 5,
                "start_epoch": 0,
                "registry": "dataset.parquet",
                "data_manifest_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.jsonl").write_text(
        '{"epoch": 5, "wall_seconds": 12.5, "best_validation_selection_score": 0.2}\n',
        encoding="utf-8",
    )
    (run / "training.complete").write_text("complete\n", encoding="utf-8")
    result = archive_runs(source, tmp_path / "archive")
    assert result["runs"] == 1
    destination = tmp_path / "archive/design/runs/design-5e"
    record = json.loads((destination / "dataset.json").read_text())
    assert record["registry"] == "dataset.parquet"
    readme = (destination / "README.md").read_text()
    assert "计划轮次" in readme and "实际记录到：5" in readme
    assert (destination / "artifacts").is_symlink()
