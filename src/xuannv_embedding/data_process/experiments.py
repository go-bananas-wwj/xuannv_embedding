"""Import and summarize historical training runs into the standard archive."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    """Write the queryable index while keeping the archive JSON as source of truth."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for record in records:
        rows.append(
            {
                "experiment_id": record["experiment_id"],
                "run_id": record["run_id"],
                "status": record["status"],
                "source": record["source"],
                "archive": record["archive"],
                "registry": record["dataset"].get("registry"),
                "data_manifest_sha256": record["dataset"].get("data_manifest_sha256"),
                "configured_epochs": record["training"].get("configured_epochs"),
                "recorded_epochs": record["metrics"].get("last_epoch"),
                "start_epoch": record["training"].get("start_epoch"),
                "first_observed_at_unix": record["timing"].get("first_observed_at_unix"),
                "last_observed_at_unix": record["timing"].get("last_observed_at_unix"),
                "last_wall_seconds": record["metrics"].get("last_wall_seconds"),
                "best_validation_selection_score": record["metrics"].get(
                    "best_validation_selection_score"
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _metrics_summary(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    epochs = [int(row["epoch"]) for row in rows if isinstance(row.get("epoch"), int)]
    walls = [
        float(row["wall_seconds"])
        for row in rows
        if isinstance(row.get("wall_seconds"), (int, float))
    ]
    return {
        "records": len(rows),
        "first_epoch": min(epochs) if epochs else None,
        "last_epoch": max(epochs) if epochs else None,
        "last_wall_seconds": walls[-1] if walls else None,
        "best_validation_selection_score": min(
            (
                float(row["best_validation_selection_score"])
                for row in rows
                if isinstance(row.get("best_validation_selection_score"), (int, float))
            ),
            default=None,
        ),
    }


def _heartbeat_window(run: Path) -> dict[str, Any]:
    values: list[float] = []
    for path in (run / "heartbeats").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("updated_at_unix"), (int, float)):
                values.append(float(value["updated_at_unix"]))
    if not values:
        return {"first_observed_at_unix": None, "last_observed_at_unix": None}
    return {"first_observed_at_unix": min(values), "last_observed_at_unix": max(values)}


def _status(run: Path, metadata: dict[str, Any]) -> str:
    if (run / "training.complete").exists() or any(run.glob("training.complete*")):
        return "completed"
    if metadata:
        return "incomplete"
    return "unknown"


def _dataset_record(metadata: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "registry",
        "observation_artifact",
        "statistics_dir",
        "data_manifest_sha256",
        "static_target_sidecars",
        "osm_reliable_negative_overlay",
        "highres_cloud_sidecar",
    )
    return {key: metadata.get(key) for key in fields if key in metadata}


def _run_readme(record: dict[str, Any]) -> str:
    timing = record["timing"]
    metrics = record["metrics"]
    configured_epochs = record["training"].get("configured_epochs", "未记录")
    recorded_epochs = metrics.get("last_epoch", "未记录")
    return f"""# 训练运行 {record['run_id']}

## 事实记录

- 状态：`{record['status']}`
- 数据集引用：`{record['dataset'].get('registry', '未记录')}`
- 数据清单摘要：`{record['dataset'].get('data_manifest_sha256', '未记录')}`
- 计划轮次：{configured_epochs}；实际记录到：{recorded_epochs}
- 续训起点：`{record['training'].get('start_epoch', '未记录')}`
- 观测时间范围（UTC Unix）：`{timing.get('first_observed_at_unix')}` →
  `{timing.get('last_observed_at_unix')}`
- 最后一条累计训练时间：`{metrics.get('last_wall_seconds', '未记录')} 秒`
- 最佳验证选择分数：`{metrics.get('best_validation_selection_score', '未记录')}`

## 训练设计

该运行由历史目录导入。原始目录名和训练配置摘要保留在 `run.json`；缺少的实验目的、
完整配置或精确启动时间不从目录名推断，需人工补充。

## 数据与复现

原始数据和处理产物通过 `dataset.json` 引用。原始日志、指标和 checkpoint 位于
`artifacts/`；可复现命令和环境信息若未被历史记录保存，则标为未恢复。
"""


def archive_runs(source_root: Path, archive_root: Path, *, mode: str = "symlink") -> dict[str, Any]:
    """Archive runs without copying large artifacts by default.

    ``mode=symlink`` preserves the complete historical tree as a read-only view;
    ``mode=copy`` makes a physical copy and is intentionally opt-in.
    """
    if mode not in {"symlink", "copy"}:
        raise ValueError("mode 必须是 symlink 或 copy")
    source_root = source_root.expanduser().resolve()
    archive_root = archive_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source in sorted(p for p in source_root.iterdir() if p.is_dir()):
        metadata = _read_json(source / "run.json")
        metrics = _metrics_summary(source / "metrics.jsonl")
        timing = _heartbeat_window(source)
        run_id = source.name
        experiment_id = run_id.removesuffix("-5e").removesuffix("-100e").removesuffix("-45e")
        destination = archive_root / experiment_id / "runs" / run_id
        artifacts = destination / "artifacts"
        if destination.exists() and not destination.is_dir():
            raise FileExistsError(destination)
        destination.mkdir(parents=True, exist_ok=True)
        source_run_json = destination / "run.json"
        if not source_run_json.exists():
            source_run_json.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        dataset = _dataset_record(metadata)
        dataset_document = {"schema": "xuannv.experiment-dataset.v1", **dataset}
        (destination / "dataset.json").write_text(
            json.dumps(dataset_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        training_fields = (
            "profile",
            "epochs",
            "configured_epochs",
            "config_training_epochs",
            "scheduler_epochs",
            "start_epoch",
            "continuation",
            "optimizer_steps_per_epoch",
            "total_optimizer_steps",
            "batch_size_per_rank",
            "effective_global_batch",
            "world_size",
        )
        config_source = source / "config.yaml"
        if config_source.is_file():
            shutil.copy2(config_source, destination / "config.yaml")
        else:
            _write(
                destination / "config.yaml",
                "# Historical run did not retain the effective YAML configuration.\n"
                + json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            )
        record = {
            "schema": "xuannv.experiment-run.v1",
            "experiment_id": experiment_id,
            "run_id": run_id,
            "source": str(source),
            "archive": str(destination),
            "status": _status(source, metadata),
            "dataset": dataset,
            "training": {key: metadata.get(key) for key in training_fields if key in metadata},
            "timing": timing,
            "metrics": metrics,
            "imported_at": _now(),
        }
        event = {"event": "imported", "at": record["imported_at"], "source": str(source)}
        (destination / "events.jsonl").write_text(
            json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _write(destination / "README.md", _run_readme(record))
        if not artifacts.exists():
            if mode == "copy":
                shutil.copytree(source, artifacts)
            else:
                artifacts.symlink_to(
                    os.path.relpath(source, artifacts.parent), target_is_directory=True
                )
        records.append(record)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["experiment_id"], []).append(record)
    for experiment_id, experiment_records in grouped.items():
        _write(
            archive_root / experiment_id / "README.md",
            _experiment_readme(experiment_id, experiment_records),
        )

    index_json = archive_root / "catalog" / "experiment_index.json"
    index_json.parent.mkdir(parents=True, exist_ok=True)
    index_document = {
        "schema": "xuannv.experiment-index.v1",
        "records": records,
        "generated_at": _now(),
    }
    index_json.write_text(
        json.dumps(index_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_parquet(archive_root / "index.parquet", records)
    _write(
        archive_root / "README.md",
        """# 玄女 Embedding 实验归档

本目录按实验设计和独立运行归档。每个运行的 `dataset.json` 记录数据集引用，
`run.json` 记录配置摘要，`README.md` 解释训练设计、时间和轮次，`artifacts/`
保留历史输出视图。历史缺失的信息明确标为未记录，不根据目录名推断。

实验索引：`catalog/experiment_index.json`。
""",
    )
    return {
        "schema": "xuannv.experiment-archive.v1",
        "source_root": str(source_root),
        "archive_root": str(archive_root),
        "runs": len(records),
        "index": str(index_json),
    }


def _experiment_readme(experiment_id: str, records: list[dict[str, Any]]) -> str:
    lines = [
        f"# 实验设计：{experiment_id}",
        "",
        "历史运行按数据集、训练轮次和观测时间归档。缺失字段标为未记录；"
        "同名指标只有在数据集和评测口径一致时才可比较。",
        "",
        "| 运行 | 状态 | 数据集 | 计划/实际轮次 | 观测时间 | 最佳验证分数 |",
        "|---|---|---|---:|---|---:|",
    ]
    for record in records:
        dataset = record["dataset"].get("registry", "未记录")
        configured = record["training"].get("configured_epochs", "未记录")
        recorded = record["metrics"].get("last_epoch", "未记录")
        timing = record["timing"]
        observed = f"{timing.get('first_observed_at_unix')} → {timing.get('last_observed_at_unix')}"
        score = record["metrics"].get("best_validation_selection_score", "未记录")
        lines.append(
            f"| [{record['run_id']}]({record['run_id']}/README.md) | {record['status']} | "
            f"{dataset} | {configured} / {recorded} | {observed} | {score} |"
        )
    lines.extend(
        [
            "",
            "## 人工说明",
            "",
            "在这里补充实验目的、基线、设计变化和结论。自动归档只记录可由原始文件证明的事实。",
        ]
    )
    return "\n".join(lines)
