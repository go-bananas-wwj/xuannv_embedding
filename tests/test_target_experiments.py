import copy
import hashlib
from pathlib import Path

import pytest
import yaml
from test_annual_model import annual_config

from xuannv_embedding.config import Config
from xuannv_embedding.data_process.target_audit import digest, write_json
from xuannv_embedding.training.target_experiments import prepare_experiments
from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest


def test_target_plan_preserves_paired_budgets_and_midpoint_checkpoints(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    positions = [f"parent_32643:{index + 100}:3200" for index in range(16384)]
    split = {"screen_patch_ids": positions[:8192], "expanded_patch_ids": positions}
    write_json(prepared / "experiment_splits.json", split)
    write_json(
        prepared / "preparation.json",
        {
            "splits_sha256": digest(prepared / "experiment_splits.json"),
            "formal_training_released": False,
        },
    )
    write_json(
        prepared / "train_statistics.screen.json",
        {"targets": {"dem_elevation": {"mean": 100, "population_std": 20}}},
    )
    raw = annual_config().to_dict()
    template = copy.deepcopy(raw["data"]["datasets"][0])
    raw["data"]["datasets"] = []
    bindings = []
    for year in (2020, 2021):
        folder = tmp_path / str(year)
        folder.mkdir()
        (folder / "statistics").mkdir()
        for name in ("summary.json", "sources.json"):
            (folder / name).write_text("{}")
        path = folder / f"{year}.train.manifest.jsonl"
        records = [
            ManifestRecord(
                patch_id=patch,
                region=f"national{year}",
                sources={"s2": None},
                provenance={"year": year},
            )
            for patch in positions
        ]
        receipt = write_manifest(
            path,
            records,
            months=[f"{year}-{month:02d}" for month in range(1, 13)],
            generator_version="fixture",
        )
        bindings.append({"path": str(path), "sha256": receipt.sha256})
        dataset = copy.deepcopy(template)
        dataset.update(
            region=f"national{year}",
            manifest_path=str(path),
            patch_grid_path=str(path),
            statistics_dir=str(folder / "statistics"),
            source_map={name: name for name in raw["model"]["input_sources"]},
        )
        raw["data"]["datasets"].append(dataset)
    initial = tmp_path / "original.yaml"
    initial.write_text(yaml.safe_dump(raw))
    write_json(
        prepared / "pretraining_split_binding.json",
        {"config_sha256": digest(initial), "original_pretraining_manifests": bindings},
    )
    plan = prepare_experiments(prepared, tmp_path / "targets", initial, tmp_path / "experiments")
    assert len(plan["configs"]) == 11
    assert not plan["training_started"] and not plan["formal_target_training_released"]
    for scope, count in [("engineering", 2048), ("screen", 8192), ("expanded", 16384)]:
        assert [row["records"] for row in plan["manifests"][scope]] == [count, count]
    for tag, receipt in plan["configs"].items():
        path = Path(receipt["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
        config = Config.from_yaml(path)
        assert config.training.gradient_checkpointing
        assert config.model.annual_exact_mask_coverage
        assert config.model.annual_context_pooling == (tag.split("/")[-1] != "R0")
        if tag.startswith("screen/"):
            batches_per_epoch = 16384 // 32
            assert config.training.epochs * batches_per_epoch == 4096
            assert config.training.save_every * batches_per_epoch == 2048
    with pytest.raises(FileExistsError):
        prepare_experiments(prepared, tmp_path / "targets", initial, tmp_path / "experiments")
