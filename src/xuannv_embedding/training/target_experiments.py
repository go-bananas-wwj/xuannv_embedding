"""Materialize already frozen target experiments; do not choose recipes or run training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xuannv_embedding.data_process.target_audit import digest, write_json


def prepare_experiments(
    prepared: Path, target_root: Path, initial_config: Path, output: Path
) -> dict:
    import yaml

    from xuannv_embedding.config import Config
    from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest

    if output.exists():
        raise FileExistsError("Target experiment directory must be new")
    preparation = json.loads((prepared / "preparation.json").read_text())
    split = json.loads((prepared / "experiment_splits.json").read_text())
    if digest(prepared / "experiment_splits.json") != preparation["splits_sha256"]:
        raise ValueError("Frozen target experiment split changed")
    binding = json.loads((prepared / "pretraining_split_binding.json").read_text())
    if digest(initial_config) != binding["config_sha256"]:
        raise ValueError("Original pretraining configuration identity changed")
    original = Config.from_yaml(initial_config)
    original_raw = yaml.safe_load(initial_config.read_text())
    engineering = sorted(
        split["screen_patch_ids"],
        key=lambda patch: hashlib.sha256(f"engineering-v1/{patch}".encode()).digest(),
    )[:2048]
    selections = {
        "engineering": set(engineering),
        "screen": set(split["screen_patch_ids"]),
        "expanded": set(split["expanded_patch_ids"]),
    }
    if len(engineering) != 2048:
        raise ValueError(
            "Engineering needs2048paired positions to exercise both years in128DDPsteps"
        )
    output.mkdir(parents=True)
    manifests = {scope: [] for scope in selections}
    source_receipts = []
    for dataset in original.data.datasets:
        source = dataset.manifest_path
        metadata = json.loads(Path(str(source) + ".meta.json").read_text())
        expected = next(
            row["sha256"]
            for row in binding["original_pretraining_manifests"]
            if row["path"] == str(source)
        )
        hasher, count = hashlib.sha256(), 0
        chosen = {scope: [] for scope in selections}
        with source.open("rb") as handle:
            for line in handle:
                hasher.update(line)
                if not line.strip():
                    continue
                count += 1
                raw = json.loads(line)
                patch = raw["patch_id"]
                if patch not in selections["expanded"]:
                    continue
                record = ManifestRecord.from_dict(raw)
                for scope, wanted in selections.items():
                    if patch in wanted:
                        chosen[scope].append(record)
        if hasher.hexdigest() != expected or count != metadata["record_count"]:
            raise ValueError("Original pretraining manifest changed")
        for scope, records in chosen.items():
            if {record.patch_id for record in records} != selections[scope]:
                raise ValueError("Paired target selection is not fully covered in each year")
            directory = output / "data" / scope
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("summary.json", "sources.json", "statistics"):
                path = directory / name
                if not path.exists():
                    path.symlink_to(source.parent / name, target_is_directory=name == "statistics")
            destination = directory / source.name
            receipt = write_manifest(
                destination,
                records,
                months=metadata["months"],
                generator_version="target-supervision-paired-v1",
            )
            manifests[scope].append(
                {
                    "region": dataset.region,
                    "year": records[0].provenance["year"],
                    "path": str(destination.resolve()),
                    "sha256": receipt.sha256,
                    "records": len(records),
                }
            )
        source_receipts.append({"path": str(source), "sha256": expected})
    statistics = json.loads((prepared / "train_statistics.screen.json").read_text())
    height = statistics["targets"]["dem_elevation"]
    target_spec = {
        "schema": "annual-dem40m-worldcover-v1",
        "target_root": str(target_root.resolve()),
        "prepared_root": str(prepared.resolve()),
        "preparation_sha256": digest(prepared / "preparation.json"),
        "scope": "screen",
        "dem_mean": height["mean"],
        "dem_std": height["population_std"],
        "head_lr_multiplier": 10.0,
    }
    configs = {}
    recipes = {
        "R0": {},
        "R1": {},
        "D1": {"dem_elevation": 0.1},
        "D2": {"worldcover": 0.1},
        "D3": {"dem_elevation": 0.1, "worldcover": 0.1},
    }
    for scope, seeds in (("engineering", (42,)), ("screen", (17, 42))):
        for seed in seeds:
            for recipe, weights in recipes.items():
                if scope == "engineering" and recipe != "D3":
                    continue
                raw = json.loads(json.dumps(original_raw))
                raw["experiment"].update(name=f"targets_{scope}_{recipe}_s{seed}", seed=seed)
                raw["model"].update(
                    annual_context_pooling=recipe != "R0", annual_exact_mask_coverage=True
                )
                raw["training"].update(
                    epochs=2 if scope == "engineering" else 8,
                    lr=3e-5,
                    warmup_epochs=1,
                    save_every=8,
                    compile=False,
                    gradient_checkpointing=True,
                    latent_prediction_weight=0.0,
                    view_consistency_weight=0.0,
                )
                if weights:
                    raw["training"]["independent_targets"] = {
                        **target_spec,
                        "weights": weights,
                        "ramp_batches": 128 if scope == "engineering" else 512,
                        "engineering_only": scope == "engineering",
                    }
                for dataset, frozen in zip(raw["data"]["datasets"], manifests[scope], strict=True):
                    dataset.update(
                        manifest_path=frozen["path"],
                        patch_grid_path=frozen["path"],
                        statistics_dir=str(Path(frozen["path"]).parent / "statistics"),
                    )
                raw["data"].update(batch_size=4, num_workers=4)
                path = output / "configs" / scope / f"seed{seed}" / f"{recipe}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
                Config.from_yaml(path)
                configs[f"{scope}/{seed}/{recipe}"] = {
                    "path": str(path.resolve()),
                    "sha256": digest(path),
                }
    result = {
        "schema": "xuannv.target-experiment-plan.v1",
        "training_started": False,
        "formal_target_training_released": preparation["formal_training_released"],
        "original_config": str(initial_config.resolve()),
        "original_config_sha256": digest(initial_config),
        "preparation_sha256": digest(prepared / "preparation.json"),
        "split_sha256": preparation["splits_sha256"],
        "manifests": manifests,
        "configs": configs,
        "engineering_patch_ids": engineering,
        "original_training_manifests": source_receipts,
        "input_normalization": "Inherited original pretrained input statistics "
        "for paired continued training; new independent DSM normalization uses only "
        "the 8192 screen training positions.",
        "expanded_recipe": "Not selected; only paired expanded data identities "
        "have been materialized.",
        "engineering_bound": "128updates plus at most2resume-checkupdates; no accuracy selection",
        "memory_policy": "All new arms enable gradient checkpointing; no parameter-schema "
        "change. Supersedes the initial no-checkpointing pilot after NCCL finalization OOM.",
    }
    write_json(output / "plan.json", result)
    return result
