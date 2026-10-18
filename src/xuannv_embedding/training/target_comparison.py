"""Freeze a bounded two-recipe comparison without reopening confirmation identities."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from xuannv_embedding.config import Config
from xuannv_embedding.training.annual_experiments import digest, write_json
from xuannv_embedding.training.cli import _git_sha, _source_code_sha256
from xuannv_embedding.utils.manifest import IndexedManifest


def prepare_comparison(local_plan: Path, evaluation_plan: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Comparison output must be new")
    local = json.loads(local_plan.read_text())
    old = json.loads(evaluation_plan.read_text())
    parent_path = Path(local["parent_plan"]["path"])
    if digest(parent_path) != local["parent_plan"]["sha256"]:
        raise ValueError("Frozen parent experiment plan changed")
    parent = json.loads(parent_path.read_text())
    raw_path = Path(local["configs"]["multisource"]["path"])
    if digest(raw_path) != local["configs"]["multisource"]["sha256"]:
        raise ValueError("Local supervision template changed")
    template = yaml.safe_load(raw_path.read_text())
    target = template["training"]["independent_targets"]
    prepared = Path(target["prepared_root"])
    if digest(prepared / "preparation.json") != target["preparation_sha256"]:
        raise ValueError("Local preparation identity changed")
    splits = json.loads((prepared / "experiment_splits.json").read_text())
    preparation = json.loads((prepared / "preparation.json").read_text())
    if digest(prepared / "experiment_splits.json") != preparation["splits_sha256"]:
        raise ValueError("Frozen positions changed")
    wanted = set(splits["screen_patch_ids"])
    labels = json.loads((Path(old["labels"]) / "labels.json").read_text())
    cities = json.loads(Path(old["cities"]).read_text())
    evaluation_ids = {row["patch_id"] for row in labels["records"]} | {row[2] for row in cities}
    if wanted & evaluation_ids:
        raise ValueError("Continuation training overlaps a downstream evaluation position")
    assets = {
        str(local_plan.resolve()): digest(local_plan),
        str(evaluation_plan.resolve()): digest(evaluation_plan),
    }
    for path in [
        prepared / "preparation.json",
        prepared / "experiment_splits.json",
        prepared / "local_targets/manifest.json",
    ]:
        assets[str(path.resolve())] = digest(path)
    for row in parent["manifests"]["screen"]:
        path = Path(row["path"])
        if digest(path) != row["sha256"] or row["records"] != 8192:
            raise ValueError("Screen manifest identity or budget changed")
        if {record.patch_id for record in IndexedManifest(path)} != wanted:
            raise ValueError("Screen manifests do not contain the frozen paired positions")
        assets[str(path)] = row["sha256"]
    # Reuse the previous development protocol, including its held-out reconstruction IDs.
    for path, value in old["assets"].items():
        if "run_200ep_" not in path and "small_exact" not in path:
            assets[path] = value
    for row in labels["records"]:
        assets[str((Path(old["labels"]) / row["file"]).resolve())] = row["sha256"]
    for patch in sorted({row[2] for row in cities}):
        path = Path(old["osm"]) / f"{patch}.json"
        assets[str(path.resolve())] = digest(path)
    for path, expected in assets.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen source asset changed: {path}")
    output.mkdir(parents=True)
    arms = {}
    for seed in (42, 17):
        for recipe in ("control", "multisource"):
            tag = f"{recipe}_s{seed}"
            raw = copy.deepcopy(template)
            raw["experiment"].update(name=f"targets_comparison_{tag}", seed=seed)
            raw["model"].update(annual_context_pooling=True, annual_exact_mask_coverage=True)
            raw["training"].update(
                epochs=2,
                save_every=2,
                warmup_epochs=1,
                lr=3e-5,
                gradient_checkpointing=True,
                compile=False,
                latent_prediction_weight=0.0,
                annual_feature_anchor_weight=0.0,
                view_consistency_weight=0.0,
            )
            raw["training"]["independent_targets"].update(
                engineering_only=False, research_max_updates=1024, ramp_batches=128
            )
            if recipe == "control":
                raw["training"]["independent_targets"] = None
            for dataset, frozen in zip(
                raw["data"]["datasets"], parent["manifests"]["screen"], strict=True
            ):
                dataset.update(
                    manifest_path=frozen["path"],
                    patch_grid_path=frozen["path"],
                    statistics_dir=str(Path(frozen["path"]).parent / "statistics"),
                )
            folder = output / tag
            folder.mkdir()
            path = folder / "config.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
            Config.from_yaml(path)
            arms[tag] = {
                "config": str(path.resolve()),
                "config_sha256": digest(path),
                "checkpoint": str((folder / "model.pt").resolve()),
                "steps_per_epoch": 512,
                "optimizer_updates": 1024,
                "changes": {"supervision": recipe == "multisource"},
            }
    result = {
        "schema": "xuannv.target-comparison.v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "git_sha": _git_sha(),
        "source_code_sha256": _source_code_sha256(),
        "training_started": False,
        "development_only": True,
        "confirmation_only": False,
        "formal_target_training_released": preparation["formal_training_released"],
        "epochs": 2,
        "seed": 42,
        "continuation_seeds": [42, 17],
        "initialize_config": old["initialize_config"],
        "initialize_checkpoint": old["initialize_checkpoint"],
        "datasets": old["datasets"],
        "labels": old["labels"],
        "cities": old["cities"],
        "osm": old["osm"],
        "probe_steps": old["probe_steps"],
        "exact_mask_coverage_targets": True,
        "assets": assets,
        "arms": arms,
        "baselines": {"original_exact": old["baselines"]["original_exact"]},
        "execution_order": ["control_s42", "multisource_s42", "multisource_s17", "control_s17"],
        "training_budget": {
            "runs": 4,
            "updates_per_run": 1024,
            "total_updates": 4096,
            "global_batch": 32,
            "annual_exposures_per_run": 32768,
            "paired_positions_per_run": 8192,
            "teacher": False,
            "EMA": False,
        },
        "evaluation_protocol": {
            "semantic": "Same124weak-label positions;5spatial folds; train-fold normalization "
            "and class weights; fixed linear head.",
            "fine": "Same203candidate positions in16cities; frozen task eligibility;5city folds;"
            "600steps; linear and conv3x3 heads; fixed logit threshold0.",
            "probe_seed_offsets": [0, 10],
            "reference_feature_mode": "pooled10m",
            "primary_semantic_score": "Mean(water:F1, landcover:mIoU)",
            "primary_fine_score": "Mean(road:conv3x3:F1, road:conv3x3:boundaryF1, "
            "building:conv3x3:F1, water:conv3x3:F1)",
            "joint_score": "Mean(primary_semantic_score, primary_fine_score)",
            "guardrails": {
                "semantic_average_drop": 0.002,
                "fine_average_drop": 0.002,
                "water_F1_drop": 0.01,
                "landcover_mIoU_drop": 0.005,
                "road_F1_drop": 0.005,
            },
            "decision": "Recommend longer supervised training only for positive semantic and fine "
            "deltas in both continuation seeds, with guardrails versus paired control and "
            "original_exact; fine gain must retain its sign in the second head seed. "
            "Otherwise retain control/original and diagnose before spending more.",
            "uncertainty": "Paired2000whole-spatial-group bootstrap per metric; "
            "two continuation seeds share one original pretrained artifact.",
            "no_posthoc_recipe_or_checkpoint_search": True,
            "new_confirmation_roles_untouched": True,
        },
        "limitations": [
            "Historical development weak labels, not an untouched test set.",
            "WorldCover and OSM now share product families with upstream supervision; scores are "
            "held-position product-related diagnostics, not independent manual accuracy.",
            "Some development positions were seen by original pretraining; "
            "report the existing unseen and HR strata separately.",
            "Training positions exclude every downstream candidate, "
            "not only the task-eligible scored subset.",
            "A short continued-training comparison does not establish "
            "full retraining gains or manual5m accuracy.",
        ],
    }
    write_json(output / "plan.json", result)
    return result
