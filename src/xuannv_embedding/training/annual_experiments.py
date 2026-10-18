"""Frozen, paired annual-model ablations and real-observation diagnostics.

Artifacts live outside version control. Existing downstream labels are development
evidence: their reuse never creates an independent, untouched test set.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import torch
import yaml

from xuannv_embedding.config import Config
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import _git_sha, _source_code_sha256, build_training_system
from xuannv_embedding.training.runtime import _autocast, _model_view, _move


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def coverage_stratum(record) -> str:
    sources = record.provenance["observations"]
    pan = any(items and "PAN" in source for source, items in sources.items())
    ms = any(
        items and source not in {"s1", "s2", "landsat"} and "PAN" not in source
        for source, items in sources.items()
    )
    zone = record.grid["parent_key"].split(":")[0]
    return f"{zone}:ms{int(ms)}:pan{int(pan)}"


def stratified_indices(records, count: int, seed: str, excluded=frozenset()) -> list[int]:
    """Proportional coverage/UTM strata, largest-remainder allocation, hashed order."""
    groups = defaultdict(list)
    for index, record in enumerate(records):
        if record.patch_id not in excluded:
            score = hashlib.sha256(f"{seed}:{record.patch_id}".encode()).digest()
            groups[coverage_stratum(record)].append((score, index))
    total = sum(map(len, groups.values()))
    if count <= 0 or count > total:
        raise ValueError(f"Requested {count} samples from {total} eligible records")
    quota = {key: count * len(values) // total for key, values in groups.items()}
    remainder = sorted(groups, key=lambda key: (-(count * len(groups[key]) % total), key))
    for key in remainder[: count - sum(quota.values())]:
        quota[key] += 1
    return sorted(
        index for key, values in groups.items() for _, index in sorted(values)[: quota[key]]
    )


def prepare(args) -> None:
    from xuannv_embedding.utils.manifest import IndexedManifest, write_manifest

    root = args.output.resolve()
    if (root / "plan.json").exists():
        raise ValueError("An existing frozen plan cannot be overwritten")
    config = Config.from_yaml(args.config)
    if config.model.architecture != "annual5m":
        raise ValueError("This workflow requires annual5m")
    labels = json.loads((args.labels / "labels.json").read_text())
    city_rows = json.loads(args.cities.read_text())
    excluded = {row["patch_id"] for row in labels["records"]} | {row[2] for row in city_rows}
    data_out = root / "data"
    data_out.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "xuannv.annual-abc.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "source_code_sha256": _source_code_sha256(),
        "development_only": True,
        "exact_mask_coverage_targets": True,
        "limitations": [
            "Historical weak labels have been evaluated previously.",
            "Pretraining-seen development patches do not establish generalization.",
            "Short continued-training ablations do not predict full retraining gains.",
        ],
        "seed": args.seed,
        "epochs": args.epochs,
        "samples_per_year": args.samples_per_year,
        "reconstruction_samples_per_year": args.validation_samples,
        "labels": str(args.labels.resolve()),
        "cities": str(args.cities.resolve()),
        "osm": str(args.osm.resolve()),
        "probe_steps": args.probe_steps,
        "initialize_config": str(args.config.resolve()),
        "initialize_checkpoint": str(args.initialize_from.resolve()),
        "teacher_config": str(args.teacher_config.resolve()),
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "assets": {},
        "datasets": [],
        "arms": {},
    }
    for path in [
        args.config,
        args.initialize_from,
        args.teacher_config,
        args.teacher_checkpoint,
        args.labels / "labels.json",
        args.cities,
    ]:
        plan["assets"][str(path.resolve())] = digest(path)
    selected_ids = {}
    for dataset in config.data.datasets:
        parent = dataset.manifest_path.parent
        for name in ["summary.json", "sources.json", "statistics"]:
            link = data_out / name
            if not link.exists():
                link.symlink_to(parent / name, target_is_directory=name == "statistics")
        year = dataset.manifest_path.name.split(".")[0]
        manifests = {}
        for split, count in [
            ("train", args.samples_per_year),
            ("validation", args.validation_samples),
        ]:
            source = parent / f"{year}.{split}.manifest.jsonl"
            records = IndexedManifest(source)
            indices = stratified_indices(
                records, count, f"abc:{args.seed}:{year}:{split}", excluded
            )
            chosen = [records[index] for index in indices]
            destination = data_out / source.name
            meta = write_manifest(
                destination,
                chosen,
                months=records.meta.months,
                generator_version="annual-abc-frozen-v1",
            )
            manifests[split] = str(destination)
            plan["assets"][str(destination)] = meta.sha256
            plan["assets"][str(source)] = records.meta.sha256
            selected_ids[f"{year}:{split}"] = [record.patch_id for record in chosen]
            print(
                json.dumps(
                    {
                        "frozen": source.name,
                        "count": len(chosen),
                        "strata": dict(Counter(map(coverage_stratum, chosen))),
                    }
                ),
                flush=True,
            )
        plan["datasets"].append({"year": int(year), "original": str(parent), **manifests})
    for path in sorted((data_out / "statistics").glob("*_stats.json")):
        plan["assets"][str(path.resolve())] = digest(path)
    for name in ["summary.json", "sources.json"]:
        plan["assets"][str((data_out / name).resolve())] = digest(data_out / name)
    write_json(root / "sample_ids.json", selected_ids)
    plan["assets"][str(root / "sample_ids.json")] = digest(root / "sample_ids.json")
    variants = {
        "control": {},
        "A_context": {"context": True},
        "AB_visible0": {"context": True, "visible": 0.0},
        "AB_visible01": {"context": True, "visible": 0.1},
        "AB_visible03": {"context": True, "visible": 0.3},
        "ABC_teacher": {"context": True, "visible": 0.1, "teacher": 0.1},
        "ABC_views": {"context": True, "visible": 0.1, "views": 0.1},
        "ABC_both": {"context": True, "visible": 0.1, "teacher": 0.1, "views": 0.1},
    }
    original = yaml.safe_load(args.config.read_text())
    for tag, changes in variants.items():
        raw = json.loads(json.dumps(original))
        raw["experiment"] = {"name": f"annual_abc_{tag}", "seed": args.seed}
        raw["model"]["annual_context_pooling"] = changes.get("context", False)
        raw["model"]["annual_exact_mask_coverage"] = True
        raw["training"].update(
            {
                "epochs": args.epochs,
                "lr": args.lr,
                "warmup_epochs": 1,
                "save_every": 2,
                "uniformity_warmup_epochs": 0,
                "latent_prediction_weight": changes.get("teacher", 0.0),
                "annual_feature_anchor_weight": 0.0,
                "view_consistency_weight": changes.get("views", 0.0),
                "highres_visible_targets": True,
                "highres_visible_weight": changes.get("visible"),
            }
        )
        if changes.get("teacher"):
            raw["training"]["latent_prediction_pool_factors"] = [2, 4, 8]
        raw["data"]["num_workers"] = args.workers
        for dc, frozen in zip(raw["data"]["datasets"], plan["datasets"], strict=True):
            dc["manifest_path"] = frozen["train"]
            dc["patch_grid_path"] = frozen["train"]
            dc["statistics_dir"] = str(data_out / "statistics")
        out = root / tag
        out.mkdir(exist_ok=True)
        path = out / "config.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        Config.from_yaml(path)
        plan["arms"][tag] = {
            "config": str(path),
            "config_sha256": digest(path),
            "checkpoint": str(out / "model.pt"),
            "changes": changes,
        }
    write_json(root / "plan.json", plan)


def load_model(config_path: Path, checkpoint: Path, device: torch.device):
    config = Config.from_yaml(config_path)
    system = build_training_system(config)
    state = load_training_checkpoint(
        checkpoint,
        model=system.model,
        expected_config_sha256=digest(config_path),
        expected_source_schema={
            key: asdict(value) for key, value in config.model.input_sources.items()
        },
        expected_regions=[dataset.region for dataset in config.data.datasets],
    )
    metadata = {key: state[key] for key in ["git_sha", "config_sha256", "epoch", "metrics"]}
    del state
    return config, system.to(device).eval(), metadata


def run(args) -> None:
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    arms = args.arms or list(plan["arms"])
    for path, expected in plan["assets"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen asset changed: {path}")
    status_file = root / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    for tag in arms:
        arm = plan["arms"][tag]
        if digest(Path(arm["config"])) != arm["config_sha256"]:
            raise ValueError(f"Configuration changed: {tag}")
        checkpoint = Path(arm["checkpoint"])
        if checkpoint.exists():
            # Only a strict final checkpoint with the complete requested epoch is reusable.
            _, _, state = load_model(Path(arm["config"]), checkpoint, torch.device("cpu"))
            if state["epoch"] + 1 != plan["epochs"]:
                raise ValueError(f"Incomplete final checkpoint: {tag}")
            continue
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(args.gpus.split(','))}",
            "-m",
            "xuannv_embedding.cli",
            "train",
            "--config",
            arm["config"],
            "--output",
            arm["checkpoint"],
            "--device",
            "cuda",
        ]
        if arm.get("steps_per_epoch"):
            command += ["--steps", str(arm["steps_per_epoch"])]
        periodic = sorted(checkpoint.parent.glob("model.epoch-*.pt"))
        remaining_epochs = plan["epochs"]
        if periodic:
            _, _, previous = load_model(Path(arm["config"]), periodic[-1], torch.device("cpu"))
            remaining_epochs -= previous["epoch"] + 1
            if remaining_epochs <= 0:
                raise ValueError("Complete periodic checkpoint exists without a final artifact")
            command += ["--resume", str(periodic[-1])]
        else:
            command += [
                "--initialize-from",
                plan["initialize_checkpoint"],
                "--initialize-config",
                plan["initialize_config"],
            ]
            if arm.get("warmed_heads"):
                head = arm["warmed_heads"]
                if digest(Path(head["path"])) != head["sha256"]:
                    raise ValueError("Frozen readout initialization changed")
                command += [
                    "--initialize-target-heads",
                    head["path"],
                    "--target-head-seed",
                    str(head["seed"]),
                ]
        command += ["--epochs", str(remaining_epochs)]
        if arm["changes"].get("teacher"):
            command += [
                "--teacher-checkpoint",
                plan["teacher_checkpoint"],
                "--teacher-config",
                plan["teacher_config"],
            ]
        env = os.environ.copy()
        env.update(
            {"CUDA_VISIBLE_DEVICES": args.gpus, "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
        )
        status[tag] = {
            "state": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "source_code_sha256": _source_code_sha256(),
        }
        write_json(status_file, status)
        print(json.dumps({"training": tag, **status[tag]}), flush=True)
        with (checkpoint.parent / "train.log").open("a") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        status[tag].update(
            {
                "state": "complete" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(status_file, status)
        if result.returncode:
            raise RuntimeError(f"Training {tag} failed; inspect {checkpoint.parent / 'train.log'}")
        print(json.dumps({"finished": tag}), flush=True)


def prepare_evaluation_batch(raw, config, seed, *, exact_targets=False):
    from xuannv_embedding.training.annual import prepare_annual_batch

    batch = prepare_annual_batch(
        raw, config, training=True, generator=torch.Generator().manual_seed(seed)
    )
    if exact_targets and not config.model.annual_exact_mask_coverage:
        target_config = replace(
            config, model=replace(config.model, annual_exact_mask_coverage=True)
        )
        common = prepare_annual_batch(
            raw, target_config, training=True, generator=torch.Generator().manual_seed(seed)
        )
        for key in ["targets", "target_masks", "highres_targets"]:
            batch[key] = common[key]
    return batch


def reconstruction(args) -> None:
    from torch.utils.data import DataLoader

    from xuannv_embedding.data.annual_dataset import collate_annual_observations
    from xuannv_embedding.training.annual import AnnualLoss
    from xuannv_embedding.training.annual_data import annual_dataset

    plan = json.loads(args.plan.read_text())
    config_path, checkpoint = artifact_paths(plan, args.arm)
    device = torch.device(args.device)
    config, system, identity = load_model(config_path, checkpoint, device)
    # Every model receives the SAME held-out target partitions and weights.
    evaluation = replace(
        config,
        training=replace(
            config.training,
            highres_visible_weight=0.0,
            latent_prediction_weight=0.0,
            annual_feature_anchor_weight=0.0,
            view_consistency_weight=0.0,
            uniformity_weight=0.0,
            independent_targets=None,
        ),
    )
    criterion = AnnualLoss(evaluation).to(device)
    rows = []
    with torch.inference_mode():
        for frozen, dc in zip(plan["datasets"], config.data.datasets, strict=True):
            dataset = annual_dataset(config, replace(dc, manifest_path=Path(frozen["validation"])))
            loader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=args.workers,
                collate_fn=collate_annual_observations,
            )
            for raw in loader:
                patch_id = raw["metadata"][0]["patch_id"]
                year = raw["metadata"][0]["year"]
                seed = int.from_bytes(
                    hashlib.sha256(f"heldout:{year}:{patch_id}".encode()).digest()[:8], "little"
                )
                batch = prepare_evaluation_batch(
                    raw,
                    evaluation,
                    seed,
                    exact_targets=plan.get("exact_mask_coverage_targets", False),
                )
                batch = _move(batch, device)
                with _autocast(device, True):
                    output = _model_view(system.model, batch)
                    losses = criterion(
                        output,
                        batch["targets"],
                        batch["target_masks"],
                        highres_targets=batch["highres_targets"],
                    )
                for name, head in config.model.target_heads.items():
                    source = head.source
                    spatial = ~batch["source_pixel_masks"][source].any(dim=1, keepdim=True)
                    # Erode two 10m pixels to isolate block interiors from decoder borders.
                    interior = torch.nn.functional.max_pool2d((~spatial).float(), 5, 1, 2) == 0
                    original_mask = raw["source_pixel_masks"][source].to(device)
                    difference = (
                        (output.reconstructions[name].float() - batch["targets"][name])
                        .abs()
                        .mean(2)
                    )
                    for part, select in [
                        ("spatial_interior", interior & original_mask),
                        ("other_heldout", batch["target_masks"][name] & ~interior),
                    ]:
                        losses[f"{part}_{source}"] = (
                            difference * select
                        ).sum() / select.sum().clamp(min=1)
                rows.append(
                    {
                        "patch_id": patch_id,
                        "year": year,
                        "highres_observations": len(batch["highres_targets"][0]),
                        **{key: float(value) for key, value in losses.items()},
                    }
                )
    result = {
        "arm": args.arm,
        "checkpoint_sha256": digest(checkpoint),
        "evaluation_source_code_sha256": _source_code_sha256(),
        "exact_mask_coverage_targets": plan.get("exact_mask_coverage_targets", False),
        "input_exact_mask_coverage": config.model.annual_exact_mask_coverage,
        "identity": identity,
        "development_only": True,
        "n": len(rows),
        "per_patch": rows,
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key not in {"patch_id", "year", "highres_observations"}
        },
    }
    hr = [row for row in rows if row["highres_observations"]]
    result["highres_supported_mean"] = (
        {
            key: float(np.mean([row[key] for row in hr]))
            for key in ["highres_recon", "highres_visible_recon", "highres_heldout_recon"]
        }
        if hr
        else {}
    )
    write_json(args.plan.parent / "evaluation" / args.arm / "reconstruction.json", result)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "n": len(rows),
                "mean": result["mean"],
                "highres_supported_mean": result["highres_supported_mean"],
            }
        ),
        flush=True,
    )


def artifact_paths(plan, tag):
    if tag == "original":
        return Path(plan["initialize_config"]), Path(plan["initialize_checkpoint"])
    if tag == "small":
        return Path(plan["teacher_config"]), Path(plan["teacher_checkpoint"])
    if tag in plan.get("baselines", {}):
        entry = plan["baselines"][tag]
        return Path(entry["config"]), Path(entry["checkpoint"])
    return Path(plan["arms"][tag]["config"]), Path(plan["arms"][tag]["checkpoint"])


def model_state_digest(model):
    sha = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        sha.update(name.encode())
        sha.update(str(value.dtype).encode())
        sha.update(str(tuple(value.shape)).encode())
        sha.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return sha.hexdigest()


def adapt_inputs(args):
    """Register an unchanged original model under an explicit corrected input policy."""
    from xuannv_embedding.training.checkpoint import save_training_checkpoint
    from xuannv_embedding.training.optimizer import build_optimizer

    if args.output.exists():
        raise ValueError("Preprocessing baseline output must be a new directory")
    config, system, original = load_model(args.config, args.checkpoint, torch.device("cpu"))
    state_sha = model_state_digest(system.model)
    raw = yaml.safe_load(args.config.read_text())
    raw["model"]["annual_exact_mask_coverage"] = True
    raw["experiment"]["name"] += "_exact_input_baseline"
    args.output.mkdir(parents=True)
    config_path = args.output / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    adapted = Config.from_yaml(config_path)
    optimizer = build_optimizer(
        system, lr=adapted.training.lr, weight_decay=adapted.training.weight_decay
    )
    save_training_checkpoint(
        args.output / "model.pt",
        model=system.model,
        criterion=system.criterion,
        optimizer=optimizer,
        scheduler=None,
        epoch=original["epoch"],
        config_sha256=digest(config_path),
        git_sha=_git_sha(),
        source_schema={key: asdict(value) for key, value in adapted.model.input_sources.items()},
        regions=[dc.region for dc in adapted.data.datasets],
        metrics={
            "checkpoint_kind": "preprocessing_only_adapter",
            "optimizer_steps": 0,
            "model_state_sha256": state_sha,
            "source_checkpoint_sha256": digest(args.checkpoint),
            "source_config_sha256": digest(args.config),
            "source_git_sha": original["git_sha"],
            "inherited_training_epoch": original["epoch"],
            "source_code_sha256": _source_code_sha256(),
        },
    )
    _, restored, _ = load_model(config_path, args.output / "model.pt", torch.device("cpu"))
    if model_state_digest(restored.model) != state_sha:
        raise RuntimeError("Preprocessing adapter unexpectedly changed model parameters")
    print(
        json.dumps(
            {"adapter": str(args.output), "model_state_sha256": state_sha, "optimizer_steps": 0}
        ),
        flush=True,
    )


def evaluation_manifest(source: Path, destination: Path, wanted: set[str]) -> Path:
    """Freeze the small labelled subset once instead of re-reading GBs per model."""
    import fcntl

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Canonical features may already exist while their shared manifest cache does
    # not. Serialize preparation across processes, including the shared symlinks.
    # Publish origin metadata last, then never rewrite a completed cache entry:
    # readers must retain IndexedManifest's size/mtime identity checks.
    with (destination.parent / ".evaluation-manifests.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _evaluation_manifest_locked(source, destination, wanted)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _evaluation_manifest_locked(source: Path, destination: Path, wanted: set[str]) -> Path:
    from xuannv_embedding.utils.manifest import IndexedManifest, write_manifest

    origin = json.loads(Path(str(source) + ".meta.json").read_text())
    identity = {
        "manifest_sha256": origin["sha256"],
        "wanted_sha256": hashlib.sha256("\n".join(sorted(wanted)).encode()).hexdigest(),
    }
    metadata = destination.with_suffix(".origin.json")
    if metadata.exists():
        if json.loads(metadata.read_text()) != identity:
            raise ValueError("Frozen evaluation manifest identity differs")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name in ["summary.json", "sources.json", "statistics"]:
        link = destination.parent / name
        if not link.exists():
            link.symlink_to(source.parent / name, target_is_directory=name == "statistics")
    records = IndexedManifest(source)
    selected = [record for record in records if record.patch_id in wanted]
    write_manifest(
        destination,
        selected,
        months=records.meta.months,
        generator_version="annual-abc-development-v1",
    )
    write_json(metadata, identity)
    return destination


def export(args) -> None:
    from torch.utils.data import DataLoader, Subset

    from xuannv_embedding.data.annual_dataset import collate_annual_observations
    from xuannv_embedding.models.annual import masked_resample
    from xuannv_embedding.training.annual import prepare_annual_batch
    from xuannv_embedding.training.annual_data import annual_dataset

    plan = json.loads(args.plan.read_text())
    config_path, checkpoint = artifact_paths(plan, args.arm)
    device = torch.device(args.device)
    config, system, identity = load_model(config_path, checkpoint, device)
    labels = json.loads((Path(plan["labels"]) / "labels.json").read_text())
    city_rows = json.loads(Path(plan["cities"]).read_text())
    semantic = {row["patch_id"] for row in labels["records"]}
    cities = {row[2]: row[0] for row in city_rows}
    want = semantic | set(cities)
    root = args.plan.parent / "evaluation" / args.arm
    found = set()
    with torch.inference_mode():
        # Export labels from their actual annual split, never infer it from label metadata.
        for split in ["train", "validation", "test"]:
            frozen, dc = next(
                (row, dataset)
                for row, dataset in zip(plan["datasets"], config.data.datasets, strict=True)
                if row["year"] == labels["year"]
            )
            manifest = Path(frozen["original"]) / f"{labels['year']}.{split}.manifest.jsonl"
            manifest = evaluation_manifest(
                manifest, args.plan.parent / "development_data" / manifest.name, want
            )
            if json.loads(Path(str(manifest) + ".meta.json").read_text())["record_count"] == 0:
                continue
            dataset = annual_dataset(config, replace(dc, manifest_path=manifest))
            indices = [
                index for index, record in enumerate(dataset.records) if record.patch_id in want
            ]
            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=4,
                num_workers=args.workers,
                collate_fn=collate_annual_observations,
            )
            for raw in loader:
                batch = _move(prepare_annual_batch(raw, config, training=False), device)
                output = _model_view(system.model, batch, decode=False)
                for i, meta in enumerate(raw["metadata"]):
                    patch_id = meta["patch_id"]
                    if patch_id in found:
                        raise ValueError("Patch appears in multiple annual splits")
                    found.add(patch_id)
                    values = output.embedding_map[i, 0].float()
                    valid = output.validity_mask[i, 0].float()
                    if patch_id in semantic:
                        f10, v10 = masked_resample(values[None], valid[None], (128, 128))
                        path = root / "semantic" / f"{patch_id}.npz"
                        path.parent.mkdir(exist_ok=True, parents=True)
                        np.savez_compressed(
                            path,
                            emb=f10[0].half().cpu().numpy(),
                            valid=v10[0, 0].cpu().numpy(),
                            split=split,
                        )
                    if patch_id in cities:
                        path = root / "city5m" / f"{patch_id}.npz"
                        path.parent.mkdir(exist_ok=True, parents=True)
                        np.savez_compressed(
                            path,
                            emb=values.half().cpu().numpy(),
                            valid=valid[0].cpu().numpy(),
                            split=split,
                            city=cities[patch_id],
                            highres_obs=len(batch["highres_observations"][i]),
                            transform=np.asarray(meta["output_grid"]["transform"]),
                        )
                print(
                    json.dumps({"export": args.arm, "split": split, "found": len(found)}),
                    flush=True,
                )
    write_json(
        root / "export.json",
        {
            "checkpoint_sha256": digest(checkpoint),
            "evaluation_source_code_sha256": _source_code_sha256(),
            "identity": identity,
            "found": len(found),
            "requested": len(want),
            "missing": sorted(want - found),
            "development_only": not plan.get("confirmation_only", False),
            "confirmation_only": plan.get("confirmation_only", False),
        },
    )
    if want - found:
        raise ValueError(f"Missing {len(want - found)} requested evaluation patches")


def probe(args) -> None:
    from xuannv_embedding.downstream.annual_ablation import city_probe, semantic_probe

    plan = json.loads(args.plan.read_text())
    seed_offset = getattr(args, "seed_offset", 0)
    if seed_offset < 0 or (seed_offset and args.suite != "city"):
        raise ValueError("Nonzero probe seed offsets require --suite city and a nonnegative seed")
    plan["probe_seed_offset"] = seed_offset
    plan["save_probe_heads"] = getattr(args, "save_heads", False)
    plan["probe_threshold_free"] = getattr(args, "threshold_free", False)
    plan["probe_feature_mode"] = getattr(args, "feature_mode", "native5m")
    if plan["probe_feature_mode"] != "native5m" and args.suite != "city":
        raise ValueError("Coarsened feature diagnostics require --suite city")
    root = args.plan.parent / "evaluation"
    device = torch.device(args.device)
    if args.suite in {"semantic", "all"}:
        semantic_probe(plan, root, args.arm, device)
    if args.suite in {"city", "all"}:
        city_probe(plan, root, args.arm, device)


def freeze_confirmation(args) -> None:
    """Freeze additional existing weak labels before inspecting candidate results."""
    from xuannv_embedding.utils.manifest import IndexedManifest

    plan = json.loads(args.plan.read_text())
    candidate_labels = json.loads((args.labels / "labels.json").read_text())
    old_labels = json.loads((Path(plan["labels"]) / "labels.json").read_text())
    excluded = {row["patch_id"] for row in old_labels["records"]} | {
        row[2] for row in json.loads(Path(plan["cities"]).read_text())
    }
    wanted = {row["patch_id"] for row in candidate_labels["records"]} - excluded
    root = args.plan.parent / "confirmation"
    if (root / "plan.json").exists():
        raise ValueError("Confirmation set is already frozen")
    year = candidate_labels["year"]
    source_root = Path(next(row["original"] for row in plan["datasets"] if row["year"] == year))
    actual_split = {}
    counts = {}
    for split in ["train", "validation", "test"]:
        source = source_root / f"{year}.{split}.manifest.jsonl"
        target = evaluation_manifest(source, root / "development_data" / source.name, wanted)
        records = IndexedManifest(target)
        counts[split] = len(records)
        for record in records:
            if record.patch_id in actual_split:
                raise ValueError("Confirmation patch appears in multiple source splits")
            actual_split[record.patch_id] = split
    selected = []
    labels_root = root / "labels"
    labels_root.mkdir(exist_ok=True)
    for row in candidate_labels["records"]:
        if row["patch_id"] not in actual_split:
            continue
        path = args.labels / row["file"]
        if digest(path) != row["sha256"]:
            raise ValueError("Confirmation label digest differs")
        destination = labels_root / row["file"]
        if not destination.exists():
            destination.symlink_to(path.resolve())
        selected.append({**row, "annual_split": actual_split[row["patch_id"]]})
    labels = {
        **candidate_labels,
        "records": selected,
        "independent_evaluation": False,
        "policy": {
            **candidate_labels["policy"],
            "current_experiment": (
                "Existing weak labels; exclude development IDs. Fit heads only on current annual "
                "train positions; evaluate once on current annual validation/test positions "
                "after choosing one candidate on development evidence. Not independent truth."
            ),
        },
    }
    write_json(labels_root / "labels.json", labels)
    write_json(root / "empty_cities.json", [])
    # Source positions absent from the annual release cannot be exported. The
    # cached records already contain exactly the remaining IDs; align its request identity.
    actual_digest = hashlib.sha256("\n".join(sorted(actual_split)).encode()).hexdigest()
    for path in (root / "development_data").glob("*.origin.json"):
        identity = json.loads(path.read_text())
        identity["wanted_sha256"] = actual_digest
        write_json(path, identity)
    frozen = {
        **plan,
        "schema": "xuannv.annual-abc.confirmation.v1",
        "confirmation_only": True,
        "development_only": False,
        "labels": str(labels_root.resolve()),
        "cities": str((root / "empty_cities.json").resolve()),
        "counts": counts,
        "absent_from_current_annual": sorted(wanted - set(actual_split)),
        "parent_plan_sha256": digest(args.plan),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "plan.json", frozen)
    print(
        json.dumps(
            {
                "confirmation": counts,
                "excluded_development": len(excluded),
                "missing": len(wanted - set(actual_split)),
            }
        ),
        flush=True,
    )


def compare(args) -> None:
    from xuannv_embedding.downstream.annual_ablation import compare as compare_probes

    plan = json.loads(args.plan.read_text())
    compare_probes(
        plan,
        args.plan.parent / "evaluation",
        args.arms or list(plan["arms"]),
        reference=args.reference,
    )


def selection(args) -> None:
    from xuannv_embedding.downstream.annual_ablation import SELECTION_PROTOCOL, select_candidate

    if args.command == "lock-selection":
        destination = args.plan.parent / "selection_protocol.json"
        if destination.exists():
            raise ValueError("Selection protocol is already locked")
        write_json(
            destination,
            {
                "protocol": SELECTION_PROTOCOL,
                "locked_utc": datetime.now(timezone.utc).isoformat(),
                "source_code_sha256": _source_code_sha256(),
            },
        )
    else:
        result = select_candidate(
            json.loads(args.plan.read_text()), args.plan.parent / "evaluation"
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)


def confirmation_osm(args) -> None:
    from xuannv_embedding.data_process.confirmation_osm import prepare_confirmation_osm

    prepare_confirmation_osm(args.plan, args.snapshot, args.snapshot_date)


def evaluate_arm(args) -> None:
    """Run one complete paired evaluation; reuse only matching evaluation code."""
    plan = json.loads(args.plan.read_text())
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(torch.device(args.device))
    destination = args.plan.parent / "evaluation" / args.arm
    source_sha = _source_code_sha256()
    _, checkpoint = artifact_paths(plan, args.arm)
    checkpoint_sha = digest(checkpoint)

    def fresh(path):
        if not path.exists():
            return False
        value = json.loads(path.read_text())
        return (
            value.get("evaluation_source_code_sha256") == source_sha
            and value.get("checkpoint_sha256") == checkpoint_sha
        )

    stages = []
    if not plan.get("confirmation_only"):
        stages.append(("reconstruction.json", reconstruction))
    stages.append(("export.json", export))
    for name, handler in stages:
        if not fresh(destination / name):
            handler(args)
        gc.collect()
        torch.cuda.empty_cache()
    for suite in (
        ["semantic"]
        if plan.get("confirmation_only") and not plan.get("fine_confirmation")
        else ["semantic", "city"]
    ):
        path = destination / f"{suite}_probe" / "metrics.json"
        if not fresh(path):
            args.suite = suite
            probe(args)
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"evaluation_complete": args.arm}), flush=True)


def evaluate_many(args) -> None:
    """Use one subprocess per GPU; canonical support is exported before parallel work."""
    plan = json.loads(args.plan.read_text())
    if args.wait_for_training:
        while not all(Path(arm["checkpoint"]).exists() for arm in plan["arms"].values()):
            status_file = args.plan.parent / "status.json"
            if status_file.exists() and any(
                arm["state"] == "failed" for arm in json.loads(status_file.read_text()).values()
            ):
                raise RuntimeError("Training failed; evaluation queue stopped")
            time.sleep(10)
    arms = args.arms or ["original", "small", *plan.get("baselines", {}), *plan["arms"]]
    gpus = args.gpus.split(",")
    command = [sys.executable, "-m", "xuannv_embedding.cli", "experiment"]
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
    canonical = args.plan.parent / "evaluation" / "original" / "export.json"
    if (
        not canonical.exists()
        or json.loads(canonical.read_text()).get("evaluation_source_code_sha256")
        != _source_code_sha256()
    ):
        subprocess.run(
            [
                *command,
                "export",
                "--plan",
                str(args.plan),
                "--arm",
                "original",
                "--device",
                f"cuda:{gpus[0]}",
                "--workers",
                str(args.workers),
            ],
            env=env,
            check=True,
        )
    pending = Queue()
    for tag in arms:
        pending.put(tag)

    def worker(gpu):
        completed = []
        while True:
            try:
                tag = pending.get_nowait()
            except Empty:
                return completed
            output = args.plan.parent / "evaluation" / tag
            output.mkdir(exist_ok=True, parents=True)
            with (output / "evaluate.log").open("a") as log:
                subprocess.run(
                    [
                        *command,
                        "evaluate-arm",
                        "--plan",
                        str(args.plan),
                        "--arm",
                        tag,
                        "--device",
                        f"cuda:{gpu}",
                        "--workers",
                        str(args.workers),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(json.dumps({"evaluated": tag, "gpu": gpu}), flush=True)
            completed.append(tag)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        results = list(pool.map(worker, gpus))
    write_json(
        args.plan.parent / "evaluation" / "completed.json",
        {
            "arms": [tag for group in results for tag in group],
            "evaluation_source_code_sha256": _source_code_sha256(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def prepare_targets(args):
    from xuannv_embedding.training.target_experiments import prepare_experiments

    result = prepare_experiments(
        args.prepared_root, args.target_root, args.initial_config, args.output
    )
    print(
        json.dumps({"target_configs_prepared": len(result["configs"]), "training_started": False})
    )


def prepare_local_supervision(args):
    from xuannv_embedding.training.target_experiments import prepare_multisource

    result = prepare_multisource(
        args.prepared_root,
        args.target_root,
        args.experiment_plan,
        args.output,
        workers=args.workers,
    )
    print(
        json.dumps({"target_configs_prepared": len(result["configs"]), "training_started": False})
    )


def prepare_training_pool(args):
    from xuannv_embedding.training.target_experiments import prepare_training_pool as prepare

    result = prepare(args.prepared_root, args.initial_config, args.output, scope=args.scope)
    print(
        json.dumps(
            {"frozen_training_positions": result["unique_positions"], "training_started": False}
        )
    )


def _development_prepare(args):
    from xuannv_embedding.downstream.spatial_development import prepare

    prepare(args)


def _development_export(args):
    from xuannv_embedding.downstream.spatial_development import export

    export(args)


def _development_probe(args):
    from xuannv_embedding.downstream.spatial_development import probe

    probe(args)


def prepare_target_comparison(args):
    from xuannv_embedding.training.target_comparison import prepare_comparison

    result = prepare_comparison(args.local_plan, args.evaluation_plan, args.output)
    print(
        json.dumps({"comparison_configs_prepared": len(result["arms"]), "training_started": False})
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv experiment", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "prepare-targets", help="Freeze supervised target configs without training"
    )
    for name in ("prepared-root", "target-root", "initial-config", "output"):
        command.add_argument(f"--{name}", required=True, type=Path)
    command.set_defaults(handler=prepare_targets)
    command = commands.add_parser(
        "prepare-multisource", help="Prepare local supervision and a bounded engineering pilot"
    )
    for name in ("prepared-root", "target-root", "experiment-plan", "output"):
        command.add_argument(f"--{name}", required=True, type=Path)
    command.add_argument("--workers", type=int, default=4)
    command.set_defaults(handler=prepare_local_supervision)
    command = commands.add_parser(
        "prepare-training-pool",
        help="Materialize a frozen training scope without choosing a recipe",
    )
    for name in ("prepared-root", "initial-config", "output"):
        command.add_argument(f"--{name}", required=True, type=Path)
    command.add_argument("--scope", choices=["screen", "expanded", "allowed"], default="allowed")
    command.set_defaults(handler=prepare_training_pool)
    command = commands.add_parser(
        "development-prepare",
        help="Freeze a new spatial-development readout protocol before label access",
    )
    for name in ("prepared-root", "target-root", "audit-contract", "matrix-plan", "output"):
        command.add_argument(f"--{name}", required=True, type=Path)
    command.set_defaults(handler=_development_prepare)
    command = commands.add_parser(
        "development-export",
        help="Export observation-only features on frozen development positions",
    )
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--arm", required=True)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--workers", type=int, default=2)
    command.set_defaults(handler=_development_export)
    command = commands.add_parser(
        "development-probe", help="Fit fixed spatial ridge readouts, using training folds only"
    )
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--arm", required=True)
    command.set_defaults(handler=_development_probe)
    command = commands.add_parser(
        "prepare-target-comparison", help="Freeze a bounded two-recipe, two-seed comparison"
    )
    for name in ("local-plan", "evaluation-plan", "output"):
        command.add_argument(f"--{name}", required=True, type=Path)
    command.set_defaults(handler=prepare_target_comparison)
    command = commands.add_parser(
        "warmup-target-heads", help="Train independent readouts with the annual encoder frozen"
    )
    for name in ("config", "initialize-config", "checkpoint", "output"):
        command.add_argument(f"--{name}", required=True, type=Path)
    command.add_argument("--steps", required=True, type=int)
    command.add_argument("--save-every", type=int, default=512)
    command.add_argument("--head-seeds", nargs="+", type=int, default=[42, 17])
    command.add_argument("--readout-state", type=Path)
    command.add_argument("--device", default="cuda")
    command.set_defaults(handler=_warmup_target_heads)
    command = commands.add_parser(
        "diagnose-target-heads", help="Attribute fixed TRAIN gradients before/after head warmup"
    )
    for name in ("config", "initialize-config", "checkpoint", "warmup", "positions", "output"):
        command.add_argument(f"--{name}", required=True, type=Path)
    command.add_argument("--midpoint", type=Path)
    command.add_argument("--mask-seed", type=int, default=9)
    command.add_argument("--noise-seed", type=int, default=2026)
    command.add_argument("--device", default="cuda:0")
    command.set_defaults(handler=_diagnose_target_heads)
    command = commands.add_parser(
        "prepare", help="Freeze paired ABC configs and stratified sample IDs"
    )
    for name in [
        "config",
        "initialize-from",
        "teacher-config",
        "teacher-checkpoint",
        "labels",
        "cities",
        "osm",
        "output",
    ]:
        command.add_argument(f"--{name}", required=True, type=Path)
    command.add_argument("--samples-per-year", type=int, default=4096)
    command.add_argument("--validation-samples", type=int, default=128)
    command.add_argument("--epochs", type=int, default=8)
    command.add_argument("--lr", type=float, default=3e-5)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--workers", type=int, default=6)
    command.add_argument("--probe-steps", type=int, default=600)
    command.set_defaults(handler=prepare)
    command = commands.add_parser(
        "run", help="Run/resume frozen ablations sequentially on specified GPUs"
    )
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    command.add_argument("--arms", nargs="+")
    command.set_defaults(handler=run)
    command = commands.add_parser(
        "freeze-confirmation", help="Lock additional weak-label spatial holdout"
    )
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--labels", required=True, type=Path)
    command.set_defaults(handler=freeze_confirmation)
    command = commands.add_parser(
        "adapt-inputs", help="Register corrected preprocessing with unchanged weights"
    )
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--checkpoint", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.set_defaults(handler=adapt_inputs)
    command = commands.add_parser(
        "freeze-confirmation-osm", help="Extract local dated OSM for fine confirmation"
    )
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--snapshot", required=True, type=Path)
    command.add_argument("--snapshot-date", required=True)
    command.set_defaults(handler=confirmation_osm)
    for name, handler in [("reconstruction", reconstruction), ("export", export), ("probe", probe)]:
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
        command.add_argument("--arm", required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--workers", type=int, default=6)
        if name == "probe":
            command.add_argument("--suite", choices=["semantic", "city", "all"], default="all")
            command.add_argument("--seed-offset", type=int, default=0)
            command.add_argument("--save-heads", action="store_true")
            command.add_argument("--threshold-free", action="store_true")
            command.add_argument(
                "--feature-mode", choices=["native5m", "pooled10m"], default="native5m"
            )
        command.set_defaults(handler=handler)
    command = commands.add_parser("compare", help="Paired city/spatial-block bootstrap differences")
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--reference", default="original")
    command.add_argument("--arms", nargs="+")
    command.set_defaults(handler=compare)
    for name in ["lock-selection", "select"]:
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
        command.set_defaults(handler=selection)
    command = commands.add_parser(
        "evaluate-arm", help="Run all paired diagnostics for one artifact"
    )
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--arm", required=True)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--workers", type=int, default=4)
    command.set_defaults(handler=evaluate_arm)
    command = commands.add_parser(
        "evaluate-many", help="Evaluate frozen artifacts across available GPUs"
    )
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--arms", nargs="+")
    command.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    command.add_argument("--workers", type=int, default=4)
    command.add_argument("--wait-for-training", action="store_true")
    command.set_defaults(handler=evaluate_many)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    args.handler(args)
    return 0


def _warmup_target_heads(args):
    from xuannv_embedding.training.target_warmup import warmup

    warmup(args)


def _diagnose_target_heads(args):
    from xuannv_embedding.training.target_gradients import diagnose

    diagnose(args)
