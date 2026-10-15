"""Frozen, paired annual-model ablations and real-observation diagnostics.

Artifacts live outside version control. Existing downstream labels are development
evidence: their reuse never creates an independent, untouched test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

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
        raw["training"].update(
            {
                "epochs": args.epochs,
                "lr": args.lr,
                "warmup_epochs": 1,
                "save_every": 2,
                "uniformity_warmup_epochs": 0,
                "latent_prediction_weight": changes.get("teacher", 0.0),
                "view_consistency_weight": changes.get("views", 0.0),
            }
        )
        if "visible" in changes:
            raw["training"]["highres_visible_weight"] = changes["visible"]
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
        periodic = sorted(checkpoint.parent.glob("model.epoch-*.pt"))
        if periodic:
            command += ["--resume", str(periodic[-1])]
        else:
            command += [
                "--initialize-from",
                plan["initialize_checkpoint"],
                "--initialize-config",
                plan["initialize_config"],
            ]
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


def reconstruction(args) -> None:
    from torch.utils.data import DataLoader

    from xuannv_embedding.data.annual_dataset import collate_annual_observations
    from xuannv_embedding.training.annual import AnnualLoss, prepare_annual_batch
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
            view_consistency_weight=0.0,
            uniformity_weight=0.0,
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
                batch = _move(
                    prepare_annual_batch(
                        raw,
                        evaluation,
                        training=True,
                        generator=torch.Generator().manual_seed(seed),
                    ),
                    device,
                )
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
    return Path(plan["arms"][tag]["config"]), Path(plan["arms"][tag]["checkpoint"])


def evaluation_manifest(source: Path, destination: Path, wanted: set[str]) -> Path:
    """Freeze the small labelled subset once instead of re-reading GBs per model."""
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
            "identity": identity,
            "found": len(found),
            "requested": len(want),
            "missing": sorted(want - found),
            "development_only": True,
        },
    )
    if want - found:
        raise ValueError(f"Missing {len(want - found)} requested evaluation patches")


def probe(args) -> None:
    from xuannv_embedding.downstream.annual_ablation import city_probe, semantic_probe

    plan = json.loads(args.plan.read_text())
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv experiment", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
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
    for name, handler in [("reconstruction", reconstruction), ("export", export), ("probe", probe)]:
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
        command.add_argument("--arm", required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--workers", type=int, default=6)
        if name == "probe":
            command.add_argument("--suite", choices=["semantic", "city", "all"], default="all")
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
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    args.handler(args)
    return 0
