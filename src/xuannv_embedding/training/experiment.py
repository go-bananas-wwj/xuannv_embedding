"""Registered regional experiments with immutable sample caches and validation logs.

Runs may use one device or DDP for one registered model. Preparation never initializes an NPU.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.distributed as dist
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import RegionRasterDataset, collate_region_batch
from xuannv_embedding.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from xuannv_embedding.training.cli import _git_sha, _setup_device, build_training_system
from xuannv_embedding.training.distributed_experiment import (
    gather_objects,
    global_means,
    rank_random_state,
    restore_rank_random_state,
)
from xuannv_embedding.training.masking import apply_input_masking
from xuannv_embedding.training.optimizer import build_optimizer, build_scheduler
from xuannv_embedding.training.runtime import _autocast, _grad_scaler, _move


def public_base_config(raw: dict, *, seed: int, lr: float) -> dict:
    result = copy.deepcopy(raw)
    model = result["model"]
    excluded = {s for s, c in model["input_sources"].items() if c["role"] == "highres"}
    model["input_sources"] = {s: c for s, c in model["input_sources"].items() if s not in excluded}
    model["target_heads"] = {
        h: c for h, c in model["target_heads"].items() if c["source"] not in excluded
    }
    model["stp"]["highres_fusion_to_embedding"] = False
    for data in result["data"]["datasets"]:
        data["source_map"] = {p: c for p, c in data["source_map"].items() if c not in excluded}
        # This target-only slot must be explicit in the production data contract.
        if "worldcover" in model["target_heads"]:
            data["source_map"]["worldcover"] = "worldcover"
    result["experiment"]["seed"] = seed
    result["experiment"]["name"] = f"public_base_seed{seed}_lr{lr:g}"
    result["training"]["lr"] = lr
    result["training"]["epochs"] = 800
    result["training"]["warmup_epochs"] = 30
    result["training"]["input_masking"]["modality_dropout_probs"] = {
        s: p
        for s, p in result["training"]["input_masking"]["modality_dropout_probs"].items()
        if s not in excluded
    }
    result["data"]["num_workers"] = 2
    return result


def spatial_partition(centers: np.ndarray, *, tile_size: float) -> dict[str, list[int]]:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2 or not np.isfinite(centers).all():
        raise ValueError("centers must be finite [N,2] projected coordinates")
    if len(centers) < 5 or tile_size <= 0:
        raise ValueError("at least five patches and positive tile size are required")
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(centers)
    order = np.lexsort((km.cluster_centers_[:, 1], km.cluster_centers_[:, 0]))
    groups = [np.flatnonzero(km.labels_ == i).tolist() for i in order]
    test, validation = groups[0], groups[-1]
    held = centers[test + validation]
    close = np.max(np.abs(centers[:, None] - held[None]), axis=-1).min(axis=1)
    candidates = set(range(len(centers))) - set(test + validation)
    train = sorted(i for i in candidates if close[i] > tile_size + 0.01)
    buffer = sorted(candidates - set(train))
    if not train:
        raise ValueError("spatial buffer leaves no training patches")
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "buffer": buffer,
        **{f"group{i}": ids for i, ids in enumerate(groups)},
    }


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config_path: Path, root: Path, workers: int, *, include_highres: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if (root / "cache.json").exists():
        raise FileExistsError("prepared cache already exists; use a new output directory")
    config = Config.from_yaml(config_path)
    if len(config.data.datasets) != 1:
        raise ValueError("prepare currently requires one region")
    if not include_highres and any(
        s.role == "highres" for s in config.model.input_sources.values()
    ):
        raise ValueError("public baseline cache cannot include highres inputs")
    dataset = RegionRasterDataset(config, config.data.datasets[0])
    centers, bounds = [], []
    for record in dataset.records:
        source = next(iter(config.model.input_sources))
        paths = dataset._record_value(record, source)
        relative = paths[0] if isinstance(paths, list) else paths
        if relative is None:
            raise ValueError(f"missing grid reference for {record.patch_id}")
        with rasterio.open(dataset._absolute(relative)) as raster:
            if not raster.crs or not raster.crs.is_projected:
                raise ValueError("spatial grouping requires a projected metric CRS")
            b = tuple(raster.bounds)
            bounds.append(b)
            centers.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    width = bounds[0][2] - bounds[0][0]
    if any(abs(b[2] - b[0] - width) > 0.01 for b in bounds):
        raise ValueError("inconsistent patch extents")
    split = spatial_partition(np.array(centers), tile_size=width)
    rng = np.random.default_rng(20260913)
    split["pilot_train"] = sorted(
        rng.choice(split["train"], min(64, len(split["train"])), replace=False).tolist()
    )
    split["pilot_validation"] = sorted(
        rng.choice(split["validation"], min(16, len(split["validation"])), replace=False).tolist()
    )
    samples = root / "samples"
    samples.mkdir(exist_ok=True)

    def materialize(index: int) -> dict:
        sample = dataset[index]
        if sample["highres_frames"] and not include_highres:
            raise ValueError("unexpected highres content in public baseline")
        destination = samples / f"{index:06d}.pt"
        temp = destination.with_suffix(".partial")
        torch.save(sample, temp)
        temp.replace(destination)
        return {
            "index": index,
            "patch_id": sample["patch_id"],
            "path": str(destination),
            "sha256": _sha(destination),
            "bounds": bounds[index],
            "target_valid_pixels": {k: float(v.sum()) for k, v in sample["target_masks"].items()},
        }

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(materialize, range(len(dataset))):
            records.append(row)
            if len(records) % 20 == 0:
                print(json.dumps({"cached": len(records), "total": len(dataset)}), flush=True)
    if any(
        sum(r["target_valid_pixels"][h] for r in records) == 0 for h in config.model.target_heads
    ):
        raise ValueError("at least one target has no effective supervision")
    _json(
        root / "cache.json",
        {
            "version": 1,
            "data": asdict(config.data),
            "source_config_sha256": _sha(config_path),
            "model_inputs": {k: asdict(v) for k, v in config.model.input_sources.items()},
            "model_targets": {k: asdict(v) for k, v in config.model.target_heads.items()},
            "manifest_sha256": _sha(config.data.datasets[0].manifest_path),
            "records": records,
            "split": split,
        },
    )


class CachedSamples(Dataset):
    def __init__(self, document: dict, indices: list[int]):
        self.records = [document["records"][i] for i in indices]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return torch.load(self.records[index]["path"], weights_only=True, mmap=True)


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run(args: argparse.Namespace) -> None:
    config = Config.from_yaml(args.config)
    document = json.loads((args.cache / "cache.json").read_text())
    if document["model_inputs"] != {k: asdict(v) for k, v in config.model.input_sources.items()}:
        raise ValueError("cache input schema mismatch")
    if document["model_targets"] != {k: asdict(v) for k, v in config.model.target_heads.items()}:
        raise ValueError("cache target schema mismatch")
    cached_data = dict(document["data"])
    cached_data.setdefault("monthly_highres", False)
    cached_data.setdefault("highres_month_assignments", {})
    if cached_data != json.loads(json.dumps(asdict(config.data), default=str)):
        raise ValueError("cache data configuration mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    resume = getattr(args, "resume", None)
    if (args.output / "run.json").exists() and resume is None:
        raise FileExistsError("run exists; never overwrite an experiment")
    git_sha = _git_sha()
    random.seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    torch.manual_seed(config.experiment.seed)
    device, distributed, _ = _setup_device(args.device)
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    if distributed:
        dist.barrier()
    torch.manual_seed(config.experiment.seed + rank)
    system = build_training_system(config)
    adaptation = None
    if getattr(args, "initialize", None) is not None:
        from xuannv_embedding.training.adaptation import initialize_adaptation

        if getattr(args, "base_config", None) is None:
            raise ValueError("adaptation requires --base-config")
        adaptation = initialize_adaptation(system, config, args, document["split"])
    if config.data.monthly_highres and adaptation is None:
        raise ValueError("monthly highres requires registered incremental adaptation")
    system = system.to(device)
    wrapped = (
        torch.nn.parallel.DistributedDataParallel(
            system,
            device_ids=[device.index] if device.type != "cpu" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if distributed
        else system
    )
    optimizer = build_optimizer(system, config.training.lr, config.training.weight_decay)
    scheduler = build_scheduler(optimizer, config.training.warmup_epochs, config.training.epochs)
    scaler = _grad_scaler(device, config.training.amp)
    prefix = "pilot_" if args.pilot else ""
    split = document["split"]
    train = CachedSamples(document, split[prefix + "train"])
    validation_indices = split[prefix + "validation"][rank::world_size]
    validation = CachedSamples(document, validation_indices)
    for record in train.records + validation.records:
        if _sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("cached sample checksum mismatch")
    loader_kwargs = dict(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        collate_fn=collate_region_batch,
    )
    if config.data.num_workers:
        loader_kwargs["multiprocessing_context"] = "spawn"
        if distributed:
            # Cached samples have no random transforms. Isolate worker seeding
            # from the checkpointed model/masking RNG, including after resume.
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["generator"] = torch.Generator().manual_seed(
                config.experiment.seed + rank
            )
    sampler = (
        DistributedSampler(
            train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.experiment.seed,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(train, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
    validation_loader = DataLoader(validation, shuffle=False, **loader_kwargs)
    metadata = {
        "git_sha": git_sha,
        "config_sha256": _sha(args.config),
        "cache_sha256": _sha(args.cache / "cache.json"),
        "device": str(device),
        "world_size": world_size,
        "global_batch_size": config.data.batch_size * world_size,
        "train_sampler_padding": (len(sampler) * world_size - len(train)) if sampler else 0,
        "validation_sharding": "rank-strided, no padding or duplicate test/validation samples",
        "seed": config.experiment.seed,
        "lr": config.training.lr,
        "initialization": "registered_base" if adaptation else "scratch",
        "adaptation": adaptation,
        "pilot": args.pilot,
        "epochs": args.epochs,
        "train_indices": split[prefix + "train"],
        "validation_indices": split[prefix + "validation"],
        "selection": "unmasked validation total loss at fixed final objective weights",
        "parameters": sum(p.numel() for p in system.parameters()),
        "trainable_parameters": sum(p.numel() for p in system.parameters() if p.requires_grad),
    }
    if resume is None and rank == 0:
        _json(args.output / "run.json", metadata)
        args.output.joinpath("config.yaml").write_bytes(args.config.read_bytes())
    elif resume is not None:
        previous = json.loads((args.output / "run.json").read_text())
        if previous.get("world_size", 1) != world_size:
            raise ValueError("resume world size mismatch")
        for key in (
            "cache_sha256",
            "config_sha256",
            "pilot",
            "train_indices",
            "validation_indices",
            "git_sha",
            "adaptation",
        ):
            if previous[key] != metadata[key]:
                raise ValueError(f"resume provenance mismatch: {key}")
    if distributed:
        dist.barrier()
    started = time.monotonic()
    best = float("inf")
    total_steps = 0
    accumulation = config.training.gradient_accumulation_steps
    start_epoch = 0

    def save(name: str, epoch: int, metrics: dict) -> None:
        states = gather_objects(rank_random_state(device, scaler))
        if rank != 0:
            return
        save_training_checkpoint(
            args.output / name,
            model=system.model,
            criterion=system.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config_sha256=metadata["config_sha256"],
            git_sha=git_sha,
            source_schema=metadata_schema,
            regions=regions,
            metrics={
                **metrics,
                "best_validation": best,
                **states[0],
                "rank_random_states": states,
                "world_size": world_size,
            },
        )

    metadata_schema = {k: asdict(v) for k, v in config.model.input_sources.items()}
    regions = [d.region for d in config.data.datasets]
    if resume is not None:
        state = load_training_checkpoint(
            resume,
            model=system.model,
            criterion=system.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            expected_config_sha256=metadata["config_sha256"],
            expected_source_schema=metadata_schema,
            expected_regions=regions,
        )
        start_epoch = state["epoch"] + 1
        best = state["metrics"]["best_validation"]
        total_steps = state["metrics"]["optimizer_steps"]
        states = state["metrics"].get("rank_random_states", [state["metrics"]])
        if len(states) != world_size:
            raise ValueError("checkpoint random states do not match world size")
        restore_rank_random_state(states[rank], device, scaler)
    if start_epoch >= args.epochs:
        raise ValueError("no epochs remaining")
    overflow_steps = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.monotonic()
        wrapped.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        system.criterion.set_epoch(epoch)
        train_sums: dict[str, float] = {}
        train_seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, raw in enumerate(train_loader):
            batch = _move(raw, device)
            batch = apply_input_masking(batch, asdict(config.training.input_masking))
            # The final accumulation window may be shorter than the configured window.
            window = min(accumulation, len(train_loader) - (step // accumulation) * accumulation)
            with _autocast(device, config.training.amp):
                losses = wrapped(batch)
                loss = losses["total"] / window
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"nonfinite training loss at epoch {epoch}, step {step}")
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            n = len(raw["patch_ids"])
            train_seen += n
            for key, value in losses.items():
                if value.numel() == 1:
                    train_sums[key] = train_sums.get(key, 0.0) + n * float(
                        value.detach().float().cpu()
                    )
            for key, value in batch.get("masking_stats", {}).items():
                train_sums[key] = train_sums.get(key, 0.0) + n * float(value.detach().cpu())
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                if scaler is None:
                    norm = torch.nn.utils.clip_grad_norm_(
                        system.parameters(), 5.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(system.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if not bool(torch.isfinite(norm).item()):
                        overflow_steps += 1
                        if overflow_steps >= 8:
                            raise FloatingPointError("eight consecutive gradient overflows")
                    else:
                        overflow_steps = 0
                optimizer.zero_grad(set_to_none=True)
                total_steps += int(bool(torch.isfinite(norm).item()))
            if rank == 0 and (step == 0 or (step + 1) % 10 == 0):
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "batch": step + 1,
                            "loss": float(losses["total"].detach().cpu()),
                        }
                    ),
                    flush=True,
                )
        _synchronize(device)
        train_seconds = time.monotonic() - epoch_started
        system.eval()
        system.criterion.set_epoch(config.training.epochs)
        validation_sums: dict[str, float] = {}
        seen = 0
        with torch.inference_mode():
            for raw in validation_loader:
                batch = _move(raw, device)
                with _autocast(device, config.training.amp):
                    losses = system(batch)
                if not bool(torch.isfinite(losses["total"]).item()):
                    raise FloatingPointError("nonfinite validation loss")
                n = len(raw["patch_ids"])
                seen += n
                for key, value in losses.items():
                    if value.numel() == 1:
                        validation_sums[key] = validation_sums.get(key, 0.0) + n * float(
                            value.detach().float().cpu()
                        )
        scheduler.step()
        _synchronize(device)
        metrics = {
            "epoch": epoch + 1,
            "optimizer_steps": total_steps,
            "train_seconds": train_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "lr_next": optimizer.param_groups[0]["lr"],
            "train": global_means(train_sums, train_seen),
            "validation": global_means(validation_sums, seen),
            "validation_samples": len(split[prefix + "validation"]),
            "world_size": world_size,
            "global_batch_size": config.data.batch_size * world_size,
        }
        if device.type == "npu":
            peaks = gather_objects(torch.npu.max_memory_allocated(device))
            metrics["peak_memory_bytes_by_rank"] = peaks
            metrics["peak_memory_bytes"] = max(peaks)
        if rank == 0:
            with (args.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            _json(args.output / "status.json", {"state": "running", **metrics})
        score = metrics["validation"]["total"]
        if score < best:
            best = score
            save("best.pt", epoch, metrics)
        save("latest.pt", epoch, metrics)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "epoch_complete": epoch + 1,
                        "validation": score,
                        "train_seconds": train_seconds,
                        "best": best,
                    }
                ),
                flush=True,
            )
        if distributed:
            dist.barrier()
    if rank == 0:
        _json(
            args.output / "status.json", {"state": "complete", **metrics, "best_validation": best}
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv experiment")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--include-highres", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--resume", type=Path)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--initialize", type=Path)
    p.add_argument("--base-config", type=Path)
    p.add_argument("--freeze-base", action="store_true")
    p.add_argument("--continue-base", action="store_true")
    p.add_argument("--highres-encoding", choices=["native", "resample"], default="native")
    p = sub.add_parser("follow")
    p.add_argument("--root", type=Path, required=True)
    p = sub.add_parser("cpu-queue")
    p.add_argument("--plan", type=Path, required=True)
    p = sub.add_parser("test-readout")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--slot-directory", type=Path)
    p = sub.add_parser("comparison")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=["raw", "alphaearth", "dinov3"], required=True)
    p.add_argument("--source", type=Path)
    p = sub.add_parser("probe")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--slot-directory", type=Path)
    p = sub.add_parser("export")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--probe-output", type=Path)
    p.add_argument("--probe-slots", type=Path)
    p = sub.add_parser("queue")
    p.add_argument("--plan", type=Path, required=True)
    p = sub.add_parser("fold-cache")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--test-group", type=int, choices=range(5), required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.action == "prepare":
        prepare(args.config, args.output, args.workers, include_highres=args.include_highres)
    elif args.action == "cpu-queue":
        from xuannv_embedding.training.evaluation_queue import run as queue_run

        queue_run(args.plan)
    elif args.action == "test-readout":
        from xuannv_embedding.downstream.final_evaluation import run as test_run

        test_run(args)
    elif args.action == "comparison":
        from xuannv_embedding.downstream.comparison_features import run as comparison_run

        comparison_run(args)
    elif args.action == "probe":
        from xuannv_embedding.downstream.development import run as probe_run

        probe_run(args)
    elif args.action == "export":
        from xuannv_embedding.training.experiment_export import run as export_run

        export_run(args)
    elif args.action == "fold-cache":
        from xuannv_embedding.training.experiment_folds import derive_fold_cache

        derive_fold_cache(args.cache, args.output, args.test_group)
    elif args.action == "follow":
        from xuannv_embedding.training.experiment_schedule import main as follow_main

        follow_main(args.root)
    elif args.action == "queue":
        from xuannv_embedding.training.experiment_queue import main as queue_main

        queue_main(args.plan)
    else:
        if args.epochs <= 0:
            parser.error("epochs must be positive")
        run(args)
    return 0
