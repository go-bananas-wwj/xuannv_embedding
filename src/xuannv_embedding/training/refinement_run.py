"""Exact-update distributed runner for a registered regional refinement."""

import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from xuannv_embedding.training.cli import _git_sha, _setup_device
from xuannv_embedding.training.distributed_experiment import (
    gather_objects,
    rank_random_state,
    restore_rank_random_state,
)
from xuannv_embedding.training.masking import apply_input_masking
from xuannv_embedding.training.regional_refinement import (
    batch_at,
    initialize,
    parameter_groups,
    read_spec,
)
from xuannv_embedding.training.runtime import _autocast, _grad_scaler, _move


def verify_samples(cache, labels):
    for record in [*cache["records"], *labels["records"]]:
        if sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("immutable image or static-target sample changed")


def _phase(completed, settings):
    return "head" if completed < settings["head_steps"] else "joint"


def run(args):
    spec, cache, labels = read_spec(args.spec)
    settings = spec["training"]
    device, distributed, _ = _setup_device(args.device)
    rank, size = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    if size != settings["world_size"]:
        raise ValueError("registered world size differs from runtime")
    resume = getattr(args, "resume", None)
    if args.output.exists() and resume is None:
        raise FileExistsError("refinement output already exists")
    if rank == 0:
        verify_samples(cache, labels)
        args.output.mkdir(parents=True, exist_ok=resume is not None)
    if distributed:
        dist.barrier()
    torch.manual_seed(settings["seed"])
    system, config = initialize(spec, cache, labels)
    system.to(device)
    schema = {k: asdict(v) for k, v in config.model.input_sources.items()}
    regions = [d.region for d in config.data.datasets]
    spec_sha, code_sha = sha(args.spec), _git_sha()
    metadata = {
        "protocol": spec["protocol"],
        "config_sha256": spec_sha,
        "git_sha": code_sha,
        "cache_sha256": spec["cache"]["sha256"],
        "spec": spec,
        "representation_training_indices": spec["training_indices"],
        "downstream_partitions": cache["split"],
        "evaluation_scope": spec["evaluation_scope"],
        "world_size": size,
        "global_effective_batch": size * settings["micro_batch"] * settings["accumulation"],
        "sampling": (
            "shuffled rank-strided cycles padded to world size; "
            "exact effective batch across cycle boundaries"
        ),
        "embedding_sampling": (
            "deterministic vMF mean as in parent adaptation; weights jointly trainable"
        ),
        "checkpoint_selection": "fixed final optimizer update, no downstream score selection",
    }
    if resume is None:
        if rank == 0:
            dump(args.output / "run.json", metadata)
            (args.output / "recipe.json").write_bytes(Path(args.spec).read_bytes())
        state = None
        completed = cycle = offset = 0
        seen = set()
        elapsed_before = 0.0
    else:
        previous = json.loads((args.output / "run.json").read_text())
        if previous != metadata:
            raise ValueError("refinement resume registration mismatch")
        state = load_training_checkpoint(
            resume,
            model=system.model,
            criterion=system.criterion,
            expected_config_sha256=spec_sha,
            expected_source_schema=schema,
            expected_regions=regions,
        )
        if state["git_sha"] != code_sha:
            raise ValueError("refinement checkpoint code changed")
        m = state["metrics"]
        completed, cycle, offset = m["optimizer_steps"], m["cycle"], m["offset"]
        seen = set(m["rank_seen_indices"][rank])
        elapsed_before = m["elapsed_seconds"]
    total = settings["head_steps"] + settings["joint_steps"]
    stop = getattr(args, "stop_after_updates", None) or total
    if not completed < stop <= total:
        raise ValueError("no registered optimizer updates remaining")
    phase = None
    wrapped = optimizer = scaler = None
    overflow = 0
    started = time.monotonic()
    history = []
    last_metrics = {}
    while completed < stop:
        next_phase = _phase(completed, settings)
        if phase != next_phase:
            if wrapped is not None:
                del wrapped
            phase = next_phase
            groups = parameter_groups(system, settings["learning_rates"], head_only=phase == "head")
            optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
            scaler = _grad_scaler(device, settings["amp"])
            system.criterion.head_only = phase == "head"
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
            system.train()
            if state is not None:
                if state["metrics"]["phase"] == phase:
                    optimizer.load_state_dict(state["optimizer"])
                    states = state["metrics"]["rank_random_states"]
                    if len(states) != size:
                        raise ValueError("resume random-state world size differs")
                    restore_rank_random_state(states[rank], device, scaler)
                else:
                    torch.manual_seed(settings["seed"] + rank + 10000)
                state = None
            else:
                # Distinct, checkpointed masking RNG; model construction was identical on ranks.
                torch.manual_seed(settings["seed"] + rank + (10000 if phase == "joint" else 0))
            if rank == 0:
                dump(
                    args.output / (phase + "_parameters.json"),
                    {
                        "trainable": {
                            name: p.numel()
                            for name, p in system.named_parameters()
                            if p.requires_grad
                        },
                        "groups": [
                            {
                                "name": g["name"],
                                "lr": g["lr"],
                                "parameters": sum(p.numel() for p in g["params"]),
                            }
                            for g in groups
                        ],
                    },
                )
        phase_step = completed if phase == "head" else completed - settings["head_steps"]
        if phase == "joint":
            warm = min(settings["ramp_steps"], settings["joint_steps"])
            factor = (
                (phase_step + 1) / warm
                if phase_step < warm
                else 0.5
                * (
                    1
                    + math.cos(
                        math.pi * (phase_step - warm) / max(1, settings["joint_steps"] - warm)
                    )
                )
            )
            for group in optimizer.param_groups:
                group["lr"] = settings["learning_rates"][group["name"]] * factor
            system.criterion.weight = settings["static_weight"] * min(
                1.0, (phase_step + 1) / settings["ramp_steps"]
            )
        else:
            system.criterion.weight = 1.0
        optimizer.zero_grad(set_to_none=True)
        values = {}
        pending_seen = set()
        for _ in range(settings["accumulation"]):
            raw, cycle, offset, indices = batch_at(
                cache, labels, spec, cycle=cycle, offset=offset, rank=rank
            )
            pending_seen.update(indices)
            batch = apply_input_masking(_move(raw, device), asdict(config.training.input_masking))
            with _autocast(device, settings["amp"]):
                losses = wrapped(batch)
                loss = losses["total"] / settings["accumulation"]
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("nonfinite regional training loss")
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            for key in ["total", "recon", "semantic_probe", "recon_" + labels["target"]]:
                if key in losses:
                    values[key] = (
                        values.get(key, 0.0)
                        + float(losses[key].detach()) / settings["accumulation"]
                    )
        if scaler is not None:
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(system.parameters(), 5.0)
        finite = torch.isfinite(norm).to(device=device, dtype=torch.int32)
        if distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            overflow += 1
            if scaler is None or overflow >= 8:
                raise FloatingPointError("regional gradient overflow")
            scaler.update(new_scale=scaler.get_scale() / 2)
            continue
        overflow = 0
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        seen.update(pending_seen)
        completed += 1
        gathered = gather_objects(values)
        last_metrics = {
            "optimizer_steps": completed,
            "phase": phase,
            "phase_steps": phase_step + 1,
            "cycle": cycle,
            "offset": offset,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "losses": {k: sum(v[k] for v in gathered) / size for k in values},
            "static_weight": system.criterion.weight,
            "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups},
        }
        if device.type == "npu":
            last_metrics["peak_memory_bytes"] = max(
                gather_objects(torch.npu.max_memory_allocated(device))
            )
        if rank == 0:
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(last_metrics) + "\n")
            dump(args.output / "status.json", {"state": "running", **last_metrics})
            if completed % 25 == 0 or completed == 1:
                print(json.dumps(last_metrics), flush=True)
        if completed % settings["save_every"] == 0 or completed in [settings["head_steps"], stop]:
            states = gather_objects(rank_random_state(device, scaler))
            seen_indices = gather_objects(sorted(seen))
            if rank == 0:
                if shutil.disk_usage(args.output).free < settings["min_free_gib"] * 2**30:
                    raise OSError("registered minimum free space reached")
                checkpoint_metrics = {
                    **last_metrics,
                    "rank_random_states": states,
                    "rank_seen_indices": seen_indices,
                    "world_size": size,
                }
                save_training_checkpoint(
                    args.output / "latest.pt",
                    model=system.model,
                    criterion=system.criterion,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=completed - 1,
                    config_sha256=spec_sha,
                    git_sha=code_sha,
                    source_schema=schema,
                    regions=regions,
                    metrics=checkpoint_metrics,
                )
                if completed in [settings["head_steps"], total]:
                    destination = args.output / (
                        "head_complete.pt" if completed == settings["head_steps"] else "final.pt"
                    )
                    shutil.copy2(args.output / "latest.pt", destination)
                history.append(
                    {"step": completed, "checkpoint_sha256": sha(args.output / "latest.pt")}
                )
                with (args.output / "checkpoint_history.jsonl").open("a") as stream:
                    stream.write(json.dumps(history[-1]) + "\n")
            if distributed:
                dist.barrier()
    seen_indices = gather_objects(sorted(seen))
    if rank == 0:
        all_seen = sorted(set(i for part in seen_indices for i in part))
        if stop == total and all_seen != sorted(spec["training_indices"]):
            raise ValueError("final model did not visit every registered training patch")
        dump(
            args.output / "status.json",
            {
                "state": "complete" if stop == total else "paused",
                **last_metrics,
                "unique_training_indices": all_seen,
                "evaluation_scope": spec["evaluation_scope"],
            },
        )
    if distributed:
        dist.barrier()


def export(args):
    from torch.utils.data import DataLoader

    from xuannv_embedding.data.raster_dataset import collate_region_batch
    from xuannv_embedding.export.embedding import export_embedding_batches
    from xuannv_embedding.training.experiment import CachedSamples

    spec, cache, labels = read_spec(args.spec)
    registration = json.loads((args.checkpoint.parent / "run.json").read_text())
    if registration["config_sha256"] != sha(args.spec):
        raise ValueError("export recipe differs from training")
    system, config = initialize(spec, cache, labels)
    state = load_training_checkpoint(
        args.checkpoint,
        model=system.model,
        criterion=system.criterion,
        expected_config_sha256=sha(args.spec),
        expected_source_schema={k: asdict(v) for k, v in config.model.input_sources.items()},
        expected_regions=[d.region for d in config.data.datasets],
    )
    if state["git_sha"] != registration["git_sha"]:
        raise ValueError("refinement export checkpoint source identity differs")
    device, distributed, _ = _setup_device(args.device)
    if distributed or args.batch_size < 1:
        raise ValueError("refinement export requires one device and positive batch size")
    args.output.mkdir(parents=True, exist_ok=False)
    verify_samples(cache, labels)
    records = []
    started = time.monotonic()
    loader = DataLoader(
        CachedSamples(cache, list(range(len(cache["records"])))),
        batch_size=args.batch_size,
        collate_fn=collate_region_batch,
        num_workers=0,
    )
    for batch in loader:
        files = export_embedding_batches(
            system.model, [batch], args.output / "embeddings", device=device
        )
        for path in files:
            original = cache["records"][len(records)]
            records.append(
                {
                    "patch_id": original["patch_id"],
                    "bounds": original["bounds"],
                    "path": str(path),
                    "sha256": sha(path),
                }
            )
        dump(
            args.output / "status.json",
            {
                "state": "running",
                "patches": len(records),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    metadata = {
        "export_git_sha": _git_sha(),
        "training_git_sha": state["git_sha"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha(args.checkpoint),
        "checkpoint_epoch": state["epoch"] + 1,
        "config_sha256": sha(args.spec),
        "cache_sha256": spec["cache"]["sha256"],
        "months": config.data.months,
        "split": cache["split"],
        "dtype": "float32",
        "masking": "none; actual availability retained",
        "selection": (
            "fixed final refinement update"
            if state["metrics"]["optimizer_steps"]
            == spec["training"]["head_steps"] + spec["training"]["joint_steps"]
            else "intermediate refinement snapshot; not a final experiment result"
        ),
        "refinement": registration,
        "representation_training_indices": spec["training_indices"],
        "evaluation_scope": spec["evaluation_scope"],
        "records": records,
    }
    dump(args.output / "manifest.json", metadata)
    dump(
        args.output / "status.json",
        {
            "state": "complete",
            "patches": len(records),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
