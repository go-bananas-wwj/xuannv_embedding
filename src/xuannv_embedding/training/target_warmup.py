"""Bounded target-readout warmup with an immutable annual encoder."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch import nn

from xuannv_embedding.config import Config
from xuannv_embedding.training.checkpoint import _atomic_torch_save, load_training_checkpoint
from xuannv_embedding.training.cli import _git_sha, _setup_device, _source_code_sha256
from xuannv_embedding.training.runtime import _autocast, _model_view, _move
from xuannv_embedding.training.target_supervision import TargetSupervision, validate_research_budget

FORMAT = "xuannv.target-head-warmup.v1"


def head_contract(config):
    target = config.training.independent_targets
    if target is None or config.model.architecture != "annual5m":
        raise ValueError("Head warmup requires annual5m and independent targets")
    return {
        "embed_dim": config.model.embed_dim,
        "schema": target.schema,
        "preparation_sha256": target.preparation_sha256,
        "scope": target.scope,
        "weights": target.weights,
        "dem_mean": target.dem_mean,
        "dem_std": target.dem_std,
        "source_schema": {k: asdict(v) for k, v in config.model.input_sources.items()},
        "context_pooling": config.model.annual_context_pooling,
        "exact_mask_coverage": config.model.annual_exact_mask_coverage,
        "vmf_training_noise": True,
        "input_masking": asdict(config.training.input_masking),
        "highres_max_observations": config.data.highres_max_observations,
        "statistics_dirs": sorted({str(row.statistics_dir) for row in config.data.datasets}),
    }


class HeadReadouts(nn.Module):
    """Independent initializations share frozen views, never encoder gradients."""

    def __init__(self, config, seeds):
        super().__init__()
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Head seeds must be nonempty and unique")
        target = config.training.independent_targets
        head_contract(config)
        self.readouts = nn.ModuleDict(
            {
                str(seed): TargetSupervision(
                    config.model.embed_dim,
                    target.weights,
                    dem_mean=target.dem_mean,
                    dem_std=target.dem_std,
                    seed=seed,
                )
                for seed in seeds
            }
        )

    def forward(self, embedding):
        return {seed: module(embedding) for seed, module in self.readouts.items()}


def freeze_encoder(model):
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None


def head_step(model, readouts, batch, optimizer, *, device, amp=False):
    """The optimizer must own exactly the readouts; reject an accidental encoder update."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("The complete encoder must be frozen and in eval mode")
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if actual != {id(p) for p in readouts.parameters()}:
        raise ValueError("Warmup optimizer may contain only all target readout parameters")
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(), _autocast(device, amp):
        embedding = _model_view(model, batch, decode=False).embedding_map[:, 0].float()
    if not bool(torch.isfinite(embedding).all()):
        raise FloatingPointError("Nonfinite frozen embedding")
    predictions = readouts(embedding)
    bare = readouts.module if hasattr(readouts, "module") else readouts
    losses = {
        seed: module.loss(predictions[seed], batch["independent_targets"], ramp=1.0)
        for seed, module in bare.readouts.items()
    }
    # Sum, rather than average: each independent replica retains its declared LR.
    total = sum(value["total"] for value in losses.values())
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite head objective")
    total.backward()
    gradients = [p.grad for p in readouts.parameters()]
    if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("Missing or nonfinite readout gradient")
    norm = float(torch.stack([g.float().norm() for g in gradients]).norm())
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite readout gradient norm")
    optimizer.step()
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError("Encoder gradient appeared during frozen warmup")
    return {
        "head_losses": {
            seed: {name: float(value.detach()) for name, value in values.items()}
            for seed, values in losses.items()
        },
        "global_weighted_losses": {
            seed: sum(
                module.weights[name] * float(losses[seed][f"target_{name}"])
                for name in module.weights
            )
            for seed, module in bare.readouts.items()
        },
        "head_gradient_norm": norm,
    }


def restore_warmed_heads(path, module, *, seed, expected_contract, expected_model_sha256):
    """Load readouts only, checking their original encoder and target identities."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state.get("format") != FORMAT
        or state.get("contract") != expected_contract
        or state.get("model_state_sha256") != expected_model_sha256
        or state.get("model_optimizer_updates") != 0
        or not state.get("encoder_unchanged")
    ):
        raise ValueError("Warmup encoder/target identity mismatch")
    values = state["heads"][str(seed)]
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError("Nonfinite warmed readout")
    module.load_state_dict(values, strict=True)
    return state


def continue_readouts(path, heads, optimizer, *, config, model_sha256, world_size, epoch_batches):
    """Continue completed frozen-view traversals; encoder and readout identity stay strict."""
    from xuannv_embedding.training.annual_experiments import digest

    state = torch.load(path, map_location="cpu", weights_only=True)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if (
        receipt["checkpoint_sha256"] != digest(path)
        or receipt["world_size"] != world_size
        or state["feature_seed"] != config.experiment.seed
        or list(state["heads"]) != list(heads.readouts)
        or receipt["global_annual_exposures"]
        != (receipt["head_optimizer_updates_per_seed"] - receipt.get("previous_head_updates", 0))
        * config.data.batch_size
        * world_size
    ):
        raise ValueError("Readout continuation sampling/checkpoint identity differs")
    step = state["head_optimizer_updates"]
    if step < 1 or step % epoch_batches:
        raise ValueError("Readout continuation requires a complete frozen-data traversal")
    for seed, module in heads.readouts.items():
        restore_warmed_heads(
            path,
            module,
            seed=int(seed),
            expected_contract=head_contract(config),
            expected_model_sha256=model_sha256,
        )
    expected_lr = config.training.lr * config.training.independent_targets.head_lr_multiplier
    optimizer.load_state_dict(state["optimizer"])
    if any(group["lr"] != expected_lr for group in optimizer.param_groups):
        raise ValueError("Readout continuation learning rate differs")
    return (
        step,
        state["head_initialization"],
        {
            "path": str(path),
            "sha256": digest(path),
            "previous_steps": step,
            "git_sha": state["git_sha"],
            "source_code_sha256": state["source_code_sha256"],
            "noise_rng_restarted": True,
        },
    )


def load_base(config, initial_config_path, checkpoint, device):
    from xuannv_embedding.training.annual_experiments import digest
    from xuannv_embedding.training.cli import build_training_system

    initial = Config.from_yaml(initial_config_path)
    schema = {key: asdict(value) for key, value in initial.model.input_sources.items()}
    if schema != {key: asdict(value) for key, value in config.model.input_sources.items()}:
        raise ValueError("Initial and warmup source schemas differ")
    if initial.model.architecture != config.model.architecture:
        raise ValueError("Initial and warmup architectures differ")
    system = build_training_system(config).to(device)
    state = load_training_checkpoint(
        checkpoint,
        model=system.model,
        device=device,
        expected_config_sha256=digest(initial_config_path),
        expected_source_schema=schema,
        expected_regions=[row.region for row in initial.data.datasets],
    )
    identity = {
        "checkpoint_sha256": digest(checkpoint),
        "config_sha256": digest(initial_config_path),
        "git_sha": state["git_sha"],
        "epoch": state["epoch"],
    }
    del state
    return system, identity


def warmup(args):
    from xuannv_embedding.training.annual_data import build_annual_stream
    from xuannv_embedding.training.annual_experiments import digest, model_state_digest, write_json

    config = Config.from_yaml(args.config)
    contract = head_contract(config)
    if args.output.exists() or any(args.output.parent.glob(f"{args.output.stem}.step-*.pt")):
        raise FileExistsError("Warmup output must be new")
    validate_research_budget(
        config.training.independent_targets,
        steps=args.steps,
        epochs=1,
        initialize_from=args.checkpoint,
        resume=None,
    )
    if not config.training.independent_targets.research_max_updates:
        raise ValueError("Head warmup requires an explicit bounded research budget")
    if args.steps < 1 or args.save_every < 1:
        raise ValueError("Warmup steps/save interval must be positive")
    device, distributed, local_rank = _setup_device(args.device)
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    try:
        torch.manual_seed(config.experiment.seed + rank)
        system, identity = load_base(config, args.initialize_config, args.checkpoint, device)
        model = system.model
        freeze_encoder(model)
        # Preserve training's stochastic bottleneck while freezing every parameter.
        model.bottleneck.train()
        before = model_state_digest(model)
        heads = HeadReadouts(config, args.head_seeds).to(device)
        initial = {
            seed: copy.deepcopy(module.cpu().state_dict())
            for seed, module in heads.readouts.items()
        }
        heads.to(device)
        wrapped = (
            nn.parallel.DistributedDataParallel(
                heads,
                device_ids=[local_rank] if device.type != "cpu" else None,
                broadcast_buffers=False,
            )
            if distributed
            else heads
        )
        lr = config.training.lr * config.training.independent_targets.head_lr_multiplier
        optimizer = torch.optim.AdamW(
            wrapped.parameters(), lr=lr, weight_decay=config.training.weight_decay
        )
        stream = build_annual_stream(
            config,
            distributed=distributed,
            max_records=None,
            max_steps=args.steps,
            start_epoch=0,
            teacher_enabled=False,
        )
        git_sha, source_sha = _git_sha(), _source_code_sha256()
        history, records, step = [], [], 0
        parent = None
        if args.readout_state is not None:
            epoch_batches = sum(len(loader) for loader in stream.loaders)
            step, initial, parent = continue_readouts(
                args.readout_state,
                heads,
                optimizer,
                config=config,
                model_sha256=before,
                world_size=world,
                epoch_batches=epoch_batches,
            )
            if step >= args.steps:
                raise ValueError("Requested total warmup budget is already complete")
            stream.epoch = step // epoch_batches
        previous_steps = step
        epoch_metrics = []
        start = time.perf_counter()

        def save(path):
            if any(not bool(torch.isfinite(p).all()) for p in heads.parameters()):
                raise FloatingPointError("Nonfinite warmed head state")
            if any(int(row["step"]) != step for row in optimizer.state.values()):
                raise AssertionError("Head optimizer step count differs")
            _atomic_torch_save(
                {
                    "format": FORMAT,
                    "contract": contract,
                    "config_sha256": digest(args.config),
                    "git_sha": git_sha,
                    "source_code_sha256": source_sha,
                    "initialization_identity": identity,
                    "model_state_sha256": before,
                    "encoder_unchanged": True,
                    "model_optimizer_updates": 0,
                    "head_optimizer_updates": step,
                    "feature_seed": config.experiment.seed,
                    "readout_parent": parent,
                    "feature_mode": (
                        "training masks and vMF noise; other frozen layers eval; "
                        "BF16 encoder if supported, FP32 readouts"
                    ),
                    "head_learning_rate": lr,
                    "head_initialization": initial,
                    "heads": {
                        seed: {k: v.detach().cpu() for k, v in module.state_dict().items()}
                        for seed, module in heads.readouts.items()
                    },
                    "optimizer": optimizer.state_dict(),
                },
                path,
            )

        while step < args.steps:
            count = 0
            data_epoch = stream.epoch
            totals = {seed: {} for seed in heads.readouts}
            for batch in stream:
                rows = [(row["year"], row["patch_id"]) for row in batch["metadata"]]
                metrics = head_step(
                    model,
                    wrapped,
                    _move(batch, device),
                    optimizer,
                    device=device,
                    amp=config.training.amp,
                )
                records.extend(rows)
                for seed, values in metrics["head_losses"].items():
                    values = {
                        name: value
                        for name, value in values.items()
                        if name in {f"target_{task}" for task in heads.readouts[seed].weights}
                    }
                    values["weighted_total"] = metrics["global_weighted_losses"][seed]
                    for name, value in values.items():
                        totals[seed][name] = totals[seed].get(name, 0.0) + value
                step += 1
                count += 1
                if step == 1 or step % 25 == 0 or step == args.steps:
                    entry = {
                        "step": step,
                        "elapsed_seconds": time.perf_counter() - start,
                        **metrics,
                    }
                    history.append(entry)
                    if rank == 0:
                        print(json.dumps(entry), flush=True)
                if step % args.save_every == 0:
                    if model_state_digest(model) != before:
                        raise AssertionError("Encoder state changed")
                    if rank == 0:
                        save(args.output.with_name(f"{args.output.stem}.step-{step}.pt"))
                if step == args.steps:
                    break
            if not count:
                raise ValueError("No training batches available")
            entry = {
                "data_epoch": data_epoch,
                "batches": count,
                "global_batch_objective_means": {
                    seed: {name: value / count for name, value in values.items()}
                    for seed, values in totals.items()
                },
            }
            epoch_metrics.append(entry)
            if rank == 0:
                print(json.dumps({"warmup_epoch": entry}), flush=True)
        if model_state_digest(model) != before:
            raise AssertionError("Frozen encoder changed")
        receipt = {
            "rank": rank,
            "head_optimizer_updates": step,
            "samples": len(records),
            "encoder_unchanged": True,
            "model_state_sha256": before,
            "head_state_sha256": model_state_digest(heads),
            "nonfinite_steps": 0,
            "records": records,
        }
        ranks = [receipt]
        if distributed:
            ranks = [None] * world
            dist.all_gather_object(ranks, receipt)
        if rank == 0:
            if (
                len({row["model_state_sha256"] for row in ranks}) != 1
                or len({row["head_state_sha256"] for row in ranks}) != 1
            ):
                raise AssertionError("Rank states differ")
            save(args.output)
            all_records = [tuple(item) for row in ranks for item in row.pop("records")]
            write_json(
                args.output.with_suffix(".json"),
                {
                    "passed": True,
                    "world_size": world,
                    "head_seeds": args.head_seeds,
                    "feature_seed": config.experiment.seed,
                    "head_optimizer_updates_per_seed": step,
                    "previous_head_updates": previous_steps,
                    "head_updates_this_run": step - previous_steps,
                    "readout_parent": parent,
                    "model_optimizer_updates": 0,
                    "global_annual_exposures": len(all_records),
                    "unique_annual_records": len(set(all_records)),
                    "unique_positions": len({patch for _, patch in all_records}),
                    "traversal_sha256": hashlib.sha256(
                        json.dumps(sorted(all_records)).encode()
                    ).hexdigest(),
                    "head_learning_rate": lr,
                    "encoder_unchanged": True,
                    "model_state_sha256": before,
                    "checkpoint_sha256": digest(args.output),
                    "wall_seconds": time.perf_counter() - start,
                    "git_sha": git_sha,
                    "source_code_sha256": source_sha,
                    "rank_metrics": ranks,
                    "training_loss_progress": history,
                    "epoch_metrics": epoch_metrics,
                    "scope": "Training-only readout warmup; no downstream/confirmation scoring.",
                },
            )
    finally:
        if distributed:
            dist.destroy_process_group()
