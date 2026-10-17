"""Build regional representations with explicit scratch/transfer initialization.

Changing the area and month window is intentional. Architecture and all tensor
keys remain strict, and optimization requires an approval tied to the exact spec.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import collate_region_batch
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.models.highres_transformer import HighResTransformerModel
from xuannv_embedding.models.model import AEFOutput
from xuannv_embedding.training.checkpoint import save_training_checkpoint
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.distributed_experiment import gather_objects, global_means
from xuannv_embedding.training.feature_preservation import (
    attach_teacher,
    freeze_public_parameters,
    load_teacher,
    preservation_loss,
)
from xuannv_embedding.training.masking import apply_input_masking
from xuannv_embedding.training.regional_diagnostics import append_training_curve
from xuannv_embedding.training.regional_refinement import StaticObjective, rename_targets
from xuannv_embedding.training.regional_resume import capture_rank, recover, registration
from xuannv_embedding.training.runtime import TrainingSystem, _autocast, _grad_scaler, _move


def require_training_approval(spec_path: Path, approval_path: Path | None) -> None:
    if approval_path is None:
        raise ValueError("Training requires explicit approval of the final preparation report")
    approval = json.loads(approval_path.read_text())
    if approval.get("approved") is not True or approval.get("spec_sha256") != sha(spec_path):
        raise ValueError("Training approval does not match the exact experiment specification")


def set_reference_window(model: nn.Module, months: list[str]) -> None:
    year, month = map(int, months[0].split("-"))
    origin = year * 12 + month - 1
    expected = [f"{(origin+i)//12:04d}-{(origin+i)%12+1:02d}" for i in range(len(months))]
    base = model.base if hasattr(model, "base") else model
    if months != expected or len(months) != base.num_months:
        raise ValueError("The new window must contain the same number of consecutive months")
    for module in (base, base.monthly_embed):
        module.ref_year, module.ref_month = year, month


def apply_initial_weights(
    model: nn.Module, criterion: nn.Module | None, *, mode: str, checkpoint: Path | None
) -> dict[str, Any]:
    if mode == "scratch":
        return {"mode": mode, "parent_weights_read": False}
    if mode != "transfer" or checkpoint is None:
        raise ValueError("Transfer initialization requires a registered checkpoint")
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state["model"], strict=True)
    if criterion is not None:
        criterion.load_state_dict(state["criterion"], strict=True)
    return {"mode": mode, "parent_weights_read": True, "checkpoint_sha256": sha(checkpoint)}


def validate_static_weight(weight: float) -> float:
    if type(weight) not in {int, float} or not math.isfinite(weight) or weight < 0:
        raise ValueError("Regional static objective weight must be finite and nonnegative")
    return float(weight)


def checkpoint_steps(settings: dict[str, Any]) -> list[int]:
    values = settings.get("checkpoint_steps", [])
    if (
        not isinstance(values, list)
        or any(type(v) is not int or not 0 < v < settings["steps"] for v in values)
        or values != sorted(set(values))
    ):
        raise ValueError("Checkpoint steps must be unique, ordered and inside the update budget")
    return values


class RegionalObjective(StaticObjective):
    def __init__(
        self,
        base: nn.Module,
        embed_dim: int,
        classes: int,
        quality_channels: dict[str, int],
        *,
        static_weight: float = 0.25,
    ) -> None:
        static_weight = validate_static_weight(static_weight)
        super().__init__(base, embed_dim, "esa_worldcover", classes)
        self.quality_channels = quality_channels
        self.head_only = False
        self.weight = static_weight
        self.teacher_weight = 0.0
        self.base.requires_grad_(True)

    def set_epoch(self, step: int) -> None:
        self.base.set_epoch(step)

    def forward(self, output, targets, masks, supervised_labels=None, supervised_label_masks=None):
        # Preserve model channel contracts while excluding categorical quality
        # carriers from the continuous reconstruction objective.
        recon, clean_targets = dict(output.reconstructions), dict(targets)
        for key, channel in self.quality_channels.items():
            if key not in recon:
                raise ValueError("Unregistered reconstruction quality channel")
            keep = [i for i in range(recon[key].shape[2]) if i != channel]
            recon[key] = recon[key][:, :, keep]
            clean_targets[key] = clean_targets[key][:, :, keep]
        cleaned = AEFOutput(output.embedding_map, output.embedding, recon)
        result = super().forward(
            cleaned, clean_targets, masks, supervised_labels, supervised_label_masks
        )
        if self.teacher_weight:
            anchor = preservation_loss(
                output, targets["public_teacher_embedding"], masks["public_teacher_embedding"]
            )
            result["public_anchor"] = anchor
            result["public_anchor_weighted"] = anchor * self.teacher_weight
            result["total"] = result["total"] + anchor * self.teacher_weight
        return result


def build_regional_system(
    config: Config,
    *,
    mode: str,
    checkpoint: Path | None,
    seed: int,
    static_weight: float = 0.25,
    initialization_config: Config | None = None,
) -> tuple[TrainingSystem, dict[str, Any]]:
    static_weight = validate_static_weight(static_weight)
    torch.manual_seed(seed)
    public_sources = {
        name: source
        for name, source in config.model.input_sources.items()
        if source.role == "temporal"
    }
    highres = {
        name: source.channels
        for name, source in config.model.input_sources.items()
        if source.role == "highres"
    }
    public_heads = {
        name: head for name, head in config.model.target_heads.items() if head.source not in highres
    }
    highres_heads = {
        name: head.channels
        for name, head in config.model.target_heads.items()
        if head.source in highres
    }
    if not highres:
        from xuannv_embedding.training.public_refinement import (
            PublicRefinementModel,
            validate_public_model,
        )

        validate_public_model(config.model)
        if initialization_config is not None:
            initial_sources = {
                k: v
                for k, v in initialization_config.model.input_sources.items()
                if v.role == "temporal"
            }
            initial_targets = {
                k: v
                for k, v in initialization_config.model.target_heads.items()
                if v.source in initial_sources or v.loss_type == "categorical"
            }
            projected = replace(
                initialization_config.model,
                input_sources=initial_sources,
                target_heads=initial_targets,
                highres_transformer=None,
            )
            if asdict(projected) != asdict(config.model):
                raise ValueError("shared initialization changed the common model architecture")
            if initialization_config.data.months != config.data.months or [
                d.region for d in initialization_config.data.datasets
            ] != [d.region for d in config.data.datasets]:
                raise ValueError("shared initialization changed the regional observation window")
            reference, _ = build_regional_system(
                initialization_config,
                mode="scratch",
                checkpoint=None,
                seed=seed,
                static_weight=static_weight,
            )
            system = TrainingSystem(
                PublicRefinementModel(reference.model.base), reference.criterion
            )
            system.criterion.base.target_cfg = {
                k: v
                for k, v in system.criterion.base.target_cfg.items()
                if k in initial_targets or k == "osm_landcover"
            }
        else:
            system = build_training_system(config)
            system.model = PublicRefinementModel(system.model)
            rename_targets(system.model, system.criterion, {"worldcover": "osm_landcover"})
            system.criterion = RegionalObjective(
                system.criterion,
                config.model.embed_dim,
                12,
                {"s2_recon": 11, "landsat_recon": 6},
                static_weight=static_weight,
            )
        receipt = apply_initial_weights(
            system.model, system.criterion, mode=mode, checkpoint=checkpoint
        )
        receipt.update(
            static_weight=system.criterion.weight,
            public_only=True,
            shared_untrained_initialization=initialization_config is not None,
        )
        set_reference_window(system.model, config.data.months)
        system.requires_grad_(True)
        return system, receipt
    if initialization_config is not None:
        raise ValueError("shared initialization is reserved for a public-observation control")
    if config.model.highres_transformer is None:
        raise ValueError("Regional base requires the registered highres transformer architecture")
    public_config = replace(
        config,
        model=replace(
            config.model,
            input_sources=public_sources,
            target_heads=public_heads,
            highres_transformer=None,
        ),
    )
    base = build_training_system(public_config)
    system = build_training_system(config)
    system.model = HighResTransformerModel(
        base.model,
        highres,
        highres_heads,
        settings=config.model.highres_transformer,
        freeze_base=True,
    )
    rename_targets(system.model, system.criterion, {"worldcover": "osm_landcover"})
    system.criterion = RegionalObjective(
        system.criterion,
        config.model.embed_dim,
        12,
        {"s2_recon": 11, "landsat_recon": 6},
        static_weight=static_weight,
    )
    receipt = apply_initial_weights(
        system.model, system.criterion, mode=mode, checkpoint=checkpoint
    )
    receipt["static_weight"] = system.criterion.weight
    set_reference_window(system.model, config.data.months)
    system.requires_grad_(True)
    return system, receipt


def validate_domains(cache: dict[str, Any]) -> None:
    """Require disjoint complete sample membership and explicit consecutive windows."""
    if "domains" not in cache:
        return
    domains = cache["domains"]
    if not isinstance(domains, list) or not domains:
        raise ValueError("Invalid training domains")
    indices, regions = [], []
    for domain in domains:
        if not isinstance(domain, dict) or set(domain) != {"region", "months", "indices"}:
            raise ValueError("Invalid domain fields")
        region, months, members = domain["region"], domain["months"], domain["indices"]
        if not isinstance(region, str) or not region or region in regions:
            raise ValueError("Invalid or duplicate domain region")
        if not isinstance(members, list) or not members or any(type(i) is not int for i in members):
            raise ValueError("Invalid domain membership")
        try:
            year, month = map(int, months[0].split("-"))
            origin = year * 12 + month - 1
            expected = [
                f"{(origin + i) // 12:04d}-{(origin + i) % 12 + 1:02d}"
                for i in range(len(cache["months"]))
            ]
            if months != expected:
                raise ValueError("Invalid domain month window")
        except (TypeError, ValueError, IndexError, KeyError, AttributeError) as exc:
            raise ValueError("Invalid domain month window") from exc
        regions.append(region)
        indices.extend(members)
    if sorted(indices) != list(range(len(cache["records"]))):
        raise ValueError("Domain membership must be disjoint and complete")
    for domain in domains:
        if any(cache["records"][i].get("region") != domain["region"] for i in domain["indices"]):
            raise ValueError("Domain region differs from sample registration")


def training_domain(cache: dict[str, Any], step: int) -> dict[str, Any]:
    """Alternate equally weighted regions; every rank uses the same step/window."""
    if type(step) is not int or step < 1:
        raise ValueError("Invalid domain training step")
    if "domains" not in cache:
        return {
            "region": "",
            "months": cache["months"],
            "indices": list(range(len(cache["records"]))),
        }
    return cache["domains"][(step - 1) % len(cache["domains"])]


def read_spec(path: Path) -> tuple[dict[str, Any], Config, dict[str, Any]]:
    spec = json.loads(path.read_text())
    required = {"schema", "mode", "config", "cache", "checkpoint", "training", "output"}
    if (
        not required.issubset(spec)
        or set(spec) - required - {"initialization_config", "teacher"}
        or spec["schema"] != "regional-base-v1"
    ):
        raise ValueError("Unknown or missing regional specification fields")
    if spec["mode"] not in {"transfer", "scratch"}:
        raise ValueError("Unknown initialization mode")
    for key in ("config", "cache"):
        if sha(Path(spec[key]["path"])) != spec[key]["sha256"]:
            raise ValueError("Registered input changed")
    if spec["mode"] == "transfer":
        if (
            not spec["checkpoint"]
            or sha(Path(spec["checkpoint"]["path"])) != spec["checkpoint"]["sha256"]
        ):
            raise ValueError("Parent checkpoint changed")
    elif spec["checkpoint"] is not None:
        raise ValueError("Scratch must not inherit a checkpoint")
    config = Config.from_yaml(spec["config"]["path"])
    if "initialization_config" in spec:
        initial = spec["initialization_config"]
        if set(initial) != {"path", "sha256"} or sha(Path(initial["path"])) != initial["sha256"]:
            raise ValueError("shared initialization configuration changed")
        if any(v.role != "temporal" for v in config.model.input_sources.values()):
            raise ValueError("shared initialization is reserved for a public-observation control")
    cache = json.loads(Path(spec["cache"]["path"]).read_text())
    if cache["state"] != "complete" or cache["months"] != config.data.months:
        raise ValueError("Cache is incomplete or its month window differs")
    if not cache["records"]:
        raise ValueError("No prepared samples")
    validate_domains(cache)
    if "domains" in cache and {d["region"] for d in cache["domains"]} != {
        d.region for d in config.data.datasets
    }:
        raise ValueError("Training domain regions differ from registered datasets")
    training = spec["training"]
    expected = {
        "steps",
        "head_steps",
        "warmup_steps",
        "seed",
        "effective_batch",
        "rates",
        "weight_decay",
        "save_every",
        "min_free_gib",
    }
    if (
        not expected.issubset(training)
        or set(training)
        - expected
        - {"static_weight", "checkpoint_steps", "freeze_public", "teacher_weight"}
        or training["effective_batch"] != 48
    ):
        raise ValueError("Invalid training settings")
    validate_static_weight(training.get("static_weight", 0.25))
    if type(training.get("freeze_public", False)) is not bool:
        raise ValueError("freeze_public must be a boolean")
    weight = training.get("teacher_weight", 0.0)
    if type(weight) not in {int, float} or not math.isfinite(weight) or weight < 0:
        raise ValueError("teacher_weight must be finite and nonnegative")
    if training.get("freeze_public", False) and config.model.highres_transformer is None:
        raise ValueError("Freezing the public backbone requires high-resolution trainable layers")
    if weight:
        if "teacher" not in spec:
            raise ValueError("Positive teacher_weight requires a registered public teacher")
        load_teacher(
            spec["teacher"], cache, channels=config.model.embed_dim, size=config.data.patch_size
        )
    elif "teacher" in spec:
        raise ValueError("An unused teacher reference is not allowed")
    checkpoint_steps(training)
    if training["steps"] < 1 or not 0 <= training["head_steps"] < training["steps"]:
        raise ValueError("Invalid update budget")
    if set(training["rates"]) != {"public", "highres", "heads"}:
        raise ValueError("The three learning-rate groups are required")
    if any(not math.isfinite(v) or v <= 0 for v in training["rates"].values()):
        raise ValueError("Learning rates must be finite and positive")
    return spec, config, cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv experiment regional-base")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["validate", "forward", "train"], default="validate")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args(argv)
    if args.phase == "train":
        require_training_approval(args.spec, args.approval)
        if "XUANNV_GIT_SHA" not in os.environ:
            raise ValueError("Training must register its actual source commit")
    spec, config, cache = read_spec(args.spec)
    system, receipt = build_regional_system(
        config,
        mode=spec["mode"],
        checkpoint=Path(spec["checkpoint"]["path"]) if spec["checkpoint"] else None,
        seed=spec["training"]["seed"],
        static_weight=spec["training"].get("static_weight", 0.25),
        **(
            {"initialization_config": Config.from_yaml(spec["initialization_config"]["path"])}
            if "initialization_config" in spec
            else {}
        ),
    )
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "validate":
        dump(
            output / "initialization.json",
            {**receipt, "strict_keys": True, "months": config.data.months, "optimizer_steps": 0},
        )
        return 0
    system.criterion.teacher_weight = spec["training"].get("teacher_weight", 0.0)
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.config.allow_internal_format = False
        torch.npu.set_device(device)
    if args.phase == "forward":
        system.to(device).eval()
        domain = training_domain(cache, 1)
        set_reference_window(system.model, domain["months"])
        sample = torch.load(
            cache["records"][domain["indices"][0]]["path"], weights_only=True, mmap=True
        )
        if system.criterion.teacher_weight:
            teacher = load_teacher(
                spec["teacher"], cache, channels=config.model.embed_dim, size=config.data.patch_size
            )
            sample = attach_teacher(sample, domain["indices"][0], teacher)
        batch = _move(collate_region_batch([sample]), device)
        with torch.inference_mode(), _autocast(device, device.type == "npu"):
            embedding = system.model(
                batch["source_frames"],
                batch["source_masks"],
                batch["timestamps"],
                batch["highres_frames"],
                batch["highres_masks"],
            ).embedding_map
            losses = system(batch)
        assert torch.isfinite(embedding).all() and torch.isfinite(losses["total"])
        dump(
            output / "forward_check.json",
            {
                "shape": list(embedding.shape),
                "loss": float(losses["total"].cpu()),
                "optimizer_steps": 0,
            },
        )
        return 0
    run_training(system, spec, config, cache, args, device)
    return 0


def run_training(
    system: TrainingSystem,
    spec: dict[str, Any],
    config: Config,
    cache: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if device.type == "npu":
            torch.npu.set_device(local)
            device = torch.device(f"npu:{local}")
        dist.init_process_group("hccl" if device.type == "npu" else "gloo")
    settings = spec["training"]
    system.criterion.teacher_weight = settings.get("teacher_weight", 0.0)
    teacher = None
    if settings.get("teacher_weight", 0.0):
        teacher = load_teacher(
            spec["teacher"], cache, channels=config.model.embed_dim, size=config.data.patch_size
        )
    if settings.get("freeze_public", False):
        freeze_public_parameters(system)
    if args.micro_batch < 1 or 48 % (world * args.micro_batch):
        raise ValueError("Micro batch and world size must divide effective batch48")
    accumulation = 48 // (world * args.micro_batch)
    system.to(device)
    groups = {name: [] for name in ["public", "highres", "heads"]}
    for name, parameter in system.named_parameters():
        if not parameter.requires_grad:
            continue
        key = (
            "heads"
            if name.startswith("criterion.") or "decoders." in name
            else ("public" if name.startswith("model.base.") else "highres")
        )
        groups[key].append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": parameters, "name": name, "lr": settings["rates"][name]}
            for name, parameters in groups.items()
            if parameters
        ],
        weight_decay=settings["weight_decay"],
    )
    wrapped = (
        DistributedDataParallel(
            system,
            device_ids=[local] if device.type == "npu" else None,
            find_unused_parameters=True,
        )
        if world > 1
        else system
    )
    scaler = _grad_scaler(device, device.type == "npu")
    generator = torch.Generator().manual_seed(settings["seed"])
    output = Path(spec["output"])
    identity = registration(spec, args.spec, world, args.micro_batch)
    recovered = {
        "completed_steps": 0,
        "elapsed_seconds": 0.0,
        "skipped_updates": 0,
        "recovery_mode": "fresh",
    }
    if getattr(args, "resume", None) is not None:
        recovered = recover(
            args.resume,
            system,
            optimizer,
            scaler,
            generator,
            device,
            config,
            spec,
            identity,
            cache,
            rank=rank,
        )
        previous = output / "status.json"
        if previous.exists():
            recovered["elapsed_seconds"] = max(
                recovered["elapsed_seconds"],
                json.loads(previous.read_text()).get("elapsed_seconds", 0),
            )
        if rank == 0:
            dump(
                output / f"recovery-{time.time_ns()}.json",
                {
                    **recovered,
                    "checkpoint": str(args.resume),
                    "checkpoint_sha256": sha(args.resume),
                    "source_git_sha": os.environ["XUANNV_GIT_SHA"],
                    "registration": identity,
                },
            )
    started = time.monotonic()
    step = recovered["completed_steps"] + 1
    skipped_updates = recovered["skipped_updates"]
    requested_stop = getattr(args, "stop_after_updates", None)
    stop = settings["steps"] if requested_stop is None else requested_stop
    if type(stop) is not int or stop < 1 or not step - 1 <= stop <= settings["steps"]:
        raise ValueError("Stop update must lie within the remaining registered budget")

    def elapsed():
        return recovered["elapsed_seconds"] + time.monotonic() - started

    def save(step, *, final=False):
        states = gather_objects(capture_rank(generator, device, scaler))
        if rank != 0:
            return
        kwargs = dict(
            model=system.model,
            criterion=system.criterion,
            optimizer=optimizer,
            scheduler=None,
            epoch=step - 1,
            config_sha256=spec["config"]["sha256"],
            git_sha=os.environ["XUANNV_GIT_SHA"],
            source_schema={k: asdict(v) for k, v in config.model.input_sources.items()},
            regions=[d.region for d in config.data.datasets],
            metrics={
                "steps": step,
                "static_weight": system.criterion.weight,
                "regional_resume": {
                    "schema": "regional-resume-v1",
                    "registration": identity,
                    "rank_states": states,
                    "elapsed_seconds": elapsed(),
                    "skipped_updates": skipped_updates,
                    "recovery_mode": recovered["recovery_mode"],
                },
            },
        )
        save_training_checkpoint(output / "latest.pt", **kwargs)
        if step in checkpoint_steps(settings):
            save_training_checkpoint(output / f"step{step:06d}.pt", **kwargs)
        if final or step == settings["steps"]:
            save_training_checkpoint(output / "final.pt", **kwargs)
        elif step == 400:
            save_training_checkpoint(output / "step0400.pt", **kwargs)

    if rank == 0:
        dump(
            output / "status.json",
            {
                "state": "running",
                "optimizer_steps": step - 1,
                "elapsed_seconds": elapsed(),
                "recovery_mode": recovered["recovery_mode"],
            },
        )
    while step <= stop:
        if __import__("shutil").disk_usage(output).free < settings["min_free_gib"] * 2**30:
            raise RuntimeError("Disk reserve reached")
        wrapped.train()
        system.criterion.set_epoch(step)
        for group in optimizer.param_groups:
            base_rate = settings["rates"][group["name"]]
            scale = min(1, step / max(settings["warmup_steps"], 1))
            scale *= (1 + math.cos(math.pi * (step - 1) / settings["steps"])) / 2
            group["lr"] = (
                0
                if step <= settings["head_steps"] and group["name"] != "heads"
                else base_rate * scale
            )
        optimizer.zero_grad(set_to_none=True)
        domain = training_domain(cache, step)
        set_reference_window(system.model, domain["months"])
        record_curve = step == 1 or step % 20 == 0 or step == stop
        curve_losses = {}
        for _ in range(accumulation):
            order = torch.randperm(len(domain["indices"]), generator=generator).tolist()
            ids = [domain["indices"][i] for i in order]
            selected = [
                ids[(rank * args.micro_batch + i) % len(ids)] for i in range(args.micro_batch)
            ]
            samples = [
                torch.load(cache["records"][i]["path"], weights_only=True, mmap=True)
                for i in selected
            ]
            if teacher is not None:
                samples = [
                    attach_teacher(sample, index, teacher)
                    for sample, index in zip(samples, selected)
                ]
            batch = _move(collate_region_batch(samples), device)
            batch = apply_input_masking(batch, asdict(config.training.input_masking))
            with _autocast(device, device.type == "npu"):
                losses = wrapped(batch)
                loss = losses["total"] / accumulation
            if record_curve:
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        curve_losses[key] = curve_losses.get(key, 0) + value.detach().float()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite regional loss")
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
        if not step_optimizer(optimizer, scaler):
            skipped_updates += 1
            if skipped_updates > 20:
                raise FloatingPointError(
                    "Repeated AMP overflows; stop without counting skipped updates"
                )
            continue
        if record_curve:
            # Some semantic statistics are absent when a rank has no valid labels.
            # Aggregate by key rather than assuming equal-length tensors across ranks.
            means = global_means(
                {key: float(value.cpu()) for key, value in curve_losses.items()}, accumulation
            )
            if rank == 0:
                append_training_curve(
                    output / "training-curve.jsonl",
                    step=step,
                    losses=means,
                    rates={group["name"]: group["lr"] for group in optimizer.param_groups},
                    elapsed_seconds=elapsed(),
                    skipped_updates=skipped_updates,
                )
        if rank == 0 and (step % 20 == 0 or step == settings["steps"]):
            dump(
                output / "status.json",
                {
                    "state": "running",
                    "optimizer_steps": step,
                    "elapsed_seconds": elapsed(),
                    "loss": float(losses["total"].detach().cpu()),
                    "active_domain": domain["region"],
                    "active_months": domain["months"],
                },
            )
        milestones = checkpoint_steps(settings)
        if step % settings["save_every"] == 0 or step == stop or step in milestones:
            save(step)
        step += 1
    if recovered["completed_steps"] == settings["steps"]:
        save(settings["steps"], final=True)
    if rank == 0:
        dump(
            output / "status.json",
            {
                "state": "complete" if stop == settings["steps"] else "checkpointed",
                "optimizer_steps": stop,
                "elapsed_seconds": elapsed(),
            },
        )
    if world > 1:
        dist.destroy_process_group()


def step_optimizer(optimizer: torch.optim.Optimizer, scaler: Any | None) -> bool:
    """Return False when dynamic scaling skips an update after an overflow."""
    if scaler is None:
        optimizer.step()
        return True
    previous = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= previous
