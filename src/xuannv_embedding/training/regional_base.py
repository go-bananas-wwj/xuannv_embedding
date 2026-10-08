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
from xuannv_embedding.training.masking import apply_input_masking
from xuannv_embedding.training.regional_refinement import StaticObjective, rename_targets
from xuannv_embedding.training.runtime import _autocast, _grad_scaler, _move


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


class RegionalObjective(StaticObjective):
    def __init__(
        self, base: nn.Module, embed_dim: int, classes: int, quality_channels: dict[str, int]
    ) -> None:
        super().__init__(base, embed_dim, "esa_worldcover", classes)
        self.quality_channels = quality_channels
        self.head_only = False
        self.weight = 0.25
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
        return super().forward(
            cleaned, clean_targets, masks, supervised_labels, supervised_label_masks
        )


def build_regional_system(
    config: Config, *, mode: str, checkpoint: Path | None, seed: int
) -> tuple[TrainingSystem, dict[str, Any]]:
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
    )
    receipt = apply_initial_weights(
        system.model, system.criterion, mode=mode, checkpoint=checkpoint
    )
    set_reference_window(system.model, config.data.months)
    system.requires_grad_(True)
    return system, receipt


def read_spec(path: Path) -> tuple[dict[str, Any], Config, dict[str, Any]]:
    spec = json.loads(path.read_text())
    required = {"schema", "mode", "config", "cache", "checkpoint", "training", "output"}
    if set(spec) != required or spec["schema"] != "regional-base-v1":
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
    cache = json.loads(Path(spec["cache"]["path"]).read_text())
    if cache["state"] != "complete" or cache["months"] != config.data.months:
        raise ValueError("Cache is incomplete or its month window differs")
    if not cache["records"]:
        raise ValueError("No prepared samples")
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
    if set(training) != expected or training["effective_batch"] != 48:
        raise ValueError("Invalid training settings")
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
    )
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "validate":
        dump(
            output / "initialization.json",
            {**receipt, "strict_keys": True, "months": config.data.months, "optimizer_steps": 0},
        )
        return 0
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.config.allow_internal_format = False
        torch.npu.set_device(device)
    if args.phase == "forward":
        system.to(device).eval()
        sample = torch.load(cache["records"][0]["path"], weights_only=True, mmap=True)
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
    if args.micro_batch < 1 or 48 % (world * args.micro_batch):
        raise ValueError("Micro batch and world size must divide effective batch48")
    accumulation = 48 // (world * args.micro_batch)
    system.to(device)
    groups = {name: [] for name in ["public", "highres", "heads"]}
    for name, parameter in system.named_parameters():
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
    started = time.monotonic()
    step = 1
    skipped_updates = 0
    while step <= settings["steps"]:
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
        for _ in range(accumulation):
            ids = torch.randperm(len(cache["records"]), generator=generator).tolist()
            selected = [
                ids[(rank * args.micro_batch + i) % len(ids)] for i in range(args.micro_batch)
            ]
            samples = [
                torch.load(cache["records"][i]["path"], weights_only=True, mmap=True)
                for i in selected
            ]
            batch = _move(collate_region_batch(samples), device)
            batch = apply_input_masking(batch, asdict(config.training.input_masking))
            with _autocast(device, device.type == "npu"):
                losses = wrapped(batch)
                loss = losses["total"] / accumulation
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
        if rank == 0 and (step % 20 == 0 or step == settings["steps"]):
            dump(
                output / "status.json",
                {
                    "state": "running",
                    "optimizer_steps": step,
                    "elapsed_seconds": time.monotonic() - started,
                    "loss": float(losses["total"].detach().cpu()),
                },
            )
        if rank == 0 and (step % settings["save_every"] == 0 or step == settings["steps"]):
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
                metrics={"steps": step},
            )
            save_training_checkpoint(output / "latest.pt", **kwargs)
            if step in [400, settings["steps"]]:
                save_training_checkpoint(
                    output / ("final.pt" if step == settings["steps"] else "step0400.pt"), **kwargs
                )
        step += 1
    if rank == 0:
        dump(
            output / "status.json",
            {
                "state": "complete",
                "optimizer_steps": settings["steps"],
                "elapsed_seconds": time.monotonic() - started,
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
