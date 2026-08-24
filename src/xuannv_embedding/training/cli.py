"""P10C 训练运行时与发布 smoke 入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import RegionRasterDataset, collate_region_batch
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from xuannv_embedding.training.losses import TotalLoss
from xuannv_embedding.training.masking import apply_input_masking
from xuannv_embedding.training.runtime import TrainingSystem, train_steps


def build_training_system(config: Config) -> TrainingSystem:
    model = AEFModel(
        sensor_channels=config.model.sensor_channels,
        embed_dim=config.model.embed_dim,
        target_heads=config.model.decoder_specs,
        stem_dim=config.model.stem_dim,
        stp=asdict(config.model.stp),
        num_months=config.model.num_months,
        ref_year=config.model.ref_year,
        ref_month=config.model.ref_month,
        gradient_checkpointing=config.training.gradient_checkpointing,
        source_roles=config.model.source_roles,
    )
    training = config.training
    criterion = TotalLoss(
        config.model.loss_specs,
        uniformity_weight=training.uniformity_weight,
        uniformity_warmup_epochs=training.uniformity_warmup_epochs,
        uniformity_temperature=training.uniformity_temperature,
        semantic_probe_embed_dim=config.model.embed_dim,
        semantic_probe_weight=training.semantic_probe_weight,
        semantic_probe_warmup_epochs=training.semantic_probe_warmup_epochs,
        semantic_probe_tasks=training.semantic_probe_tasks,
        semantic_probe_task_weights=training.semantic_probe_task_weights,
        semantic_probe_pos_weight=training.semantic_probe_pos_weight,
        semantic_probe_pos_weights=training.semantic_probe_pos_weights,
        semantic_probe_hidden_dim=training.semantic_probe_hidden_dim,
        semantic_probe_hard_negative_ratio=training.semantic_probe_hard_negative_ratio,
        semantic_probe_hard_negative_weight=training.semantic_probe_hard_negative_weight,
        semantic_probe_hard_negative_warmup_epochs=(
            training.semantic_probe_hard_negative_warmup_epochs
        ),
    )
    return TrainingSystem(model, criterion)


def synthetic_batch(
    config: Config,
    *,
    batch_size: int,
    spatial_size: int,
    missing_sources: set[str] = frozenset(),
) -> dict[str, Any]:
    """构造遵守 source role 和缺模态合同的无持久化 smoke batch。"""
    if batch_size <= 0 or spatial_size < 16:
        raise ValueError("batch_size 必须为正，spatial_size 必须至少为 16")
    months = torch.tensor(
        [int(month.replace("-", "")) for month in config.data.months], dtype=torch.long
    )
    timestamps = months.unsqueeze(0).repeat(batch_size, 1)
    month_count = len(config.data.months)
    source_frames: dict[str, torch.Tensor] = {}
    source_masks: dict[str, torch.Tensor] = {}
    highres_frames: dict[str, torch.Tensor] = {}
    highres_masks: dict[str, torch.Tensor] = {}
    for source, source_config in config.model.input_sources.items():
        missing = source in missing_sources
        if source_config.role == "temporal":
            source_frames[source] = (
                torch.zeros(
                    batch_size,
                    month_count,
                    source_config.channels,
                    spatial_size,
                    spatial_size,
                )
                if missing
                else torch.randn(
                    batch_size,
                    month_count,
                    source_config.channels,
                    spatial_size,
                    spatial_size,
                )
            )
            source_masks[source] = (
                torch.zeros(batch_size, month_count)
                if missing
                else torch.ones(batch_size, month_count)
            )
        else:
            highres_frames[source] = (
                torch.zeros(batch_size, source_config.channels, spatial_size, spatial_size)
                if missing
                else torch.randn(batch_size, source_config.channels, spatial_size, spatial_size)
            )
            highres_masks[source] = (
                torch.zeros(batch_size, 1, spatial_size, spatial_size)
                if missing
                else torch.ones(batch_size, 1, spatial_size, spatial_size)
            )

    targets: dict[str, torch.Tensor] = {}
    target_masks: dict[str, torch.Tensor] = {}
    for name, head in config.model.target_heads.items():
        source_missing = head.source in missing_sources
        if head.loss_type == "continuous":
            targets[name] = torch.randn(
                batch_size,
                month_count,
                head.channels,
                spatial_size,
                spatial_size,
            )
        else:
            targets[name] = torch.randint(
                1,
                head.channels,
                (batch_size, month_count, spatial_size, spatial_size),
            )
        target_masks[name] = (
            torch.zeros(batch_size, month_count, spatial_size, spatial_size)
            if source_missing
            else torch.ones(batch_size, month_count, spatial_size, spatial_size)
        )

    labels = {
        task: torch.zeros(batch_size, spatial_size, spatial_size)
        for task in config.training.semantic_probe_tasks
    }
    for value in labels.values():
        value[:, 0, 0] = 1.0
    return {
        "patch_ids": [f"synthetic_{index:06d}" for index in range(batch_size)],
        "source_frames": source_frames,
        "source_masks": source_masks,
        "timestamps": timestamps,
        "highres_frames": highres_frames,
        "highres_masks": highres_masks,
        "targets": targets,
        "target_masks": target_masks,
        "supervised_labels": labels,
        "supervised_label_masks": {
            task: torch.ones(batch_size) for task in config.training.semantic_probe_tasks
        },
    }


class RegionBatchStream:
    """按 sampling_weight 确定性轮转区域 loader，每个 epoch 可重新迭代。"""

    def __init__(
        self,
        loaders: list[DataLoader],
        weights: list[float],
        *,
        seed: int,
        max_steps: int | None,
        masking_config: dict[str, Any],
    ) -> None:
        if not loaders or len(loaders) != len(weights):
            raise ValueError("loaders 与 weights 必须非空且长度一致")
        self.loaders = loaders
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.seed = seed
        self.max_steps = max_steps
        self.masking_config = masking_config
        self.epoch = 0

    def __iter__(self):
        for loader in self.loaders:
            sampler = loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
        iterators = [iter(loader) for loader in self.loaders]
        natural_steps = sum(len(loader) for loader in self.loaders)
        steps = self.max_steps or natural_steps
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(steps):
            index = int(torch.multinomial(self.weights, 1, generator=generator).item())
            try:
                batch = next(iterators[index])
            except StopIteration:
                iterators[index] = iter(self.loaders[index])
                batch = next(iterators[index])
            yield apply_input_masking(batch, self.masking_config)


def build_region_batch_stream(
    config: Config,
    *,
    distributed: bool,
    max_records: int | None,
    max_steps: int | None,
) -> RegionBatchStream:
    loaders: list[DataLoader] = []
    weights: list[float] = []
    for dataset_config in config.data.datasets:
        dataset = RegionRasterDataset(config, dataset_config, max_records=max_records)
        sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
        loaders.append(
            DataLoader(
                dataset,
                batch_size=config.data.batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                num_workers=config.data.num_workers,
                collate_fn=collate_region_batch,
                pin_memory=True,
                drop_last=False,
            )
        )
        weights.append(dataset_config.sampling_weight)
    return RegionBatchStream(
        loaders,
        weights,
        seed=config.experiment.seed,
        max_steps=max_steps,
        masking_config=asdict(config.training.input_masking),
    )


def _setup_device(requested: str | None) -> tuple[torch.device, bool, int]:
    distributed = "RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        import torch_npu  # noqa: F401

        torch.npu.set_device(local_rank)
        dist.init_process_group(backend="hccl")
        return torch.device(f"npu:{local_rank}"), True, local_rank
    if requested is not None:
        device = torch.device(requested)
    else:
        try:
            import torch_npu  # noqa: F401

            device = torch.device("npu:0") if torch.npu.is_available() else torch.device("cpu")
        except ImportError:
            device = torch.device("cpu")
    if device.type == "npu":
        torch.npu.set_device(device)
    return device, False, local_rank


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _epoch_count(configured_epochs: int, requested_epochs: int | None, start_epoch: int) -> int:
    """无显式覆盖时，把配置 epochs 解释为最终总 epoch 数。"""
    epochs = requested_epochs if requested_epochs is not None else configured_epochs - start_epoch
    if epochs <= 0:
        raise ValueError(
            f"没有待训练 epoch: configured={configured_epochs}, start_epoch={start_epoch}"
        )
    return epochs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--spatial-size", type=int, default=128)
    parser.add_argument("--missing-source", action="append", default=[])
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error("--steps 不得为负")
    if args.synthetic and args.steps == 0:
        parser.error("--synthetic 需要显式提供正数 --steps")

    config = Config.from_yaml(args.config)
    device, distributed, local_rank = _setup_device(args.device)
    torch.manual_seed(config.experiment.seed + (dist.get_rank() if distributed else 0))
    system = build_training_system(config).to(device)
    optimizer = torch.optim.AdamW(
        system.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    start_epoch = 0
    if args.resume is not None:
        state = load_training_checkpoint(
            args.resume,
            model=system.model,
            criterion=system.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = int(state["epoch"]) + 1
    wrapped: nn.Module = system
    if distributed:
        wrapped = nn.parallel.DistributedDataParallel(
            system, device_ids=[local_rank], broadcast_buffers=False
        )
    if args.synthetic:
        batch = synthetic_batch(
            config,
            batch_size=args.batch_size,
            spatial_size=args.spatial_size,
            missing_sources=set(args.missing_source),
        )
        batches: Any = [batch] * args.steps
    else:
        batches = build_region_batch_stream(
            config,
            distributed=distributed,
            max_records=args.max_records,
            max_steps=args.steps or None,
        )
    try:
        epoch_count = _epoch_count(config.training.epochs, args.epochs, start_epoch)
    except ValueError as exc:
        parser.error(str(exc))
    summary = train_steps(
        wrapped,
        batches,
        optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epoch_count,
        start_epoch=start_epoch,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        amp=config.training.amp and not args.no_amp,
    )
    summary.update(
        {
            "world_size": dist.get_world_size() if distributed else 1,
            "spatial_size": args.spatial_size,
            "device_type": device.type,
        }
    )
    rank = dist.get_rank() if distributed else 0
    if rank == 0:
        save_training_checkpoint(
            args.output,
            model=system.model,
            criterion=system.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=int(summary["end_epoch"]),
            config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
            git_sha=_git_sha(),
            source_schema={
                name: asdict(value) for name, value in config.model.input_sources.items()
            },
            regions=[dataset.region for dataset in config.data.datasets],
            metrics=summary,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0
