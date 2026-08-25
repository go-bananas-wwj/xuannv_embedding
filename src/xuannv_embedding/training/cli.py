"""P10C 训练运行时与发布 smoke 入口。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from xuannv_embedding.config import Config, V2Config
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.checkpoint import (
    load_training_checkpoint,
    load_v2_training_checkpoint,
    save_training_checkpoint,
    save_v2_training_checkpoint,
)
from xuannv_embedding.training.losses import TotalLoss
from xuannv_embedding.training.masking import apply_input_masking
from xuannv_embedding.training.optimizer import build_optimizer, build_scheduler
from xuannv_embedding.training.runtime import (
    TrainingSystem,
    V2TrainingSystem,
    train_steps,
    train_v2_steps,
)


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
        start_epoch: int = 0,
    ) -> None:
        if not loaders or len(loaders) != len(weights):
            raise ValueError("loaders 与 weights 必须非空且长度一致")
        self.loaders = loaders
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.seed = seed
        self.max_steps = max_steps
        self.masking_config = masking_config
        if start_epoch < 0:
            raise ValueError("start_epoch 必须是非负整数")
        self.epoch = start_epoch

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
    start_epoch: int = 0,
) -> RegionBatchStream:
    from xuannv_embedding.data.raster_dataset import RegionRasterDataset, collate_region_batch

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
        start_epoch=start_epoch,
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


def _git_sha_from_repository(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    candidate = result.stdout.strip()
    if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", candidate):
        return candidate
    return None


def _repository_path_from_direct_url(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    return Path(unquote(parsed.path))


def _git_sha() -> str:
    explicit = os.environ.get("XUANNV_GIT_SHA")
    if explicit is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", explicit):
            raise RuntimeError("XUANNV_GIT_SHA 必须是 7-64 位十六进制 Git commit")
        return explicit.lower()

    project_root = Path(__file__).resolve().parents[3]
    candidate = _git_sha_from_repository(project_root)
    if candidate is not None:
        return candidate

    try:
        direct_url = importlib.metadata.distribution("xuannv-embedding").read_text(
            "direct_url.json"
        )
        metadata = json.loads(direct_url) if direct_url is not None else {}
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        metadata = {}
    candidate = str(metadata.get("vcs_info", {}).get("commit_id", ""))
    if re.fullmatch(r"[0-9a-fA-F]{7,64}", candidate):
        return candidate.lower()
    source_repository = _repository_path_from_direct_url(str(metadata.get("url", "")))
    if source_repository is not None:
        candidate = _git_sha_from_repository(source_repository)
        if candidate is not None:
            return candidate
    raise RuntimeError("无法证明训练代码的 Git SHA；请设置 XUANNV_GIT_SHA 后再启动训练")


def _epoch_count(configured_epochs: int, requested_epochs: int | None, start_epoch: int) -> int:
    """无显式覆盖时，把配置 epochs 解释为最终总 epoch 数。"""
    epochs = requested_epochs if requested_epochs is not None else configured_epochs - start_epoch
    if epochs <= 0:
        raise ValueError(
            f"没有待训练 epoch: configured={configured_epochs}, start_epoch={start_epoch}"
        )
    return epochs


def _periodic_checkpoint_path(final_path: Path, completed_epoch: int) -> Path:
    suffix = final_path.suffix or ".pt"
    stem = final_path.name[: -len(suffix)] if final_path.suffix else final_path.name
    return final_path.with_name(f"{stem}.epoch-{completed_epoch:04d}{suffix}")


def _v2_registry(config: V2Config, profile_name: str) -> Path:
    names = {
        "mini-real": "mini_16.parquet",
        "smoke": "smoke_620_plus_32.parquet",
        "npu-smoke": "smoke_620_plus_32.parquet",
    }
    if profile_name not in names:
        raise ValueError(f"未知 V2 validation profile: {profile_name}")
    return config.paths.data_root / "registry" / names[profile_name]


def _run_v2_training(args: argparse.Namespace) -> int:
    from xuannv_embedding.data.v2_dataset import V2LocalZipDataset, collate_v2
    from xuannv_embedding.training.validation_profiles import (
        build_profile_model,
        build_v2_criterion,
        data_manifest_sha256,
        v2_product_schema,
        v2_temporal_contract,
    )

    if args.profile is None:
        raise ValueError("V2 训练必须显式提供 --profile")
    config = V2Config.from_yaml(args.config)
    if args.profile not in config.validation_profiles:
        raise ValueError(f"配置中不存在 validation profile: {args.profile}")
    profile = config.validation_profiles[args.profile]
    if args.profile != "mini-real":
        raise ValueError("Stage 04 仅执行 mini-real；smoke profile 由后续阶段入口执行")
    device, distributed, _ = _setup_device(args.device)
    if distributed:
        raise ValueError("mini-real 是单进程恢复门禁，不接受分布式环境")
    torch.manual_seed(42)
    dataset = V2LocalZipDataset(
        config,
        _v2_registry(config, args.profile),
        spatial_size=profile.spatial_size,
        max_records=profile.records,
        january_pair_only=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=profile.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_v2,
    )
    model = build_profile_model(config, profile)
    criterion = build_v2_criterion(config)
    system = V2TrainingSystem(model, criterion)
    effective_lr = (
        max(config.training.lr, 1.0e-3) if profile.model_profile == "mini" else config.training.lr
    )
    optimizer = torch.optim.AdamW(
        system.parameters(), lr=effective_lr, weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    manifest_sha = data_manifest_sha256(config.paths.data_root)
    product_schema = v2_product_schema(config)
    temporal_contract = v2_temporal_contract(config)
    if args.resume is not None:
        state = load_v2_training_checkpoint(
            args.resume,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config_sha256=config_sha,
            expected_data_manifest_sha256=manifest_sha,
            expected_product_schema=product_schema,
            expected_temporal_contract=temporal_contract,
            device=device,
        )
        start_step = int(state["step"])
    else:
        start_step = 0
    first_parameter = next(model.parameters())
    parameter_before = first_parameter.detach().clone()
    summary = train_v2_steps(
        system,
        loader,
        optimizer,
        device=device,
        max_steps=profile.steps,
        amp=config.training.amp and not args.no_amp and device.type != "cpu",
        scheduler=scheduler,
    )
    if torch.equal(parameter_before, first_parameter.detach()):
        raise RuntimeError("mini-real 模型参数未发生变化")
    if not optimizer.state:
        raise RuntimeError("mini-real optimizer 状态为空")
    completed_step = start_step + profile.steps
    summary.update(
        {
            "profile": args.profile,
            "records": len(dataset),
            "effective_lr": effective_lr,
            "data_manifest_sha256": manifest_sha,
            "config_sha256": config_sha,
            "git_sha": _git_sha(),
            "checkpoint_restore_verified": False,
        }
    )
    save_v2_training_checkpoint(
        args.output,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        step=completed_step,
        config_sha256=config_sha,
        git_sha=summary["git_sha"],
        data_manifest_sha256=manifest_sha,
        product_schema=product_schema,
        temporal_contract=temporal_contract,
        metrics=summary,
    )

    restored_model = build_profile_model(config, profile)
    restored_criterion = build_v2_criterion(config)
    restored = V2TrainingSystem(restored_model, restored_criterion)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=effective_lr, weight_decay=config.training.weight_decay
    )
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    load_v2_training_checkpoint(
        args.output,
        model=restored_model,
        criterion=restored_criterion,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_config_sha256=config_sha,
        expected_data_manifest_sha256=manifest_sha,
        expected_product_schema=product_schema,
        expected_temporal_contract=temporal_contract,
        device=device,
    )
    resume_summary = train_v2_steps(
        restored,
        loader,
        restored_optimizer,
        device=device,
        max_steps=profile.resume_steps,
        amp=False,
        scheduler=restored_scheduler,
    )
    completed_step += profile.resume_steps
    summary["checkpoint_restore_verified"] = True
    summary["resume_loss"] = resume_summary["loss"]

    if profile.overfit_steps:
        overfit_model = build_profile_model(config, profile)
        overfit_criterion = build_v2_criterion(config)
        overfit_system = V2TrainingSystem(overfit_model, overfit_criterion)
        overfit_optimizer = torch.optim.AdamW(overfit_system.parameters(), lr=5.0e-3)
        fixed_batch = next(iter(loader))
        overfit = train_v2_steps(
            overfit_system,
            [fixed_batch],
            overfit_optimizer,
            device=device,
            max_steps=profile.overfit_steps,
            amp=False,
        )
        decline = (overfit["first_loss"] - overfit["last_loss"]) / max(
            abs(overfit["first_loss"]), 1.0e-8
        )
        if decline < 0.05:
            raise RuntimeError(f"mini-real 单 batch 过拟合下降不足 5%: {decline:.3%}")
        summary["overfit_loss_decline"] = decline

    save_v2_training_checkpoint(
        args.output,
        model=restored_model,
        criterion=restored_criterion,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        step=completed_step,
        config_sha256=config_sha,
        git_sha=summary["git_sha"],
        data_manifest_sha256=manifest_sha,
        product_schema=product_schema,
        temporal_contract=temporal_contract,
        metrics=summary,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--profile")
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

    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if isinstance(document, dict) and document.get("schema_version") == "2":
        return _run_v2_training(args)
    config = Config.from_yaml(args.config)
    git_sha = _git_sha()
    device, distributed, local_rank = _setup_device(args.device)
    torch.manual_seed(config.experiment.seed + (dist.get_rank() if distributed else 0))
    system = build_training_system(config).to(device)
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    source_schema = {name: asdict(value) for name, value in config.model.input_sources.items()}
    regions = [dataset.region for dataset in config.data.datasets]
    optimizer = build_optimizer(
        system, lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=config.training.warmup_epochs,
        total_epochs=config.training.epochs,
    )
    start_epoch = 0
    if args.resume is not None:
        state = load_training_checkpoint(
            args.resume,
            model=system.model,
            criterion=system.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            expected_config_sha256=config_sha256,
            expected_source_schema=source_schema,
            expected_regions=regions,
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
            start_epoch=start_epoch,
        )
    try:
        epoch_count = _epoch_count(config.training.epochs, args.epochs, start_epoch)
    except ValueError as exc:
        parser.error(str(exc))
    rank = dist.get_rank() if distributed else 0

    def save_periodic(epoch: int) -> None:
        completed_epoch = epoch + 1
        if completed_epoch % config.training.save_every != 0:
            return
        if distributed:
            dist.barrier()
        if rank == 0:
            save_training_checkpoint(
                _periodic_checkpoint_path(args.output, completed_epoch),
                model=system.model,
                criterion=system.criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config_sha256=config_sha256,
                git_sha=git_sha,
                source_schema=source_schema,
                regions=regions,
                metrics={"checkpoint_kind": "periodic", "completed_epoch": completed_epoch},
            )
        if distributed:
            dist.barrier()

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
        epoch_end_callback=save_periodic,
    )
    summary.update(
        {
            "world_size": dist.get_world_size() if distributed else 1,
            "spatial_size": args.spatial_size,
            "device_type": device.type,
        }
    )
    if rank == 0:
        save_training_checkpoint(
            args.output,
            model=system.model,
            criterion=system.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=int(summary["end_epoch"]),
            config_sha256=config_sha256,
            git_sha=git_sha,
            source_schema=source_schema,
            regions=regions,
            metrics=summary,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0
