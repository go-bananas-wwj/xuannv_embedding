"""P10C 训练运行时与发布 smoke 入口。"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Subset

from xuannv_embedding.config import Config, V2Config
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.checkpoint import (
    capture_rng_state,
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
    train_v2_accumulation_steps,
    train_v2_steps,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_v2_archives(config: V2Config, distributed: bool) -> None:
    from xuannv_embedding.data.local_archives import verify_archive_lock

    rank = dist.get_rank() if distributed else 0
    result: list[str | None] = [None]
    if rank == 0:
        try:
            verify_archive_lock(
                config.paths.data_root / "locks" / "local_archive_sha256.jsonl",
                expected_count=72,
            )
            cache_path = (
                config.paths.data_root / "observations" / "dense_2020_2021" / "smoke_652.zarr"
            )
            if cache_path.is_dir():
                from xuannv_embedding.data_process.v2_zarr_cache import (
                    verify_smoke_zarr_cache,
                )

                verify_smoke_zarr_cache(cache_path, full=True)
        except ValueError as exc:
            result[0] = str(exc)
    if distributed:
        dist.broadcast_object_list(result, src=0)
    if result[0] is not None:
        raise ValueError(f"V2 训练前 archive lock 校验失败: {result[0]}")


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


def _v2_subset_loader(dataset, start: int, stop: int, batch_size: int = 1) -> DataLoader:
    from xuannv_embedding.data.v2_dataset import collate_v2

    return DataLoader(
        Subset(dataset, range(start, stop)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_v2,
    )


def _move_nested(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_nested(child, device) for key, child in value.items()}
    return value


def _validate_missing_dense_products(
    system: V2TrainingSystem,
    raw_batch: dict[str, Any],
    products: Sequence[str],
    device: torch.device,
) -> list[str]:
    verified = []
    system.train()
    for product_id in products:
        candidate = copy.deepcopy(raw_batch)
        inputs = candidate["model_inputs"]
        inputs["source_frames"][product_id].zero_()
        inputs["source_pixel_masks"][product_id].zero_()
        inputs["source_observation_masks"][product_id].zero_()
        candidate["targets"][product_id].zero_()
        candidate["target_masks"][product_id].zero_()
        system.zero_grad(set_to_none=True)
        result = system(_move_nested(candidate, device))
        if not bool(torch.isfinite(result["total"]).item()):
            raise FloatingPointError(f"缺失 {product_id} 时 loss 非有限值")
        result["total"].backward()
        if not any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
            for parameter in system.parameters()
        ):
            raise RuntimeError(f"缺失 {product_id} 时没有有限梯度")
        verified.append(product_id)
    system.zero_grad(set_to_none=True)
    return verified


def _run_v2_smoke(
    args: argparse.Namespace,
    config: V2Config,
    profile,
    device: torch.device,
) -> int:
    from xuannv_embedding.data.v2_dataset import V2LocalZipDataset, collate_v2
    from xuannv_embedding.export.v2_sharded import export_v2_sharded, model_state_sha256
    from xuannv_embedding.training.validation_profiles import (
        assert_macro_disjoint,
        build_profile_model,
        build_v2_criterion,
        data_manifest_sha256,
        v2_product_schema,
        v2_temporal_contract,
    )

    if args.resume is not None:
        raise ValueError("smoke 自带中断恢复对照，不接受外部 --resume")
    registry_path = _v2_registry(config, "smoke")
    split_counts = assert_macro_disjoint(registry_path)
    dataset = V2LocalZipDataset(
        config,
        registry_path,
        spatial_size=profile.spatial_size,
        max_records=profile.records,
        output_selection="random_single",
        random_seed=42,
        context_days=config.temporal.dense_lookback_days,
        allow_incomplete_statistics=True,
    )
    if len(dataset) != 652:
        raise RuntimeError(f"smoke registry 必须为 652 条，实际 {len(dataset)}")
    torch.manual_seed(42)
    model = build_profile_model(config, profile)
    criterion = build_v2_criterion(config)
    system = V2TrainingSystem(model, criterion)
    optimizer = torch.optim.AdamW(
        system.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    amp = config.training.amp and not args.no_amp and device.type != "cpu"
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    manifest_sha = data_manifest_sha256(config.paths.data_root)
    product_schema = v2_product_schema(config)
    temporal_contract = v2_temporal_contract(config)
    git_sha = _git_sha()
    boundary_step = profile.steps // 2
    boundary_path = args.output.with_name(
        f"{args.output.stem}.step-{boundary_step:04d}{args.output.suffix or '.pt'}"
    )
    first = train_v2_steps(
        system,
        _v2_subset_loader(dataset, 0, boundary_step),
        optimizer,
        device=device,
        max_steps=boundary_step,
        amp=amp,
        scheduler=scheduler,
    )
    save_v2_training_checkpoint(
        boundary_path,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        step=boundary_step,
        config_sha256=config_sha,
        git_sha=git_sha,
        data_manifest_sha256=manifest_sha,
        product_schema=product_schema,
        temporal_contract=temporal_contract,
        metrics=first,
    )
    next_loader = _v2_subset_loader(dataset, boundary_step, boundary_step + 1)
    uninterrupted = train_v2_steps(
        system,
        next_loader,
        optimizer,
        device=device,
        max_steps=1,
        amp=amp,
        scheduler=scheduler,
    )
    expected_probe = next(model.parameters()).detach().float().cpu().clone()
    del system, model, criterion, optimizer, scheduler
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()

    restored_model = build_profile_model(config, profile)
    restored_criterion = build_v2_criterion(config)
    restored = V2TrainingSystem(restored_model, restored_criterion)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    load_v2_training_checkpoint(
        boundary_path,
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
    resumed = train_v2_steps(
        restored,
        _v2_subset_loader(dataset, boundary_step, boundary_step + 1),
        restored_optimizer,
        device=device,
        max_steps=1,
        amp=amp,
        scheduler=restored_scheduler,
    )
    loss_close = math.isclose(
        uninterrupted["loss"], resumed["loss"], rel_tol=1.0e-4, abs_tol=1.0e-3
    )
    parameter_close = torch.allclose(
        expected_probe,
        next(restored_model.parameters()).detach().float().cpu(),
        rtol=1.0e-4,
        atol=1.0e-5,
    )
    if not loss_close or not parameter_close:
        raise RuntimeError(
            "checkpoint 恢复后的下一 step 与未中断运行不一致: "
            f"loss={uninterrupted['loss']}/{resumed['loss']}, parameter={parameter_close}"
        )
    remaining = profile.steps - boundary_step - 1
    tail = train_v2_steps(
        restored,
        _v2_subset_loader(dataset, boundary_step + 1, profile.steps),
        restored_optimizer,
        device=device,
        max_steps=remaining,
        amp=amp,
        scheduler=restored_scheduler,
    )
    missing_verified = _validate_missing_dense_products(
        restored,
        next(iter(_v2_subset_loader(dataset, 0, 1))),
        dataset.dense_products,
        device,
    )
    all_losses = first["losses"] + resumed["losses"] + tail["losses"]
    summary = {
        "profile": "smoke",
        "records": len(dataset),
        "steps": profile.steps,
        "loss": sum(all_losses) / len(all_losses),
        "first_loss": all_losses[0],
        "last_loss": all_losses[-1],
        "checkpoint_restore_verified": True,
        "checkpoint_loss_delta": abs(uninterrupted["loss"] - resumed["loss"]),
        "checkpoint_parameter_close": parameter_close,
        "missing_products_verified": missing_verified,
        "split_counts": split_counts,
        "network_remote_pixels": config.network_policy.allow_remote_pixels,
        "data_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "git_sha": git_sha,
    }
    save_v2_training_checkpoint(
        args.output,
        model=restored_model,
        criterion=restored_criterion,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        step=profile.steps,
        config_sha256=config_sha,
        git_sha=git_sha,
        data_manifest_sha256=manifest_sha,
        product_schema=product_schema,
        temporal_contract=temporal_contract,
        metrics=summary,
    )
    export_dataset = V2LocalZipDataset(
        config,
        registry_path,
        spatial_size=profile.spatial_size,
        max_records=profile.records,
        output_selection="fixed",
        fixed_output_months=((2020, 12), (2021, 12)),
        context_days=config.temporal.dense_lookback_days,
        include_targets=False,
        allow_incomplete_statistics=True,
    )
    export_loader = DataLoader(
        export_dataset,
        batch_size=profile.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_v2,
    )
    product_version = f"xuannv-v2-smoke-{git_sha[:12]}"
    catalog = export_v2_sharded(
        restored_model,
        export_loader,
        config.paths.data_root / "products",
        device=device,
        product_version=product_version,
        data_manifest_sha256=manifest_sha,
        config_sha256=config_sha,
        git_sha=git_sha,
        checkpoint_sha256=_sha256_file(args.output),
        model_state_sha256=model_state_sha256(restored_model),
        run_id="smoke",
    )
    import pyarrow.parquet as pq

    catalog_rows = pq.read_metadata(catalog).num_rows
    if catalog_rows != len(export_dataset) * 2:
        raise RuntimeError(f"V2 导出 catalog 行数错误: {catalog_rows}")
    summary["catalog"] = str(catalog)
    summary["catalog_rows"] = catalog_rows
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


class _CyclingLoader:
    def __init__(
        self,
        loader: DataLoader,
        sampler: DistributedSampler,
        *,
        start_micro_batches: int = 0,
    ) -> None:
        self.loader = loader
        self.sampler = sampler
        self.start_micro_batches = start_micro_batches

    def __iter__(self):
        batches_per_epoch = len(self.loader)
        epoch, offset = divmod(self.start_micro_batches, batches_per_epoch)
        while True:
            self.sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(self.loader):
                if batch_index >= offset:
                    yield batch
            offset = 0
            epoch += 1


def _run_v2_npu_smoke(
    args: argparse.Namespace,
    config: V2Config,
    profile,
    device: torch.device,
    local_rank: int,
) -> int:
    import pyarrow.parquet as pq
    from torch.nn.parallel import DistributedDataParallel

    from xuannv_embedding.data.v2_dataset import V2LocalZipDataset, collate_v2
    from xuannv_embedding.export.v2_sharded import export_v2_sharded, model_state_sha256
    from xuannv_embedding.training.validation_profiles import (
        assert_macro_disjoint,
        build_profile_model,
        build_v2_criterion,
        data_manifest_sha256,
        v2_product_schema,
        v2_temporal_contract,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8 or device.type != "npu":
        raise ValueError(f"npu-smoke 必须使用 8×NPU，实际 world_size={world_size}, device={device}")
    effective_batch = world_size * profile.batch_size * config.training.gradient_accumulation_steps
    if effective_batch != 64:
        raise ValueError(f"npu-smoke 等效全局 batch 必须为 64，实际 {effective_batch}")
    registry_path = _v2_registry(config, "npu-smoke")
    split_counts = assert_macro_disjoint(registry_path)
    cache_path = config.paths.data_root / "observations" / "dense_2020_2021" / "smoke_652.zarr"
    active_cache = cache_path if cache_path.is_dir() else None
    input_backend = "zarr" if active_cache is not None else "zip"
    dataset = V2LocalZipDataset(
        config,
        registry_path,
        spatial_size=profile.spatial_size,
        max_records=profile.records,
        output_selection="random_single",
        random_seed=42,
        context_days=config.temporal.dense_lookback_days,
        zarr_cache_path=active_cache,
        allow_incomplete_statistics=True,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=profile.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_v2,
    )
    torch.manual_seed(42)
    model = build_profile_model(config, profile)
    criterion = build_v2_criterion(config)
    system = V2TrainingSystem(model, criterion).to(device)
    distributed_system = DistributedDataParallel(
        system,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    optimizer = torch.optim.AdamW(
        distributed_system.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    manifest_sha = data_manifest_sha256(config.paths.data_root)
    product_schema = v2_product_schema(config)
    temporal_contract = v2_temporal_contract(config)
    git_sha = _git_sha()
    boundary_step = profile.steps // 2
    boundary_path = args.output.with_name(
        f"{args.output.stem}.step-{boundary_step:04d}{args.output.suffix or '.pt'}"
    )
    torch.npu.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    start_micro_batches = boundary_step * config.training.gradient_accumulation_steps
    if args.npu_phase == "first":
        first_stream = iter(_CyclingLoader(loader, sampler))
        first = train_v2_accumulation_steps(
            distributed_system,
            first_stream,
            optimizer,
            device=device,
            optimizer_steps=boundary_step,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            amp=config.training.amp and not args.no_amp,
            scheduler=scheduler,
        )
        probe_stream = iter(
            _CyclingLoader(loader, sampler, start_micro_batches=start_micro_batches)
        )
        probe_batch = next(probe_stream)
        distributed_system.eval()
        with torch.inference_mode():
            resume_probe_loss = float(
                distributed_system(_move_nested(probe_batch, device))["total"].float().item()
            )
        distributed_system.train()
        rank_metrics: list[dict[str, Any] | None] = [None] * world_size
        rank_rng_states: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(rank_metrics, first)
        dist.all_gather_object(rank_rng_states, capture_rng_state())
        probe_losses: list[float | None] = [None] * world_size
        dist.all_gather_object(probe_losses, resume_probe_loss)
        dist.barrier()
        checkpoint_start = time.perf_counter()
        if rank == 0:
            save_v2_training_checkpoint(
                boundary_path,
                model=system.model,
                criterion=system.criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                step=boundary_step,
                config_sha256=config_sha,
                git_sha=git_sha,
                data_manifest_sha256=manifest_sha,
                product_schema=product_schema,
                temporal_contract=temporal_contract,
                metrics={
                    "phase": "first",
                    "rank_metrics": rank_metrics,
                    "resume_probe_losses": probe_losses,
                },
                sampler_state={"micro_batches_per_rank": start_micro_batches},
                rank_rng_states=[state for state in rank_rng_states if state is not None],
            )
        dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "profile": "npu-smoke",
                        "phase": "first",
                        "checkpoint": str(boundary_path),
                        "checkpoint_seconds": time.perf_counter() - checkpoint_start,
                        "process_exit_required": True,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        dist.destroy_process_group()
        return 0
    if args.npu_phase != "resume":
        raise ValueError("npu-smoke 必须由 first、resume 两个独立 torchrun 阶段执行")
    checkpoint_start = time.perf_counter()
    state = load_v2_training_checkpoint(
        boundary_path,
        model=system.model,
        criterion=system.criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_config_sha256=config_sha,
        expected_data_manifest_sha256=manifest_sha,
        expected_product_schema=product_schema,
        expected_temporal_contract=temporal_contract,
        device=device,
        rng_rank=rank,
    )
    if int(state["step"]) != boundary_step:
        raise RuntimeError(f"npu-smoke 恢复 step 错误: {state['step']}")
    start_micro_batches = int(state["sampler_state"]["micro_batches_per_rank"])
    first = state["metrics"]["rank_metrics"][rank]
    stream = iter(_CyclingLoader(loader, sampler, start_micro_batches=start_micro_batches))
    first_resumed_batch = next(stream)
    distributed_system.eval()
    with torch.inference_mode():
        resumed_probe_loss = float(
            distributed_system(_move_nested(first_resumed_batch, device))["total"].float().item()
        )
    distributed_system.train()
    expected_probe_loss = float(state["metrics"]["resume_probe_losses"][rank])
    if not math.isclose(resumed_probe_loss, expected_probe_loss, rel_tol=1.0e-4, abs_tol=1.0e-3):
        raise RuntimeError(
            f"跨进程恢复首 batch loss 不一致: {resumed_probe_loss}/{expected_probe_loss}"
        )
    checkpoint_seconds = time.perf_counter() - checkpoint_start
    second = train_v2_accumulation_steps(
        distributed_system,
        itertools.chain((first_resumed_batch,), stream),
        optimizer,
        device=device,
        optimizer_steps=profile.steps - boundary_step,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        amp=config.training.amp and not args.no_amp,
        scheduler=scheduler,
    )
    dist.barrier()
    wall_seconds = time.perf_counter() - wall_start
    probe = next(system.model.parameters()).detach().float().sum()
    minimum = probe.clone()
    maximum = probe.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not bool(torch.isclose(minimum, maximum, rtol=1.0e-5, atol=1.0e-6).item()):
        raise RuntimeError(f"DDP 参数不同步: min={minimum.item()}, max={maximum.item()}")
    local = torch.tensor(
        [
            profile.steps,
            first["micro_batches"] + second["micro_batches"],
            first["samples"] + second["samples"],
            first["loss"] * boundary_step + second["loss"] * (profile.steps - boundary_step),
            first["data_seconds"] + second["data_seconds"],
            first["compute_seconds"] + second["compute_seconds"],
            float(torch.npu.max_memory_allocated(device)),
            wall_seconds,
            checkpoint_seconds,
            abs(resumed_probe_loss - expected_probe_loss),
        ],
        dtype=torch.float32,
        device=device,
    )
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    matrix = torch.stack(gathered).cpu()
    if not bool((matrix[:, :3] == matrix[0, :3]).all()):
        raise RuntimeError(f"DDP rank step/样本计数不一致: {matrix[:, :3].tolist()}")
    summary = {
        "profile": "npu-smoke",
        "world_size": world_size,
        "device_name": torch.npu.get_device_name(device),
        "optimizer_steps": profile.steps,
        "micro_batches_per_rank": int(matrix[0, 1]),
        "samples_per_rank": int(matrix[0, 2]),
        "effective_global_batch": effective_batch,
        "global_samples": int(matrix[:, 2].sum()),
        "loss": float(matrix[:, 3].sum() / (world_size * profile.steps)),
        "data_seconds_sum": float(matrix[:, 4].sum()),
        "compute_seconds_sum": float(matrix[:, 5].sum()),
        "data_wait_fraction": float(matrix[:, 4].sum() / (matrix[:, 4].sum() + matrix[:, 5].sum())),
        "wall_seconds_max": float(matrix[:, 7].max()),
        "throughput_samples_per_second": float(matrix[:, 2].sum() / matrix[:, 7].max()),
        "peak_memory_bytes_max": int(matrix[:, 6].max()),
        "checkpoint_seconds_max": float(matrix[:, 8].max()),
        "checkpoint_restore_step": boundary_step,
        "process_restart_verified": True,
        "resume_probe_loss_delta_max": float(matrix[:, 9].max()),
        "rank_sync_verified": True,
        "split_counts": split_counts,
        "config_sha256": config_sha,
        "data_manifest_sha256": manifest_sha,
        "git_sha": git_sha,
        "network_remote_pixels": config.network_policy.allow_remote_pixels,
        "input_backend": input_backend,
        "zarr_metadata_sha256": (
            hashlib.sha256((active_cache / ".zmetadata").read_bytes()).hexdigest()
            if active_cache is not None
            else None
        ),
        "zarr_repack_required": input_backend == "zip"
        and float(matrix[:, 4].sum() / (matrix[:, 4].sum() + matrix[:, 5].sum())) > 0.1,
    }
    if rank == 0:
        save_v2_training_checkpoint(
            args.output,
            model=system.model,
            criterion=system.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            step=profile.steps,
            config_sha256=config_sha,
            git_sha=git_sha,
            data_manifest_sha256=manifest_sha,
            product_schema=product_schema,
            temporal_contract=temporal_contract,
            metrics=summary,
            sampler_state={
                "micro_batches_per_rank": profile.steps
                * config.training.gradient_accumulation_steps
            },
        )
    dist.barrier()
    free_memory = torch.tensor(
        [float(torch.npu.mem_get_info(device)[0])], dtype=torch.float32, device=device
    )
    free_by_rank = [torch.zeros_like(free_memory) for _ in range(world_size)]
    dist.all_gather(free_by_rank, free_memory)
    export_rank = int(torch.stack(free_by_rank).cpu().argmax().item())
    product_version = f"xuannv-v2-npu-smoke-{git_sha[:12]}"
    catalog = config.paths.data_root / "products" / product_version / "catalog.parquet"
    export_metrics = torch.zeros(2, dtype=torch.float32, device=device)
    if rank == export_rank:
        export_start = time.perf_counter()
        export_dataset = V2LocalZipDataset(
            config,
            registry_path,
            spatial_size=profile.spatial_size,
            max_records=profile.records,
            output_selection="fixed",
            fixed_output_months=((2020, 12), (2021, 12)),
            context_days=config.temporal.dense_lookback_days,
            include_targets=False,
            zarr_cache_path=active_cache,
            allow_incomplete_statistics=True,
        )
        export_loader = DataLoader(
            export_dataset,
            batch_size=profile.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_v2,
        )
        written_catalog = export_v2_sharded(
            system.model,
            export_loader,
            config.paths.data_root / "products",
            device=device,
            product_version=product_version,
            data_manifest_sha256=manifest_sha,
            config_sha256=config_sha,
            git_sha=git_sha,
            checkpoint_sha256=_sha256_file(args.output),
            model_state_sha256=model_state_sha256(system.model),
            run_id="npu-smoke",
        )
        export_metrics[0] = time.perf_counter() - export_start
        export_metrics[1] = pq.read_metadata(written_catalog).num_rows
    dist.broadcast(export_metrics, src=export_rank)
    if rank == 0:
        summary["export_seconds"] = float(export_metrics[0].cpu())
        summary["export_rank"] = export_rank
        summary["catalog"] = str(catalog)
        summary["catalog_rows"] = int(export_metrics[1].cpu())
        report_path = config.paths.data_root / "runs" / "npu-smoke" / "metrics.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return 0


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
    device, distributed, local_rank = _setup_device(args.device)
    _verify_v2_archives(config, distributed)
    if args.profile == "smoke":
        if distributed:
            raise ValueError("smoke 是单卡门禁；8 卡请使用 npu-smoke")
        return _run_v2_smoke(args, config, profile, device)
    if args.profile == "npu-smoke":
        if not distributed:
            raise ValueError("npu-smoke 必须由 torchrun 启动")
        return _run_v2_npu_smoke(args, config, profile, device, local_rank)
    if args.profile != "mini-real":
        raise ValueError(f"未知 V2 profile: {args.profile}")
    if distributed:
        raise ValueError("mini-real 是单进程恢复门禁，不接受分布式环境")
    torch.manual_seed(42)
    dataset = V2LocalZipDataset(
        config,
        _v2_registry(config, args.profile),
        spatial_size=profile.spatial_size,
        max_records=profile.records,
        january_pair_only=True,
        allow_incomplete_statistics=True,
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
    parser.add_argument("--npu-phase", choices=("first", "resume"))
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
