"""Annual data loading with bounded metadata memory and non-padding DDP sampling."""

from __future__ import annotations

import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader, DistributedSampler

from xuannv_embedding.data.annual_dataset import (
    AnnualObservationDataset,
    collate_annual_observations,
)
from xuannv_embedding.training.annual import prepare_annual_batch


class TargetAnnualDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, reader, tasks):
        self.dataset, self.reader, self.tasks = dataset, reader, tuple(tasks)
        identities = getattr(dataset.records, "identities", None)
        patches = (
            [patch for _, patch in identities]
            if identities is not None
            else [record.patch_id for record in dataset.records]
        )
        if set(patches) - reader.roles["train"]:
            raise ValueError(
                "Annual training manifest contains positions outside the frozen target scope"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raw = self.dataset[index]
        raw["independent_targets"] = self.reader.read(
            raw["patch_id"], raw["year"], tasks=self.tasks
        )
        return raw


class AnnualBatchStream:
    def __init__(
        self, config, loaders, *, max_steps, start_epoch, teacher_enabled, teacher_config=None
    ):
        self.config = config
        self.loaders = loaders
        self.max_steps = max_steps
        self.epoch = start_epoch
        self.teacher_enabled = teacher_enabled
        self.teacher_config = teacher_config or config

    def __iter__(self):
        rank = distributed.get_rank() if distributed.is_initialized() else 0
        generator = torch.Generator().manual_seed(
            self.config.experiment.seed + self.epoch * 1009 + rank
        )
        view_generator = torch.Generator().manual_seed(
            self.config.experiment.seed + self.epoch * 1009 + rank + 1_000_003
        )
        steps = 0
        for loader in self.loaders:
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(self.epoch)
            for raw in loader:
                batch = prepare_annual_batch(raw, self.config, training=True, generator=generator)
                if self.teacher_enabled:
                    batch["teacher_view"] = prepare_annual_batch(
                        raw, self.teacher_config, training=False
                    )
                if self.config.training.view_consistency_weight > 0:
                    batch["consistency_view"] = prepare_annual_batch(
                        raw, self.config, training=True, generator=view_generator
                    )
                yield batch
                steps += 1
                if self.max_steps is not None and steps >= self.max_steps:
                    break
            if self.max_steps is not None and steps >= self.max_steps:
                break
        self.epoch += 1


def annual_dataset(config, dataset_config, *, max_records=None):
    if any(source != canonical for source, canonical in dataset_config.source_map.items()):
        raise ValueError("Annual configuration currently requires explicit identity source_map")
    dataset = AnnualObservationDataset(
        dataset_config.manifest_path,
        indexed=True,
        max_records=max_records,
        max_observations=config.data.highres_max_observations,
    )
    dataset.selected_sources = set(config.model.input_sources)
    for source, specification in config.model.input_sources.items():
        schema = dataset.schemas.get(source)
        if (
            schema is None
            or schema["channels"] != specification.channels
            or schema["role"] != specification.role
        ):
            raise ValueError(f"Annual source schema mismatch: {source}")
        if source not in dataset.statistics:
            raise ValueError(f"Missing train-only annual statistics: {source}")
        if specification.role == "highres":
            pan = source in config.model.annual_pan_sources
            if pan != (schema.get("product_group") == "PAN_2m"):
                raise ValueError(f"Annual PAN branch/schema mismatch: {source}")
    return dataset


def build_annual_stream(
    config,
    *,
    distributed,
    max_records,
    max_steps,
    start_epoch,
    teacher_enabled,
    teacher_config=None,
):
    loaders = []
    reader = None
    target_config = config.training.independent_targets
    if target_config is not None:
        from xuannv_embedding.data.targets import AnnualTargetReader

        reader = AnnualTargetReader(
            target_config.target_root,
            target_config.prepared_root,
            preparation_sha256=target_config.preparation_sha256,
            scope=target_config.scope,
        )
        from xuannv_embedding.data.target_contract import LEGACY_TARGET_SCHEMA

        if reader.preparation.get("target_schema", LEGACY_TARGET_SCHEMA) != target_config.schema:
            raise ValueError("Configured target schema differs from the pinned preparation")
        if (reader.dem_mean, reader.dem_std) != (target_config.dem_mean, target_config.dem_std):
            raise ValueError(
                "Configured target normalization differs from the pinned training statistics"
            )
        if (
            not target_config.engineering_only
            and not target_config.research_max_updates
            and not reader.preparation["formal_training_released"]
        ):
            raise ValueError(
                "Formal target-supervised training has not been released by preparation"
            )
    for dataset_config in config.data.datasets:
        if ".train." not in dataset_config.manifest_path.name:
            raise ValueError("Training requires explicitly named annual train manifests")
        if dataset_config.sampling_weight != 1.0:
            raise ValueError("Annual epoch traversal requires sampling_weight=1")
        dataset = annual_dataset(config, dataset_config, max_records=max_records)
        if reader is not None:
            dataset = TargetAnnualDataset(dataset, reader, target_config.weights)
        sampler = (
            DistributedSampler(dataset, shuffle=True, drop_last=True, seed=config.experiment.seed)
            if distributed
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=config.data.num_workers,
            persistent_workers=config.data.num_workers > 0,
            collate_fn=collate_annual_observations,
            pin_memory=True,
            drop_last=distributed,
        )
        if not len(loader):
            raise ValueError("Annual dataset is too small for the configured batch/world size")
        loaders.append(loader)
    return AnnualBatchStream(
        config,
        loaders,
        max_steps=max_steps,
        start_epoch=start_epoch,
        teacher_enabled=teacher_enabled,
        teacher_config=teacher_config,
    )
