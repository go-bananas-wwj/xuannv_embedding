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


class AnnualBatchStream:
    def __init__(self, config, loaders, *, max_steps, start_epoch, teacher_enabled):
        self.config = config
        self.loaders = loaders
        self.max_steps = max_steps
        self.epoch = start_epoch
        self.teacher_enabled = teacher_enabled

    def __iter__(self):
        rank = distributed.get_rank() if distributed.is_initialized() else 0
        generator = torch.Generator().manual_seed(
            self.config.experiment.seed + self.epoch * 1009 + rank
        )
        steps = 0
        for loader in self.loaders:
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(self.epoch)
            for raw in loader:
                batch = prepare_annual_batch(raw, self.config, training=True, generator=generator)
                if self.teacher_enabled:
                    batch["teacher_view"] = prepare_annual_batch(raw, self.config, training=False)
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
    config, *, distributed, max_records, max_steps, start_epoch, teacher_enabled
):
    loaders = []
    for dataset_config in config.data.datasets:
        if ".train." not in dataset_config.manifest_path.name:
            raise ValueError("Training requires explicitly named annual train manifests")
        if dataset_config.sampling_weight != 1.0:
            raise ValueError("Annual epoch traversal requires sampling_weight=1")
        dataset = annual_dataset(config, dataset_config, max_records=max_records)
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
    )
