"""Geographically shared masking and date-conditioned annual self-supervision."""

from __future__ import annotations

import calendar
import math
from collections import defaultdict
from datetime import date

import torch
from torch import nn

from xuannv_embedding.models.annual import masked_resample
from xuannv_embedding.training.losses import batch_uniformity_loss, reconstruction_loss


def prepare_annual_batch(batch: dict, config, *, training: bool, generator=None) -> dict:
    """Hide targets before all encoders; share time/geographic masks across sensors."""
    timestamps = batch["timestamps"]
    batch_size, months = timestamps.shape
    base_size = max(
        (tuple(values.shape[-2:]) for values in batch["source_frames"].values()),
        key=lambda shape: math.prod(shape),
    )
    output_size = (base_size[0] * 2, base_size[1] * 2)
    masking = config.training.input_masking
    visible = torch.ones(batch_size, 1, *base_size)
    hidden_months = torch.zeros(batch_size, months, dtype=torch.bool)
    if training:
        for sample in range(batch_size):
            available = torch.stack([mask[sample] for mask in batch["source_masks"].values()]).any(
                dim=0
            )
            candidates = available.nonzero().flatten()
            if candidates.numel() > 1:
                count = min(
                    masking.max_months_per_sample,
                    max(1, round(candidates.numel() * masking.month_dropout_prob)),
                    candidates.numel() - 1,
                )
                permutation = torch.randperm(candidates.numel(), generator=generator)
                hidden_months[sample, candidates[permutation[:count]]] = True
            if torch.rand((), generator=generator) < masking.spatial_block_prob:
                rows = math.ceil(base_size[0] / masking.spatial_block_size)
                columns = math.ceil(base_size[1] / masking.spatial_block_size)
                count = min(
                    rows * columns - 1, max(1, round(rows * columns * masking.spatial_block_ratio))
                )
                block_keep = torch.ones(rows * columns)
                block_keep[torch.randperm(rows * columns, generator=generator)[:count]] = 0
                expanded = (
                    block_keep.view(rows, columns)
                    .repeat_interleave(masking.spatial_block_size, 0)
                    .repeat_interleave(masking.spatial_block_size, 1)
                )
                visible[sample, 0] = expanded[: base_size[0], : base_size[1]]
    frames, pixel_masks, temporal_masks, targets, target_masks = {}, {}, {}, {}, {}
    for source, values in batch["source_frames"].items():
        if source not in config.model.input_sources:
            continue
        original_mask = batch["source_pixel_masks"][source].bool()
        _, coverage = masked_resample(visible, visible, tuple(values.shape[-2:]))
        keep = (coverage[:, 0] >= 1 - 1e-6)[:, None] & ~hidden_months[:, :, None, None]
        mask = original_mask & keep
        frames[source] = torch.where(mask[:, :, None], values, 0.0)
        pixel_masks[source] = mask
        temporal_masks[source] = mask.flatten(2).any(dim=2)
        hidden = hidden_months[:, :, None, None] | (coverage[:, 0, None] <= 1e-6)
        for name, head in config.model.target_heads.items():
            if head.source == source:
                targets[name] = values
                target_masks[name] = original_mask & (
                    hidden if training else torch.ones_like(hidden)
                )
    observations, highres_targets, requests = [], [], []
    for sample_index, sample in enumerate(batch["highres_observations"]):
        inputs, sample_targets, sample_requests = [], [], []
        for item in sample:
            metadata = item["metadata"]
            source = metadata["source"]
            if source not in config.model.input_sources:
                continue
            acquisition = date.fromisoformat(metadata["date"][:10])
            phase = (
                2
                * math.pi
                * (acquisition.timetuple().tm_yday - 1)
                / (366 if calendar.isleap(acquisition.year) else 365)
            )
            date_features = torch.tensor([math.sin(phase), math.cos(phase)])
            registration = metadata.get("relative_registration", {})
            if registration.get("status") != "measured":
                raise ValueError("Unmeasured highres registration cannot enter dense annual fusion")
            offset = float(registration["offset_m"])
            if not math.isfinite(offset) or not 0 <= offset <= 10:
                raise ValueError(
                    "Highres registration exceeds the released 10 m diagnostic threshold"
                )
            quality = math.exp(-offset / 5.0)
            original_mask = item["mask"].bool()
            _, coverage = masked_resample(
                visible[sample_index : sample_index + 1],
                visible[sample_index : sample_index + 1],
                tuple(item["values"].shape[-2:]),
            )
            month_hidden = bool(hidden_months[sample_index, acquisition.month - 1])
            keep = (coverage[0, 0] >= 1 - 1e-6) & ~torch.tensor(month_hidden)
            input_mask = original_mask & keep
            inputs.append(
                {
                    "source": source,
                    "values": torch.where(input_mask[None], item["values"], 0.0),
                    "mask": input_mask,
                    "date_features": date_features,
                    "quality_weight": quality,
                }
            )
            heldout = (coverage[0, 0] <= 1e-6) | month_hidden
            target_mask = original_mask & (heldout if training else torch.ones_like(heldout))
            values5, support5 = masked_resample(
                item["values"][None], target_mask[None, None].float(), output_size
            )
            sample_targets.append(
                {
                    "source": source,
                    "values": values5[0],
                    "mask": support5[0, 0],
                    "quality_weight": quality,
                }
            )
            sample_requests.append({"source": source, "date_features": date_features})
        observations.append(inputs)
        highres_targets.append(sample_targets)
        requests.append(sample_requests)
    return {
        "source_frames": frames,
        "source_masks": temporal_masks,
        "source_pixel_masks": pixel_masks,
        "timestamps": timestamps,
        "highres_observations": observations,
        "reconstruction_requests": requests,
        "targets": targets,
        "target_masks": target_masks,
        "highres_targets": highres_targets,
        "metadata": batch["metadata"],
        "patch_ids": [item["patch_id"] for item in batch["metadata"]],
    }


class AnnualLoss(nn.Module):
    """Per-source held-out reconstruction plus cross-parent uniformity."""

    def __init__(self, config) -> None:
        super().__init__()
        self.target_cfg = config.model.loss_specs
        self.uniformity_weight = config.training.uniformity_weight
        self.uniformity_temperature = config.training.uniformity_temperature
        self.warmup = config.training.uniformity_warmup_epochs
        self.latent_prediction_weight = config.training.latent_prediction_weight
        self.latent_predictor = (
            nn.Conv2d(config.model.embed_dim, config.model.embed_dim, 1)
            if self.latent_prediction_weight
            else None
        )
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def forward(
        self,
        output,
        targets,
        masks,
        supervised_labels=None,
        supervised_label_masks=None,
        *,
        teacher_output=None,
        highres_targets=None,
    ) -> dict[str, torch.Tensor]:
        zero = output.embedding_map.sum() * 0
        reconstruction = zero
        result = {}
        for name, specification in self.target_cfg.items():
            loss = reconstruction_loss(
                output.reconstructions[name], targets[name], masks[name], specification["loss_type"]
            )
            reconstruction = reconstruction + specification["weight"] * loss
            result[f"recon_{name}"] = loss
        source_losses = defaultdict(list)
        for predictions, sample_targets in zip(
            output.highres_reconstructions, highres_targets or [], strict=True
        ):
            for prediction, target in zip(predictions, sample_targets, strict=True):
                if bool((target["mask"] > 0).any()):
                    loss = reconstruction_loss(
                        prediction[None], target["values"][None], target["mask"][None], "l1"
                    )
                    source_losses[target["source"]].append(loss * target["quality_weight"])
        highres_loss = zero
        if source_losses:
            highres_loss = torch.stack(
                [torch.stack(losses).mean() for losses in source_losses.values()]
            ).mean()
        valid = output.validity_mask.flatten(2).any(dim=2)
        uniformity = batch_uniformity_loss(output.embedding, self.uniformity_temperature, valid)
        weight = self.uniformity_weight * (
            min(1.0, (self.current_epoch + 1) / self.warmup) if self.warmup else 1.0
        )
        latent = zero
        if self.latent_predictor is not None:
            if teacher_output is None:
                raise ValueError("Annual latent prediction requires a teacher")
            student = self.latent_predictor(output.embedding_map[:, 0])
            teacher = teacher_output.embedding_map[:, 0].detach()
            support = teacher_output.validity_mask[:, 0].float()
            student, pooled_support = masked_resample(student, support, (8, 8))
            teacher, _ = masked_resample(teacher, support, (8, 8))
            student = torch.nn.functional.normalize(student.float(), dim=1)
            teacher = torch.nn.functional.normalize(teacher.float(), dim=1)
            errors = 1 - (student * teacher).sum(dim=1, keepdim=True)
            latent = (errors * pooled_support).sum() / pooled_support.sum().clamp(min=1)
        result.update(
            {
                "total": reconstruction
                + highres_loss
                + weight * uniformity
                + self.latent_prediction_weight * latent,
                "recon": reconstruction,
                "highres_recon": highres_loss,
                "uniformity": uniformity,
                "uniformity_weighted": weight * uniformity,
                "latent_prediction": latent,
            }
        )
        return result
