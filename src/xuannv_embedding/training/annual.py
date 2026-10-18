"""Geographically shared masking and date-conditioned annual self-supervision."""

from __future__ import annotations

import calendar
import math
from collections import defaultdict
from datetime import date

import torch
import torch.nn.functional as F
from torch import distributed, nn

from xuannv_embedding.models.annual import masked_resample, visibility_coverage
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
    validity_masks = {}
    for source, values in batch["source_frames"].items():
        if source not in config.model.input_sources:
            continue
        original_mask = batch["source_pixel_masks"][source].bool()
        validity_masks[source] = original_mask
        coverage = visibility_coverage(
            visible, tuple(values.shape[-2:]), exact=config.model.annual_exact_mask_coverage
        )
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
            coverage = visibility_coverage(
                visible[sample_index : sample_index + 1],
                tuple(item["values"].shape[-2:]),
                exact=config.model.annual_exact_mask_coverage,
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
            # 默认只监督被遮住的高分像素；打开 highres_visible_targets 后可见像素也作为目标，
            # 要求 64 维瓶颈保留高分细节，而不仅是从低分推断缺失的高分。
            supervise_all = not training or config.training.highres_visible_targets
            target_mask = original_mask & (torch.ones_like(heldout) if supervise_all else heldout)
            values5, support5 = masked_resample(
                item["values"][None], target_mask[None, None].float(), output_size
            )
            target = {
                "source": source,
                "values": values5[0],
                "mask": support5[0, 0],
                "quality_weight": quality,
            }
            if training and config.training.highres_visible_weight is not None:
                target["partitions"] = {}
                # Partition actual encoder visibility, including any partially
                # covered native pixels conservatively removed from the input.
                for name, partition in (("heldout", ~keep), ("visible", keep)):
                    values_part, support_part = masked_resample(
                        item["values"][None],
                        (original_mask & partition)[None, None].float(),
                        output_size,
                    )
                    target["partitions"][name] = {
                        "values": values_part[0],
                        "mask": support_part[0, 0],
                    }
            sample_targets.append(target)
            sample_requests.append({"source": source, "date_features": date_features})
        observations.append(inputs)
        highres_targets.append(sample_targets)
        requests.append(sample_requests)
    return {
        **(
            {"independent_targets": batch["independent_targets"]}
            if "independent_targets" in batch
            else {}
        ),
        "source_frames": frames,
        "source_masks": temporal_masks,
        "source_pixel_masks": pixel_masks,
        **(
            {"source_validity_masks": validity_masks} if config.model.annual_context_pooling else {}
        ),
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
        self.feature_anchor_weight = config.training.annual_feature_anchor_weight
        self.feature_anchor_mode = config.training.annual_feature_anchor_mode
        if not isinstance(self.feature_anchor_mode, str) or self.feature_anchor_mode not in {
            "native",
            "highpass_10m",
            "weighted_10m",
        }:
            raise ValueError(
                "Annual feature anchor mode must be native, highpass_10m or weighted_10m"
            )
        self.latent_pool_factors = config.training.latent_prediction_pool_factors
        self.view_consistency_weight = config.training.view_consistency_weight
        self.view_pool_factors = config.training.view_consistency_pool_factors
        self.view_predictor = (
            nn.Conv2d(config.model.embed_dim, config.model.embed_dim, 1)
            if self.view_consistency_weight
            else None
        )
        self.highres_loss_weight = config.training.highres_loss_weight
        self.highres_visible_weight = config.training.highres_visible_weight
        self.latent_predictor = (
            nn.Conv2d(config.model.embed_dim, config.model.embed_dim, 1)
            if self.latent_prediction_weight
            else None
        )
        self.current_epoch = 0
        self.target_supervision = None
        self.target_ramp_batches = 0
        if config.training.independent_targets is not None:
            from xuannv_embedding.training.target_supervision import TargetSupervision

            target = config.training.independent_targets
            self.target_supervision = TargetSupervision(
                config.model.embed_dim,
                target.weights,
                dem_mean=target.dem_mean,
                dem_std=target.dem_std,
                seed=config.experiment.seed,
            )
            self.target_ramp_batches = target.ramp_batches
            self.register_buffer("target_batches", torch.tensor(0, dtype=torch.long))

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
        consistency_output=None,
        highres_targets=None,
        independent_targets=None,
        anchor_output=None,
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
        partition_losses = {name: defaultdict(list) for name in ("heldout", "visible")}
        for predictions, sample_targets in zip(
            output.highres_reconstructions, highres_targets or [], strict=True
        ):
            for prediction, target in zip(predictions, sample_targets, strict=True):
                if "partitions" in target:
                    if self.highres_visible_weight is None:
                        raise ValueError("Partitioned highres targets require an explicit weight")
                    numerator, denominator = zero, zero
                    for name, weight in (
                        ("heldout", 1.0),
                        ("visible", self.highres_visible_weight),
                    ):
                        part = target["partitions"][name]
                        count = part["mask"].sum()
                        loss = reconstruction_loss(
                            prediction[None], part["values"][None], part["mask"][None], "l1"
                        )
                        numerator = numerator + loss * count * weight
                        denominator = denominator + count * weight
                        if bool(count > 0):
                            partition_losses[name][target["source"]].append(
                                loss * target["quality_weight"]
                            )
                    if bool(denominator > 0):
                        source_losses[target["source"]].append(
                            numerator / denominator.clamp(min=1e-8) * target["quality_weight"]
                        )
                elif bool((target["mask"] > 0).any()):
                    loss = reconstruction_loss(
                        prediction[None], target["values"][None], target["mask"][None], "l1"
                    )
                    source_losses[target["source"]].append(loss * target["quality_weight"])
        highres_loss = zero
        if source_losses:
            highres_loss = torch.stack(
                [torch.stack(losses).mean() for losses in source_losses.values()]
            ).mean()
        for name, sources in partition_losses.items():
            result[f"highres_{name}_recon"] = (
                torch.stack([torch.stack(values).mean() for values in sources.values()]).mean()
                if sources
                else zero
            )
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
            support = (teacher_output.validity_mask[:, 0] & output.validity_mask[:, 0]).float()
            latent = multiscale_latent_loss(student, teacher, support, self.latent_pool_factors)
        consistency = zero
        if self.view_predictor is not None:
            if consistency_output is None:
                raise ValueError("Annual multi-view consistency requires an EMA target")
            support = (consistency_output.validity_mask[:, 0] & output.validity_mask[:, 0]).float()
            consistency = multiscale_latent_loss(
                self.view_predictor(output.embedding_map[:, 0]),
                consistency_output.embedding_map[:, 0],
                support,
                self.view_pool_factors,
            )
        result.update(
            {
                "total": reconstruction
                + self.highres_loss_weight * highres_loss
                + weight * uniformity
                + self.latent_prediction_weight * latent
                + self.view_consistency_weight * consistency,
                "recon": reconstruction,
                "highres_recon": highres_loss,
                "uniformity": uniformity,
                "uniformity_weighted": weight * uniformity,
                "latent_prediction": latent,
                "view_consistency": consistency,
            }
        )
        if self.feature_anchor_weight > 0:
            if anchor_output is None or teacher_output is None:
                raise ValueError(
                    "Annual feature anchor requires clean student and frozen reference"
                )
            support = anchor_output.validity_mask[:, 0] & teacher_output.validity_mask[:, 0]
            if self.feature_anchor_mode == "native":
                anchor_loss = native_feature_anchor_loss
            elif self.feature_anchor_mode == "highpass_10m":
                anchor_loss = highpass_feature_anchor_loss
            else:
                anchor_loss = weighted_feature_anchor_loss
            anchor = anchor_loss(
                anchor_output.embedding_map[:, 0], teacher_output.embedding_map[:, 0], support
            )
            weighted_anchor = self.feature_anchor_weight * anchor
            result["total"] = result["total"] + weighted_anchor
            result["feature_anchor"] = anchor
            result["feature_anchor_weighted"] = weighted_anchor
        if self.target_supervision is not None:
            if independent_targets is None:
                raise ValueError("Configured independent targets are missing from annual batch")
            ramp = (
                min(1.0, int(self.target_batches) / self.target_ramp_batches)
                if self.target_ramp_batches
                else 1.0
            )
            auxiliary = self.target_supervision.loss(
                self.target_supervision(output.embedding_map[:, 0]),
                independent_targets,
                ramp=ramp,
            )
            result["total"] = result["total"] + auxiliary.pop("total")
            result.update(auxiliary)
            result["target_ramp"] = zero.detach() + ramp
            if self.training:
                self.target_batches.add_(1)
        return result


def native_feature_anchor_loss(student, reference, support):
    """Preserve deterministic native-grid directions without a trainable predictor.

    Directions already come from the sole vMF bottleneck; no further projection,
    pooling or normalization is applied. Invalid observations contribute nothing.
    """
    if student.shape != reference.shape or student.ndim != 4:
        raise ValueError("Native anchor feature maps must have the same BCHW shape")
    if support.shape != (student.shape[0], 1, *student.shape[-2:]):
        raise ValueError("Native anchor validity must have shape B1HW")
    mask = support.float()
    error = torch.where(mask > 0, student.float() - reference.detach().float(), 0.0)
    numerator = (0.5 * error.square().sum(1, keepdim=True) * mask).sum()
    count = mask.sum()
    if distributed.is_available() and distributed.is_initialized():
        totals = torch.stack((numerator.detach(), count.detach()))
        distributed.all_reduce(totals)
        denominator = totals[1].clamp(min=1)
        # DDP averages gradients; scale the local graph by world size while
        # reporting the same true global valid-pixel mean on every rank.
        proxy = numerator * distributed.get_world_size() / denominator
        mean = totals[0] / denominator
        return proxy + (mean - proxy.detach())
    return numerator / count.clamp(min=1)


def highpass_feature_anchor_loss(student, reference, support):
    """Retain within-10m detail while leaving each valid block mean unconstrained.

    The feature grid is native 5m. Both views use the same support-weighted 2x2
    projection, so whole feature error decomposes into block-mean and residual
    errors. This loss keeps only the residual term with no extra normalization.
    Partial edge blocks and missing pixels use only their actual valid support.
    """
    if student.shape != reference.shape or student.ndim != 4:
        raise ValueError("Highpass anchor feature maps must have the same BCHW shape")
    if support.shape != (student.shape[0], 1, *student.shape[-2:]):
        raise ValueError("Highpass anchor validity must have shape B1HW")
    mask = support.float()
    count = F.avg_pool2d(mask, 2, stride=2, ceil_mode=True, count_include_pad=False)

    def residual(values):
        clean = torch.where(mask > 0, values.float(), 0.0)
        pooled = F.avg_pool2d(
            clean * mask, 2, stride=2, ceil_mode=True, count_include_pad=False
        ) / count.clamp(min=1e-6)
        mean = pooled.repeat_interleave(2, -2).repeat_interleave(2, -1)
        return clean - mean[..., : values.shape[-2], : values.shape[-1]]

    return native_feature_anchor_loss(residual(student), residual(reference.detach()), support)


def weighted_feature_anchor_loss(student, reference, support):
    """Keep the fine penalty and add one tenth of its orthogonal block-mean term.

    On the same valid support, native error = block-mean error + highpass error.
    The fixed composition 0.1 * native + 0.9 * highpass therefore assigns weights
    0.1 and 1 to those two terms, respectively. It adds no normalization or model
    parameters. Both constituent means use the existing global DDP pixel count;
    the reference is detached in each. The coefficient is fixed by this mode.
    """
    whole = native_feature_anchor_loss(student, reference, support)
    fine = highpass_feature_anchor_loss(student, reference, support)
    return 0.1 * whole + 0.9 * fine


def multiscale_latent_loss(student, teacher, support, pool_factors=None):
    """Compare coarse supported features; stop gradients at the target only.

    Pool factors 2/4/8 correspond to 10/20/40 m on the public 5 m grid.
    Omitting the factors retains the historical 8 x 8 target grid.
    """
    if student.shape != teacher.shape:
        raise ValueError("Annual teacher and student map shapes must agree")
    height, width = student.shape[-2:]
    sizes = (
        [(8, 8)]
        if pool_factors is None
        else [(max(1, height // factor), max(1, width // factor)) for factor in pool_factors]
    )
    losses = []
    for size in dict.fromkeys(sizes):
        pooled_student, pooled_support = masked_resample(student, support, size)
        pooled_teacher, _ = masked_resample(teacher.detach(), support, size)
        pooled_student = torch.nn.functional.normalize(pooled_student.float(), dim=1)
        pooled_teacher = torch.nn.functional.normalize(pooled_teacher.float(), dim=1)
        errors = 1 - (pooled_student * pooled_teacher).sum(dim=1, keepdim=True)
        losses.append((errors * pooled_support).sum() / pooled_support.sum().clamp(min=1))
    return torch.stack(losses).mean()
