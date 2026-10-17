"""Independent targets read only after the public embedding, at their supervision grids."""

from __future__ import annotations

import hashlib
import math

import torch
import torch.distributed as distributed
from torch import nn
from torch.nn import functional as F

from xuannv_embedding.data.target_contract import TARGET_TASKS


def validate_research_budget(target, *, steps, epochs, initialize_from, resume):
    """Bound explicitly requested research without changing formal data release status."""
    if target is None or not target.research_max_updates:
        return
    if target.engineering_only:
        raise ValueError("Engineering and bounded research modes are mutually exclusive")
    if steps < 1 or epochs < 1 or steps * epochs > target.research_max_updates:
        raise ValueError(
            "Bounded target research requires explicit steps within research_max_updates"
        )
    if initialize_from is None or resume is not None:
        raise ValueError(
            "Bounded target research requires a fresh explicit initialization, not resume"
        )


def research_budget_record(target, *, steps, completed_epochs):
    if target is None or not target.research_max_updates:
        return None
    return {
        "limit": target.research_max_updates,
        "step_limit_per_epoch": steps,
        "completed_epochs": completed_epochs,
        "completed_batch_upper_bound": steps * completed_epochs,
        "scope": "Conservative cumulative batch bound, including prior resumed epochs.",
    }


def validate_research_resume(target, state, *, steps, epochs):
    """Resume only new checkpoints carrying the original cumulative budget ledger."""
    if target is None or not target.research_max_updates:
        return
    ledger = state["metrics"].get("research_budget")
    initial = state["metrics"].get("initialization_identity")
    completed = int(state["epoch"]) + 1
    expected = research_budget_record(target, steps=steps, completed_epochs=completed)
    if not initial or not initial.get("checkpoint_sha256") or ledger != expected:
        raise ValueError("Bounded resume lacks a matching cumulative budget and initialization")
    if steps < 1 or epochs < 1 or steps * (completed + epochs) > target.research_max_updates:
        raise ValueError("Bounded resume exceeds its original cumulative research budget")


class TargetSupervision(nn.Module):
    """Isolated readouts; target tensors enter loss(), never the encoder or forward()."""

    def __init__(
        self,
        embed_dim: int,
        weights: dict[str, float],
        *,
        dem_mean: float,
        dem_std: float,
        seed: int = 42,
    ):
        super().__init__()
        if not weights or set(weights) - set(TARGET_TASKS):
            raise ValueError("Unsupported independent target task")
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
            raise ValueError("Target weights must be finite and positive")
        if not math.isfinite(dem_mean) or not math.isfinite(dem_std) or dem_std <= 0:
            raise ValueError("Target DSM normalization must be finite with positive std")
        self.weights = dict(sorted(weights.items()))
        self.register_buffer("dem_mean", torch.tensor(dem_mean))
        self.register_buffer("dem_std", torch.tensor(dem_std))
        self.heads = nn.ModuleDict()
        for name in self.weights:
            isolated = int.from_bytes(
                hashlib.sha256(f"{seed}/{name}".encode()).digest()[:8], "little"
            )
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(isolated)
                self.heads[name] = nn.Conv2d(embed_dim, TARGET_TASKS[name][0], 1, device="cpu")

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        factors = [8, *[TARGET_TASKS[name][1] for name in self.heads]]
        if embedding.ndim != 4 or any(size % max(factors) for size in embedding.shape[-2:]):
            raise ValueError("Public embedding does not divide the configured supervision grid")
        return {
            name: head(F.avg_pool2d(embedding, TARGET_TASKS[name][1]))
            for name, head in self.heads.items()
        }

    def _terms(self, name, prediction, target):
        channels, _, kind = TARGET_TASKS[name]
        values, valid = target["values"].detach(), target["valid"].bool()
        grid = (prediction.shape[0], *prediction.shape[-2:])
        valid_shape = (grid[0], channels, *grid[1:]) if kind == "binary" else grid
        value_shape = tuple(prediction.shape) if kind in {"binary", "soft_categorical"} else grid
        if tuple(values.shape) != value_shape or tuple(valid.shape) != valid_shape:
            raise ValueError(f"Target/prediction supervision grid mismatch: {name}")
        weight = target.get("weight", valid).detach().float()
        if weight.shape != valid.shape or not bool(torch.isfinite(weight[valid]).all()):
            raise ValueError("Invalid target confidence shape or values")
        if bool(((weight[valid] < 0) | (weight[valid] > 1)).any()):
            raise ValueError("Target confidence must be between0and1")
        weight = torch.where(valid, weight, 0)
        if kind == "categorical":
            safe = torch.where(valid, values.long(), -100)
            if bool((valid & ((safe < 0) | (safe >= channels))).any()):
                raise ValueError(f"Invalid mapped categorical label: {name}")
            error = F.cross_entropy(prediction.float(), safe, ignore_index=-100, reduction="none")
        elif kind == "soft_categorical":
            safe = torch.where(valid[:, None], values.float(), 0)
            if not bool(torch.isfinite(safe).all()) or bool(((safe < 0) | (safe > 1)).any()):
                raise ValueError("Invalid soft categorical probabilities")
            if not bool(
                torch.allclose(safe.sum(1)[valid], torch.ones_like(weight[valid]), atol=1e-4)
            ):
                raise ValueError("Valid class fractions must sum to1")
            logits = torch.where(valid[:, None], prediction.float(), 0)
            error = -(safe * F.log_softmax(logits, 1)).sum(1)
        elif kind == "binary":
            safe = torch.where(valid, values.float(), 0)
            if not bool(torch.isfinite(safe).all()) or bool(((safe < 0) | (safe > 1)).any()):
                raise ValueError("Invalid OSM coverage target")
            logits = torch.where(valid, prediction.float(), 0)
            positive, negative = weight * safe, weight * (1 - safe)
            axes = (0, 2, 3)
            counts = torch.stack([positive.sum(axes), negative.sum(axes)], 1).flatten()
            numerators = torch.stack(
                [
                    (F.softplus(-logits) * positive).sum(axes),
                    (F.softplus(logits) * negative).sum(axes),
                ],
                1,
            ).flatten()
            return numerators, counts, valid.sum().float()
        else:
            if not bool(torch.isfinite(values[valid]).all()):
                raise ValueError(f"Nonfinite supervised continuous values: {name}")
            safe = torch.where(valid, values.float(), 0)
            if name == "dem_elevation":
                safe = (safe - self.dem_mean) / self.dem_std
            elif name == "dem_slope":
                if bool(((safe[valid] < 0) | (safe[valid] > 90)).any()):
                    raise ValueError("Slope must be in degrees between0and90")
                safe = safe / 90
            error = (prediction[:, 0].float() - safe).abs()
        error = torch.where(valid, error, 0)
        return (error * weight).sum().reshape(1), weight.sum().reshape(1), valid.sum().float()

    def loss(self, predictions: dict, targets: dict, *, ramp: float = 1.0) -> dict:
        if not math.isfinite(ramp) or not 0 <= ramp <= 1:
            raise ValueError("Auxiliary ramp must be between0and1")
        if set(predictions) != set(self.heads):
            raise ValueError("All configured target heads must participate on every rank")
        numbers, counts, raw_counts, sizes = [], [], [], []
        for name in self.heads:
            if name not in targets:
                raise ValueError(
                    f"Missing task tensor (use a false mask for missing labels): {name}"
                )
            num, count, raw = self._terms(name, predictions[name], targets[name])
            numbers.append(num)
            counts.append(count)
            raw_counts.append(raw)
            sizes.append(num.numel())
        local, count = torch.cat(numbers), torch.cat(counts)
        length = local.numel()
        totals = torch.cat([count.detach(), local.detach(), torch.stack(raw_counts).detach()])
        world = distributed.get_world_size() if distributed.is_initialized() else 1
        if world > 1:
            distributed.all_reduce(totals)
        global_counts, global_numbers = totals[:length], totals[length : 2 * length]
        result, objective, offset = {}, [], 0
        for i, name in enumerate(self.heads):
            stop = offset + sizes[i]
            denominator = global_counts[offset:stop].clamp(min=1e-12)
            factors = torch.ones_like(denominator)
            if TARGET_TASKS[name][2] == "binary":
                active = (global_counts[offset:stop] > 0).view(-1, 2)
                factors = (
                    active.float()
                    / active.sum(1).clamp(min=1)[:, None]
                    / active.any(1).sum().clamp(min=1)
                ).flatten()
            objective.append(
                self.weights[name]
                * ramp
                * (local[offset:stop] * world / denominator * factors).sum()
            )
            result[f"target_{name}"] = (global_numbers[offset:stop] / denominator * factors).sum()
            result[f"target_{name}_valid_pixels"] = totals[2 * length + i]
            result[f"target_{name}_weighted_support"] = global_counts[offset:stop].sum()
            if TARGET_TASKS[name][2] == "binary":
                result[f"target_{name}_positive_mass"] = global_counts[offset:stop:2].sum()
                result[f"target_{name}_negative_mass"] = global_counts[offset + 1 : stop : 2].sum()
            offset = stop
        result["total"] = sum(objective)
        if "dem_elevation" in self.heads:
            result["target_dem_elevation_MAE_m"] = result["target_dem_elevation"] * self.dem_std
        if "dem_slope" in self.heads:
            result["target_dem_slope_MAE_degrees"] = result["target_dem_slope"] * 90
        return result
