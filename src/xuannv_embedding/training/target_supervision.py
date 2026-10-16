"""Independent targets read only after the public embedding, at their supervision grids."""

from __future__ import annotations

import hashlib
import math

import torch
import torch.distributed as distributed
from torch import nn
from torch.nn import functional as F


class TargetSupervision(nn.Module):
    """Small isolated readouts; target tensors are accepted by loss(), never forward()."""

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
        if not weights or set(weights) - {"dem_elevation", "worldcover"}:
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
                # Seed CPU only: torch.manual_seed would also perturb accelerator RNG streams.
                torch.random.default_generator.manual_seed(isolated)
                self.heads[name] = nn.Conv2d(
                    embed_dim, 11 if name == "worldcover" else 1, kernel_size=1, device="cpu"
                )

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        if embedding.ndim != 4 or any(size % 8 for size in embedding.shape[-2:]):
            raise ValueError("Public 5m embedding dimensions must divide by8")
        return {
            name: head(F.avg_pool2d(embedding, 2 if name == "worldcover" else 8))
            for name, head in self.heads.items()
        }

    def loss(self, predictions: dict, targets: dict, *, ramp: float = 1.0) -> dict:
        if not math.isfinite(ramp) or not 0 <= ramp <= 1:
            raise ValueError("Auxiliary ramp must be between0and1")
        if set(predictions) != set(self.heads):
            raise ValueError("All configured target heads must participate on every rank")
        predictions = {name: predictions[name] for name in self.heads}
        numerators, counts = [], []
        for name, prediction in predictions.items():
            if name not in targets:
                raise ValueError(
                    f"Missing task tensor (use a false mask for missing labels): {name}"
                )
            values, valid = targets[name]["values"], targets[name]["valid"].bool()
            expected = (prediction.shape[0], *prediction.shape[-2:])
            if values.shape != expected or valid.shape != expected:
                raise ValueError(f"Target/prediction supervision grid mismatch: {name}")
            if name == "worldcover":
                safe_values = torch.where(valid, values.long(), -100)
                if bool((valid & ((safe_values < 0) | (safe_values >= 11))).any()):
                    raise ValueError("Invalid mapped WorldCover label")
                numerator = F.cross_entropy(
                    prediction.float(), safe_values, ignore_index=-100, reduction="sum"
                )
            else:
                if not bool(torch.isfinite(values[valid]).all()):
                    raise ValueError("Nonfinite supervised DSM values")
                safe_values = torch.where(valid, values.float(), self.dem_mean)
                normalized = (safe_values - self.dem_mean) / self.dem_std
                error = (prediction[:, 0].float() - normalized).abs()
                numerator = torch.where(valid, error, 0).sum()
            numerators.append(numerator)
            counts.append(valid.sum().to(dtype=torch.float32))
        local = torch.stack(numerators)
        totals = torch.stack([torch.stack(counts), local.detach().float()])
        world = distributed.get_world_size() if distributed.is_initialized() else 1
        if world > 1:
            distributed.all_reduce(totals)
        denominators = totals[0].clamp(min=1)
        losses = local * world / denominators
        result = {
            "total": sum(
                self.weights[name] * ramp * loss for name, loss in zip(predictions, losses)
            )
        }
        for i, name in enumerate(predictions):
            result[f"target_{name}"] = totals[1, i] / denominators[i]
            result[f"target_{name}_valid_pixels"] = totals[0, i]
        if "dem_elevation" in predictions:
            result["target_dem_elevation_MAE_m"] = result["target_dem_elevation"] * self.dem_std
        return result
