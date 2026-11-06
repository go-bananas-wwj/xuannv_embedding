"""Interval-query attention for independent product timelines."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

TemporalMode = Literal["within_period", "causal_window", "centered_window"]


def interval_observation_mask(
    time_bounds: torch.Tensor,
    available_at: torch.Tensor,
    observation_mask: torch.Tensor,
    output_intervals: torch.Tensor,
    *,
    mode: TemporalMode,
    window_days: int,
) -> torch.Tensor:
    """Return ``[B,M,N]`` validity without inventing point timestamps."""
    if time_bounds.ndim != 3 or time_bounds.shape[-1] != 2:
        raise ValueError("time_bounds 必须为 [B,N,2]")
    if output_intervals.ndim != 3 or output_intervals.shape[-1] != 2:
        raise ValueError("output_intervals 必须为 [B,M,2]")
    start = time_bounds[:, None, :, 0]
    end = time_bounds[:, None, :, 1]
    output_start = output_intervals[:, :, None, 0]
    output_end = output_intervals[:, :, None, 1]
    base = observation_mask[:, None, :].bool()
    if mode == "within_period":
        selected = (start < output_end) & (end > output_start)
    elif mode == "causal_window":
        selected = (
            (available_at[:, None, :] <= output_end)
            & (start < output_end)
            & (end > output_end - float(window_days))
        )
    elif mode == "centered_window":
        observation_center = (start + end) * 0.5
        output_center = (output_start + output_end) * 0.5
        selected = (observation_center - output_center).abs() <= float(window_days)
    else:
        raise ValueError(f"未知 temporal mode: {mode}")
    return base & selected


def limit_observations(
    selected: torch.Tensor,
    observation_times: torch.Tensor,
    output_intervals: torch.Tensor,
    quality: torch.Tensor,
    max_observations: int,
) -> torch.Tensor:
    """Deterministically retain quality-first, time-nearest observations."""
    if max_observations <= 0:
        raise ValueError("max_observations 必须大于 0")
    limited = selected.clone()
    centers = output_intervals.mean(dim=-1)
    for batch_index in range(selected.shape[0]):
        for output_index in range(selected.shape[1]):
            indices = torch.nonzero(selected[batch_index, output_index], as_tuple=False).flatten()
            if indices.numel() <= max_observations:
                continue
            center = float(centers[batch_index, output_index])
            ranked = sorted(
                (int(index) for index in indices),
                key=lambda index: (
                    -float(quality[batch_index, index]),
                    abs(float(observation_times[batch_index, index]) - center),
                    float(observation_times[batch_index, index]),
                    index,
                ),
            )
            keep = set(ranked[:max_observations])
            for index in indices.tolist():
                limited[batch_index, output_index, index] = int(index) in keep
    return limited


class IntervalAttention(nn.Module):
    """Multi-head attention whose values retain their spatial feature maps."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("IntervalAttention dim 必须能被 num_heads 整除")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.key = nn.Linear(dim + 3, dim)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)
        self.output = nn.Conv2d(dim, dim, kernel_size=1)

    @staticmethod
    def _time_descriptor(bounds: torch.Tensor) -> torch.Tensor:
        center = bounds.mean(dim=-1)
        duration = bounds[..., 1] - bounds[..., 0]
        return torch.stack(
            (torch.sin(center / 365.25), torch.cos(center / 365.25), duration / 365.25),
            dim=-1,
        )

    def forward(
        self,
        features: torch.Tensor,
        time_bounds: torch.Tensor,
        output_intervals: torch.Tensor,
        selected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, observations, channels, height, width = features.shape
        outputs = output_intervals.shape[1]
        pooled = features.mean(dim=(-2, -1))
        keys = self.key(torch.cat((pooled, self._time_descriptor(time_bounds)), dim=-1))
        queries = self.query(self._time_descriptor(output_intervals))
        keys = keys.view(batch, observations, self.num_heads, self.head_dim)
        queries = queries.view(batch, outputs, self.num_heads, self.head_dim)
        scores = torch.einsum("bmhd,bnhd->bhmn", queries, keys) / math.sqrt(self.head_dim)
        valid = selected[:, None, :, :]
        scores = scores.masked_fill(~valid, -1.0e4)
        weights = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        values = self.value(features.reshape(batch * observations, channels, height, width))
        values = values.view(batch, observations, self.num_heads, self.head_dim, height, width)
        summary = torch.einsum("bhmn,bnhdxy->bmhdxy", weights, values)
        summary = summary.reshape(batch * outputs, channels, height, width)
        summary = self.output(summary).view(batch, outputs, channels, height, width)
        available = selected.any(dim=-1)
        return summary, available


class ProductGatedFusion(nn.Module):
    """Fuse product summaries while retaining explicit missing-product masks."""

    def __init__(self, product_ids: list[str], dim: int) -> None:
        super().__init__()
        self.product_ids = tuple(product_ids)
        self.gates = nn.ModuleDict(
            {
                product_id: nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
                for product_id in product_ids
            }
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not features:
            raise ValueError("ProductGatedFusion 至少需要一个产品")
        ordered = [product_id for product_id in self.product_ids if product_id in features]
        maps = torch.stack([features[product_id] for product_id in ordered], dim=2)
        logits = torch.stack(
            [
                self.gates[product_id](features[product_id].mean(dim=(-2, -1))).squeeze(-1)
                for product_id in ordered
            ],
            dim=2,
        )
        valid = torch.stack([available[product_id] for product_id in ordered], dim=2)
        logits = logits.masked_fill(~valid, -1.0e4)
        weights = torch.softmax(logits, dim=2) * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        return (maps * weights[:, :, :, None, None, None]).sum(dim=2)
