"""Streaming physical-band moments with bounded, reproducible quantile samples."""

from __future__ import annotations

import numpy as np


class StreamingBandStatistics:
    def __init__(self, bands: int, *, reservoir_size: int = 4096, seed: int = 42):
        if bands <= 0 or reservoir_size <= 0:
            raise ValueError("statistics dimensions must be positive")
        self.count = np.zeros(bands, dtype=np.int64)
        self.mean = np.zeros(bands, dtype=np.float64)
        self.m2 = np.zeros(bands, dtype=np.float64)
        self.minimum = np.full(bands, np.inf)
        self.maximum = np.full(bands, -np.inf)
        self.reservoir_size = reservoir_size
        self.samples = [np.empty(0, dtype=np.float64) for _ in range(bands)]
        self.priorities = [np.empty(0, dtype=np.float64) for _ in range(bands)]
        self.generator = np.random.default_rng(seed)
        self.seed = seed

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        if values.shape != valid.shape or values.shape[0] != len(self.count):
            raise ValueError("statistics input/mask bands or shape disagree")
        for i, (band, mask) in enumerate(zip(values, valid, strict=True)):
            selected = np.asarray(band[mask & np.isfinite(band)], dtype=np.float64)
            n = selected.size
            if not n:
                continue
            average = selected.mean()
            delta = average - self.mean[i]
            total = self.count[i] + n
            self.m2[i] += np.square(selected - average).sum() + delta**2 * self.count[i] * n / total
            self.mean[i] += delta * n / total
            self.count[i] = total
            self.minimum[i] = min(self.minimum[i], selected.min())
            self.maximum[i] = max(self.maximum[i], selected.max())
            keys = self.generator.random(n)
            if len(self.priorities[i]) == self.reservoir_size:
                keep = keys < self.priorities[i].max()
                selected, keys = selected[keep], keys[keep]
            candidates = np.concatenate([self.samples[i], selected])
            priorities = np.concatenate([self.priorities[i], keys])
            if len(candidates) > self.reservoir_size:
                keep = np.argpartition(priorities, self.reservoir_size - 1)[: self.reservoir_size]
                candidates, priorities = candidates[keep], priorities[keep]
            self.samples[i], self.priorities[i] = candidates, priorities

    def finish(self) -> dict:
        if (self.count < 2).any():
            raise ValueError("empty or insufficient valid channel samples")
        std = np.sqrt(self.m2 / self.count)
        if not np.isfinite(std).all() or (std <= 1e-12).any():
            raise ValueError("constant or invalid channel variance")
        return {
            "count": self.count.tolist(),
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "quantile_probabilities": [0.01, 0.05, 0.5, 0.95, 0.99],
            "quantiles": [
                np.quantile(sample, [0.01, 0.05, 0.5, 0.95, 0.99]).tolist()
                for sample in self.samples
            ],
            "quantile_method": "uniform random-priority reservoir per valid pixel",
            "quantile_reservoir_size": self.reservoir_size,
            "seed": self.seed,
            "variance_ddof": 0,
        }
