"""Gradient directions and embedding drift for non-persistent adaptation diagnostics."""

from __future__ import annotations

from itertools import combinations

import torch


def gradient_geometry(losses: dict, groups: dict) -> dict:
    """Keep zero/unused gradients explicit; undefined cosine is never treated as zero."""
    named = [(group, name, p) for group, values in groups.items() for name, p in values.items()]
    if (
        not named
        or any(not values for values in groups.values())
        or any(not p.requires_grad for _, _, p in named)
    ):
        raise ValueError("gradient geometry requires nonempty trainable parameter groups")
    if len({id(p) for _, _, p in named}) != len(named):
        raise ValueError("parameter groups must be disjoint")
    vectors = {group: {} for group in groups}
    for objective, loss in losses.items():
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError("nonfinite diagnostic loss")
        gradients = torch.autograd.grad(
            loss, [p for _, _, p in named], retain_graph=True, allow_unused=True
        )
        pieces = {group: [] for group in groups}
        for (group, _, parameter), gradient in zip(named, gradients, strict=True):
            value = (
                torch.zeros(parameter.numel(), dtype=torch.float64)
                if gradient is None
                else gradient.detach().cpu().double().flatten()
            )
            if not torch.isfinite(value).all():
                raise FloatingPointError("nonfinite diagnostic gradient")
            pieces[group].append(value)
        for group in groups:
            vectors[group][objective] = torch.cat(pieces[group])
    report = {}
    for group, objectives in vectors.items():
        norms = {name: float(value.norm()) for name, value in objectives.items()}
        cosines = {}
        for left, right in combinations(objectives, 2):
            denominator = norms[left] * norms[right]
            cosines[f"{left}|{right}"] = (
                float((objectives[left] @ objectives[right] / denominator).clamp(-1, 1))
                if denominator > 0
                else None
            )
        report[group] = {"norms": norms, "cosines": cosines}
    return report


def embedding_geometry(embedding: torch.Tensor, base: torch.Tensor, *, stride: int = 8) -> dict:
    if stride < 1 or embedding.ndim != 5 or embedding.shape != base.shape:
        raise ValueError("embedding geometry requires equal [B,T,D,H,W] and positive stride")
    if not torch.isfinite(embedding).all() or not torch.isfinite(base).all():
        raise ValueError("embedding geometry requires finite inputs")
    difference = embedding.detach().cpu().double() - base.detach().cpu().double()
    denominator = base.detach().cpu().double().norm()
    features = embedding.detach()[..., ::stride, ::stride].permute(0, 1, 3, 4, 2)
    features = features.reshape(-1, embedding.shape[2]).cpu().double()
    centered = features - features.mean(0)
    covariance = centered.T @ centered / max(1, len(features) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    mass = float(eigenvalues.sum())
    if mass > 0:
        probabilities = eigenvalues[eigenvalues > 0] / mass
        rank = float((-(probabilities * probabilities.log()).sum()).exp())
    else:
        rank = 0.0
    return {
        "relative_l2": float(difference.norm() / denominator) if denominator > 0 else None,
        "absolute_rms": float(difference.square().mean().sqrt()),
        "effective_rank": rank,
        "sampled_vectors": len(features),
        "sampling_stride": stride,
        "definition": "entropy rank of centered feature covariance; descriptive, not sample count",
    }
