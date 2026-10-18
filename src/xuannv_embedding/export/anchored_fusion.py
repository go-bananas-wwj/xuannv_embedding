"""Label-free paired alignment with an exact public-only fallback."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def align_samples(public: np.ndarray, semantic: np.ndarray, *, ridge: float = 0.01) -> dict:
    public, semantic = np.asarray(public), np.asarray(semantic)
    if (
        public.ndim != 2
        or public.shape != semantic.shape
        or public.shape[1] != 64
        or len(public) < 65
        or not np.isfinite(public).all()
        or not np.isfinite(semantic).all()
        or type(ridge) not in {int, float}
        or not math.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("Paired finite 64D unlabelled samples and positive ridge are required")
    u, v = public.astype(np.float64), semantic.astype(np.float64)
    mu, mv = u.mean(0), v.mean(0)
    x, y = v - mv, u - mu
    gram = x.T @ x
    strength = max(float(np.trace(gram)) / 64, 1e-8) * ridge
    matrix = np.linalg.solve(gram + np.eye(64) * strength, x.T @ y)
    return {
        "matrix": matrix.astype(np.float32),
        "public_mean": mu.astype(np.float32),
        "semantic_mean": mv.astype(np.float32),
    }


def blend_aligned(public, semantic, model, *, strength: float, kernel: int = 1) -> np.ndarray:
    u, v = np.asarray(public, np.float32), np.asarray(semantic, np.float32)
    if (
        u.ndim != 3
        or u.shape != v.shape
        or u.shape[0] != 64
        or not np.isfinite(u).all()
        or not np.isfinite(v).all()
        or type(strength) not in {int, float}
        or not math.isfinite(strength)
        or not 0 <= strength <= 1
        or type(kernel) is not int
        or kernel not in {1, 3}
        or set(model) != {"matrix", "public_mean", "semantic_mean"}
        or np.shape(model["matrix"]) != (64, 64)
        or any(np.shape(model[k]) != (64,) for k in ["public_mean", "semantic_mean"])
        or any(not np.isfinite(value).all() for value in model.values())
    ):
        raise ValueError("Invalid public-anchored residual inputs or recipe")
    if strength == 0:
        return u.copy()
    observed = np.linalg.norm(v, axis=0) > 1e-8
    if not observed.any():
        return u.copy()
    if kernel > 1:
        with torch.inference_mode():
            weights = torch.from_numpy(observed.astype(np.float32))[None, None]
            numerator = F.avg_pool2d(torch.from_numpy(v)[None] * weights, 3, 1, 1)
            denominator = F.avg_pool2d(weights, 3, 1, 1)
            v = (numerator / denominator.clamp_min(1e-8))[0].numpy()
    h, w = u.shape[-2:]
    prediction = (
        (v.reshape(64, -1).T - model["semantic_mean"]) @ model["matrix"] + model["public_mean"]
    ).T.reshape(64, h, w)
    output = u.copy()
    output[:, observed] += strength * (prediction[:, observed] - u[:, observed])
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite anchored fusion")
    return output


def minimax_rank(scores: dict, reference: dict) -> tuple[float, float]:
    if not reference or set(scores) != set(reference):
        raise ValueError("Every registered region must participate in shared selection")
    gains = []
    for region in sorted(reference):
        if set(scores[region]) != {"mlp", "conv3x3"} or set(reference[region]) != set(
            scores[region]
        ):
            raise ValueError("Both readers are required in every region")
        for head in ["mlp", "conv3x3"]:
            baseline = reference[region][head]
            if set(baseline) != {"aef", "public"}:
                raise ValueError("Both AEF and genuine public references are required")
            value = scores[region][head]
            if any(
                type(v) not in {int, float} or not math.isfinite(v) or not 0 <= v <= 1
                for v in [value, *baseline.values()]
            ):
                raise ValueError("Invalid validation score")
            gains.append(value - max(baseline.values()))
    return min(gains), float(np.mean(gains))


def fit_tail(public, semantic, raw, *, semantic_dimensions: int, seed: int) -> dict:
    arrays = [np.asarray(x, np.float32) for x in [public, semantic, raw]]
    if (
        any(
            a.ndim != 2 or a.shape != (len(arrays[0]), d) or not np.isfinite(a).all()
            for a, d in zip(arrays, [64, 64, 95])
        )
        or len(arrays[0]) < 65
        or type(semantic_dimensions) is not int
        or not 1 <= semantic_dimensions < 32
    ):
        raise ValueError("Invalid unlabelled tail-injection samples")
    u, v, r = arrays
    centered = u.astype(np.float64) - u.mean(0, dtype=np.float64)
    eigenvalues, basis = np.linalg.eigh(centered.T @ centered)
    result = {
        "basis": basis[:, :32].astype(np.float32),
        "semantic_dimensions": np.asarray(semantic_dimensions),
        "public_tail_variance_fraction": np.asarray(
            eigenvalues[:32].sum() / max(eigenvalues.sum(), 1e-12)
        ),
    }
    for name, values, dimensions, standardize in [
        ("semantic", v, semantic_dimensions, False),
        ("raw", r, 32 - semantic_dimensions, True),
    ]:
        active = values[np.linalg.norm(values, axis=1) > 1e-8]
        channels = values.shape[1]
        mean, scale = np.zeros(channels, np.float32), np.ones(channels, np.float32)
        pca_mean = np.zeros(channels, np.float32)
        components, deviation = np.zeros((dimensions, channels), np.float32), np.ones(
            dimensions, np.float32
        )
        if len(active):
            if len(active) < dimensions:
                raise ValueError("Insufficient optional-source observations")
            if standardize:
                scaler = StandardScaler().fit(active)
                mean, scale = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
            values = (active - mean) / scale
            pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=seed).fit(
                values
            )
            pca_mean, components = pca.mean_, pca.components_
            deviation = np.sqrt(np.maximum(pca.explained_variance_, 1e-12) * dimensions)
        result.update(
            {
                f"{name}_mean": mean,
                f"{name}_scale": scale,
                f"{name}_pca_mean": pca_mean.astype(np.float32),
                f"{name}_components": components.astype(np.float32),
                f"{name}_deviation": deviation.astype(np.float32),
            }
        )
    return result


def inject_tail(public, semantic, raw, model, *, strength: float):
    arrays = [np.asarray(x, np.float32) for x in [public, semantic, raw]]
    u, v, r = arrays
    if (
        u.ndim != 3
        or u.shape[0] != 64
        or v.shape != u.shape
        or r.shape != (95, *u.shape[-2:])
        or any(not np.isfinite(a).all() for a in arrays)
        or type(strength) not in {int, float}
        or not 0 <= strength <= 1
        or not math.isfinite(strength)
        or np.shape(model["basis"]) != (64, 32)
    ):
        raise ValueError("Invalid public-tail injection inputs")
    if strength == 0 or not (np.any(v) or np.any(r)):
        return u.copy()
    blocks = []
    for name, values in [("semantic", v), ("raw", r)]:
        observed = np.linalg.norm(values, axis=0).ravel() > 1e-8
        matrix = values.reshape(values.shape[0], -1).T
        matrix = (matrix - model[f"{name}_mean"]) / model[f"{name}_scale"]
        projected = (
            (matrix - model[f"{name}_pca_mean"])
            @ model[f"{name}_components"].T
            / model[f"{name}_deviation"]
        )
        projected[~observed] = 0
        blocks.append(projected)
    delta = np.concatenate(blocks, axis=1) @ model["basis"].T
    result = u + strength * delta.T.reshape(u.shape)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite public-tail residual")
    return result.astype(np.float32)
