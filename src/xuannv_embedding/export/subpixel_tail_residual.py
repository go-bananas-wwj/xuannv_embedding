"""Bounded within-cell RGB detail in the low-variance tail of a frozen public space."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from xuannv_embedding.data_process import subpixel_rgb
from xuannv_embedding.export import public_visual_subspace as shared
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.frozen_fusion import source_array, verified


def _centered_rgb(values, coverage):
    x, m = np.asarray(values, np.float32), np.asarray(coverage, np.float32)
    if (
        x.ndim != 3
        or len(x) != 75
        or m.shape != (25, *x.shape[-2:])
        or not np.isfinite(m).all()
        or (m < 0).any()
        or (m > 1).any()
    ):
        raise ValueError("Invalid RGB coverage or subpixel geometry")
    x = x.reshape(3, 25, *x.shape[-2:])
    m = m[None]
    clean = np.where(m > 0, x, 0)
    if not np.isfinite(clean).all() or (clean < 0).any() or (clean > 255).any():
        raise ValueError("Nonfinite or out-of-range observed RGB")
    mean = (clean * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1e-12)
    return np.where(m > 0, clean - mean, 0), m


def highpass(values, coverage, scale):
    centered, m = _centered_rgb(values, coverage)
    scale = np.asarray(scale, np.float32)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale < 1).any():
        raise ValueError("Invalid highpass scale")
    x = np.clip(centered / scale[:, None, None, None], -6, 6)
    mean = (x * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1e-12)
    return np.where(m > 0, x - mean, 0).reshape(75, *x.shape[-2:])


def _signed_basis(basis):
    result = basis.copy()
    pivots = np.abs(result).argmax(0)
    result *= np.where(result[pivots, np.arange(result.shape[1])] < 0, -1, 1)[None]
    return result


def fit_parameters(public_samples, rgb_samples, coverage_samples):
    p = np.asarray(public_samples, np.float32)
    rgb, masks = np.asarray(rgb_samples, np.float32), np.asarray(coverage_samples, np.float32)
    if (
        p.ndim != 2
        or p.shape[1] != 64
        or not np.isfinite(p).all()
        or rgb.ndim != 2
        or rgb.shape[1] != 75
        or masks.shape != (len(rgb), 25)
    ):
        raise ValueError("Invalid public or RGB projection samples")
    mean, scale = np.zeros(64, np.float32), np.ones(64, np.float32)
    public_center, eigenvalues, basis = np.zeros(64, np.float32), np.zeros(64), np.eye(64)
    if len(p):
        mean, scale = p.mean(0, dtype=np.float64).astype(np.float32), p.std(
            0, dtype=np.float64
        ).astype(np.float32)
        scale[scale < 1e-6] = 1
        z = np.clip((p - mean) / scale, -6, 6).astype(np.float64)
        public_center = z.mean(0).astype(np.float32)
        z -= public_center
        eigenvalues, basis = np.linalg.eigh(z.T @ z / max(1, len(z) - 1))
        basis = _signed_basis(basis)
    model = {
        "public_mean": mean,
        "public_scale": scale,
        "public_center": public_center,
        "public_basis": basis.astype(np.float32),
        "public_eigenvalues": np.maximum(eigenvalues, 0).astype(np.float32),
        "hp_scale": np.ones(3, np.float32),
        "private_center": np.zeros(75, np.float32),
        "private_components": np.zeros((16, 75), np.float32),
        "private_std": np.full(16, 0.01, np.float32),
        "private_eigenvalues": np.zeros(16, np.float32),
        "private_explained_variance_ratio": np.zeros(16, np.float32),
    }
    if len(rgb):
        centered, observed = _centered_rgb(rgb.T[:, :, None], masks.T[:, :, None])
        denominator = np.maximum(observed.sum((1, 2, 3)), 1)
        hp_scale = np.sqrt(
            (centered.astype(np.float64) ** 2 * observed).sum((1, 2, 3)) / denominator
        )
        model["hp_scale"] = np.maximum(hp_scale, 1).astype(np.float32)
        x = (
            highpass(rgb.T[:, :, None], masks.T[:, :, None], model["hp_scale"])
            .reshape(75, -1)
            .T.astype(np.float64)
        )
        center = x.mean(0).astype(np.float32)
        x -= center
        variance, vectors = np.linalg.eigh(x.T @ x / max(1, len(x) - 1))
        order = np.argsort(variance)[::-1][:16]
        values = np.maximum(variance[order], 0)
        components = _signed_basis(vectors[:, order]).T
        components[values <= 1e-8] = 0
        model.update(
            private_center=center,
            private_components=components.astype(np.float32),
            private_std=np.sqrt(np.maximum(values, 0.01**2)).astype(np.float32),
            private_eigenvalues=values.astype(np.float32),
            private_explained_variance_ratio=(values / max(variance.sum(), 1e-12)).astype(
                np.float32
            ),
        )
    return model


def transform(public, rgb, coverage, domain, model, *, strength):
    if type(strength) not in {int, float} or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Invalid subpixel residual strength")
    p, m = np.asarray(public, np.float32), np.asarray(coverage, np.float32)
    if (
        p.ndim != 3
        or len(p) != 64
        or not np.isfinite(p).all()
        or np.shape(domain) != (1, *p.shape[-2:])
    ):
        raise ValueError("Public residual geometry differs")
    if np.shape(rgb) != (75, *p.shape[-2:]) or m.shape != (25, *p.shape[-2:]):
        raise ValueError("Subpixel residual geometry differs")
    if (
        not np.isfinite(domain).all()
        or (np.asarray(domain) < 0).any()
        or (np.asarray(domain) > 1).any()
    ):
        raise ValueError("Invalid public observation domain")
    features = highpass(rgb, m, model["hp_scale"]).reshape(75, -1).T
    scores = ((features - model["private_center"]) @ model["private_components"].T) / model[
        "private_std"
    ]
    confidence = m.mean(0) * ((m > 0).sum(0) >= 2) * np.asarray(domain)[0]
    coefficients = strength * np.tanh(scores).T * confidence.reshape(1, -1)
    delta = (model["public_basis"][:, :16] @ coefficients).reshape(p.shape)
    return (p + model["public_scale"][:, None, None] * delta).astype(np.float32)


def validate_specification(spec):
    if (
        not isinstance(spec, dict)
        or set(spec)
        != {"schema", "cache", "seed", "fit_samples", "output", "reserve_gib", "budgets"}
        or spec["schema"] != "subpixel-tail-residual-v1"
        or type(spec["seed"]) is not int
        or not 0 <= spec["seed"] < 2**32
        or type(spec["fit_samples"]) is not int
        or spec["fit_samples"] < 64
        or type(spec["reserve_gib"]) is not int
        or spec["reserve_gib"] < 0
        or not isinstance(spec["output"], str)
        or not Path(spec["output"]).is_absolute()
        or not isinstance(spec["cache"], dict)
        or set(spec["cache"]) != {"path", "sha256"}
        or not isinstance(spec["cache"]["path"], str)
        or not Path(spec["cache"]["path"]).is_absolute()
        or not isinstance(spec["cache"]["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", spec["cache"]["sha256"]) is None
        or not isinstance(spec["budgets"], list)
        or not spec["budgets"]
    ):
        raise ValueError("Invalid self-contained subpixel tail specification")
    names = set()
    for b in spec["budgets"]:
        if (
            not isinstance(b, dict)
            or set(b) != {"name", "strength"}
            or not isinstance(b["name"], str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", b["name"])
            or b["name"] in names
            or type(b["strength"]) not in {int, float}
            or not math.isfinite(b["strength"])
            or not 0 <= b["strength"] <= 1
        ):
            raise ValueError("Invalid common strength specification")
        names.add(b["name"])
    return spec


def specification(path):
    return validate_specification(shared.read_json(path))


class SourceData:
    def __init__(self, spec):
        self.cache = subpixel_rgb.load_manifest(verified(spec["cache"]))
        c = self.cache
        self.index = shared.read_json(verified(c["index"]))
        self.n = len(self.index["records"])
        if c["registered_tiles"] != self.n:
            raise ValueError("Subpixel input index count differs")
        digest = c["index"]["sha256"]
        self.public = source_array(c["public"]["array"], digest, self.n)
        self.rgb = source_array(c["rgb"], digest, self.n)
        self.coverage = source_array(c["coverage"], digest, self.n)
        self.physical = source_array(c["public"]["domain"]["array"], digest, self.n)
        columns = shared.read_json(verified(c["public"]["domain"]["columns"]))["columns"]
        self.available = [i for i, name in enumerate(columns) if name.endswith(":availability")]
        shape = tuple(c["output_shape"])
        if (
            self.public.shape != (self.n, 64, *shape)
            or self.physical.shape != (self.n, len(columns), *shape)
            or not self.available
        ):
            raise ValueError("Subpixel public/physical grid differs")
        self.spec = spec

    def domain(self, position):
        values = np.asarray(self.physical[position, self.available], np.float32)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid public observation availability")
        return (values > 0).any(0, keepdims=True)

    def samples(self):
        public, public_positions = shared.sample_rows(
            [self.public[i] for i in range(self.n)],
            [self.domain(i) for i in range(self.n)],
            limit=self.spec["fit_samples"],
            seed=self.spec["seed"],
        )
        masks = []
        for i in range(self.n):
            m = np.asarray(self.coverage[i], np.float32)
            if not np.isfinite(m).all() or (m < 0).any() or (m > 1).any():
                raise ValueError("Invalid subpixel observation coverage")
            masks.append(((m > 0).sum(0, keepdims=True) >= 2) & self.domain(i))
        rgb, rgb_positions = shared.sample_rows(
            [self.rgb[i] for i in range(self.n)],
            masks,
            limit=self.spec["fit_samples"],
            seed=self.spec["seed"],
        )
        coverage = []
        for i in range(self.n):
            ids = rgb_positions[rgb_positions[:, 0] == i, 1]
            coverage.append(np.asarray(self.coverage[i]).reshape(25, -1)[:, ids].T)
        return (
            public,
            rgb,
            np.concatenate(coverage).astype(np.float32),
            public_positions,
            rgb_positions,
        )


def identity(path):
    return {
        "spec_sha256": sha(path),
        "implementation": {
            "subpixel_tail": sha(__file__),
            "subpixel_rgb": sha(subpixel_rgb.__file__),
            "sampling": sha(shared.__file__),
        },
    }


def load_parameters(root):
    status = shared.read_json(root / "status.json")
    if (
        status.get("state") != "complete"
        or status.get("identity") != shared.read_json(root / "identity.json")
        or sha(root / "model.npz") != status["model_sha256"]
    ):
        raise ValueError("Subpixel parameters identity changed")
    with np.load(root / "model.npz", allow_pickle=False) as a:
        result = {k: a[k] for k in a.files}
    shapes = {
        "public_mean": (64,),
        "public_scale": (64,),
        "public_center": (64,),
        "public_basis": (64, 64),
        "public_eigenvalues": (64,),
        "hp_scale": (3,),
        "private_center": (75,),
        "private_components": (16, 75),
        "private_std": (16,),
        "private_eigenvalues": (16,),
        "private_explained_variance_ratio": (16,),
    }
    if set(result) != set(shapes) or any(
        result[k].shape != v or not np.isfinite(result[k]).all() for k, v in shapes.items()
    ):
        raise ValueError("Invalid subpixel parameters")
    if (
        (result["public_scale"] <= 0).any()
        or (result["hp_scale"] < 1).any()
        or (result["private_std"] <= 0).any()
    ):
        raise ValueError("Invalid subpixel normalization parameters")
    return result


def fit(path):
    spec = specification(path)
    root, ident = Path(spec["output"]), identity(path)
    with shared.ownership(root, ident), threadpool_limits(limits=2):
        if (root / "status.json").exists():
            load_parameters(root)
            return shared.read_json(root / "status.json")
        if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30:
            raise RuntimeError("Subpixel fitting disk reserve")
        data = SourceData(spec)
        public, rgb, masks, pp, rp = data.samples()
        params = fit_parameters(public, rgb, masks)
        with (root / "model.partial.npz").open("wb") as f:
            np.savez(f, **params)
        (root / "model.partial.npz").replace(root / "model.npz")
        np.savez(root / "sample-positions.npz", public=pp, rgb=rp)
        result = {
            "state": "complete",
            "identity": ident,
            "model_sha256": sha(root / "model.npz"),
            "registered_input_tiles": data.n,
            "public_sample_rows": len(public),
            "private_sample_rows": len(rgb),
            "public_observed_tiles": len(np.unique(pp[:, 0])),
            "private_observed_tiles": len(np.unique(rp[:, 0])),
            "sample_positions_sha256": sha(root / "sample-positions.npz"),
            "primary_or_weak_maps_loaded": False,
            "neural_encoder_updated": False,
        }
        dump(root / "status.json", result)
        load_parameters(root)
        return result


def export(path):
    spec = specification(path)
    root, ident = Path(spec["output"]), identity(path)
    with shared.ownership(root, ident), threadpool_limits(limits=2):
        params = load_parameters(root)
        data = SourceData(spec)
        shape = (data.n, 64, *data.public.shape[-2:])
        arrays, results, diagnostics = {}, [], {}
        for b in spec["budgets"]:
            dest = root / "representations" / b["name"]
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "status.json").exists():
                record = shared.read_json(dest / "status.json")
                if (
                    record.get("state") != "complete"
                    or record.get("identity") != ident
                    or record.get("model_sha256") != sha(root / "model.npz")
                    or record.get("index_sha256") != data.cache["index"]["sha256"]
                    or record.get("shape") != list(shape)
                    or sha(dest / "annual.npy") != record["sha256"]
                ):
                    raise ValueError("Cached subpixel export changed")
                results.append(record)
                continue
            if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30 + math.prod(shape) * 2:
                raise RuntimeError("Subpixel export disk reserve")
            arrays[b["name"]] = np.lib.format.open_memmap(
                dest / "annual.partial.npy", mode="w+", dtype=np.float16, shape=shape
            )
            diagnostics[b["name"]] = {
                "max_protected_error_fp32": 0.0,
                "max_protected_error_fp16": 0.0,
                "max_normalized_delta_l2": 0.0,
            }
        for i in range(data.n):
            if not arrays:
                break
            if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30:
                raise RuntimeError("Subpixel export disk reserve")
            p = np.asarray(data.public[i], np.float32)
            for b in spec["budgets"]:
                name = b["name"]
                if name not in arrays:
                    continue
                actual = transform(
                    p, data.rgb[i], data.coverage[i], data.domain(i), params, strength=b["strength"]
                )
                quantized = actual.astype(np.float16)
                if not np.isfinite(quantized).all():
                    raise FloatingPointError("Nonfinite subpixel embedding")
                d = diagnostics[name]
                for value, key in [
                    (actual, "max_protected_error_fp32"),
                    (quantized.astype(np.float32), "max_protected_error_fp16"),
                ]:
                    delta = ((value - p) / params["public_scale"][:, None, None]).reshape(64, -1)
                    error = np.max(np.abs(params["public_basis"][:, 16:].T @ delta))
                    d[key] = max(d[key], float(error))
                    if key.endswith("fp32"):
                        d["max_normalized_delta_l2"] = max(
                            d["max_normalized_delta_l2"], float(np.linalg.norm(delta, axis=0).max())
                        )
                if (
                    d["max_protected_error_fp32"] > 1e-4
                    or d["max_normalized_delta_l2"] > 4 * b["strength"] + 1e-4
                ):
                    raise FloatingPointError("Subpixel projection protection or bound failed")
                arrays[name][i] = quantized
        for name, a in arrays.items():
            a.flush()
            dest = root / "representations" / name
            (dest / "annual.partial.npy").replace(dest / "annual.npy")
            result = {
                "state": "complete",
                "identity": ident,
                "shape": list(shape),
                "index_sha256": data.cache["index"]["sha256"],
                "model_sha256": sha(root / "model.npz"),
                "sha256": sha(dest / "annual.npy"),
                "primary_or_weak_maps_loaded": False,
                "neural_encoder_updated": False,
                "unit_normalized": False,
                **diagnostics[name],
            }
            dump(dest / "status.json", result)
            results.append(result)
        return results


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment subpixel-tail-residual")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["fit", "export"], required=True)
    args = parser.parse_args(argv)
    fit(args.spec) if args.phase == "fit" else export(args.spec)
    return 0
