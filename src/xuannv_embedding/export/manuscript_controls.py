"""Frozen image-guidance and simple-concatenation attribution controls."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.export import rgb_guided_refinement as legacy
from xuannv_embedding.export import viewwise_guided_refinement as viewwise
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.cli import _setup_device
from xuannv_embedding.training.frozen_fusion import verified


def binary_coverage(value):
    if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError("Invalid observed coverage")
    return (value > 0).to(value.dtype)


def pca_projection(samples):
    if samples.ndim != 2 or min(samples.shape) < 64 or not np.isfinite(samples).all():
        raise ValueError("Insufficient finite unlabeled PCA samples")
    mean, scale = samples.mean(0), samples.std(0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    z = ((samples - mean) / scale).astype(np.float64)
    _, vectors = np.linalg.eigh(z.T @ z / len(z))
    vectors = vectors[:, -64:][:, ::-1]
    signs = np.sign(vectors[np.abs(vectors).argmax(0), np.arange(64)])
    return mean, scale, (vectors * np.where(signs == 0, 1, signs)).astype("float32")


def export(path, device):
    spec = json.loads(path.read_text())
    if (
        set(spec) != {"schema", "guide", "features", "kind", "output", "reserve_gib"}
        or spec["schema"] != "manuscript-frozen-control-v1"
        or spec["kind"] not in {"guided", "binary_coverage", "simple_concat"}
        or spec["reserve_gib"] < 100
        or not Path(spec["output"]).is_absolute()
    ):
        raise ValueError("Invalid self-contained frozen control")
    guide = copy.deepcopy(viewwise.specification(verified(spec["guide"])))
    guide["features"] = spec["features"]
    guide["canonical_export"] = None
    data = viewwise.SourceData(guide)
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "spec_sha256": sha(path),
        "implementation": sha(__file__),
        "viewwise": sha(viewwise.__file__),
        "legacy": sha(legacy.__file__),
    }
    previous_hash = None
    if (root / "status.json").exists():
        status = json.loads((root / "status.json").read_text())
        if status.get("identity") != identity:
            raise ValueError("Changed frozen-control source")
        previous_hash = status["features"]["sha256"]
        if Path(status["features"]["path"]).exists():
            verified(status["features"])
            return status
    if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30 + math.prod(data.base.shape) * 2:
        raise RuntimeError("Control allocation violates disk reserve")
    dev, _, _ = _setup_device(device)
    model = None
    if spec["kind"] != "simple_concat":
        model = legacy.load_anyup(
            guide["upsampler"]["source_bundle"], guide["upsampler"]["checkpoint"]
        ).to(dev)

    def descriptor(i):
        return (
            np.concatenate(
                [
                    np.asarray(data.base[i], np.float32),
                    np.asarray(data.rgb[i], np.float32) / 255,
                    np.asarray(data.coverage[i], np.float32),
                ],
                0,
            )
            .reshape(164, -1)
            .T
        )

    if spec["kind"] == "simple_concat":
        rng = np.random.default_rng(41)
        samples = np.concatenate(
            [descriptor(i)[rng.choice(128 * 128, 256, replace=False)] for i in range(data.n)]
        )[:65536]
        mean, scale, vectors = pca_projection(samples)
        np.savez(root / "projection.npz", mean=mean, scale=scale, components=vectors)
    output = np.lib.format.open_memmap(
        root / "annual.partial.npy", mode="w+", dtype="float16", shape=data.base.shape
    )
    with torch.inference_mode():
        for i in range(data.n):
            if spec["kind"] == "simple_concat":
                result = (((descriptor(i) - mean) / scale) @ vectors).T.reshape(64, 128, 128)
            else:
                base = torch.tensor(np.array(data.base[i : i + 1], np.float32), device=dev)
                if spec["kind"] == "guided":
                    result = viewwise.refine_views(
                        base,
                        data.batch_views([i]),
                        model,
                        guide["normalization"],
                        guide["q_chunk_size"],
                        0.75,
                        data.view_count,
                    )
                else:
                    delta = torch.zeros_like(base)
                    for image, mask in data.batch_views([i]):
                        image, coverage = legacy.reconstruct_guide(
                            image, mask, guide["normalization"]
                        )
                        refined = model(
                            image.to(dev),
                            base,
                            output_size=base.shape[-2:],
                            q_chunk_size=guide["q_chunk_size"],
                        )
                        c = binary_coverage(coverage.to(dev))
                        if data.view_count == 1:
                            result = legacy.blend(base, refined, c, 0.75)
                        else:
                            delta += c * (refined - base) / data.view_count
                    if data.view_count != 1:
                        result = base + 0.75 * delta
                result = result[0].float().cpu().numpy()
            if not np.isfinite(result).all():
                raise FloatingPointError("Nonfinite frozen control")
            output[i] = result.astype("float16")
    output.flush()
    (root / "annual.partial.npy").replace(root / "annual.npy")
    status = {
        "state": "complete",
        "identity": identity,
        "shape": list(data.base.shape),
        "features": {"path": str(root / "annual.npy"), "sha256": sha(root / "annual.npy")},
        "primary_maps_loaded": False,
        "kind": spec["kind"],
        "actual_views": data.view_count,
        "neural_weights_updated": False,
    }
    if previous_hash is not None and status["features"]["sha256"] != previous_hash:
        raise ValueError("Rebuilt control differs from the frozen calibration representation")
    dump(root / "status.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    export(args.spec, args.device)
    return 0
