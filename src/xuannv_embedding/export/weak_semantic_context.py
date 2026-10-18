"""Recover auxiliary class probabilities without replacing observation embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from xuannv_embedding.downstream.worldcover_reference import CLASSES
from xuannv_embedding.export.context import dump, sha


def _verified(ref):
    if set(ref) != {"path", "sha256"} or sha(ref["path"]) != ref["sha256"]:
        raise ValueError("Weak context source identity changed")
    return Path(ref["path"])


def _inputs(path):
    spec = json.loads(Path(path).read_text())
    fields = {
        "training",
        "model",
        "embedding",
        "embedding_identity",
        "embedding_status",
        "index",
    }
    if set(spec) != fields | {"schema", "output"} or spec["schema"] != "weak-semantic-context-v1":
        raise ValueError("Invalid weak semantic context specification")
    paths = {name: _verified(spec[name]) for name in fields}
    root = Path(spec["output"]).resolve()
    if any(p.resolve().is_relative_to(root) for p in paths.values()):
        raise ValueError("Weak context output would contain source artifacts")
    training = json.loads(paths["training"].read_text())
    index = json.loads(paths["index"].read_text())
    saved = torch.load(str(paths["model"]), weights_only=True, map_location="cpu")
    header = json.loads(paths["embedding_identity"].read_text())
    status = json.loads(paths["embedding_status"].read_text())
    summary = saved["training_summary"]
    if (
        saved["identity"].get("spec_sha256") != spec["training"]["sha256"]
        or header != {"training": saved["identity"], "model_sha256": spec["model"]["sha256"]}
        or training["index"] != spec["index"]
        or status.get("state") != "complete"
        or status.get("sha256") != spec["embedding"]["sha256"]
        or status.get("index_sha256") != spec["index"]["sha256"]
        or status.get("evaluation_maps_used") is not False
        or saved["step"] != training["steps"]
        or summary.get("state") != "complete"
    ):
        raise ValueError("Weak context model, embedding or training identity differs")
    uses_local = training["inputs"]["semantic"] is not None
    if summary.get("high_resolution_used") is not uses_local:
        raise ValueError("Weak context checkpoint input role differs")
    array = np.load(paths["embedding"], mmap_mode="r")
    if (
        array.ndim != 4
        or array.shape[1] != 64
        or len(array) != len(index["records"])
        or list(array.shape) != status["shape"]
        or not np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError("Weak context embedding shape differs from its geographic index")
    weight, bias = [
        saved["model"]["worldcover_head." + name].detach().float().numpy()
        for name in ["weight", "bias"]
    ]
    if (
        weight.shape != (12, 64, 1, 1)
        or bias.shape != (12,)
        or not np.isfinite(weight).all()
        or not np.isfinite(bias).all()
    ):
        raise ValueError("Weak context requires the twelve-class auxiliary WorldCover head")
    return spec, array, weight[1:, :, 0, 0], bias[1:], uses_local


def export(path):
    """Read archived observations and head weights; never open map targets.

    Probabilities are conditional on the eleven foreground WorldCover classes.
    Zero embedding vectors remain zero. These are predictions, not reference labels.
    """
    spec, array, weight, bias, uses_local = _inputs(path)
    root = Path(spec["output"])
    identity = {"spec_sha256": sha(path), "implementation": sha(__file__)}
    header, status = root / "identity.json", root / "status.json"
    if root.exists() and any(root.iterdir()) and not header.exists():
        raise ValueError("Unowned weak context output directory")
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError("Weak context output identity changed")
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "features.npy"
    if dest.exists() or status.exists():
        previous = json.loads(status.read_text()) if status.exists() else {}
        if (
            previous.get("state") == "complete"
            and dest.exists()
            and sha(dest) == previous.get("features_sha256")
            and sha(root / "metadata.json") == previous.get("metadata_sha256")
            and sha(root / "index.json") == spec["index"]["sha256"]
        ):
            return dest
        raise ValueError("Completed weak context features changed")
    dump(header, identity)
    pending = root / "features.partial.npy"
    result = np.lib.format.open_memmap(
        pending, mode="w+", dtype=np.float32, shape=(len(array), 11, *array.shape[-2:])
    )
    with threadpool_limits(limits=2):
        for i in range(len(array)):
            value = np.asarray(array[i], np.float32)
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite weak context input embedding")
            logits = weight @ value.reshape(64, -1) + bias[:, None]
            if not np.isfinite(logits).all():
                raise FloatingPointError("Nonfinite weak semantic context logits")
            logits -= logits.max(0, keepdims=True)
            probability = np.exp(logits)
            probability /= probability.sum(0, keepdims=True)
            probability *= (np.linalg.norm(value.reshape(64, -1), axis=0) > 1e-8)[None]
            result[i] = probability.reshape(11, *value.shape[-2:])
    result.flush()
    pending.replace(dest)
    (root / "index.json").write_bytes(Path(spec["index"]["path"]).read_bytes())
    dump(
        root / "metadata.json",
        {
            "index": spec["index"],
            "source_embedding": spec["embedding"],
            "model": spec["model"],
            "training": spec["training"],
            "classes": [{"official_code": k, "name": v} for k, v in CLASSES.items()],
            "encoding": "softmax over foreground logits 1..11; zero embeddings remain zero",
            "high_resolution_used": uses_local,
            "reference_maps_opened": False,
            "role": "predicted auxiliary semantic context; not ground truth",
        },
    )
    dump(
        status,
        {
            "state": "complete",
            "features_sha256": sha(dest),
            "metadata_sha256": sha(root / "metadata.json"),
            "shape": list(result.shape),
            "index_sha256": spec["index"]["sha256"],
            "reference_maps_opened": False,
        },
    )
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment weak-semantic-context")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.spec)
    return 0
