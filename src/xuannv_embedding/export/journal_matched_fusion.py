"""Direct fusion of the same pretrained base and already computed guided increment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.manuscript_controls import pca_projection
from xuannv_embedding.training.frozen_fusion import verified


def shared_feature_parts(base, final):
    if (
        base.shape != final.shape
        or base.ndim != 3
        or base.shape[0] != 64
        or base.dtype != np.float16
        or final.dtype != np.float16
        or not np.isfinite(base).all()
        or not np.isfinite(final).all()
    ):
        raise ValueError("Matched fusion requires equal native float16 embedding grids")
    b, e = base.astype("float32"), final.astype("float32")
    increment = e - b
    if not np.array_equal(b + increment, e):
        raise ValueError("Shared feature components do not reconstruct the deployed feature")
    return np.concatenate([b, increment], axis=0)


def run(path, device):
    from xuannv_embedding.training.manuscript_followup import score

    spec = json.loads(Path(path).read_text())
    if (
        set(spec) != {"schema", "base", "final", "region", "output", "seed", "fit_samples"}
        or spec["schema"] != "journal-matched-feature-fusion-v1"
    ):
        raise ValueError("Invalid pretrained-feature fusion control")
    if spec["seed"] != 41 or spec["fit_samples"] != 65536:
        raise ValueError("Matched fusion sample budget changed")
    configured = json.loads(verified(spec["region"]["aef_jobs"]["mlp"]).read_text())
    if device != configured["device"]:
        raise ValueError("Device differs from the registered matched readout contract")
    index = json.loads(verified(spec["region"]["index"]).read_text())
    arrays = []
    for key in ["base", "final"]:
        ref = spec[key]
        if ref["layout"] != "nchw" or ref["index_sha256"] != spec["region"]["index"]["sha256"]:
            raise ValueError("Matched fusion coordinate binding changed")
        arrays.append(
            np.load(verified({"path": ref["path"], "sha256": ref["sha256"]}), mmap_mode="r")
        )
    base, final = arrays
    if base.shape != final.shape or len(base) != len(index["records"]):
        raise ValueError("Matched feature/region shapes differ")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    embedding = root / "embedding"
    embedding.mkdir(exist_ok=True)
    identity = {"spec": sha(path), "code": sha(__file__)}
    status = embedding / "status.json"
    if status.exists():
        result = json.loads(status.read_text())
        if result["identity"] != identity:
            raise ValueError("Matched fusion control binding changed")
        verified(result["features"])
    else:
        rng = np.random.default_rng(spec["seed"])
        samples = []
        for i in range(len(base)):
            parts = shared_feature_parts(base[i], final[i]).reshape(128, -1).T
            positions = rng.choice(len(parts), min(256, len(parts)), replace=False)
            samples.append(parts[positions])
        samples = np.concatenate(samples)
        if len(samples) > spec["fit_samples"]:
            samples = samples[rng.choice(len(samples), spec["fit_samples"], replace=False)]
        mean, scale, components = pca_projection(samples)
        np.savez(embedding / "projection.npz", mean=mean, scale=scale, components=components)
        destination = embedding / "annual.npy"
        output = np.lib.format.open_memmap(
            embedding / "annual.partial.npy", mode="w+", dtype="float16", shape=base.shape
        )
        for i in range(len(base)):
            parts = shared_feature_parts(base[i], final[i]).reshape(128, -1).T
            values = (((parts - mean) / scale) @ components).T.reshape(base.shape[1:])
            if not np.isfinite(values).all() or np.max(np.abs(values)) > np.finfo(np.float16).max:
                raise ValueError("Nonfinite matched fusion")
            output[i] = values.astype("float16")
        output.flush()
        del output
        (embedding / "annual.partial.npy").replace(destination)
        result = dict(
            state="complete",
            identity=identity,
            features={"path": str(destination), "sha256": sha(destination)},
            shared_pretrained_features=True,
            base_plus_increment_exact=True,
            encoder_weights_refitted=False,
            input_dimensions=128,
            output_dimensions=64,
            fit_samples=len(samples),
        )
        dump(status, result)
    follow = dict(
        schema="manuscript-followup-v1",
        region=spec["region"],
        case="matched_feature_concat",
        producer={"kind": "frozen", "features": result["features"]},
        output=str(root / "comparison"),
        retire_arrays=False,
    )
    follow_path = root / "score.json"
    if follow_path.exists() and json.loads(follow_path.read_text()) != follow:
        raise ValueError("Matched fusion readout contract changed")
    dump(follow_path, follow)
    return score(follow_path)
