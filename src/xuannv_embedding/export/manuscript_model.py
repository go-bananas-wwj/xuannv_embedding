"""Export a terminal paired model, optionally removing local input before encoding."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.cli import _setup_device
from xuannv_embedding.training.frozen_fusion import verified
from xuannv_embedding.training.manuscript_study import (
    Observations,
    build,
    load_study_checkpoint,
    specification,
)
from xuannv_embedding.training.runtime import _autocast, _move


def coverage_mask(shape, fraction, position):
    if fraction not in [0.0, 0.25, 0.5, 1.0]:
        raise ValueError("Unregistered optical coverage")
    order = np.random.default_rng(41 + position).permutation(16)
    keep = np.zeros(16, np.float32)
    keep[order[: round(16 * fraction)]] = 1
    return F.interpolate(torch.from_numpy(keep.reshape(1, 1, 4, 4)), shape, mode="nearest")[0]


def export(spec_path, phase, device, *, fraction=1.0):
    spec = specification(spec_path)
    if phase not in {"public", "highres"}:
        raise ValueError("Only terminal paired models can be exported")
    root = Path(spec["output"])
    checkpoint = root / "runs" / phase / "final.pt"
    state = load_study_checkpoint(checkpoint)
    if state["step"] != 1600 or state["identity"]["spec_sha256"] != sha(spec_path):
        raise ValueError("Export requires the registered terminal model")
    data = Observations(spec, highres=phase == "highres")
    model, _ = build(spec, data.meta, phase)
    model.load_state_dict(state["model"], strict=True)
    dev, _, _ = _setup_device(device)
    model.to(dev).eval()
    out = root / "exports" / (phase + f"_coverage{round(100*fraction)}")
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "checkpoint_sha256": sha(checkpoint),
        "spec_sha256": sha(spec_path),
        "implementation": sha(__file__),
        "coverage": fraction,
        "index": spec["index"],
    }
    previous_hash = None
    if (out / "status.json").exists():
        result = json.loads((out / "status.json").read_text())
        if result.get("identity") != identity:
            raise ValueError("Export identity changed")
        previous_hash = result["features"]["sha256"]
        if Path(result["features"]["path"]).exists():
            verified(result["features"])
            return result
    required = spec["reserve_gib"] * 2**30 + data.n * 64 * 128 * 128 * 2
    if shutil.disk_usage(out).free < required:
        raise RuntimeError("Export allocation would violate disk reserve")
    shape = (data.n, 64, 128, 128)
    values = np.lib.format.open_memmap(
        out / "annual.partial.npy", mode="w+", dtype="float16", shape=shape
    )
    with torch.inference_mode():
        for i in range(data.n):
            batch = _move(default_collate([data.sample(i)]), dev)
            if phase == "highres":
                for key, mask in batch["highres_masks"].items():
                    batch["highres_masks"][key] = mask * coverage_mask(
                        mask.shape[-2:], fraction, i
                    ).to(dev)
            with _autocast(dev, dev.type == "npu"):
                z = model(
                    batch["source_frames"],
                    batch["source_masks"],
                    batch["timestamps"],
                    batch["highres_frames"],
                    batch["highres_masks"],
                ).embedding_map.mean(1)
            if not torch.isfinite(z).all():
                raise FloatingPointError("Nonfinite regional export")
            values[i] = z[0].float().cpu().numpy().astype("float16")
    values.flush()
    (out / "annual.partial.npy").replace(out / "annual.npy")
    result = {
        "state": "complete",
        "identity": identity,
        "shape": list(shape),
        "features": {"path": str(out / "annual.npy"), "sha256": sha(out / "annual.npy")},
        "primary_maps_loaded": False,
        "monthly_outputs_averaged": 12,
        "full_local_input_masked_before_encoder": phase == "highres",
    }
    if previous_hash is not None and result["features"]["sha256"] != previous_hash:
        raise ValueError("Rebuilt export differs from the frozen calibrated representation")
    dump(out / "status.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["public", "highres"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coverage", type=float, default=1.0)
    args = parser.parse_args(argv)
    export(args.spec, args.phase, args.device, fraction=args.coverage)
    return 0
