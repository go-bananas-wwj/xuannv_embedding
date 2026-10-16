"""Export registered regional weights without retaining redundant monthly cubes."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import collate_region_batch
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import _setup_device
from xuannv_embedding.training.regional_base import build_regional_system, read_spec
from xuannv_embedding.training.runtime import _autocast, _move


def representations(values: torch.Tensor) -> dict[str, torch.Tensor]:
    if values.ndim != 5 or values.shape[1] != 12 or not torch.isfinite(values).all():
        raise ValueError("Finite January-December monthly maps are required")
    return {
        "annual": F.normalize(values.float().mean(1), dim=1),
        "april": F.normalize(values[:, 3].float(), dim=1),
        "mar_may": F.normalize(values[:, 2:5].float().mean(1), dim=1),
    }


def export(spec_path: Path, checkpoint: Path, output: Path, *, device_name: str) -> dict:
    spec, config, cache = read_spec(spec_path)
    years = {value[:4] for value in config.data.months}
    if len(years) != 1 or [value[5:] for value in config.data.months] != [
        f"{i:02d}" for i in range(1, 13)
    ]:
        raise ValueError("Annual export requires January through December of one year")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "spec_sha256": sha(spec_path),
        "checkpoint_sha256": sha(checkpoint),
        "representations": ["annual", "april", "mar_may"],
    }
    path = output / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Export identity changed")
    dump(path, identity)
    complete = output / "status.json"
    if complete.exists() and json.loads(complete.read_text()).get("state") == "complete":
        return json.loads(complete.read_text())
    system, _ = build_regional_system(
        config,
        mode="scratch",
        checkpoint=None,
        seed=spec["training"]["seed"],
        static_weight=spec["training"].get("static_weight", 0.25),
        initialization_config=(
            Config.from_yaml(spec["initialization_config"]["path"])
            if "initialization_config" in spec
            else None
        ),
    )
    state = load_training_checkpoint(
        checkpoint,
        model=system.model,
        criterion=system.criterion,
        expected_config_sha256=spec["config"]["sha256"],
        expected_source_schema={k: asdict(v) for k, v in config.model.input_sources.items()},
        expected_regions=[d.region for d in config.data.datasets],
    )
    step = state["epoch"] + 1
    del state
    device, _, _ = _setup_device(device_name)
    if device.type == "npu":
        torch.npu.config.allow_internal_format = False
    model = system.model.to(device).eval()
    arrays = {}
    shape = (
        len(cache["records"]),
        config.model.embed_dim,
        config.data.patch_size,
        config.data.patch_size,
    )
    for name in identity["representations"]:
        arrays[name] = np.lib.format.open_memmap(
            output / f"{name}.partial.npy", mode="w+", dtype=np.float16, shape=shape
        )
    keys = ("source_frames", "source_masks", "timestamps", "highres_frames", "highres_masks")
    with torch.inference_mode():
        for i, record in enumerate(cache["records"]):
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("Registered sample changed")
            sample = torch.load(record["path"], weights_only=True, mmap=True)
            batch = collate_region_batch([sample])
            inputs = _move({key: batch[key] for key in keys}, device)
            with _autocast(device, device.type == "npu"):
                result = model(**inputs).embedding_map
            for name, values in representations(result).items():
                norm = values.norm(dim=1)
                if not torch.allclose(norm, torch.ones_like(norm), atol=1e-4, rtol=1e-4):
                    raise ValueError("Degenerate regional embedding")
                arrays[name][i] = values[0].cpu().numpy().astype(np.float16)
            if i % 20 == 0:
                dump(
                    complete,
                    {
                        "state": "exporting",
                        "completed": i + 1,
                        "total": len(cache["records"]),
                        "optimizer_steps": step,
                    },
                )
    for name, array in arrays.items():
        array.flush()
        (output / f"{name}.partial.npy").replace(output / f"{name}.npy")
    report = {
        "state": "complete",
        **identity,
        "optimizer_steps": step,
        "shape": list(shape),
        "patch_ids": [r["patch_id"] for r in cache["records"]],
        "files": {
            name: {"path": str(output / f"{name}.npy"), "sha256": sha(output / f"{name}.npy")}
            for name in arrays
        },
    }
    dump(complete, report)
    return report
