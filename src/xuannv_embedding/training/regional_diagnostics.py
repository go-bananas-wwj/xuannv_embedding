"""Training telemetry and fixed-sample diagnostics without downstream test labels."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def append_training_curve(
    path: Path,
    *,
    step: int,
    losses: dict,
    rates: dict,
    elapsed_seconds: float,
    skipped_updates: int,
) -> None:
    record = {
        "optimizer_steps": step,
        "elapsed_seconds": elapsed_seconds,
        "losses": losses,
        "learning_rates": rates,
        "skipped_updates": skipped_updates,
        "aggregation": "mean of microbatches and distributed ranks at this update",
    }
    # Serialize before opening the file: invalid telemetry cannot damage prior history.
    line = json.dumps(record, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(line)


def initialize_trained_public(system, state: dict) -> None:
    """Copy a complete public backbone and auxiliary heads, preserving fresh RGB layers."""
    weights = state["model"]
    if not weights or any(not key.startswith("base.") for key in weights):
        raise ValueError("Public initialization must not contain high-resolution parameters")
    system.model.base.load_state_dict(
        {key[5:]: value for key, value in weights.items()}, strict=True
    )
    system.criterion.load_state_dict(state["criterion"], strict=True)


def checkpoint_diagnostics(
    spec_path: Path, checkpoints: list[Path], output: Path, device_name: str
):
    """Recompute unmasked objective snapshots; never claim an original training curve."""
    from xuannv_embedding.config import Config
    from xuannv_embedding.data.raster_dataset import collate_region_batch
    from xuannv_embedding.export.context import dump, sha
    from xuannv_embedding.training.cli import _setup_device
    from xuannv_embedding.training.regional_base import build_regional_system, read_spec
    from xuannv_embedding.training.runtime import _autocast, _move

    spec, config, cache = read_spec(spec_path)
    device, _, _ = _setup_device(device_name)
    if device.type == "npu":
        torch.npu.config.allow_internal_format = False
    indices = sorted(
        {
            0,
            len(cache["records"]) // 4,
            len(cache["records"]) // 2,
            3 * len(cache["records"]) // 4,
            len(cache["records"]) - 1,
        }
    )
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
    system.to(device).eval()
    result = {
        "kind": "fixed_sample_unmasked_objective_snapshots_not_original_training_curve",
        "sample_indices": indices,
        "spec_sha256": sha(spec_path),
        "snapshots": [],
        "downstream_labels_read": False,
    }
    for checkpoint in [None, *checkpoints]:
        if checkpoint is not None:
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
            if state["config_sha256"] != spec["config"]["sha256"]:
                raise ValueError("Diagnostic checkpoint configuration changed")
            system.model.load_state_dict(state["model"], strict=True)
            system.criterion.load_state_dict(state["criterion"], strict=True)
            step = state["epoch"] + 1
            del state
        else:
            step = 0
        system.criterion.set_epoch(step)
        sums = {}
        for index in indices:
            sample = torch.load(cache["records"][index]["path"], weights_only=True, mmap=True)
            batch = _move(collate_region_batch([sample]), device)
            with torch.inference_mode(), _autocast(device, device.type == "npu"):
                losses = system(batch)
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    sums[key] = sums.get(key, 0.0) + float(value.detach().float().cpu())
        result["snapshots"].append(
            {
                "step": step,
                "checkpoint_sha256": sha(checkpoint) if checkpoint else None,
                "losses": {key: value / len(indices) for key, value in sums.items()},
            }
        )
        dump(output, result)
    return result
