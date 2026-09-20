"""Small collective helpers for registered single-model distributed experiments."""

from __future__ import annotations

import torch
import torch.distributed as dist


def gather_objects(value):
    if not dist.is_initialized():
        return [value]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def global_means(sums: dict[str, float], count: int) -> dict[str, float]:
    parts = gather_objects((sums, count))
    total = sum(n for _, n in parts)
    if total <= 0:
        raise ValueError("distributed metric has no samples")
    keys = set().union(*(values for values, _ in parts))
    return {k: sum(values.get(k, 0.0) for values, _ in parts) / total for k in sorted(keys)}


def rank_random_state(device: torch.device, scaler) -> dict:
    return {
        "torch_rng_state": torch.get_rng_state(),
        "device_rng_state": torch.npu.get_rng_state(device).cpu() if device.type == "npu" else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }


def restore_rank_random_state(state: dict, device: torch.device, scaler) -> None:
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if device.type == "npu":
        torch.npu.set_rng_state(state["device_rng_state"].cpu(), device)
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
