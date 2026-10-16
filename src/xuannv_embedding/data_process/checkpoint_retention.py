"""Lossless inference-only copies of terminal training checkpoints."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch

from xuannv_embedding.training.checkpoint import _atomic_torch_save


def read_checkpoint(path: Path) -> dict:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc):
            raise
        return torch.load(str(path), map_location="cpu", weights_only=True)


def update_number(state: dict, name: str) -> int:
    metrics = state.get("metrics", {})
    for value in [
        metrics.get("optimizer_steps"),
        metrics.get("steps"),
        state.get("global_step"),
        state.get("step"),
        state.get("epoch"),
    ]:
        if type(value) is int and value >= 0:
            return value
    match = re.search(r"(?:epoch|step)[_-]?(\d+)", name)
    return int(match[1]) if match else -1


def model_digest(state: dict) -> str:
    digest = hashlib.sha256()
    found = False
    for group in ["model", "state_dict", "model_state_dict", "criterion"]:
        values = state.get(group)
        if not isinstance(values, dict):
            continue
        for key, value in sorted(values.items()):
            if not isinstance(value, torch.Tensor):
                raise ValueError("model state contains unsupported non-tensor values")
            found = True
            digest.update(f"{group}/{key}/{value.dtype}/{tuple(value.shape)}".encode())
            digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy())
    if not found:
        raise ValueError("checkpoint does not contain an identifiable model state")
    return digest.hexdigest()


def compact_checkpoint(source: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    state = read_checkpoint(source)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must be a state mapping")
    before = model_digest(state)
    removed = []
    for key in [
        "optimizer",
        "optimizer_state_dict",
        "optimizers",
        "scheduler",
        "scheduler_state_dict",
        "lr_scheduler",
        "scaler",
        "grad_scaler",
        "amp",
        "rng_state",
        "random_states",
    ]:
        if key in state:
            state[key] = None
            removed.append(key)
    state["inference_only"] = True
    _atomic_torch_save(state, destination)
    after = model_digest(read_checkpoint(destination))
    if after != before:
        destination.unlink()
        raise ValueError("compaction changed model or auxiliary weights")
    return {
        "source": str(source),
        "destination": str(destination),
        "model_tensor_digest": before,
        "verified_digest": after,
        "removed_training_state": removed,
        "source_bytes": source.stat().st_size,
        "retained_bytes": destination.stat().st_size,
        "optimizer_step": update_number(state, source.name),
    }
