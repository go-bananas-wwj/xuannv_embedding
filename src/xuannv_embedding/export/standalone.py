"""Strict inference restoration from a complete final state, never from ancestors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from xuannv_embedding.config import Config
from xuannv_embedding.training.regional_base import build_regional_system


def verified_file(root: Path, reference: dict) -> Path:
    if set(reference) != {"path", "sha256"}:
        raise ValueError("invalid bundle file reference")
    relative = Path(reference["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("bundle reference must stay inside the bundle")
    path = root / relative
    if not path.is_file():
        raise ValueError("bundle file is missing")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(data)
    if digest.hexdigest() != reference["sha256"]:
        raise ValueError("bundle content identity differs")
    return path


def load_bundle(path: Path, *, device: str = "cpu"):
    path = Path(path)
    spec = json.loads(path.read_text())
    if spec.get("protocol") != "standalone-regional-model-v1":
        raise ValueError("unsupported standalone model protocol")
    config = Config.from_yaml(verified_file(path.parent, spec["config"]))
    checkpoint = verified_file(path.parent, spec["checkpoint"])
    system, _ = build_regional_system(
        config,
        mode="scratch",
        checkpoint=None,
        seed=spec["seed"],
        static_weight=spec["static_weight"],
    )
    state = torch.load(str(checkpoint), weights_only=True, map_location="cpu", mmap=True)
    system.model.load_state_dict(state["model"], strict=True)
    system.criterion.load_state_dict(state["criterion"], strict=True)
    system.to(device).eval().requires_grad_(False)
    return system, config
