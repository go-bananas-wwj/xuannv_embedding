"""Exact public-parameter protection and geographically registered feature anchoring."""

from __future__ import annotations

import json
import re

import numpy as np
import torch
import torch.nn.functional as F

from xuannv_embedding.export.context import sha


def freeze_public_parameters(system) -> None:
    system.model.base.requires_grad_(False)


def audit_frozen_checkpoint(checkpoint, initializer) -> dict:
    before = torch.load(str(initializer), map_location="cpu", weights_only=True, mmap=True)["model"]
    after = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)["model"]
    names = [key for key in before if key.startswith("base.")]
    if not names or any(
        key not in after or not torch.equal(before[key], after[key]) for key in names
    ):
        raise ValueError("Frozen public checkpoint changed its registered backbone")
    return {
        "public_tensors": len(names),
        "public_tensors_unchanged": True,
        "initializer_sha256": sha(initializer),
    }


def load_teacher(reference: dict, cache: dict, *, channels: int, size: int) -> np.ndarray:
    if set(reference) != {"path", "sha256", "index", "checkpoint_sha256"}:
        raise ValueError("Unknown public teacher registration")
    index = reference["index"]
    if set(index) != {"path", "sha256"} or any(
        sha(ref["path"]) != ref["sha256"] for ref in [reference, index]
    ):
        raise ValueError("Public teacher or geographic index changed")
    if not re.fullmatch(r"[a-f0-9]{64}", reference["checkpoint_sha256"]):
        raise ValueError("Invalid teacher checkpoint identity")
    with open(index["path"]) as handle:
        source = json.load(handle)
    if source["months"] != cache["months"] or len(source["records"]) != len(cache["records"]):
        raise ValueError("Teacher geographic/time index differs")
    for a, b in zip(source["records"], cache["records"]):
        if any(a.get(k) != b.get(k) for k in ["patch_id", "bounds", "crs", "region"]):
            raise ValueError("Teacher geographic index differs")
    values = np.load(reference["path"], mmap_mode="r")
    if values.shape != (len(cache["records"]), channels, size, size):
        raise ValueError("Teacher shape differs from the registered region")
    return values


def attach_teacher(sample: dict, index: int, teacher: np.ndarray) -> dict:
    values = np.array(teacher[index], dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite teacher embedding")
    masks = [v.bool().any(dim=0) for k, v in sample["target_masks"].items() if k.endswith("_recon")]
    if not masks:
        raise ValueError("Feature anchoring requires public observation masks")
    return {
        **sample,
        "targets": {**sample["targets"], "public_teacher_embedding": torch.from_numpy(values)},
        "target_masks": {
            **sample["target_masks"],
            "public_teacher_embedding": torch.stack(masks).any(0).float(),
        },
    }


def preservation_loss(output, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    student = F.normalize(output.embedding_map.float().mean(dim=1), dim=1)
    if teacher.shape != student.shape or mask.shape != (student.shape[0], *student.shape[-2:]):
        raise ValueError("Teacher features and observation masks must match student geography")
    clean = torch.where(mask[:, None].bool(), teacher.detach().float(), 0)
    target = F.normalize(clean, dim=1)
    errors = (1 - (student * target).sum(dim=1)).clamp_min(0)
    return (errors * mask).sum() / mask.sum().clamp_min(1)
