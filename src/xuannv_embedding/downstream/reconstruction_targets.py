"""Strictly paired external targets; their observations never become encoder inputs."""

from __future__ import annotations

import math

import torch

from xuannv_embedding.config import TargetHeadConfig
from xuannv_embedding.downstream.multitask import check_partition
from xuannv_embedding.training.experiment import CachedSamples


def paired_target_schema(document: dict, targets: dict, name: str) -> TargetHeadConfig:
    """Require an identical registered grid, month order and complete split topology."""
    check_partition(targets["split"], len(targets["records"]))
    if (
        document["split"] != targets["split"]
        or len(document["records"]) != len(targets["records"])
        or not document.get("manifest_sha256")
        or document["manifest_sha256"] != targets.get("manifest_sha256")
        or any(document["data"][k] != targets["data"][k] for k in ("months", "patch_size"))
    ):
        raise ValueError("external target grid, months or partition differ")
    for a, b in zip(document["records"], targets["records"], strict=True):
        if any(a[k] != b[k] for k in ("index", "patch_id", "bounds")):
            raise ValueError("external target grid/order differs")
    head = targets.get("model_targets", {}).get(name, {})
    if (
        set(head) != {"source", "loss_type", "channels", "weight"}
        or not isinstance(head["source"], str)
        or not head["source"]
        or head["source"] not in targets.get("model_inputs", {})
        or head["loss_type"] != "continuous"
        or type(head["channels"]) is not int
        or head["channels"] < 1
        or type(head["weight"]) not in (float, int)
        or not math.isfinite(head["weight"])
        or head["weight"] < 0
    ):
        raise ValueError("external target must declare a valid continuous target schema")
    return TargetHeadConfig(**head)


def paired_samples(document, targets, indices, *, name, channels):
    """Yield model observations and separate targets, reading only requested records."""
    if targets is document:
        for sample in CachedSamples(document, indices):
            yield sample, sample
        return
    timestamps = torch.tensor([int(m.replace("-", "")) for m in document["data"]["months"]])
    side = document["data"]["patch_size"]
    for index, sample, reference in zip(
        indices, CachedSamples(document, indices), CachedSamples(targets, indices), strict=True
    ):
        if (
            sample["patch_id"] != document["records"][index]["patch_id"]
            or sample["patch_id"] != reference["patch_id"]
            or sample["region"] != reference["region"]
            or not torch.equal(sample["timestamps"], timestamps)
            or not torch.equal(reference["timestamps"], timestamps)
        ):
            raise ValueError("external target sample identity or timestamps differ")
        values, mask = reference["targets"][name], reference["target_masks"][name]
        if values.shape != (len(timestamps), channels, side, side) or mask.shape != (
            len(timestamps),
            side,
            side,
        ):
            raise ValueError("external target sample geometry differs")
        yield sample, reference
