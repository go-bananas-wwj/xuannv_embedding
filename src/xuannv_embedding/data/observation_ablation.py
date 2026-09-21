"""Deterministic inference input ablations; cached targets remain untouched."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import torch


def ablate_inputs(
    batch: dict[str, Any], sources: Sequence[str], *, last_month: int | None = None
) -> dict[str, Any]:
    count = batch["timestamps"].shape[1]
    names = set(batch["source_frames"]) | set(batch.get("highres_frames", {}))
    if len(set(sources)) != len(sources) or not set(sources) <= names:
        raise ValueError("ablation sources must be unique registered inputs")
    if last_month is not None and (type(last_month) is not int or not 0 <= last_month < count):
        raise ValueError("invalid last visible month")
    result = dict(batch)
    for group in ("source", "highres"):
        frames_key, masks_key = group + "_frames", group + "_masks"
        result[frames_key] = dict(batch.get(frames_key, {}))
        result[masks_key] = dict(batch.get(masks_key, {}))
        for name, original in batch.get(frames_key, {}).items():
            if name not in sources and last_month is None:
                continue
            frame, mask = original.clone(), batch[masks_key][name].clone()
            if frame.ndim not in (4, 5):
                raise ValueError("unsupported observation dimensions")
            if name in sources or frame.ndim == 4:
                # Undated static inputs cannot certify any prefix time boundary.
                frame.zero_()
                mask.zero_()
            else:
                if frame.shape[1] != count or mask.shape[:2] != frame.shape[:2]:
                    raise ValueError("monthly frame/mask dimensions differ")
                frame[:, last_month + 1 :] = 0
                mask[:, last_month + 1 :] = 0
            result[frames_key][name], result[masks_key][name] = frame, mask
    return result


def mask_audit(batch: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for index, patch_id in enumerate(batch["patch_ids"]):
        digest = hashlib.sha256()
        availability = {}
        for group in ("source", "highres"):
            for name, value in sorted(batch.get(group + "_masks", {}).items()):
                mask = value[index].detach().cpu().contiguous()
                if not torch.isfinite(mask).all():
                    raise ValueError("nonfinite input availability mask")
                key = group + ":" + name
                digest.update(json.dumps([key, list(mask.shape), str(mask.dtype)]).encode())
                digest.update(mask.numpy().tobytes())
                monthly = batch[group + "_frames"][name].ndim == 5
                if monthly:
                    fractions = (mask > 0).reshape(len(mask), -1).float().mean(1).tolist()
                else:
                    fractions = [float((mask > 0).float().mean())]
                availability[key] = fractions
        result.append(
            {"patch_id": patch_id, "sha256": digest.hexdigest(), "availability": availability}
        )
    return result
