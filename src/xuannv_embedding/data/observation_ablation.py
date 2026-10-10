"""Deterministic inference input ablations; cached targets remain untouched."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

import torch


def retain_highres_inputs(batch, sources, *, fraction, seed):
    """Keep an edge-anchored prefix of each month's originally valid HR pixels.

    A fixed hash selects the edge for a tile/source. Fractions are nested and keep
    floor(fraction * valid_count) pixels. No global RNG or target arrays are used.
    """
    frames = batch.get("highres_frames", {})
    ids = batch["patch_ids"]
    if (
        not sources
        or len(set(sources)) != len(sources)
        or not set(sources) <= set(frames)
        or isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
        or type(seed) is not int
        or seed < 0
        or len(set(ids)) != len(ids)
        or any(not isinstance(i, str) or not i for i in ids)
    ):
        raise ValueError("invalid registered high-resolution retention request")
    result = dict(batch)
    result["highres_frames"] = dict(frames)
    result["highres_masks"] = dict(batch["highres_masks"])
    for source in sources:
        original, mask = frames[source], batch["highres_masks"][source]
        if original.ndim not in (4, 5) or original.shape[0] != len(ids):
            raise ValueError("invalid high-resolution retention frame shape")
        h, w = original.shape[-2:]
        leading = original.shape[:2] if original.ndim == 5 else original.shape[:1]
        if (
            min(h, w) < 1
            or mask.shape != (*leading, 1, h, w)
            or mask.device != original.device
            or not ((mask == 0) | (mask == 1)).all()
            or (original.ndim == 5 and original.shape[1] != batch["timestamps"].shape[1])
        ):
            raise ValueError("invalid monthly high-resolution retention mask")
        if fraction == 1:
            continue
        output, kept_masks = original.clone(), mask.clone()
        for b, patch_id in enumerate(ids):
            digest = hashlib.sha256(json.dumps([seed, patch_id, source]).encode()).digest()
            order = torch.arange(h * w, device=original.device).reshape(h, w)
            if digest[0] % 2:
                order = order.T
            if digest[1] % 2:
                order = order.flip(0)
            order = order.reshape(-1)
            for month in range(original.shape[1] if original.ndim == 5 else 1):
                index = (b, month) if original.ndim == 5 else b
                valid = mask[index].reshape(-1) > 0
                eligible = order[valid[order]]
                keep = torch.zeros(h * w, dtype=torch.bool, device=original.device)
                keep[eligible[: int(fraction * len(eligible))]] = True
                keep = keep.reshape(1, h, w)
                output[index] = torch.where(keep, original[index], 0)
                kept_masks[index] = torch.where(keep, mask[index], 0)
        result["highres_frames"][source], result["highres_masks"][source] = output, kept_masks
    return result


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
