"""Observation-matched raw features with explicit native-grid area aggregation."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.downstream import multitask_features as feature_contract
from xuannv_embedding.downstream import paired_multitask as contracts
from xuannv_embedding.export.context import dump, sha

PROTOCOL = "monthly-input-raw-v1"


@lru_cache(maxsize=16)
def _area_weights(native: int, target: int) -> np.ndarray:
    edges = np.linspace(0, native, target + 1)
    source = np.arange(native)
    overlap = np.maximum(
        0, np.minimum(edges[1:, None], source + 1) - np.maximum(edges[:-1, None], source)
    )
    return overlap / (native / target)


def _array(value):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("raw input must be a CPU tensor")
    return value.detach().numpy()


def features(sample, inputs, sources, months, size):
    """Use forward inputs only; source-major/month-major bands then availability.

    High-resolution masked area means assume the same audited north-up footprint.
    Target labels, target validity and loss-only quality masks are never accessed.
    """
    if (
        not sources
        or len(set(sources)) != len(sources)
        or set(sources) != set(inputs)
        or not months
        or months != sorted(set(months))
        or not all(feature_contract._month(m) for m in months)
        or type(size) is not int
        or size < 1
    ):
        raise ValueError("invalid raw source/month/grid contract")
    temporal = {k for k, v in inputs.items() if v["role"] == "temporal"}
    highres = {k for k, v in inputs.items() if v["role"] == "highres"}
    if temporal | highres != set(inputs):
        raise ValueError("raw sources must be model temporal or highres inputs")
    for prefix, expected in [("source", temporal), ("highres", highres)]:
        if any(
            set(sample.get(prefix + suffix, {})) != expected for suffix in ["_frames", "_masks"]
        ):
            raise ValueError("raw sample sources differ from model inputs")
    if not np.array_equal(_array(sample["timestamps"]), [int(m.replace("-", "")) for m in months]):
        raise ValueError("raw sample timestamps differ from registered months")
    pieces, channel_names = [], []
    valid = np.zeros((size, size), bool)
    for source in sources:
        cfg = inputs[source]
        prefix = "source" if source in temporal else "highres"
        x = _array(sample[prefix + "_frames"][source])
        mask = _array(sample[prefix + "_masks"][source])
        channels = cfg["channels"]
        if (
            type(channels) is not int
            or channels < 1
            or x.ndim != 4
            or x.shape[:2] != (len(months), channels)
            or min(x.shape[2:]) < 1
            or not np.issubdtype(x.dtype, np.floating)
            or not np.isin(mask, [0, 1]).all()
        ):
            raise ValueError("invalid raw frame shape, dtype or binary availability")
        if source in temporal:
            if x.shape[2:] != (size, size) or mask.shape != (len(months),):
                raise ValueError("public raw frame grid or time mask differs")
            availability = np.broadcast_to(mask[:, None, None, None], (len(months), 1, size, size))
        else:
            if mask.shape != (len(months), 1, *x.shape[2:]):
                raise ValueError("monthly highres availability grid differs")
            availability = mask
        clean = np.where(availability.astype(bool), x, 0)
        if not np.isfinite(clean).all():
            raise ValueError("nonfinite visible raw observation")
        if source in highres:
            wy, wx = (_area_weights(n, size) for n in x.shape[2:])
            fraction = wy @ availability.astype(np.float64) @ wx.T
            total = wy @ clean.astype(np.float64) @ wx.T
            means = np.divide(total, fraction, out=np.zeros_like(total), where=fraction > 0)
        else:
            means, fraction = clean, availability
        valid |= (fraction > 0).any(axis=(0, 1))
        for t, month in enumerate(months):
            pieces.extend([means[t], fraction[t]])
            channel_names.extend(
                {"source": source, "month": month, "band": b} for b in range(channels)
            )
            channel_names.append({"source": source, "month": month, "band": "availability"})
    return np.concatenate(pieces).astype(np.float32), valid, channel_names


def export(spec_path):
    spec = contracts._load(spec_path)
    if (
        set(spec) != {"protocol", "cache", "sources", "period", "splits", "output"}
        or spec["protocol"] != PROTOCOL
    ):
        raise ValueError("invalid monthly raw specification")
    cache = contracts._registered(spec["cache"])
    feature_contract._grid(cache, cache)
    splits, sources = spec["splits"], spec["sources"]
    if (
        not isinstance(splits, list)
        or not splits
        or len(set(splits)) != len(splits)
        or set(splits) - {"train", "validation", "test", "buffer"}
        or not isinstance(sources, list)
        or len(set(sources)) != len(sources)
        or set(sources) != set(cache["model_inputs"])
        or spec["period"] != cache["data"]["months"][-1]
    ):
        raise ValueError("raw source order, period or splits differ from cache contract")
    indices = [i for split in splits for i in cache["split"][split]]
    if not indices:
        raise ValueError("raw export selected no records")
    stage = Path(spec["output"])
    stage.mkdir(parents=True, exist_ok=False)
    tic, digests, source_digests = time.monotonic(), {}, {}
    records = [
        {"patch_id": r["patch_id"], "bounds": r["bounds"], "path": f"tile_{i:06d}.npz"}
        for i, r in enumerate(cache["records"])
    ]
    dump(stage / "status.json", {"state": "running", "completed": 0})
    try:
        for i in indices:
            record = cache["records"][i]
            path = Path(record["path"])
            if not path.is_absolute():
                path = Path(spec["cache"]["path"]).parent / path
            if sha(path) != record["sha256"]:
                raise ValueError("raw source sample digest changed")
            sample = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
            if sample["patch_id"] != record["patch_id"]:
                raise ValueError("raw sample tile identity differs")
            values, valid, channel_names = features(
                sample,
                cache["model_inputs"],
                sources,
                cache["data"]["months"],
                cache["data"]["patch_size"],
            )
            out = stage / records[i]["path"]
            np.savez_compressed(out, embedding=values[None], valid_mask=valid)
            records[i]["sha256"] = digests[record["patch_id"]] = sha(out)
            source_digests[record["patch_id"]] = record["sha256"]
            dump(stage / "status.json", {"state": "running", "completed": len(digests)})
        manifest = {
            "kind": "raw",
            "protocol": PROTOCOL,
            "cache_sha256": spec["cache"]["sha256"],
            "months": [spec["period"]],
            "observation_months": cache["data"]["months"],
            "temporal_semantics": (
                "all registered input months concatenated; not a monthly embedding"
            ),
            "aggregation": (
                "same-footprint exact area weighted masked mean; area availability fraction"
            ),
            "channel_names": channel_names,
            "channels": len(channel_names),
            "split": cache["split"],
            "records": records,
            "exported_indices": indices,
            "source_sample_sha256": source_digests,
        }
        dump(stage / "manifest.json", manifest)
        result = {
            "state": "complete",
            "spec_sha256": sha(Path(spec_path)),
            "manifest_sha256": sha(stage / "manifest.json"),
            "tile_sha256": digests,
            "exported_indices": indices,
            "channels": len(channel_names),
            "test_records_read": bool(set(indices) & set(cache["split"]["test"])),
            "implementation_sha256": sha(Path(__file__)),
            "runtime": {"numpy": np.__version__, "torch": str(torch.__version__)},
            "elapsed_seconds": time.monotonic() - tic,
            "scope": (
                "raw observation preparation only; no label use, model fitting or accuracy score"
            ),
        }
        dump(stage / "verification.json", result)
        dump(stage / "status.json", {"state": "complete", "completed": len(digests)})
        return result
    except BaseException as exc:
        dump(
            stage / "status.json",
            {"state": "failed", "completed": len(digests), "error": repr(exc)},
        )
        raise
