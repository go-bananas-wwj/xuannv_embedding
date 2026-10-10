"""A frozen public embedding plus same-month native high-resolution statistics."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.downstream import multitask_features as features
from xuannv_embedding.downstream import paired_multitask as contracts
from xuannv_embedding.downstream.raw_monthly import _area_weights, _array
from xuannv_embedding.export.context import dump, sha

PROTOCOL = "monthly-late-fusion-v1"


def highres_statistics(sample, inputs, sources, months, month, size):
    """Area-weighted means, population deviations and valid fractions on one month.

    Native rasters must have the same audited north-up footprint as the output grid.
    Invalid values are ignored; only forward input masks are used. No label or target
    mask is inspected. Centering before second moments avoids large-offset cancellation.
    """
    expected = {k for k, v in inputs.items() if v["role"] == "highres"}
    if (
        not sources
        or len(set(sources)) != len(sources)
        or set(sources) != expected
        or not months
        or months != sorted(set(months))
        or not all(features._month(m) for m in months)
        or month not in months
        or type(size) is not int
        or size < 1
        or set(sample["highres_frames"]) != expected
        or set(sample["highres_masks"]) != expected
        or not np.array_equal(
            _array(sample["timestamps"]), [int(m.replace("-", "")) for m in months]
        )
    ):
        raise ValueError("invalid high-resolution statistic source/month/grid contract")
    index = months.index(month)
    pieces, channels = [], []
    for source in sources:
        x = _array(sample["highres_frames"][source])
        mask = _array(sample["highres_masks"][source])
        count = inputs[source]["channels"]
        if (
            type(count) is not int
            or count < 1
            or x.ndim != 4
            or x.shape[:2] != (len(months), count)
            or min(x.shape[2:]) < 1
            or not np.issubdtype(x.dtype, np.floating)
            or mask.shape != (len(months), 1, *x.shape[2:])
            or not np.isin(mask, [0, 1]).all()
        ):
            raise ValueError("invalid high-resolution frame or binary input mask")
        valid = mask[index].astype(bool)
        clean = np.where(valid, x[index], 0).astype(np.float64)
        if not np.isfinite(clean).all():
            raise ValueError("nonfinite visible high-resolution input")
        wy, wx = (_area_weights(n, size) for n in x.shape[2:])
        fraction = wy @ valid.astype(np.float64) @ wx.T
        offset = clean.sum(axis=(-2, -1), keepdims=True) / max(int(valid.sum()), 1)
        centered = np.where(valid, clean - offset, 0)
        first = wy @ centered @ wx.T
        first = np.divide(first, fraction, out=np.zeros_like(first), where=fraction > 0)
        second = wy @ (centered * centered) @ wx.T
        second = np.divide(second, fraction, out=np.zeros_like(second), where=fraction > 0)
        mean = np.where(fraction > 0, first + offset, 0)
        deviation = np.sqrt(np.maximum(second - first * first, 0))
        pieces.extend([mean, deviation, fraction])
        for statistic in ("mean", "std"):
            channels.extend(
                {"source": source, "month": month, "band": band, "statistic": statistic}
                for band in range(count)
            )
        channels.append(
            {"source": source, "month": month, "band": None, "statistic": "availability"}
        )
    result = np.concatenate(pieces).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite float32 high-resolution statistic")
    return result, channels


def export(spec_path):
    spec = contracts._load(spec_path)
    if (
        set(spec) != {"protocol", "cache", "base", "sources", "month", "splits", "output"}
        or spec["protocol"] != PROTOCOL
        or set(spec["base"])
        != {
            "manifest_path",
            "manifest_sha256",
            "cache_path",
            "cache_sha256",
            "tile_sha256",
            "channels",
        }
    ):
        raise ValueError("invalid late-fusion specification")
    cache = contracts._registered(spec["cache"])
    _, layout = features._grid(cache, cache)
    base = spec["base"]
    base_cache = features._registered_json(
        Path(base["cache_path"]), base["cache_sha256"], "base cache"
    )
    base_manifest = features._registered_json(
        Path(base["manifest_path"]), base["manifest_sha256"], "base manifest"
    )
    _, base_layout = features._grid(base_cache, base_manifest)
    if layout != base_layout or cache["data"]["months"] != base_cache["data"]["months"]:
        raise ValueError("late-fusion base and native-input geography or period differ")
    if not isinstance(spec["sources"], list) or set(spec["sources"]) != {
        k for k, v in cache["model_inputs"].items() if v["role"] == "highres"
    }:
        raise ValueError("late-fusion requires registered high-resolution sources")
    selection = features.FeatureSelection("monthly", spec["month"], spec["month"], base["channels"])
    batch = features.read_features(
        base["manifest_path"],
        base["cache_path"],
        manifest_sha256=base["manifest_sha256"],
        cache_sha256=base["cache_sha256"],
        tile_sha256=base["tile_sha256"],
        selection=selection,
        splits=spec["splits"],
    )
    stage = Path(spec["output"])
    stage.mkdir(parents=True, exist_ok=False)
    records = [
        {"patch_id": r["patch_id"], "bounds": r["bounds"], "path": f"tile_{i:06d}.npz"}
        for i, r in enumerate(cache["records"])
    ]
    tic, digests, sample_digests = time.monotonic(), {}, {}
    dump(stage / "status.json", {"state": "running", "completed": 0})
    try:
        for position, index in enumerate(batch.indices):
            record = cache["records"][index]
            path = Path(record["path"])
            if not path.is_absolute():
                path = Path(spec["cache"]["path"]).parent / path
            if sha(path) != record["sha256"]:
                raise ValueError("late-fusion native sample digest changed")
            sample = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
            if sample["patch_id"] != record["patch_id"]:
                raise ValueError("late-fusion sample tile identity differs")
            statistics, channel_names = highres_statistics(
                sample,
                cache["model_inputs"],
                spec["sources"],
                cache["data"]["months"],
                spec["month"],
                cache["data"]["patch_size"],
            )
            values = np.concatenate([batch.values[position].transpose(2, 0, 1), statistics])
            values[:, ~batch.valid[position]] = 0
            target = stage / records[index]["path"]
            np.savez_compressed(
                target,
                embedding=values[None],
                valid_mask=batch.valid[position],
                timestamps=np.array([int(spec["month"].replace("-", ""))]),
            )
            records[index]["sha256"] = digests[record["patch_id"]] = sha(target)
            sample_digests[record["patch_id"]] = record["sha256"]
            dump(stage / "status.json", {"state": "running", "completed": len(digests)})
        manifest = {
            "kind": "monthly",
            "protocol": PROTOCOL,
            "months": [spec["month"]],
            "cache_sha256": spec["cache"]["sha256"],
            "split": cache["split"],
            "records": records,
            "exported_indices": list(batch.indices),
            "channels": values.shape[0],
            "base_dimensions": base["channels"],
            "appended_channel_names": channel_names,
            "source_sample_sha256": sample_digests,
            "base_identity": batch.identity,
            "aggregation": (
                "same-footprint area-weighted native mean, "
                "population standard deviation, availability"
            ),
            "validity": (
                "base feature validity; missing highres adds zero statistics and availability"
            ),
            "temporal_semantics": (
                "whole-window public embedding at requested month plus same-month HR statistics"
            ),
        }
        dump(stage / "manifest.json", manifest)
        result = {
            "state": "complete",
            "protocol": PROTOCOL,
            "spec_sha256": sha(Path(spec_path)),
            "manifest_sha256": sha(stage / "manifest.json"),
            "tile_sha256": digests,
            "exported_indices": list(batch.indices),
            "channels": values.shape[0],
            "test_records_read": bool(set(batch.indices) & set(cache["split"]["test"])),
            "labels_read": False,
            "implementation_sha256": sha(Path(__file__)),
            "area_weights_implementation_sha256": sha(Path(__file__).with_name("raw_monthly.py")),
            "elapsed_seconds": time.monotonic() - tic,
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
