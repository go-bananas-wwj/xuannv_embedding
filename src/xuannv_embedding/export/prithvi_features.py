"""Frozen multispectral temporal features with one projection per complete region."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import shutil
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from xuannv_embedding.data_process import public_monthly_tensor as monthly
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS
from xuannv_embedding.export.semantic_features import align_patch_features
from xuannv_embedding.models import prithvi
from xuannv_embedding.training.cli import _setup_device
from xuannv_embedding.training.runtime import _autocast

S2_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")


def normalize_reflectance(
    frames: np.ndarray,
    valid: np.ndarray,
    bands: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert already calibrated reflectance to published scaled-reflectance units."""
    frames, valid = np.asarray(frames), np.asarray(valid)
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if (
        frames.ndim != 4
        or len(frames) != 12
        or frames.shape[1] != len(bands)
        or len(set(bands)) != len(bands)
        or not set(S2_BANDS).issubset(bands)
        or valid.shape != (len(frames), *frames.shape[-2:])
        or valid.dtype != bool
        or not np.issubdtype(frames.dtype, np.number)
        or np.iscomplexobj(frames)
        or mean.shape != (6,)
        or std.shape != (6,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Invalid named S2 reflectance, mask or normalization")
    selected = frames[:, [list(bands).index(b) for b in S2_BANDS]].astype(np.float32)
    if not np.isfinite(selected.transpose(0, 2, 3, 1)[valid]).all():
        raise ValueError("Observed S2 reflectance must be finite")
    selected = np.where(valid[:, None], selected, mean[None, :, None, None] / 10000)
    values = (selected * 10000 - mean[None, :, None, None]) / std[None, :, None, None]
    values = np.where(valid[:, None], values, 0).astype(np.float32)
    return torch.from_numpy(values.transpose(1, 0, 2, 3).copy()), torch.from_numpy(valid.copy())


def read_public_s2(
    record: dict, observations: dict, stats: dict, months: list[int], mean: list, std: list
) -> tuple[torch.Tensor, torch.Tensor]:
    # This reader ALREADY applies radiometry and returns canonical band order.
    # Applying calibrate_public again would corrupt China's scaled/offset input.
    frames, masks = monthly.read_record(record, observations, stats, months)
    return normalize_reflectance(frames["s2"], masks["s2"], PUBLIC_BANDS["s2"], mean, std)


def pool_temporal(features: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool using observed patch area; counts are observed-frame equivalents."""
    if (
        features.ndim != 5
        or valid.ndim != 4
        or valid.dtype != torch.bool
        or features.shape[0] != valid.shape[0]
        or features.shape[2] != valid.shape[1]
        or valid.shape[-2] % 16
        or valid.shape[-1] % 16
        or features.shape[-2:] != (valid.shape[-2] // 16, valid.shape[-1] // 16)
        or not torch.isfinite(features).all()
    ):
        raise ValueError("Temporal feature, observation mask or patch grid differs")
    fractions = F.avg_pool3d(valid[:, None].float(), (1, 16, 16))
    counts = fractions.sum(2)
    numerator = (features.float() * fractions).sum(2)
    return torch.where(counts > 0, numerator / counts.clamp_min(1e-6), 0), counts


def validate_ranges(ranges: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    ordered = sorted(ranges)
    cursor = 0
    for start, stop in ordered:
        if (
            type(start) is not int
            or type(stop) is not int
            or start != cursor
            or not start < stop <= count
        ):
            raise ValueError("Shard ranges must cover the region without gaps or overlap")
        cursor = stop
    if cursor != count:
        raise ValueError("Shard ranges do not cover the complete region")
    return ordered


def _unique(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key: " + key)
        value[key] = item
    return value


def _json(path: Path | str) -> dict:
    value = json.loads(Path(path).read_text(), object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise ValueError("Prithvi JSON root must be an object")
    return value


def _ref(path: Path | str) -> dict:
    return {"path": str(path), "sha256": sha(path)}


def _verified(value: dict) -> Path:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256"}
        or not isinstance(value["path"], str)
        or not Path(value["path"]).is_absolute()
        or sha(value["path"]) != value["sha256"]
    ):
        raise ValueError("Prithvi source identity changed or is not self-contained")
    return Path(value["path"])


def read_spec(path: Path | str) -> tuple[dict, tuple, prithvi.PrithviEncoderConfig, dict]:
    spec = _json(path)
    required = {
        "schema",
        "monthly_spec",
        "weights",
        "model_config",
        "output",
        "seed",
        "fit_samples",
        "batch_size",
        "reserve_gib",
        "inference_dtype",
        "output_shape",
        "fit_min_frames",
        "output_min_coverage",
    }
    if set(spec) != required or spec["schema"] != "prithvi-public-features-v1":
        raise ValueError("Unknown or incomplete Prithvi feature specification")
    if (
        any(type(spec[k]) is not int for k in ["seed", "fit_samples", "batch_size", "reserve_gib"])
        or spec["fit_samples"] < 64
        or not 1 <= spec["batch_size"] <= 64
        or spec["reserve_gib"] < 0
        or spec["inference_dtype"] not in ["float32", "bfloat16"]
        or not isinstance(spec["output"], str)
        or not Path(spec["output"]).is_absolute()
        or not isinstance(spec["output_shape"], list)
        or len(spec["output_shape"]) != 2
        or any(type(n) is not int or n < 1 for n in spec["output_shape"])
        or any(
            type(spec[k]) not in [int, float] or not math.isfinite(spec[k])
            for k in ["fit_min_frames", "output_min_coverage"]
        )
        or not 0 < spec["fit_min_frames"] <= 12
        or not 0 <= spec["output_min_coverage"] < 1
    ):
        raise ValueError("Invalid Prithvi feature parameters")
    _verified(spec["weights"])
    published = _json(_verified(spec["model_config"]))
    fields = {
        "img_size",
        "num_frames",
        "patch_size",
        "in_chans",
        "embed_dim",
        "depth",
        "num_heads",
        "decoder_embed_dim",
        "decoder_depth",
        "decoder_num_heads",
        "mlp_ratio",
        "coords_encoding",
        "coords_scale_learn",
        "mask_ratio",
        "norm_pix_loss",
        "bands",
        "mean",
        "std",
        "origin_url",
        "paper_ids",
    }
    cfg = published.get("pretrained_cfg", {})
    if not isinstance(cfg, dict):
        raise ValueError("Invalid published Prithvi configuration")
    if (
        set(published) != {"architecture", "num_features", "pretrained_cfg"}
        or set(cfg) != fields
        or cfg["coords_encoding"]
        or cfg["coords_scale_learn"] is not False
        or cfg["in_chans"] != 6
        or cfg["bands"] != ["B02", "B03", "B04", "B05", "B06", "B07"]
        or cfg["patch_size"] != [1, 16, 16]
        or published["num_features"] != cfg["embed_dim"]
        or any(len(cfg[k]) != 6 or not np.isfinite(cfg[k]).all() for k in ["mean", "std"])
        or min(cfg["std"]) <= 0
    ):
        raise ValueError("Unsupported published non-TL Prithvi configuration")
    model = prithvi.PrithviEncoderConfig(
        image_size=cfg["img_size"],
        num_frames=cfg["num_frames"],
        in_channels=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        patch_size=tuple(cfg["patch_size"]),
        mlp_ratio=cfg["mlp_ratio"],
    )
    return spec, monthly.read_inputs(_verified(spec["monthly_spec"])), model, cfg


@contextmanager
def _lock(path: Path, *, blocking: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def _owned_root(path: Path | str, spec: dict) -> Path:
    root = Path(spec["output"])
    identity = {
        "spec_sha256": sha(path),
        "implementation_sha256": sha(__file__),
        "encoder_sha256": sha(prithvi.__file__),
        "monthly_reader_sha256": sha(monthly.__file__),
    }
    root.mkdir(parents=True, exist_ok=True)
    header = root / "identity.json"
    ownership_lock = root / "ownership.lock"
    if header.exists() and _json(header) != identity:
        raise ValueError("Prithvi export identity changed")
    if not header.exists() and not ownership_lock.exists() and any(root.iterdir()):
        raise ValueError("Unowned Prithvi output directory")
    with _lock(ownership_lock, blocking=True):
        if header.exists():
            if _json(header) != identity:
                raise ValueError("Prithvi export identity changed")
        elif any(p.name != "ownership.lock" for p in root.iterdir()):
            raise ValueError("Unowned Prithvi output directory")
        else:
            dump(header, identity)
    return root


def _reserve(root: Path, spec: dict, extra: int = 0) -> None:
    if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30 + extra:
        raise RuntimeError("Prithvi feature export would exceed disk reserve")


def load_model(spec: dict, config: prithvi.PrithviEncoderConfig, device: torch.device):
    model = prithvi.PrithviEncoder(config)
    weights = torch.load(spec["weights"]["path"], map_location="cpu", weights_only=True, mmap=True)
    receipt = prithvi.load_encoder_state(model, weights)
    model.eval().requires_grad_(False).to(device)
    return model, receipt


def _prefix(array: np.ndarray, count: int) -> str:
    return hashlib.sha256(np.ascontiguousarray(array[:count]).tobytes()).hexdigest()


def _checked_array(path: Path, shape: tuple, dtype: np.dtype) -> np.ndarray:
    value = np.load(path, mmap_mode="r+")
    if value.shape != shape or value.dtype != dtype:
        raise ValueError("Prithvi progress array shape or dtype changed")
    return value


def _checked_shard(root: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    state = _json(root / "status.json")
    if state["stage"] != "complete":
        raise ValueError("Incomplete Prithvi shard; extraction must finish before projection")
    for key in ["native", "counts"]:
        if sha(root / (key + ".npy")) != state[key + "_sha256"]:
            raise ValueError("Completed Prithvi shard changed")
    native, counts = (np.load(root / (k + ".npy"), mmap_mode="r") for k in ["native", "counts"])
    expected = (state["stop"] - state["start"], state["channels"], *state["native_grid"])
    if (
        native.shape != expected
        or counts.shape != (expected[0], 1, *expected[-2:])
        or native.dtype != np.float16
        or counts.dtype != np.float32
        or state["completed"] != expected[0]
    ):
        raise ValueError("Completed Prithvi shard shape changed")
    return state, native, counts


def extract(path: Path | str, *, device_name: str, start: int, stop: int) -> Path:
    spec, inputs, config, cfg = read_spec(path)
    _, observations, records, stats, months = inputs
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(records):
        raise ValueError("Invalid Prithvi extraction range")
    root = _owned_root(path, spec)
    shard = root / "shards" / f"{start:06d}-{stop:06d}"
    with _lock(shard / "extract.lock"):
        status_path = shard / "status.json"
        state = _json(status_path) if status_path.exists() else {}
        if state.get("stage") == "complete":
            _checked_shard(shard)
            return shard
        first, _ = read_public_s2(
            records[start], observations, stats, months, cfg["mean"], cfg["std"]
        )
        t, h, w = first.shape[1:]
        if h % 16 or w % 16:
            raise ValueError("Public source shape must be divisible by native patches")
        size = stop - start
        shape = (size, config.embed_dim, h // 16, w // 16)
        completed = state.get("completed", 0)
        if type(completed) is not int or not 0 <= completed <= size:
            raise ValueError("Invalid Prithvi resume count")
        _reserve(root, spec, 0 if completed else int(np.prod(shape)) * 2 + size * h * w // 64)
        arrays = []
        for key, wanted, dtype in [
            ("native", shape, np.dtype("float16")),
            ("counts", (size, 1, h // 16, w // 16), np.dtype("float32")),
        ]:
            pending = shard / (key + ".partial.npy")
            if completed:
                chosen = pending if pending.exists() else shard / (key + ".npy")
                if chosen.suffix == ".npy" and not pending.exists() and completed != size:
                    raise ValueError("Unexpected finalized partial Prithvi shard")
                a = _checked_array(chosen, wanted, dtype)
                if _prefix(a, completed) != state[key + "_prefix_sha256"]:
                    raise ValueError("Corrupt committed Prithvi progress")
            else:
                a = np.lib.format.open_memmap(pending, mode="w+", dtype=dtype, shape=wanted)
            arrays.append(a)
        native, counts = arrays
        consumption = state.get("encoder_consumption")
        totals = np.asarray(state.get("valid_pixels_per_month", [0] * 12), np.int64)
        if completed < size:
            device, _, _ = _setup_device(device_name)
            actual_dtype = (
                "bfloat16"
                if device.type in ["npu", "cuda"] and spec["inference_dtype"] == "bfloat16"
                else "float32"
            )
            if completed and (
                state["actual_inference_dtype"] != actual_dtype
                or state["device_type"] != device.type
            ):
                raise ValueError("Prithvi progress numerical mode changed")
            if device.type == "npu":
                torch.npu.config.allow_internal_format = False
            model, consumption = load_model(spec, config, device)
            with torch.inference_mode():
                for offset in range(completed, size, spec["batch_size"]):
                    _reserve(root, spec)
                    end = min(size, offset + spec["batch_size"])
                    samples = [
                        read_public_s2(
                            records[i], observations, stats, months, cfg["mean"], cfg["std"]
                        )
                        for i in range(start + offset, start + end)
                    ]
                    x, valid = (torch.stack([s[j] for s in samples]) for j in [0, 1])
                    if tuple(x.shape[1:]) != (6, t, h, w):
                        raise ValueError("Public source shape changes within a region")
                    with _autocast(device, spec["inference_dtype"] == "bfloat16"):
                        features = model(x.to(device)).float().cpu()
                    pooled, area = pool_temporal(features, valid)
                    stored = pooled.numpy().astype(np.float16)
                    if not np.isfinite(stored).all():
                        raise FloatingPointError("Nonfinite stored Prithvi features")
                    native[offset:end], counts[offset:end] = stored, area.numpy()
                    totals += valid.sum((0, 2, 3)).numpy()
                    native.flush()
                    counts.flush()
                    state = {
                        "stage": "extracting",
                        "start": start,
                        "stop": stop,
                        "completed": end,
                        "channels": config.embed_dim,
                        "native_grid": [h // 16, w // 16],
                        "source_shape": [t, h, w],
                        "months": months,
                        "valid_pixels_per_month": totals.tolist(),
                        "encoder_consumption": consumption,
                        "actual_inference_dtype": actual_dtype,
                        "device_type": device.type,
                        "native_prefix_sha256": _prefix(native, end),
                        "counts_prefix_sha256": _prefix(counts, end),
                    }
                    dump(status_path, state)
        for key in ["native", "counts"]:
            pending = shard / (key + ".partial.npy")
            if pending.exists():
                pending.replace(shard / (key + ".npy"))
        state.update(
            stage="complete",
            native_sha256=sha(shard / "native.npy"),
            counts_sha256=sha(shard / "counts.npy"),
        )
        dump(status_path, state)
    return shard


def _sample(shards: list[tuple], threshold: float, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    buffers, priorities, count = [], [], 0
    for _, native, counts in shards:
        for i in range(len(native)):
            positions = np.flatnonzero(counts[i, 0].ravel() >= threshold)
            if not len(positions):
                continue
            values = np.asarray(native[i], np.float32).reshape(native.shape[1], -1)[:, positions].T
            buffers.append(values)
            priorities.append(rng.random(len(values)))
            count += len(values)
            if count > 2 * cap:
                all_values, keys = np.concatenate(buffers), np.concatenate(priorities)
                chosen = np.argpartition(keys, cap - 1)[:cap]
                buffers, priorities, count = [all_values[chosen]], [keys[chosen]], cap
    if not buffers:
        raise ValueError("No observed Prithvi patch samples for regional fitting")
    values, keys = np.concatenate(buffers), np.concatenate(priorities)
    if len(values) > cap:
        values = values[np.argpartition(keys, cap - 1)[:cap]]
    if len(values) < 64 or not np.isfinite(values).all():
        raise ValueError("Insufficient finite Prithvi samples for 64D projection")
    return values


def project(path: Path | str) -> Path:
    spec, inputs, config, _ = read_spec(path)
    root = _owned_root(path, spec)
    _, observations, records, _, months = inputs
    index = _json(_verified(_json(spec["monthly_spec"]["path"])["index"]))
    with _lock(root / "project.lock"):
        status_path = root / "status.json"
        if status_path.exists() and _json(status_path).get("stage") == "complete":
            saved = _json(status_path)
            for key in ["annual", "projection", "index"]:
                if sha(root / saved[key + "_file"]) != saved[key + "_sha256"]:
                    raise ValueError("Completed Prithvi projection changed")
            return root / "annual.npy"
        directories = sorted(p for p in (root / "shards").glob("*") if p.is_dir())
        if not directories:
            raise ValueError("Extract all Prithvi shards before regional projection")
        shards = [_checked_shard(p) for p in directories]
        ordered = validate_ranges([(s[0]["start"], s[0]["stop"]) for s in shards], len(records))
        by_start = {s[0]["start"]: s for s in shards}
        shards = [by_start[start] for start, _ in ordered]
        shapes = {tuple(s[0]["source_shape"]) for s in shards}
        if len(shapes) != 1 or next(iter(shapes))[0] != len(months):
            raise ValueError("Prithvi shard source/calendar grids differ")
        modes = {(s[0]["actual_inference_dtype"], s[0]["device_type"]) for s in shards}
        if len(modes) != 1:
            raise ValueError("Prithvi shards use different numerical modes")
        for state, native, counts in shards:
            if (
                state["months"] != months
                or native.shape[1] != config.embed_dim
                or not np.isfinite(native).all()
                or not np.isfinite(counts).all()
                or (counts < 0).any()
                or (counts > len(months)).any()
            ):
                raise ValueError("Invalid completed Prithvi shard observations")
        sample = _sample(shards, spec["fit_min_frames"], spec["fit_samples"], spec["seed"])
        dump(status_path, {"stage": "regional_projection", "samples": len(sample)})
        pca = PCA(n_components=64, svd_solver="randomized", random_state=spec["seed"])
        with threadpool_limits(limits=4):
            pca.fit(sample)
        np.savez_compressed(
            root / "projection.npz",
            mean=pca.mean_,
            components=pca.components_,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )
        t, h, w = next(iter(shapes))
        oh, ow = spec["output_shape"]
        shape = (len(records), 64, oh, ow)
        _reserve(root, spec, int(np.prod(shape)) * 2)
        pending = root / "annual.partial.npy"
        output = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float16, shape=shape)
        for state, native, counts in shards:
            for i in range(len(native)):
                _reserve(root, spec)
                a = np.asarray(native[i], np.float32)
                with threadpool_limits(limits=4):
                    value = pca.transform(a.reshape(config.embed_dim, -1).T).T
                value = torch.from_numpy(value.reshape(1, 64, *a.shape[-2:]))
                coverage = torch.from_numpy(np.asarray(counts[i : i + 1]).copy()) / t
                numerator = align_patch_features(value * coverage, (h, w), (oh, ow), patch=16)
                denominator = align_patch_features(coverage, (h, w), (oh, ow), patch=16)
                result = (
                    torch.where(
                        denominator > spec["output_min_coverage"],
                        numerator / denominator.clamp_min(1e-6),
                        0,
                    )[0]
                    .numpy()
                    .astype(np.float16)
                )
                if not np.isfinite(result).all():
                    raise FloatingPointError("Nonfinite stored regional Prithvi projection")
                output[state["start"] + i] = result
        output.flush()
        pending.replace(root / "annual.npy")
        dump(root / "index.json", {**index, "months": months})
        dump(
            status_path,
            {
                "stage": "complete",
                "shape": list(shape),
                "fit_samples": len(sample),
                "per_region_projection": True,
                "map_labels_used": False,
                "radiometry_application": "monthly reader exactly once",
                "bands": list(S2_BANDS),
                "physical_scale_to_model": 10000,
                "months": months,
                "source_grid": [h, w],
                "native_patch_pixels": 16,
                "model_inputs": "S2 only; other sources remain separate fusion inputs",
                "cloud_attention_mask_supported": False,
                "observed_temporal_pooling": True,
                "per_pixel_unit_normalization": False,
                "requested_inference_dtype": spec["inference_dtype"],
                "inference_dtype": next(iter(modes))[0],
                "device_type": next(iter(modes))[1],
                "torch_version": torch.__version__,
                "shard_provenance": [
                    {
                        "start": state["start"],
                        "stop": state["stop"],
                        "native_sha256": state["native_sha256"],
                        "counts_sha256": state["counts_sha256"],
                        "valid_pixels_per_month": state["valid_pixels_per_month"],
                    }
                    for state, _, _ in shards
                ],
                "annual_file": "annual.npy",
                "annual_sha256": sha(root / "annual.npy"),
                "projection_file": "projection.npz",
                "projection_sha256": sha(root / "projection.npz"),
                "index_file": "index.json",
                "index_sha256": sha(root / "index.json"),
                "weights": spec["weights"],
                "source_observations": observations["inputs"],
                "explained_variance_64": float(pca.explained_variance_ratio_.sum()),
            },
        )
    return root / "annual.npy"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv experiment prithvi-features")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["extract", "project"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.phase == "extract":
        if args.start is None or args.stop is None:
            parser.error("extract requires an explicit start and stop")
        extract(args.spec, device_name=args.device, start=args.start, stop=args.stop)
    else:
        if args.start is not None or args.stop is not None:
            parser.error("project always fits the complete region, not a shard")
        project(args.spec)
    return 0
