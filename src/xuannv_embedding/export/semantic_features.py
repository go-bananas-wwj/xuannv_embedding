"""Frozen satellite-vision features with explicit radiometry and patch-center geometry."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.cli import _setup_device
from xuannv_embedding.training.runtime import _autocast


def restore_rgb(image, mask, source_mean, source_std, model_mean, model_std):
    def channel(values):
        return image.new_tensor(values)[:, None, None]

    rgb = (image.float() * channel(source_std) + channel(source_mean)).clamp(0, 255) / 255
    clean = torch.where(mask.bool(), rgb, channel(model_mean))
    return (clean - channel(model_mean)) / channel(model_std)


def input_reduction(spec):
    factor = spec.get("input_reduction", 1)
    if type(factor) is not int or factor not in (1, 2, 4, 8):
        raise ValueError("Semantic input reduction must be one of 1, 2, 4, 8")
    return factor


def reduce_observed_rgb(image, mask, factor):
    """Expand physical window context by averaging only observed source pixels.

    The complete footprint is retained. Fractional masks record observed area;
    the default returns the original tensors without altering legacy numerics.
    """
    input_reduction({"input_reduction": factor})
    if image.ndim != 3 or image.shape[0] != 3 or mask.shape != (1, *image.shape[-2:]):
        raise ValueError("Semantic RGB/mask shapes differ")
    if not torch.isfinite(mask).all() or ((mask < 0) | (mask > 1)).any():
        raise ValueError("Semantic observation mask must be finite in [0, 1]")
    if not torch.isfinite(image[:, mask[0] > 0]).all():
        raise ValueError("Observed semantic RGB values must be finite")
    if any(n < factor or n % factor for n in image.shape[-2:]):
        raise ValueError("Semantic input extent must divide exactly by the reduction")
    if factor == 1:
        return image, mask
    weights = mask.float()
    observed = torch.where(weights > 0, image.float(), 0)
    coverage = F.avg_pool2d(weights, factor)
    numerator = F.avg_pool2d(observed * weights, factor)
    values = torch.where(coverage > 0, numerator / coverage.clamp_min(1e-12), 0)
    return values, coverage


def align_patch_features(values, source_shape, output_shape, *, patch):
    h, w = source_shape
    if values.ndim != 4 or values.shape[-2:] != (math.ceil(h / patch), math.ceil(w / patch)):
        raise ValueError("Native patch grid does not match the padded source footprint")
    oh, ow = output_shape
    yy = (torch.arange(oh, device=values.device).float() + 0.5) * h / oh
    xx = (torch.arange(ow, device=values.device).float() + 0.5) * w / ow
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack(
        (2 * xx / (values.shape[-1] * patch) - 1, 2 * yy / (values.shape[-2] * patch) - 1), dim=-1
    )
    return F.grid_sample(
        values.float(),
        grid[None].expand(len(values), -1, -1, -1),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def window_starts(length: int, side: int) -> list[int]:
    if side < 16 or side > length or length % 16 or side % 16:
        raise ValueError("Semantic windows must follow the native patch grid")
    stride = max(16, (side * 3 // 4) // 16 * 16)
    return sorted(set([*range(0, length - side + 1, stride), length - side]))


def stitch_patch_windows(pieces, shape, *, patch):
    if not pieces:
        raise ValueError("Missing semantic window coverage")
    if len(pieces) == 1 and pieces[0][:2] == (0, 0) and pieces[0][2].shape[-2:] == shape:
        return pieces[0][2].float()
    total = torch.zeros((pieces[0][2].shape[0], *shape), dtype=torch.float32)
    denominator = torch.zeros((1, *shape), dtype=torch.float32)
    for y, x, values in pieces:
        if y % patch or x % patch:
            raise ValueError("Semantic window offset is not patch-aligned")
        h, w = values.shape[-2:]
        wy = (1 - torch.linspace(-1, 1, h).abs()).clamp_min(0.1)
        wx = (1 - torch.linspace(-1, 1, w).abs()).clamp_min(0.1)
        weight = wy[:, None] * wx[None, :]
        sy, sx = y // patch, x // patch
        total[:, sy : sy + h, sx : sx + w] += values.float() * weight
        denominator[:, sy : sy + h, sx : sx + w] += weight
    if (denominator <= 0).any():
        raise ValueError("Incomplete semantic window coverage")
    return total / denominator


def read_aerial_window(path, record):
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.Env(GDAL_CACHEMAX=128 * 1024**2), rasterio.open(path) as source:
        if str(source.crs) != record["crs"] or source.count != 3:
            raise ValueError("Aerial raster CRS/channel contract differs")
        window = from_bounds(*record["bounds"], transform=source.transform)
        aligned = window.round_offsets().round_lengths()
        if any(abs(a - b) > 1e-6 for a, b in zip(window.flatten(), aligned.flatten())):
            raise ValueError("Aerial record is not aligned to native source pixels")
        values = torch.from_numpy(source.read(window=aligned, out_dtype="float32"))
        valid = torch.from_numpy(source.dataset_mask(window=aligned) > 0)
        if values.shape[-2:] != (int(aligned.height), int(aligned.width)):
            raise ValueError("Aerial raster does not cover the requested tile extent")
        mask = valid & torch.isfinite(values).all(0)
        return values, mask[None].float()


def external_output_shape(spec):
    shape = spec.get("output_shape")
    if shape is None:
        return None
    if "raster" not in spec and "rasters" not in spec:
        raise ValueError("An explicit output shape requires external raster inputs")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(v) is not int or v < 1 for v in shape)
    ):
        raise ValueError("Invalid external raster output shape")
    return tuple(shape)


def registered_rasters(spec, cache):
    if "rasters" not in spec:
        return None
    ref = spec["rasters"]
    if set(ref) != {"path", "sha256"} or sha(ref["path"]) != ref["sha256"]:
        raise ValueError("Raster manifest identity changed")
    manifest = json.loads(Path(ref["path"]).read_text())
    if (
        set(manifest) not in ({"schema", "records"}, {"schema", "records", "dependencies"})
        or manifest["schema"] != "semantic-raster-inputs-v1"
        or len(manifest["records"]) != len(cache["records"])
    ):
        raise ValueError("Invalid semantic raster manifest")
    for dependency in manifest.get("dependencies", []):
        if set(dependency) != {"path", "sha256"} or sha(dependency["path"]) != dependency["sha256"]:
            raise ValueError("Semantic raster dependency identity changed")
    paths, seen = [], set()
    for record, item in zip(cache["records"], manifest["records"]):
        if (
            set(item) != {"patch_id", "bounds", "crs", "raster"}
            or any(item[k] != record[k] for k in ["patch_id", "bounds", "crs"])
            or item["patch_id"] in seen
        ):
            raise ValueError("Semantic raster geography or order differs")
        source = item["raster"]
        if set(source) != {"path", "sha256"} or sha(source["path"]) != source["sha256"]:
            raise ValueError("Semantic raster identity changed")
        paths.append(source["path"])
        seen.add(item["patch_id"])
    return paths


def read_spec(path: Path):
    spec = json.loads(path.read_text())
    expected = {
        "schema",
        "cache",
        "statistics",
        "weights",
        "model",
        "source",
        "output",
        "seed",
        "fit_samples",
        "components",
    }
    if (
        not expected.issubset(spec)
        or set(spec)
        - expected
        - {
            "raster",
            "rasters",
            "raster_mask",
            "max_pixels",
            "crop_batch",
            "output_shape",
            "input_reduction",
        }
        or spec["schema"] != "regional-semantic-features-v1"
    ):
        raise ValueError("Unknown semantic feature specification")
    input_reduction(spec)
    for name in [
        "cache",
        "statistics",
        "weights",
        *[k for k in ["raster", "raster_mask"] if k in spec],
    ]:
        ref = spec[name]
        if set(ref) != {"path", "sha256"} or sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Semantic feature source identity changed")
    side, batch = spec.get("max_pixels", 512), spec.get("crop_batch", 4)
    if (
        type(side) is not int
        or not 128 <= side <= 1024
        or side % 16
        or type(batch) is not int
        or not 1 <= batch <= 8
    ):
        raise ValueError("Invalid semantic window size or batch")
    if (
        spec["components"] != 64
        or type(spec["fit_samples"]) is not int
        or spec["fit_samples"] < 64
        or type(spec["seed"]) is not int
    ):
        raise ValueError("Semantic projection requires 64 components and sufficient samples")
    cache = json.loads(Path(spec["cache"]["path"]).read_text())
    if cache["state"] != "complete" or not cache["records"]:
        raise ValueError("Incomplete input cache")
    if "raster" in spec and "rasters" in spec:
        raise ValueError("Single and per-tile raster inputs are mutually exclusive")
    external_output_shape(spec)
    if "rasters" in spec and "output_shape" not in spec:
        raise ValueError("Per-tile raster inputs require an explicit output shape")
    registered_rasters(spec, cache)
    return spec, cache


def extract_native(spec, cache, *, device_name):
    import timm
    from timm.models import load_checkpoint

    root, source = Path(spec["output"]), spec["source"]
    rasters = registered_rasters(spec, cache)
    if "raster" in spec or rasters is not None:
        path = rasters[0] if rasters is not None else spec["raster"]["path"]
        image, _ = read_aerial_window(path, cache["records"][0])
        shape = image.shape
    else:
        first = torch.load(cache["records"][0]["path"], weights_only=True, mmap=True)
        shape = first["highres_frames"][source].shape
    if len(shape) != 3 or shape[0] != 3:
        raise ValueError("Satellite vision extraction requires one static RGB acquisition")
    original_shape = tuple(shape[-2:])
    reduction = input_reduction(spec)
    if any(n < reduction or n % reduction for n in original_shape):
        raise ValueError("Semantic input extent must divide exactly by the reduction")
    shape = (3, *(n // reduction for n in original_shape))
    h, w = shape[-2:]
    gh, gw = math.ceil(h / 16), math.ceil(w / 16)
    crop_h = min(gh * 16, spec.get("max_pixels", 512))
    crop_w = min(gw * 16, spec.get("max_pixels", 512))
    native_shape = (len(cache["records"]), 1024, gh, gw)
    model = timm.create_model(
        spec["model"], pretrained=False, img_size=(crop_h, crop_w), dynamic_img_size=True
    )
    load_checkpoint(model, spec["weights"]["path"], strict=True)
    if model.num_features != 1024 or tuple(model.patch_embed.patch_size) != (16, 16):
        raise ValueError("Unexpected satellite encoder feature/patch contract")
    stats = json.loads(Path(spec["statistics"]["path"]).read_text())
    device, _, _ = _setup_device(device_name)
    if device.type == "npu":
        torch.npu.config.allow_internal_format = False
    model.to(device).eval()
    native_file, mask_file = root / "native.partial.npy", root / "native-mask.partial.npy"
    status_path = root / "native-status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    start = previous.get("completed", 0)
    mode = "r+" if start else "w+"
    native = np.lib.format.open_memmap(native_file, mode=mode, dtype=np.float16, shape=native_shape)
    masks = np.lib.format.open_memmap(
        mask_file, mode=mode, dtype=np.float32, shape=(len(native), 1, gh, gw)
    )
    with torch.inference_mode():
        for i in range(start, len(native)):
            record = cache["records"][i]
            if "raster" in spec or rasters is not None:
                path = rasters[i] if rasters is not None else spec["raster"]["path"]
                image, mask = read_aerial_window(path, record)
                source_mean, source_std = [0, 0, 0], [1, 1, 1]
            else:
                if sha(record["path"]) != record["sha256"]:
                    raise ValueError("Registered aerial sample changed")
                sample = torch.load(record["path"], weights_only=True, mmap=True)
                image, mask = sample["highres_frames"][source], sample["highres_masks"][source]
                source_mean, source_std = stats["mean"], stats["std"]
            if image.shape[-2:] != original_shape:
                raise ValueError("Static aerial source extent differs")
            image, mask = reduce_observed_rgb(image, mask, reduction)
            if image.shape != shape or mask.shape != (1, h, w):
                raise ValueError("Static aerial sample shape differs")
            pad = (0, gw * 16 - w, 0, gh * 16 - h)
            fraction = F.avg_pool2d(F.pad(mask[None].float(), pad), 16)
            masks[i] = fraction[0].numpy()
            if mask.any():
                rgb = restore_rgb(
                    image,
                    mask,
                    source_mean,
                    source_std,
                    model.pretrained_cfg["mean"],
                    model.pretrained_cfg["std"],
                )
                rgb = F.pad(rgb[None], pad, mode="reflect")
                offsets = [
                    (y, x)
                    for y in window_starts(gh * 16, crop_h)
                    for x in window_starts(gw * 16, crop_w)
                ]
                pieces = []
                for start_crop in range(0, len(offsets), spec.get("crop_batch", 4)):
                    selected = offsets[start_crop : start_crop + spec.get("crop_batch", 4)]
                    inputs = torch.cat(
                        [rgb[:, :, y : y + crop_h, x : x + crop_w] for y, x in selected]
                    ).to(device)
                    with _autocast(device, device.type == "npu"):
                        values = model.forward_features(inputs)[:, model.num_prefix_tokens :]
                    values = (
                        values.transpose(1, 2)
                        .reshape(-1, 1024, crop_h // 16, crop_w // 16)
                        .float()
                        .cpu()
                    )
                    pieces.extend((y, x, values[j]) for j, (y, x) in enumerate(selected))
                values = stitch_patch_windows(pieces, (gh, gw), patch=16)[None]
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Nonfinite satellite semantic features")
                native[i] = values[0].cpu().numpy().astype(np.float16)
            else:
                native[i] = 0
            if (i + 1) % 10 == 0 or i + 1 == len(native):
                native.flush()
                masks.flush()
                dump(
                    status_path,
                    {
                        "stage": "extracting",
                        "completed": i + 1,
                        "total": len(native),
                        "native_shape": list(native_shape),
                        "source_shape": [h, w],
                        "original_source_shape": list(original_shape),
                        "input_reduction": reduction,
                        "prefix_tokens_removed": model.num_prefix_tokens,
                        "normalization_mean": list(model.pretrained_cfg["mean"]),
                        "normalization_std": list(model.pretrained_cfg["std"]),
                        "window_shape": [crop_h, crop_w],
                        "crop_batch": spec.get("crop_batch", 4),
                    },
                )
    native.flush()
    masks.flush()
    native_file.replace(root / "native.npy")
    mask_file.replace(root / "native-mask.npy")
    dump(
        status_path,
        {
            "stage": "complete",
            "completed": len(native),
            "source_shape": [h, w],
            "original_source_shape": list(original_shape),
            "input_reduction": reduction,
            "native_shape": list(native_shape),
            "prefix_tokens_removed": model.num_prefix_tokens,
            "normalization_mean": list(model.pretrained_cfg["mean"]),
            "normalization_std": list(model.pretrained_cfg["std"]),
            "window_shape": [crop_h, crop_w],
            "crop_batch": spec.get("crop_batch", 4),
        },
    )


def project_native(spec, cache):
    root = Path(spec["output"])
    native = np.load(root / "native.npy", mmap_mode="r")
    masks = np.load(root / "native-mask.npy", mmap_mode="r")
    chunks = []
    for i in range(len(native)):
        valid = np.flatnonzero(masks[i, 0] >= 0.9)
        rng = np.random.default_rng(spec["seed"] + i * 997)
        selected = rng.choice(valid, min(256, len(valid)), replace=False)
        chunks.append(np.asarray(native[i], np.float32).reshape(1024, -1)[:, selected].T)
    samples = np.concatenate(chunks)
    rng = np.random.default_rng(spec["seed"])
    if len(samples) > spec["fit_samples"]:
        samples = samples[rng.choice(len(samples), spec["fit_samples"], replace=False)]
    if len(samples) < 64:
        raise ValueError("Insufficient unlabelled aerial tokens for PCA")
    dump(root / "status.json", {"stage": "unlabelled_projection", "samples": len(samples)})
    projection = PCA(n_components=64, svd_solver="randomized", random_state=spec["seed"])
    with threadpool_limits(limits=8):
        projection.fit(samples)
    np.savez_compressed(
        root / "projection.npz",
        mean=projection.mean_,
        components=projection.components_,
        explained_variance_ratio=projection.explained_variance_ratio_,
    )
    source_shape = json.loads((root / "native-status.json").read_text())["source_shape"]
    output_shape = external_output_shape(spec)
    if output_shape is None:
        first = torch.load(cache["records"][0]["path"], weights_only=True, mmap=True)
        output_shape = next(iter(first["source_frames"].values())).shape[-2:]
    shape = (len(native), 64, *output_shape)
    path = root / "annual.partial.npy"
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    for i in range(len(native)):
        values = np.asarray(native[i], np.float32)
        values = projection.transform(values.reshape(1024, -1).T).T.reshape(64, *values.shape[-2:])
        coverage = torch.from_numpy(np.asarray(masks[i : i + 1]).copy())
        numerator = align_patch_features(
            torch.from_numpy(values)[None] * coverage, source_shape, output_shape, patch=16
        )
        denominator = align_patch_features(coverage, source_shape, output_shape, patch=16)
        result = F.normalize(numerator / denominator.clamp_min(1e-6), dim=1)
        result = torch.where(denominator > 0.05, result, 0)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite projected semantic features")
        output[i] = result[0].numpy().astype(np.float16)
    output.flush()
    path.replace(root / "annual.npy")
    dump(root / "index.json", {"months": cache["months"], "records": cache["records"]})
    dump(
        root / "status.json",
        {
            "stage": "complete",
            "shape": list(shape),
            "fit_samples": len(samples),
            "label_data_used": False,
            "explained_variance_64": float(projection.explained_variance_ratio_.sum()),
            "weights_sha256": spec["weights"]["sha256"],
            "annual_sha256": sha(root / "annual.npy"),
            "pretraining": "DINOv3 SAT-493M global satellite pretraining, not regional scratch",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment semantic-features")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="npu:1")
    parser.add_argument("--phase", choices=["all", "extract", "project"], default="all")
    args = parser.parse_args(argv)
    spec, cache = read_spec(args.spec)
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {"spec_sha256": sha(args.spec)}
    if (root / "identity.json").exists() and json.loads(
        (root / "identity.json").read_text()
    ) != identity:
        raise ValueError("Semantic feature recipe changed")
    dump(root / "identity.json", identity)
    torch.set_num_threads(2)
    with (root / "extract.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase != "extract" and (root / "annual.npy").exists():
            previous = (
                json.loads((root / "status.json").read_text())
                if (root / "status.json").exists()
                else {}
            )
            if previous.get("stage") == "complete":
                if previous.get("annual_sha256") != sha(root / "annual.npy"):
                    raise ValueError("Completed semantic features changed")
                return 0
        if args.phase != "project" and not (root / "native.npy").exists():
            extract_native(spec, cache, device_name=args.device)
        if (root / "native-mask.partial.npy").exists() and not (root / "native-mask.npy").exists():
            (root / "native-mask.partial.npy").replace(root / "native-mask.npy")
        if not (root / "native.npy").exists() or not (root / "native-mask.npy").exists():
            raise ValueError("Native features and masks must be extracted before projection")
        native_status = json.loads((root / "native-status.json").read_text())
        native = np.load(root / "native.npy", mmap_mode="r")
        masks = np.load(root / "native-mask.npy", mmap_mode="r")
        if (
            native_status.get("completed") != len(cache["records"])
            or len(native) != len(cache["records"])
            or native.ndim != 4
            or native.shape[1] != 1024
            or masks.shape != (len(native), 1, *native.shape[-2:])
        ):
            raise ValueError("Native extraction is incomplete or its geometry differs")
        if native_status.get("stage") != "complete":
            # Both files were atomically published after the final flushed row.
            dump(root / "native-status.json", {**native_status, "stage": "complete"})
        if args.phase == "extract":
            return 0
        complete = (
            json.loads((root / "status.json").read_text()).get("stage") == "complete"
            if (root / "status.json").exists()
            else False
        )
        if not (root / "annual.npy").exists() or not complete:
            project_native(spec, cache)
    return 0
