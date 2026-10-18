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
    if set(spec) != expected or spec["schema"] != "regional-semantic-features-v1":
        raise ValueError("Unknown semantic feature specification")
    for name in ["cache", "statistics", "weights"]:
        ref = spec[name]
        if set(ref) != {"path", "sha256"} or sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Semantic feature source identity changed")
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
    return spec, cache


def extract_native(spec, cache, *, device_name):
    import timm
    from timm.models import load_checkpoint

    root, source = Path(spec["output"]), spec["source"]
    first = torch.load(cache["records"][0]["path"], weights_only=True, mmap=True)
    shape = first["highres_frames"][source].shape
    if len(shape) != 3 or shape[0] != 3:
        raise ValueError("Satellite vision extraction requires one static RGB acquisition")
    h, w = shape[-2:]
    gh, gw = math.ceil(h / 16), math.ceil(w / 16)
    native_shape = (len(cache["records"]), 1024, gh, gw)
    model = timm.create_model(
        spec["model"], pretrained=False, img_size=(gh * 16, gw * 16), dynamic_img_size=True
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
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("Registered aerial sample changed")
            sample = torch.load(record["path"], weights_only=True, mmap=True)
            image, mask = sample["highres_frames"][source], sample["highres_masks"][source]
            if image.shape != shape or mask.shape != (1, h, w):
                raise ValueError("Static aerial sample shape differs")
            pad = (0, gw * 16 - w, 0, gh * 16 - h)
            fraction = F.avg_pool2d(F.pad(mask[None].float(), pad), 16)
            masks[i] = fraction[0].numpy()
            if mask.any():
                rgb = restore_rgb(
                    image,
                    mask,
                    stats["mean"],
                    stats["std"],
                    model.pretrained_cfg["mean"],
                    model.pretrained_cfg["std"],
                )
                rgb = F.pad(rgb[None], pad, mode="reflect").to(device)
                with _autocast(device, device.type == "npu"):
                    values = model.forward_features(rgb)[:, model.num_prefix_tokens :]
                values = values.transpose(1, 2).reshape(1, 1024, gh, gw).float()
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
                        "prefix_tokens_removed": model.num_prefix_tokens,
                        "normalization_mean": list(model.pretrained_cfg["mean"]),
                        "normalization_std": list(model.pretrained_cfg["std"]),
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
            "native_shape": list(native_shape),
            "prefix_tokens_removed": model.num_prefix_tokens,
            "normalization_mean": list(model.pretrained_cfg["mean"]),
            "normalization_std": list(model.pretrained_cfg["std"]),
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
        if not (root / "native.npy").exists():
            extract_native(spec, cache, device_name=args.device)
        if (root / "native-mask.partial.npy").exists() and not (root / "native-mask.npy").exists():
            (root / "native-mask.partial.npy").replace(root / "native-mask.npy")
        complete = (
            json.loads((root / "status.json").read_text()).get("stage") == "complete"
            if (root / "status.json").exists()
            else False
        )
        if not (root / "annual.npy").exists() or not complete:
            project_native(spec, cache)
    return 0
