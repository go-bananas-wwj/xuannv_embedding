"""Raw-image missingness with frozen vision weights and original PCA coordinates."""

from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_bounds
from torch.nn import functional as F

from xuannv_embedding.export import semantic_features as vision
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.jstars_revision import retain_observation
from xuannv_embedding.training.frozen_fusion import verified


def frozen_projection(native, mask, projection, source_shape, output_shape):
    """Apply the registered PCA without fitting or changing feature coordinates."""
    native, mask = np.asarray(native, np.float32), np.asarray(mask, np.float32)
    if (
        native.ndim != 4
        or native.shape[1] != 1024
        or mask.shape != (len(native), 1, *native.shape[-2:])
    ):
        raise ValueError("Invalid frozen projection source geometry")
    if not np.isfinite(native).all() or not np.isfinite(mask).all():
        raise ValueError("Nonfinite frozen projection input")
    mean, components = projection["mean"], projection["components"]
    if mean.shape != (1024,) or components.shape != (64, 1024):
        raise ValueError("Original PCA basis must map 1024 features to 64")
    b, _, h, w = native.shape
    from sklearn.decomposition import PCA

    mapper = PCA(n_components=64, whiten=False)
    mapper.mean_, mapper.components_ = mean, components
    mapper.n_features_in_, mapper.n_components_ = 1024, 64
    mapper.explained_variance_ = np.ones(64, dtype=components.dtype)
    values = np.stack([mapper.transform(x.reshape(1024, -1).T).T.reshape(64, h, w) for x in native])
    coverage = torch.from_numpy(mask.copy())
    numerator = vision.align_patch_features(
        torch.from_numpy(values) * coverage, source_shape, output_shape, patch=16
    )
    denominator = vision.align_patch_features(coverage, source_shape, output_shape, patch=16)
    result = F.normalize(numerator / denominator.clamp_min(1e-6), dim=1)
    return torch.where(denominator > 0.05, result, 0)


def reencode(path, device):
    """Mask native RGB before every encoder, then reuse its original projection."""
    from xuannv_embedding.export.public_visual_subspace import read_json

    spec = read_json(path)
    required = {"schema", "feature_spec", "projection", "fraction", "seed", "raw_output", "output"}
    if (
        set(spec) != required
        or spec["schema"] != "jstars-raw-vision-v1"
        or spec["fraction"] not in {0.25, 0.5, 1}
    ):
        raise ValueError("Invalid raw-observation intervention")
    original, cache = vision.read_spec(verified(spec["feature_spec"]))
    root = Path(spec["output"])
    raw = Path(spec["raw_output"])
    root.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    identity = {
        "spec_sha256": sha(path),
        "code_sha256": sha(__file__),
        "frozen_weights": original["weights"],
        "original_projection": spec["projection"],
    }
    status = root / "status.json"
    if status.exists():
        result = json.loads(status.read_text())
        if result["identity"] != identity:
            raise ValueError("Completed raw source changed")
        for ref in result["arrays"].values():
            verified(ref)
        return result
    if shutil.disk_usage(root).free < 120 * 2**30:
        raise RuntimeError("Raw extraction requires the registered 120 GiB working reserve")
    started = time.monotonic()
    paths = vision.registered_rasters(original, cache)
    if paths is None:
        raise ValueError("Raw intervention requires the actual external raster inventory")
    raw_identity = {
        "rasters": original["rasters"],
        "fraction": spec["fraction"],
        "seed": spec["seed"],
    }
    header = raw / "identity.json"
    if header.exists() and json.loads(header.read_text()) != raw_identity:
        raise ValueError("Masked raster directory belongs to a different intervention")
    dump(header, raw_identity)
    rows = []
    for i, (record, source) in enumerate(zip(cache["records"], paths, strict=True)):
        target = raw / f"{i:04d}.tif"
        if not target.exists():
            image, mask = vision.read_aerial_window(source, record)
            image, mask = retain_observation(
                image.numpy(), mask.numpy(), spec["fraction"], str(i), spec["seed"]
            )
            h, w = image.shape[-2:]
            pending = target.with_suffix(".partial.tif")
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(
                    pending,
                    "w",
                    driver="GTiff",
                    height=h,
                    width=w,
                    count=3,
                    dtype="float32",
                    crs=record["crs"],
                    transform=from_bounds(*record["bounds"], w, h),
                    compress="deflate",
                    tiled=True,
                ) as dst:
                    dst.write(image)
                    dst.write_mask((mask[0] > 0).astype("uint8") * 255)
            pending.replace(target)
        rows.append(
            {
                **{k: record[k] for k in ["patch_id", "bounds", "crs"]},
                "raster": {"path": str(target), "sha256": sha(target)},
            }
        )
        if i % 40 == 0:
            print("masked raw", root.name, i + 1, len(paths), flush=True)
    manifest = raw / "rasters.json"
    dump(
        manifest,
        {
            "schema": "semantic-raster-inputs-v1",
            "records": rows,
            "dependencies": [spec["feature_spec"]],
        },
    )
    modified = copy.deepcopy(original)
    modified["rasters"] = {"path": str(manifest), "sha256": sha(manifest)}
    modified["output"] = str(root)
    dump(root / "feature-spec.json", modified)
    native_status = root / "native-status.json"
    if (
        not native_status.exists()
        or json.loads(native_status.read_text()).get("stage") != "complete"
    ):
        vision.extract_native(modified, cache, device_name=device)
    refs = {}
    for name in ["native", "native-mask"]:
        p = root / (name + ".npy")
        refs[name] = {"path": str(p), "sha256": sha(p)}
    if spec["projection"] is not None:
        with np.load(verified(spec["projection"])) as z:
            projection = {k: z[k] for k in ["mean", "components"]}
        native = np.load(root / "native.npy", mmap_mode="r")
        masks = np.load(root / "native-mask.npy", mmap_mode="r")
        source_shape = json.loads(native_status.read_text())["source_shape"]
        output_shape = modified["output_shape"]
        output = np.lib.format.open_memmap(
            root / "annual.partial.npy",
            mode="w+",
            dtype="float16",
            shape=(len(native), 64, *output_shape),
        )
        for i in range(len(native)):
            with torch.inference_mode():
                value = frozen_projection(
                    native[i : i + 1], masks[i : i + 1], projection, source_shape, output_shape
                )
            output[i] = value[0].numpy().astype("float16")
        output.flush()
        del output
        (root / "annual.partial.npy").replace(root / "annual.npy")
        refs["annual"] = {"path": str(root / "annual.npy"), "sha256": sha(root / "annual.npy")}
    result = {
        "state": "complete",
        "identity": identity,
        "arrays": refs,
        "feature_spec": {
            "path": str(root / "feature-spec.json"),
            "sha256": sha(root / "feature-spec.json"),
        },
        "rasters": {"path": str(manifest), "sha256": sha(manifest)},
        "raw_mask_before_visual_encoder": True,
        "pca_refitted": False,
        "model_weights_refitted": False,
        "wall_seconds": time.monotonic() - started,
    }
    dump(status, result)
    return result


def fuse_reencoded(path, device):
    """Replay both original regional fusers with re-encoded missing-observation inputs."""
    from xuannv_embedding.export.journal_sources import archived_semantic_runtime
    from xuannv_embedding.training import decoupled_geometry as geometry
    from xuannv_embedding.training import semantic_refinement as semantic
    from xuannv_embedding.training.cli import _setup_device

    spec = json.loads(Path(path).read_text())
    if (
        set(spec) != {"schema", "original", "sources", "output"}
        or spec["schema"] != "jstars-raw-fusion-v1"
    ):
        raise ValueError("Invalid re-encoded regional fusion contract")
    original = json.loads(verified(spec["original"]).read_text())
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    ident = {"spec": sha(path), "code": sha(__file__)}
    if (root / "status.json").exists():
        result = json.loads((root / "status.json").read_text())
        if result["identity"] != ident:
            raise ValueError("Re-encoded fusion identity changed")
        verified(result["features"])
        return result
    sources = spec["sources"]
    loaded = {k: json.loads(verified(sources[k]).read_text()) for k in ["detail", "context"]}
    views = [json.loads(verified(ref).read_text()) for ref in sources["views"]]
    for row in [*loaded.values(), *views]:
        if (
            row["state"] != "complete"
            or row["pca_refitted"]
            or not row["raw_mask_before_visual_encoder"]
        ):
            raise ValueError("Every local source must be removed before vision encoding")
    dev, _, _ = _setup_device(device)
    started = time.monotonic()
    outputs = []
    index_ref = None
    for name, runtime in [
        ("semantic", archived_semantic_runtime(original["semantic_source"])),
        ("geometry", geometry.RUNTIME),
    ]:
        settings, index = runtime.specification(verified(original[name]))
        if index_ref is not None and settings["index"] != index_ref:
            raise ValueError("Regional fusers use different geographic coordinates")
        index_ref = settings["index"]
        checkpoint = Path(settings["output"]) / "model.weights.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            saved["identity"] != runtime.identity(Path(original[name]["path"]))
            or saved["step"] != settings["steps"]
        ):
            raise ValueError("Re-encoded inference requires original weights and source identities")
        data = runtime.data(settings, index)
        data.detail = np.load(verified(loaded["detail"]["arrays"]["annual"]), mmap_mode="r")
        data.extra["context_target"] = None
        if name == "semantic":
            data.extra["context"] = np.load(
                verified(loaded["context"]["arrays"]["native"]), mmap_mode="r"
            )
            data.extra["context_mask"] = np.load(
                verified(loaded["context"]["arrays"]["native-mask"]), mmap_mode="r"
            )
        else:
            if len(views) != len(data.views):
                raise ValueError("Actual observation count changed")
            data.views = [np.load(verified(v["arrays"]["native"]), mmap_mode="r") for v in views]
            data.view_masks = [
                np.load(verified(v["arrays"]["native-mask"]), mmap_mode="r") for v in views
            ]
            data.extra["context"], data.extra["context_mask"] = data.views[0], data.view_masks[0]
        model = runtime.model(**settings["model"]).to(dev).eval()
        model.load_state_dict(saved["model"], strict=True)
        target = root / (name + ".npy")
        output = np.lib.format.open_memmap(
            target.with_suffix(".partial.npy"),
            mode="w+",
            dtype="float16",
            shape=(data.n, 64, 128, 128),
        )
        with torch.inference_mode():
            for start in range(0, data.n, 2):
                ids = list(range(start, min(start + 2, data.n)))
                batch = semantic._move(data.batch(ids), dev)
                with semantic._autocast(settings, dev):
                    value = runtime.forward(model, batch, data.context_shape).embedding
                values = value.float().cpu().numpy().astype("float16")
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite masked regional fusion")
                output[ids] = values
                if start % 40 == 0:
                    dump(
                        root / "progress.json",
                        {"component": name, "completed": start + len(ids), "total": data.n},
                    )
        output.flush()
        del output
        target.with_suffix(".partial.npy").replace(target)
        outputs.append(target)
        del data, model
        if dev.type == "npu":
            torch.npu.empty_cache()
    a, b = [np.load(p, mmap_mode="r") for p in outputs]
    target = root / "annual.npy"
    output = np.lib.format.open_memmap(
        root / "annual.partial.npy", mode="w+", dtype="float16", shape=a.shape
    )
    for i in range(len(a)):
        output[i] = (
            np.float32(0.85) * a[i].astype("float32") + np.float32(0.15) * b[i].astype("float32")
        ).astype("float16")
    output.flush()
    del output
    (root / "annual.partial.npy").replace(target)
    result = {
        "state": "complete",
        "identity": ident,
        "features": {"path": str(target), "sha256": sha(target)},
        "index": index_ref,
        "all_local_visual_sources_reencoded": True,
        "normalization_and_projection_refitted": False,
        "wall_seconds": time.monotonic() - started,
    }
    dump(root / "status.json", result)
    return result


def guide_reencoded(path, device):
    """Bind altered raw dates, caches and regional base before the final image guide."""
    from xuannv_embedding.data_process import subpixel_rgb as rgb
    from xuannv_embedding.export import viewwise_guided_refinement as viewwise

    spec = json.loads(Path(path).read_text())
    if (
        set(spec) != {"schema", "original_guide", "fusion", "views", "output"}
        or spec["schema"] != "jstars-raw-guidance-v1"
    ):
        raise ValueError("Invalid raw-source guide contract")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    ident = {"spec": sha(path), "code": sha(__file__)}
    if (root / "status.json").exists():
        result = json.loads((root / "status.json").read_text())
        if result["identity"] != ident:
            raise ValueError("Raw guide intervention identity changed")
        verified(result["features"])
        return result
    started = time.monotonic()
    guide = viewwise.specification(verified(spec["original_guide"]))
    fusion = json.loads(verified(spec["fusion"]).read_text())
    if fusion["state"] != "complete" or not fusion["all_local_visual_sources_reencoded"]:
        raise ValueError("Terminal guide requires the re-encoded upstream representation")
    raw = rgb.read_json(verified(guide["rgb_sources"]))
    views = [json.loads(verified(ref).read_text()) for ref in spec["views"]]
    if len(views) != len(raw["views"]):
        raise ValueError("Guidance observation inventory changed")
    for row, value in zip(raw["views"], views, strict=True):
        row["producer_spec"] = value["feature_spec"]
        row["rasters"] = value["rasters"]
    raw["output"] = str(root / "rgb-cache")
    rawpath = root / "rgb-spec.json"
    dump(rawpath, raw)
    rgb.prepare(rawpath)
    cache = root / "rgb-cache/manifest.json"
    guide.update(
        features={**fusion["features"], "layout": "nchw", "index_sha256": guide["index"]["sha256"]},
        rgb_sources={"path": str(rawpath), "sha256": sha(rawpath)},
        rgb_cache={"path": str(cache), "sha256": sha(cache)},
        canonical_export=None,
        output=str(root / "guided"),
    )
    guidepath = root / "guide-spec.json"
    dump(guidepath, guide)
    exports = viewwise.export(guidepath, device=device, batch_size=2)
    dest = Path(guide["output"]) / "representations/viewwise075/annual.npy"
    result = {
        "state": "complete",
        "identity": ident,
        "features": {"path": str(dest), "sha256": sha(dest)},
        "exports": exports,
        "scope": "raw_images_removed_before_all_local_encoders_and_terminal_guidance",
        "parameters_and_readouts_refitted": False,
        "wall_seconds": time.monotonic() - started,
    }
    dump(root / "status.json", result)
    return result


def run_raw_case(path, device):
    """Run a complete pre-registered raw missingness pipeline on one leased device."""
    from xuannv_embedding.export.jstars_revision import score_frozen

    spec = json.loads(Path(path).read_text())
    required = {"schema", "original", "original_guide", "sources", "region", "readers", "output"}
    if set(spec) != required or spec["schema"] != "jstars-raw-case-v1":
        raise ValueError("Invalid complete raw-observation case")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ["detail", "context"]:
        source = verified(spec["sources"][name])
        reencode(source, device)
        output = Path(json.loads(source.read_text())["output"]) / "status.json"
        manifest[name] = {"path": str(output), "sha256": sha(output)}
    manifest["views"] = []
    for ref in spec["sources"]["views"]:
        source = verified(ref)
        reencode(source, device)
        output = Path(json.loads(source.read_text())["output"]) / "status.json"
        manifest["views"].append({"path": str(output), "sha256": sha(output)})

    def bind(name, value):
        target = root / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError("Completed pipeline input binding changed")
        dump(target, value)
        return target

    fusion = bind(
        "fusion.json",
        {
            "schema": "jstars-raw-fusion-v1",
            "original": spec["original"],
            "sources": manifest,
            "output": str(root / "fusion"),
        },
    )
    fuse_reencoded(fusion, device)
    fusion_status = root / "fusion/status.json"
    guide = bind(
        "guide.json",
        {
            "schema": "jstars-raw-guidance-v1",
            "original_guide": spec["original_guide"],
            "fusion": {"path": str(fusion_status), "sha256": sha(fusion_status)},
            "views": manifest["views"],
            "output": str(root / "guidance"),
        },
    )
    result = guide_reencoded(guide, device)
    score = bind(
        "score.json",
        {
            "schema": "jstars-frozen-intervention-v1",
            "region": spec["region"],
            "feature": result["features"],
            "readers": spec["readers"],
            "output": str(root / "evaluation"),
        },
    )
    return score_frozen(score, device)
