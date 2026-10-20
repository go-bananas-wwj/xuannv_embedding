"""Frozen AnyUp RGB guidance on the original feature grid, with explicit provenance.

AnyUp is an optional external dependency, not redistributed or relicensed here.
See docs/operations/rgb-guided-refinement.md for attribution and rebuild instructions.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from xuannv_embedding.data_process import subpixel_rgb
from xuannv_embedding.export import public_visual_subspace as shared
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.cli import _setup_device
from xuannv_embedding.training.frozen_fusion import source_array, verified


def _reference(value, *, array=False):
    keys = {"path", "sha256"} | ({"layout", "index_sha256"} if array else set())
    return (
        isinstance(value, dict)
        and set(value) == keys
        and isinstance(value["path"], str)
        and Path(value["path"]).is_absolute()
        and isinstance(value["sha256"], str)
        and re.fullmatch("[a-f0-9]{64}", value["sha256"]) is not None
        and (
            not array
            or (
                value["layout"] == "nchw"
                and isinstance(value["index_sha256"], str)
                and re.fullmatch("[a-f0-9]{64}", value["index_sha256"]) is not None
            )
        )
    )


def _normalization(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"mean", "std", "RGB_divisor"}
        or type(value["RGB_divisor"]) not in {int, float}
        or value["RGB_divisor"] != 255
        or any(
            not isinstance(value[k], list)
            or len(value[k]) != 3
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in value[k])
            for k in ["mean", "std"]
        )
        or any(not 0 <= v <= 1 for v in value["mean"])
        or any(v <= 0 for v in value["std"])
    ):
        raise ValueError("Invalid RGB normalization")


def validate_specification(spec):
    if (
        not isinstance(spec, dict)
        or set(spec)
        != {
            "schema",
            "index",
            "features",
            "rgb_cache",
            "upsampler",
            "normalization",
            "q_chunk_size",
            "output",
            "reserve_gib",
            "budgets",
        }
        or spec["schema"] != "rgb-guided-refinement-v1"
        or not _reference(spec["index"])
        or not _reference(spec["features"], array=True)
        or not _reference(spec["rgb_cache"])
        or spec["features"]["index_sha256"] != spec["index"]["sha256"]
        or type(spec["q_chunk_size"]) is not int
        or spec["q_chunk_size"] < 1
        or type(spec["reserve_gib"]) is not int
        or spec["reserve_gib"] < 0
        or not isinstance(spec["output"], str)
        or not Path(spec["output"]).is_absolute()
        or not isinstance(spec["budgets"], list)
        or not spec["budgets"]
    ):
        raise ValueError("Invalid self-contained RGB refinement specification")
    up = spec["upsampler"]
    if (
        not isinstance(up, dict)
        or set(up) != {"family", "source_bundle", "checkpoint", "use_natten"}
        or up["family"] != "anyup_multi_backbone"
        or up["use_natten"] is not False
        or not _reference(up["source_bundle"])
        or not _reference(up["checkpoint"])
    ):
        raise ValueError("Invalid frozen upsampler")
    _normalization(spec["normalization"])
    names = set()
    for b in spec["budgets"]:
        if (
            not isinstance(b, dict)
            or set(b) != {"name", "strength"}
            or not isinstance(b["name"], str)
            or re.fullmatch("[a-z][a-z0-9_]{0,63}", b["name"]) is None
            or b["name"] in names
            or type(b["strength"]) not in {int, float}
            or not math.isfinite(b["strength"])
            or not 0 <= b["strength"] <= 1
        ):
            raise ValueError("Invalid common refinement strength")
        names.add(b["name"])
    return spec


def specification(path):
    return validate_specification(shared.read_json(path))


def load_anyup(bundle_ref, checkpoint_ref):
    """Verify an explicitly trusted source tree before executing it in a private namespace."""
    bundle = shared.read_json(verified(bundle_ref))
    if (
        bundle.get("repository") != "https://github.com/wimmerth/anyup"
        or bundle.get("license") != "CC-BY-4.0"
        or re.fullmatch("[a-f0-9]{40}", str(bundle.get("commit"))) is None
        or not isinstance(bundle.get("files"), dict)
        or not {"LICENSE", "anyup/__init__.py", "anyup/model.py"} <= set(bundle["files"])
    ):
        raise ValueError("Invalid AnyUp source bundle")
    root = Path(bundle["source_path"])
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("Invalid AnyUp source directory")
    root = root.resolve()
    for name, digest in bundle["files"].items():
        path = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not path.is_file()
            or any(p.is_symlink() for p in [path, *path.parents] if p != root)
        ):
            raise ValueError("Invalid AnyUp source entry")
        if not path.resolve().is_relative_to(root) or sha(path) != digest:
            raise ValueError("AnyUp source identity changed")
    actual = {str(p.relative_to(root)) for p in (root / "anyup").rglob("*.py")}
    expected = {n for n in bundle["files"] if n.startswith("anyup/") and n.endswith(".py")}
    if actual != expected:
        raise ValueError("Unregistered AnyUp source files")
    alias = "_xuannv_anyup_" + bundle_ref["sha256"]
    if alias not in sys.modules:
        definition = importlib.util.spec_from_file_location(
            alias, root / "anyup/__init__.py", submodule_search_locations=[str(root / "anyup")]
        )
        package = importlib.util.module_from_spec(definition)
        sys.modules[alias] = package
        definition.loader.exec_module(package)
    model_type = importlib.import_module(alias + ".model").AnyUp
    model = model_type(use_natten=False)
    weights = torch.load(str(verified(checkpoint_ref)), weights_only=True, map_location="cpu")
    model.load_state_dict(weights, strict=True)
    return model.eval().requires_grad_(False)


def reconstruct_guide(rgb, coverage, normalization):
    _normalization(normalization)
    if (
        rgb.ndim != 4
        or rgb.shape[1] != 75
        or coverage.shape != (len(rgb), 25, *rgb.shape[-2:])
        or not torch.isfinite(coverage).all()
        or (coverage < 0).any()
        or (coverage > 1).any()
    ):
        raise ValueError("Invalid packed RGB geometry or coverage")
    rgb, coverage = rgb.float(), coverage.float()
    image, mask = F.pixel_shuffle(rgb, 5), F.pixel_shuffle(coverage, 5)
    observed = torch.where(mask > 0, image, 0)
    if not torch.isfinite(observed).all() or (observed < 0).any() or (observed > 255).any():
        raise ValueError("Invalid observed RGB")
    mean = image.new_tensor(normalization["mean"])[None, :, None, None]
    std = image.new_tensor(normalization["std"])[None, :, None, None]
    image = torch.where(mask > 0, (observed / normalization["RGB_divisor"] - mean) / std, 0)
    return image, coverage.mean(1, keepdim=True)


def blend(base, refined, coverage, strength):
    if (
        type(strength) not in {int, float}
        or not math.isfinite(strength)
        or not 0 <= strength <= 1
        or base.ndim != 4
        or base.shape != refined.shape
        or coverage.shape != (len(base), 1, *base.shape[-2:])
        or not all(torch.isfinite(x).all() for x in [base, refined, coverage])
        or (coverage < 0).any()
        or (coverage > 1).any()
    ):
        raise ValueError("Invalid refinement result, coverage or strength")
    return base + strength * coverage * (refined - base)


class SourceData:
    def __init__(self, spec):
        self.index = shared.read_json(verified(spec["index"]))
        self.cache = subpixel_rgb.load_manifest(verified(spec["rgb_cache"]))
        self.n = len(self.index["records"])
        if (
            self.n < 1
            or self.cache["index"] != spec["index"]
            or self.cache.get("factor") != 5
            or self.cache.get("working_pixel_m") != 2
            or self.cache.get("output_pixel_m") != 10
            or self.cache.get("primary_or_weak_maps_loaded") is not False
        ):
            raise ValueError("RGB and base feature geography differs")
        digest = spec["index"]["sha256"]
        self.base = source_array(spec["features"], digest, self.n)
        self.rgb = source_array(self.cache["rgb"], digest, self.n)
        self.coverage = source_array(self.cache["coverage"], digest, self.n)
        if self.base.shape != (self.n, 64, *self.rgb.shape[-2:]):
            raise ValueError("RGB and base feature grid differs")


def identity(path, batch_size):
    return {
        "spec_sha256": sha(path),
        "implementation": {"refinement": sha(__file__), "rgb": sha(subpixel_rgb.__file__)},
        "batch_size": batch_size,
        "compute_dtype": "float32",
    }


def export(path, *, device="cpu", batch_size=1):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Refinement batch size must be positive")
    spec = specification(path)
    root, ident = Path(spec["output"]), identity(path, batch_size)
    with shared.ownership(root, ident):
        data = SourceData(spec)
        model = load_anyup(spec["upsampler"]["source_bundle"], spec["upsampler"]["checkpoint"])
        shape, arrays, results = data.base.shape, {}, []
        missing = sum(
            not (root / "representations" / b["name"] / "status.json").exists()
            for b in spec["budgets"]
        )
        reserve = spec["reserve_gib"] * 2**30
        if shutil.disk_usage(root).free < reserve + missing * math.prod(shape) * 2:
            raise RuntimeError("RGB refinement export disk reserve")
        for b in spec["budgets"]:
            dest = root / "representations" / b["name"]
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "status.json").exists():
                status = shared.read_json(dest / "status.json")
                if (
                    status.get("state") != "complete"
                    or status.get("identity") != ident
                    or status.get("shape") != list(shape)
                    or status.get("index_sha256") != spec["index"]["sha256"]
                    or sha(dest / "annual.npy") != status["sha256"]
                ):
                    raise ValueError("Cached refinement export changed")
                results.append(status)
            else:
                arrays[b["name"]] = np.lib.format.open_memmap(
                    dest / "annual.partial.npy", mode="w+", dtype=np.float16, shape=shape
                )
        if not arrays:
            return results
        dev, _, _ = _setup_device(device)
        model = model.to(dev)
        start, max_error, missing_pixels = time.monotonic(), 0.0, 0
        with torch.inference_mode():
            for first in range(0, data.n, batch_size):
                if shutil.disk_usage(root).free < reserve:
                    raise RuntimeError("RGB refinement disk reserve")
                last = min(first + batch_size, data.n)
                base = torch.from_numpy(np.array(data.base[first:last], dtype=np.float32)).to(dev)
                rgb = torch.from_numpy(np.array(data.rgb[first:last], dtype=np.float32))
                cov = torch.from_numpy(np.array(data.coverage[first:last], dtype=np.float32))
                guide, confidence = reconstruct_guide(rgb, cov, spec["normalization"])
                confidence = confidence.to(dev)
                refined = model(
                    guide.to(dev),
                    base,
                    output_size=base.shape[-2:],
                    q_chunk_size=spec["q_chunk_size"],
                ).float()
                if (
                    not torch.isfinite(refined).all()
                    or (refined < base.amin((2, 3), keepdim=True) - 0.005).any()
                    or (refined > base.amax((2, 3), keepdim=True) + 0.005).any()
                ):
                    raise FloatingPointError("Refinement convex value range failed")
                absent = (confidence == 0).expand_as(base)
                missing_pixels += int((confidence == 0).sum())
                for b in spec["budgets"]:
                    if b["name"] not in arrays:
                        continue
                    value = blend(base, refined, confidence, b["strength"])
                    if not torch.equal(value[absent], base[absent]):
                        raise FloatingPointError("Missing RGB fallback changed")
                    quantized = value.cpu().numpy().astype(np.float16)
                    if not np.isfinite(quantized).all():
                        raise FloatingPointError("Nonfinite stored refinement")
                    max_error = max(
                        max_error, float(np.max(np.abs(quantized - value.cpu().numpy())))
                    )
                    arrays[b["name"]][first:last] = quantized
                dump(root / "progress.json", {"completed_tiles": last, "total_tiles": data.n})
        for name, values in arrays.items():
            values.flush()
            dest = root / "representations" / name
            (dest / "annual.partial.npy").replace(dest / "annual.npy")
            result = {
                "state": "complete",
                "identity": ident,
                "shape": list(shape),
                "index_sha256": spec["index"]["sha256"],
                "sha256": sha(dest / "annual.npy"),
                "primary_or_weak_maps_loaded": False,
                "neural_encoder_updated": False,
                "unit_normalized": False,
                "missing_rgb_pixels": missing_pixels,
                "max_fp16_error_across_candidates": max_error,
                "paired_export_wall_seconds": time.monotonic() - start,
                "frozen_additional_parameters": sum(p.numel() for p in model.parameters()),
            }
            dump(dest / "status.json", result)
            results.append(result)
        return results


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment rgb-guided-refinement")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    export(args.spec, device=args.device, batch_size=args.batch_size)
    return 0
