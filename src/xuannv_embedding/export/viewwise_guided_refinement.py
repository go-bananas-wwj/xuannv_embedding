"""Stream real RGB observations through a frozen guide before averaging corrections."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.data_process import subpixel_rgb as rgb
from xuannv_embedding.export import public_visual_subspace as shared
from xuannv_embedding.export import rgb_guided_refinement as legacy
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.cli import _setup_device
from xuannv_embedding.training.frozen_fusion import verified


def validate_specification(spec):
    if (
        not isinstance(spec, dict)
        or spec.get("schema") != "viewwise-rgb-guided-refinement-v1"
        or not {"rgb_sources", "canonical_export"} <= set(spec)
        or not legacy._reference(spec["rgb_sources"])
    ):
        raise ValueError("Invalid self-contained viewwise specification")
    old = {k: v for k, v in spec.items() if k not in {"rgb_sources", "canonical_export"}}
    old["schema"] = "rgb-guided-refinement-v1"
    legacy.validate_specification(old)
    canonical = spec["canonical_export"]
    if canonical is not None and (
        not isinstance(canonical, dict)
        or set(canonical) != {"spec", "case", "implementation", "status"}
        or not legacy._reference(canonical["spec"])
        or not legacy._reference(canonical["status"])
        or not isinstance(canonical["case"], str)
        or re.fullmatch("[a-z][a-z0-9_]{0,63}", canonical["case"]) is None
        or not isinstance(canonical["implementation"], dict)
        or set(canonical["implementation"]) != {"refinement", "rgb"}
        or any(not legacy._reference(v) for v in canonical["implementation"].values())
        or len(spec["budgets"]) != 1
    ):
        raise ValueError("Invalid canonical export reference")
    return spec


def specification(path):
    return validate_specification(shared.read_json(path))


def refine_views(base, frames, model, normalization, q_chunk_size, strength, view_count):
    if (
        type(view_count) is not int
        or view_count < 0
        or type(strength) not in {int, float}
        or not math.isfinite(strength)
        or not 0 <= strength <= 1
        or not isinstance(base, torch.Tensor)
        or base.ndim != 4
        or base.shape[1] != 64
        or base.dtype != torch.float32
        or not torch.isfinite(base).all()
    ):
        raise ValueError("Invalid registered view count, base or strength")
    delta, seen, singleton = torch.zeros_like(base), 0, None
    for image, mask in frames:
        seen += 1
        if seen > view_count:
            raise ValueError("Actual view count differs from registered count")
        if image.shape != (len(base), 75, *base.shape[-2:]):
            raise ValueError("View and base batch geometry differs")
        guide, coverage = legacy.reconstruct_guide(image, mask, normalization)
        coverage = coverage.to(base.device)
        refined = model(
            guide.to(base.device), base, output_size=base.shape[-2:], q_chunk_size=q_chunk_size
        ).float()
        if (
            refined.shape != base.shape
            or not torch.isfinite(refined).all()
            or (refined < base.amin((2, 3), keepdim=True) - 0.005).any()
            or (refined > base.amax((2, 3), keepdim=True) + 0.005).any()
        ):
            raise ValueError("Invalid refined feature shape, value or convex range")
        if view_count == 1:
            singleton = legacy.blend(base, refined, coverage, strength)
        else:
            delta += coverage * (refined - base) / view_count
    if seen != view_count:
        raise ValueError("Actual view count differs from registered count")
    return singleton if singleton is not None else base + strength * delta


class SourceData(legacy.SourceData):
    def __init__(self, spec):
        super().__init__(spec)
        self.raw_spec, index = rgb.specification(verified(spec["rgb_sources"]))
        if (
            self.raw_spec["index"] != spec["index"]
            or self.raw_spec["public"] != self.cache["public"]
            or self.raw_spec["time_contract"] != self.cache["source_time_contract"]
            or [v["producer_spec"] for v in self.raw_spec["views"]]
            != self.cache["source_producers"]
            or len(self.raw_spec["views"]) != self.cache["actual_observations"]
        ):
            raise ValueError("Raw observations differ from the registered RGB cache")
        self.raw = rgb.Sources(self.raw_spec, index)
        self.view_count = len(self.raw.paths)

    def batch_views(self, positions):
        for frames in zip(*(self.raw.frames(i) for i in positions), strict=True):
            images, masks = [], []
            for image, mask in frames:
                clean, coverage = rgb.pool_rgb([(image, mask)], self.raw_spec["source_shape"])
                packed, coverage = rgb.pack_subpixels(clean, coverage, factor=5)
                images.append(packed.astype(np.float16).astype(np.float32))
                masks.append(coverage.astype(np.float16).astype(np.float32))
            yield torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(masks))


def identity(path, batch_size):
    source = rgb.identity(path)["implementation"]
    return {
        "spec_sha256": sha(path),
        "implementation": {"viewwise": sha(__file__), "legacy": sha(legacy.__file__), **source},
        "batch_size": batch_size,
        "compute_dtype": "float32",
    }


def check_canonical_recipe(spec, old):
    for k in ["index", "features", "rgb_cache", "upsampler", "normalization", "q_chunk_size"]:
        if spec[k] != old[k]:
            raise ValueError("Changed canonical numerical recipe")
    canonical = spec["canonical_export"]
    if canonical is None:
        raise ValueError("Missing canonical reference")
    budgets = [b for b in old["budgets"] if b["name"] == canonical["case"]]
    if len(budgets) != 1 or budgets[0]["strength"] != spec["budgets"][0]["strength"]:
        raise ValueError("Changed canonical refinement strength")


def singleton_equivalence(spec, data, ident, root):
    canonical = spec["canonical_export"]
    if data.view_count != 1 or canonical is None:
        raise ValueError("Only an exact single-view canonical export can be reused")
    old_path = verified(canonical["spec"])
    old = legacy.specification(old_path)
    check_canonical_recipe(spec, old)
    implementation = {k: sha(verified(v)) for k, v in canonical["implementation"].items()}
    if (
        implementation["refinement"] != sha(legacy.__file__)
        or data.cache["identity"]["implementation"]["subpixel_rgb"] != implementation["rgb"]
    ):
        raise ValueError("Unverified canonical implementation")
    source = Path(old["output"]) / "representations" / canonical["case"]
    status_path, array = source / "status.json", source / "annual.npy"
    if verified(canonical["status"]) != status_path:
        raise ValueError("Changed canonical status reference")
    status = shared.read_json(status_path)
    payload = np.load(array, mmap_mode="r", allow_pickle=False)
    expected = {
        "spec_sha256": sha(old_path),
        "implementation": implementation,
        "batch_size": ident["batch_size"],
        "compute_dtype": "float32",
    }
    if (
        status.get("state") != "complete"
        or status.get("identity") != expected
        or status.get("shape") != list(data.base.shape)
        or status.get("index_sha256") != spec["index"]["sha256"]
        or status.get("sha256") != sha(array)
        or payload.dtype != np.float16
        or payload.shape != data.base.shape
    ):
        raise ValueError("Canonical export identity changed")
    legacy.load_anyup(spec["upsampler"]["source_bundle"], spec["upsampler"]["checkpoint"])
    binding = {"identity": ident, "canonical_status_sha256": sha(status_path)}
    proof_path = root / "singleton-equivalence.json"
    if proof_path.exists():
        proof = shared.read_json(proof_path)
        if (
            proof.get("state") != "passed"
            or proof.get("binding") != binding
            or proof.get("all_tiles_checked") != data.n
            or proof.get("canonical_array") != {"path": str(array), "sha256": status["sha256"]}
            or proof.get("all_source_RGB_and_coverage_exact") is not True
            or proof.get("exact_legacy_singleton_arithmetic") is not True
            or re.fullmatch("[a-f0-9]{64}", str(proof.get("source_input_digest"))) is None
        ):
            raise ValueError("Singleton proof identity changed")
        return array, proof_path
    digest = hashlib.sha256()
    for i in range(data.n):
        views = list(data.batch_views([i]))
        if len(views) != 1:
            raise ValueError("Singleton observation count changed")
        image, coverage = views[0]
        for actual, cached in [(image[0], data.rgb[i]), (coverage[0], data.coverage[i])]:
            values = actual.numpy().astype(np.float16)
            if not np.array_equal(values, cached):
                raise ValueError("Singleton source differs from canonical cached input")
            digest.update(values.tobytes())
    dump(
        proof_path,
        {
            "state": "passed",
            "binding": binding,
            "all_tiles_checked": data.n,
            "canonical_array": {"path": str(array), "sha256": status["sha256"]},
            "source_input_digest": digest.hexdigest(),
            "all_source_RGB_and_coverage_exact": True,
            "exact_legacy_singleton_arithmetic": True,
            "task_scores_computed": 0,
            "scope": (
                "All registered RGB inputs plus identical frozen base/operator; "
                "not raw upstream encoder replay"
            ),
        },
    )
    return array, proof_path


def export(path, *, device="cpu", batch_size=2):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Viewwise batch size must be positive")
    spec = specification(path)
    root, ident = Path(spec["output"]), identity(path, batch_size)
    with shared.ownership(root, ident):
        data = SourceData(spec)
        canonical = None
        if spec["canonical_export"] is not None:
            canonical = singleton_equivalence(spec, data, ident, root)
        arrays, results = {}, []
        reserve = spec["reserve_gib"] * 2**30
        if shutil.disk_usage(root).free < reserve:
            raise RuntimeError("Viewwise export disk reserve")
        missing = sum(
            not (root / "representations" / b["name"] / "status.json").exists()
            for b in spec["budgets"]
        )
        if (
            canonical is None
            and shutil.disk_usage(root).free < reserve + missing * data.base.nbytes
        ):
            raise RuntimeError("Viewwise export allocation reserve")
        for b in spec["budgets"]:
            dest = root / "representations" / b["name"]
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "status.json").exists():
                status = shared.read_json(dest / "status.json")
                if (
                    status.get("state") != "complete"
                    or status.get("identity") != ident
                    or status.get("shape") != list(data.base.shape)
                    or status.get("index_sha256") != spec["index"]["sha256"]
                    or status.get("sha256") != sha(dest / "annual.npy")
                ):
                    raise ValueError("Cached viewwise export changed")
                if canonical is not None:
                    array, proof = canonical
                    if status.get("canonical_array") != {
                        "path": str(array),
                        "sha256": sha(array),
                    } or status.get("equivalence") != {"path": str(proof), "sha256": sha(proof)}:
                        raise ValueError("Cached singleton equivalence changed")
                results.append(status)
            elif canonical is not None:
                array, proof = canonical
                pending = dest / "annual.partial.npy"
                if pending.exists():
                    pending.unlink()
                os.link(array, pending)
                pending.replace(dest / "annual.npy")
                status = {
                    "state": "complete",
                    "identity": ident,
                    "shape": list(data.base.shape),
                    "index_sha256": spec["index"]["sha256"],
                    "sha256": sha(array),
                    "canonical_array": {"path": str(array), "sha256": sha(array)},
                    "equivalence": {"path": str(proof), "sha256": sha(proof)},
                    "view_count": 1,
                    "primary_or_weak_maps_loaded": False,
                    "neural_encoder_updated": False,
                    "unit_normalized": False,
                }
                dump(dest / "status.json", status)
                results.append(status)
            else:
                arrays[b["name"]] = np.lib.format.open_memmap(
                    dest / "annual.partial.npy", mode="w+", dtype=np.float16, shape=data.base.shape
                )
        if not arrays:
            return results
        dev, _, _ = _setup_device(device)
        model = legacy.load_anyup(
            spec["upsampler"]["source_bundle"], spec["upsampler"]["checkpoint"]
        ).to(dev)
        start = time.monotonic()
        with torch.inference_mode():
            for first in range(0, data.n, batch_size):
                if shutil.disk_usage(root).free < reserve:
                    raise RuntimeError("Viewwise export disk reserve")
                positions = list(range(first, min(data.n, first + batch_size)))
                base = torch.from_numpy(np.array(data.base[positions], dtype=np.float32)).to(dev)
                # Registered experiments use one strength; multiple strengths stream afresh.
                for b in spec["budgets"]:
                    if b["name"] not in arrays:
                        continue
                    values = (
                        refine_views(
                            base,
                            data.batch_views(positions),
                            model,
                            spec["normalization"],
                            spec["q_chunk_size"],
                            b["strength"],
                            data.view_count,
                        )
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
                    if not np.isfinite(values).all():
                        raise FloatingPointError("Nonfinite stored viewwise features")
                    arrays[b["name"]][positions] = values
                dump(
                    root / "progress.json",
                    {"completed_tiles": positions[-1] + 1, "total_tiles": data.n},
                )
        for name, values in arrays.items():
            values.flush()
            dest = root / "representations" / name
            (dest / "annual.partial.npy").replace(dest / "annual.npy")
            status = {
                "state": "complete",
                "identity": ident,
                "shape": list(data.base.shape),
                "index_sha256": spec["index"]["sha256"],
                "sha256": sha(dest / "annual.npy"),
                "view_count": data.view_count,
                "primary_or_weak_maps_loaded": False,
                "neural_encoder_updated": False,
                "unit_normalized": False,
                "export_wall_seconds": time.monotonic() - start,
                "frozen_additional_parameters": sum(p.numel() for p in model.parameters()),
            }
            dump(dest / "status.json", status)
            results.append(status)
        return results


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment viewwise-guided-refinement")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    export(args.spec, device=args.device, batch_size=args.batch_size)
    return 0
