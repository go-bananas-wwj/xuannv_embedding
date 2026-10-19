"""Fit one bounded gate without updating its source encoders or weak-label heads."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from xuannv_embedding.data_process import osm_weak_targets
from xuannv_embedding.downstream import worldcover_reference
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.models import content_adaptive_residual as model_source
from xuannv_embedding.models.content_adaptive_residual import (
    ContentAdaptiveResidual,
    adaptive_residual_loss,
)
from xuannv_embedding.training import frozen_fusion, resident_features
from xuannv_embedding.training import semantic_refinement as base
from xuannv_embedding.training.frozen_fusion import (
    atomic_save,
    batch_indices,
    source_array,
    verified,
)
from xuannv_embedding.training.resident_features import ResidentFeatures


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _reference(value, *, array=False):
    expected = {"path", "sha256"} | ({"layout", "index_sha256"} if array else set())
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or not isinstance(value["path"], str)
        or not value["path"]
        or not Path(value["path"]).is_absolute()
        or not isinstance(value["sha256"], str)
        or len(value["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in value["sha256"])
        or (array and not isinstance(value["layout"], str))
        or (array and value["layout"] not in {"nchw", "nhwc"})
        or (array and not isinstance(value["index_sha256"], str))
        or (array and len(value["index_sha256"]) != 64)
        or (array and any(c not in "0123456789abcdef" for c in value["index_sha256"]))
    ):
        raise ValueError("Invalid explicit adaptive source reference")


def specification(path):
    return validate_specification(_read(path))


def validate_specification(spec):
    expected = {
        "schema",
        "index",
        "inputs",
        "worldcover",
        "osm",
        "teacher",
        "model",
        "loss",
        "steps",
        "warmup_steps",
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip",
        "seed",
        "amp",
        "device",
        "reserve_gib",
        "output",
    }
    if (
        not isinstance(spec, dict)
        or set(spec) != expected
        or spec["schema"] != "content-adaptive-residual-v1"
        or not isinstance(spec["inputs"], dict)
        or set(spec["inputs"]) != {"public", "semantic", "geometry", "spectral"}
        or not isinstance(spec["model"], dict)
        or set(spec["model"]) != {"width", "initial_alpha"}
        or not isinstance(spec["loss"], dict)
        or set(spec["loss"]) != {"preservation_weight"}
        or not isinstance(spec["teacher"], dict)
        or set(spec["teacher"])
        != {
            f"{source}_{kind}"
            for source in ["semantic", "geometry"]
            for kind in ["checkpoint", "spec", "export"]
        }
        or any(
            type(spec[k]) is not int or spec[k] < 1 for k in ["steps", "warmup_steps", "batch_size"]
        )
        or any(type(spec[k]) is not int or spec[k] < 0 for k in ["seed", "reserve_gib"])
        or spec["seed"] >= 2**32
        or any(
            type(spec[k]) not in {int, float} or not math.isfinite(spec[k]) or spec[k] <= 0
            for k in ["lr", "grad_clip"]
        )
        or type(spec["weight_decay"]) not in {int, float}
        or not math.isfinite(spec["weight_decay"])
        or spec["weight_decay"] < 0
        or type(spec["loss"]["preservation_weight"]) not in {int, float}
        or not math.isfinite(spec["loss"]["preservation_weight"])
        or spec["loss"]["preservation_weight"] < 0
        or type(spec["amp"]) is not bool
        or not isinstance(spec["device"], str)
        or not isinstance(spec["output"], str)
        or not Path(spec["output"]).is_absolute()
        or not isinstance(spec["inputs"]["spectral"], list)
        or len(spec["inputs"]["spectral"]) != 1
    ):
        raise ValueError("Invalid self-contained adaptive residual specification")
    for value in [spec["index"], spec["worldcover"], spec["osm"], *spec["teacher"].values()]:
        _reference(value)
    for key in ["public", "semantic", "geometry"]:
        _reference(spec["inputs"][key], array=True)
    for block in spec["inputs"]["spectral"]:
        if not isinstance(block, dict) or set(block) != {"array", "columns"}:
            raise ValueError("Invalid adaptive physical block")
        _reference(block["array"], array=True)
        _reference(block["columns"])
    ContentAdaptiveResidual(**spec["model"])
    index = _read(verified(spec["index"]))
    records = index.get("records", [])
    if (
        not records
        or len({r["patch_id"] for r in records}) != len(records)
        or spec["steps"] * spec["batch_size"] < len(records)
    ):
        raise ValueError("Adaptive budget or geographic index differs")
    return spec, index


def identity(path):
    return {
        "spec_sha256": sha(path),
        "implementation": {
            "training": sha(__file__),
            "model": sha(model_source.__file__),
            "weak_data": sha(base.__file__),
            "source_arrays": sha(frozen_fusion.__file__),
            "resident": sha(resident_features.__file__),
            "weak_osm": sha(osm_weak_targets.__file__),
            "weak_worldcover": sha(worldcover_reference.__file__),
        },
    }


class TrainingData:
    def __init__(self, spec, index, *, load_weak=True):
        self.n = len(index["records"])
        self.index_sha256 = spec["index"]["sha256"]
        self.load_weak = load_weak
        self.teacher_state, self.source_summaries = self._teachers(spec)
        if load_weak:
            weak = base.TrainingData(
                {
                    "index": spec["index"],
                    "inputs": {
                        "public": spec["inputs"]["public"],
                        "spectral": spec["inputs"]["spectral"],
                        "detail": None,
                        "context": None,
                        "context_mask": None,
                        "context_target": None,
                    },
                    "worldcover": spec["worldcover"],
                    "osm": spec["osm"],
                    "context_source_shape": [1, 1],
                },
                index,
            )
            self.public, self.physical, self.groups = weak.public, weak.spectral[0], weak.groups
            self.labels, self.domain = weak.labels, weak.domain
            self.class_weights, self.domain_sha256 = weak.class_weights, weak.domain_sha256
            for summary in self.source_summaries:
                if summary["weak_domain_sha256"] != self.domain_sha256 or any(
                    summary["class_weights"][k] != v.tolist() for k, v in self.class_weights.items()
                ):
                    raise ValueError("Frozen teacher weak-label exposure changed")
        else:
            self.public = source_array(spec["inputs"]["public"], self.index_sha256, self.n)
            block = spec["inputs"]["spectral"][0]
            self.physical = source_array(block["array"], self.index_sha256, self.n)
            columns = _read(verified(block["columns"]))["columns"]
            if (
                len(columns) != 95
                or len(set(columns)) != 95
                or any(c.split(":")[0] not in {"s2", "s1", "landsat"} for c in columns)
            ):
                raise ValueError("Adaptive public physical columns differ")
            self.groups = [
                (
                    columns.index(source + ":availability"),
                    [i for i, c in enumerate(columns) if c.startswith(source + ":")],
                )
                for source in ["s2", "s1", "landsat"]
            ]
        self.semantic = source_array(spec["inputs"]["semantic"], self.index_sha256, self.n)
        self.geometry = source_array(spec["inputs"]["geometry"], self.index_sha256, self.n)
        if (
            self.public.shape[1] != 64
            or min(self.public.shape) < 1
            or self.semantic.shape != self.public.shape
            or self.geometry.shape != self.public.shape
            or self.physical.shape != (self.n, 95, *self.public.shape[-2:])
            or any(
                a.dtype not in [np.float16, np.float32]
                for a in [self.public, self.semantic, self.geometry, self.physical]
            )
            or any(tuple(shape) != self.public.shape for shape in self.source_shapes)
        ):
            raise ValueError("Adaptive source coordinate shapes differ")

    def _teachers(self, spec):
        models, summaries = [], []
        self.source_shapes = []
        for source in ["semantic", "geometry"]:
            fields = spec["teacher"]
            original = _read(verified(fields[source + "_spec"]))
            original = original.get("base", original)
            saved = torch.load(
                str(verified(fields[source + "_checkpoint"])), weights_only=True, map_location="cpu"
            )
            exported = _read(verified(fields[source + "_export"]))
            if (
                saved["identity"]["spec_sha256"] != fields[source + "_spec"]["sha256"]
                or saved["step"] != original["steps"]
                or original["index"]["sha256"] != self.index_sha256
                or original["inputs"]["public"]["sha256"] != spec["inputs"]["public"]["sha256"]
                or original["inputs"]["spectral"] != spec["inputs"]["spectral"]
                or any(original[k]["sha256"] != spec[k]["sha256"] for k in ["worldcover", "osm"])
                or exported.get("state") != "complete"
                or exported.get("evaluation_maps_used") is not False
                or not isinstance(exported.get("shape"), list)
                or exported["index_sha256"] != self.index_sha256
                or exported["sha256"] != spec["inputs"][source]["sha256"]
                or exported["model_sha256"] != fields[source + "_checkpoint"]["sha256"]
            ):
                raise ValueError("Frozen teacher or source export binding changed")
            models.append(saved["model"])
            summaries.append(saved["summary"])
            self.source_shapes.append(exported["shape"])
        prefix = "base." if "base.public_mean" in models[1] else ""
        for key in ["public_mean", "public_std", "spectral_mean", "spectral_std"]:
            if not torch.equal(models[0][key], models[1][prefix + key]):
                raise ValueError("Frozen source coordinate statistics differ")
        names = ["public_mean", "public_std", "spectral_mean", "spectral_std"] + [
            name + "_head." + field
            for name in ["worldcover", "area", "geometry"]
            for field in ["weight", "bias"]
        ]
        state = {key: models[0][key].clone() for key in names}
        check = ContentAdaptiveResidual(**spec["model"])
        check.set_teacher(state)
        return state, summaries

    def batch(self, positions):
        if not positions or any(type(i) is not int or not 0 <= i < self.n for i in positions):
            raise ValueError("Adaptive batch positions must be valid registered tile indices")

        def take(array):
            return torch.from_numpy(np.array(array[positions], dtype=np.float32, copy=True))

        public, semantic, geometry = [take(a) for a in [self.public, self.semantic, self.geometry]]
        physical = take(self.physical)
        visible = torch.zeros_like(physical, dtype=torch.bool)
        for availability, group in self.groups:
            visible[:, group] = physical[:, availability : availability + 1] > 0
        physical = torch.where(visible, physical, 0)
        labels = (
            {
                k: torch.from_numpy(np.array(y[positions], np.int64, copy=True))
                for k, y in self.labels.items()
            }
            if self.load_weak
            else None
        )
        domain = (
            torch.from_numpy(np.array(self.domain[positions], bool, copy=True))
            if self.load_weak
            else None
        )
        return public, semantic, geometry, physical, visible, labels, domain


def _forward(model, batch, spec, current):
    return model(*batch[:5])


def update(model, batch, weights, optimizer, spec, current, *, forward_function=None):
    optimizer.zero_grad(set_to_none=True)
    with base._autocast(spec, batch[0].device):
        output = (forward_function or _forward)(model, batch, spec, current)
        losses = adaptive_residual_loss(output, batch[5], batch[6], weights, **spec["loss"])
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError("Nonfinite adaptive gate loss")
    losses["loss"].backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, spec["grad_clip"])
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite adaptive gate gradient")
    rate = (
        spec["lr"]
        * min(1, (current + 1) / spec["warmup_steps"])
        * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * current / spec["steps"])))
    )
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.step()
    metrics = {
        **{k: v.detach() for k, v in losses.items()},
        "gradient_norm": norm.detach(),
        "mean_alpha": output.alpha.detach().mean(),
    }
    if hasattr(output, "expert_weights"):
        means = output.expert_weights.detach().mean((0, 2, 3, 4))
        metrics.update(
            {
                name + "_weight": means[i]
                for i, name in enumerate(["public", "semantic", "geometry"])
            }
        )
    return metrics


@dataclass(frozen=True)
class AdaptiveRuntime:
    specification: Callable = specification
    identity: Callable = identity
    data: Callable = TrainingData
    model: Callable = ContentAdaptiveResidual
    forward: Callable = _forward


def train(path, *, max_updates=None, runtime: AdaptiveRuntime | None = None):
    runtime = runtime or AdaptiveRuntime()
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("Partial training limit must be positive")
    spec, index = runtime.specification(path)
    root, ident = Path(spec["output"]), runtime.identity(path)
    with base.ownership(root, ident, spec["reserve_gib"]):
        data = runtime.data(spec, index)
        final, latest, status = root / "model.weights.pt", root / "latest.pt", root / "status.json"
        if final.exists():
            saved = torch.load(str(final), weights_only=True, map_location="cpu")
            if (
                saved["identity"] != ident
                or saved["step"] != spec["steps"]
                or saved["summary"]["weak_domain_sha256"] != data.domain_sha256
            ):
                raise ValueError("Adaptive final identity changed")
            digest = sha(final)
            if (
                status.exists()
                and _read(status).get("state") == "complete"
                and _read(status)["model_sha256"] != digest
            ):
                raise ValueError("Adaptive final payload changed")
            cold = runtime.model(**spec["model"])
            cold.load_state_dict(saved["model"], strict=True)
            result = {**saved["summary"], "model_sha256": digest}
            dump(status, result)
            return result
        device = base._device(spec)
        torch.manual_seed(spec["seed"])
        model = runtime.model(**spec["model"]).to(device)
        model.set_teacher(data.teacher_state)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=spec["lr"],
            weight_decay=spec["weight_decay"],
        )
        step, seen = 0, set()
        if latest.exists():
            saved = torch.load(str(latest), weights_only=True, map_location="cpu")
            if saved["identity"] != ident:
                raise ValueError("Adaptive recovery identity changed")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            step, seen = saved["step"], set(saved["seen"])

        def checkpoint():
            atomic_save(
                latest,
                {
                    "identity": ident,
                    "step": step,
                    "seen": sorted(seen),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
            )

        checkpoint()
        resident = ResidentFeatures(data, device)
        dump(
            root / "input-residency.json",
            {
                "state": "complete",
                "tiles": data.n,
                "bytes": resident.resident_bytes,
                "device": str(device),
                "canonical_streaming_dtype_preserved": True,
            },
        )
        weights = {k: v.to(device) for k, v in data.class_weights.items()}
        model.train()
        limit = min(spec["steps"], step + max_updates) if max_updates is not None else spec["steps"]
        for current in range(step, limit):
            ids = batch_indices(data.n, spec["batch_size"], spec["seed"], current)
            batch = resident.batch(ids)
            metrics = update(
                model, batch, weights, optimizer, spec, current, forward_function=runtime.forward
            )
            step = current + 1
            seen.update(ids)
            if step % 25 == 0 or step == limit:
                with (root / "curve.jsonl").open("a") as f:
                    f.write(
                        json.dumps(
                            {"step": step, **{k: float(v.cpu()) for k, v in metrics.items()}}
                        )
                        + "\n"
                    )
                checkpoint()
                dump(
                    status,
                    {
                        "state": "training",
                        "step": step,
                        "steps": spec["steps"],
                        "unique_training_tiles": len(seen),
                    },
                )
                if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30:
                    raise RuntimeError("Adaptive disk reserve reached")
            del batch, metrics
        if step < spec["steps"]:
            result = {"state": "checkpointed", "step": step, "unique_training_tiles": len(seen)}
            dump(status, result)
            return result
        if len(seen) != data.n:
            raise RuntimeError("Adaptive fit did not cover all registered tiles")
        if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
            raise FloatingPointError("Nonfinite adaptive terminal weights")
        summary = {
            "state": "complete",
            "step": step,
            "unique_training_tiles": len(seen),
            "weak_domain_sha256": data.domain_sha256,
            "class_weights": {k: v.tolist() for k, v in data.class_weights.items()},
            "only_gate_parameters_optimized": all(
                not p.requires_grad
                for head in [model.worldcover_head, model.area_head, model.geometry_head]
                for p in head.parameters()
            ),
            "auxiliary_heads_optimized": any(
                p.requires_grad
                for head in [model.worldcover_head, model.area_head, model.geometry_head]
                for p in head.parameters()
            ),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "teacher_sources": spec["teacher"],
        }
        atomic_save(
            final,
            {"identity": ident, "step": step, "model": model.state_dict(), "summary": summary},
        )
        result = {**summary, "model_sha256": sha(final)}
        dump(status, result)
        latest.unlink(missing_ok=True)
        return result


def export(path, *, runtime: AdaptiveRuntime | None = None):
    runtime = runtime or AdaptiveRuntime()
    spec, index = runtime.specification(path)
    root, ident = Path(spec["output"]), runtime.identity(path)
    with base.ownership(root, ident, spec["reserve_gib"]):
        final = root / "model.weights.pt"
        terminal_status = root / "status.json"
        if terminal_status.exists():
            previous = _read(terminal_status)
            if previous.get("state") == "complete" and previous["model_sha256"] != sha(final):
                raise ValueError("Adaptive terminal payload changed before export")
        saved = torch.load(str(final), weights_only=True, map_location="cpu")
        if saved["identity"] != ident or saved["step"] != spec["steps"]:
            raise ValueError("Adaptive export requires terminal source-bound weights")
        data = runtime.data(spec, index, load_weak=False)
        device = base._device(spec)
        model = runtime.model(**spec["model"]).to(device)
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        out = root / "embedding"
        out.mkdir(exist_ok=True)
        dest, status, header = out / "annual.npy", out / "status.json", out / "identity.json"
        export_ident = {"training": ident, "model_sha256": sha(final)}
        if header.exists() and _read(header) != export_ident:
            raise ValueError("Adaptive export model changed")
        dump(header, export_ident)
        if dest.exists() and status.exists():
            value = _read(status)
            if value.get("state") != "complete" or sha(dest) != value["sha256"]:
                raise ValueError("Adaptive exported payload changed")
            return value
        shape = data.public.shape
        if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30 + math.prod(shape) * 2:
            raise RuntimeError("Adaptive export disk reserve reached")
        pending = out / "annual.partial.npy"
        array = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float16, shape=shape)
        with torch.inference_mode():
            for start in range(0, data.n, min(spec["batch_size"], 8)):
                ids = list(range(start, min(data.n, start + min(spec["batch_size"], 8))))
                batch = base._move(data.batch(ids), device)
                with base._autocast(spec, device):
                    output = runtime.forward(model, batch, spec, spec["steps"]).embedding
                value = output.float().cpu().numpy().astype(np.float16)
                if not np.isfinite(value).all():
                    raise FloatingPointError("Adaptive export overflow")
                array[ids] = value
                del batch, output, value
        array.flush()
        pending.replace(dest)
        value = {
            "state": "complete",
            "shape": list(shape),
            "sha256": sha(dest),
            "index_sha256": spec["index"]["sha256"],
            "model_sha256": sha(final),
            "evaluation_maps_used": False,
            "weak_labels_loaded": False,
        }
        dump(status, value)
        return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment content-adaptive-residual")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["train", "export"], required=True)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args(argv)
    if args.phase == "export" and args.max_updates is not None:
        parser.error("--max-updates applies only to training")
    train(args.spec, max_updates=args.max_updates) if args.phase == "train" else export(args.spec)
    return 0
