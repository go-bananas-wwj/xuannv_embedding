"""Matched regional scale-aware refinement from frozen sensor observations."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from pyproj import CRS

from xuannv_embedding.data_process.osm_weak_targets import (
    AREA_CLASSES,
    GEOMETRY_CLASSES,
    _json,
)
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.models import semantic_residual
from xuannv_embedding.models.semantic_residual import SemanticResidualFusion, semantic_loss
from xuannv_embedding.training import frozen_fusion, resident_features
from xuannv_embedding.training.cli import _setup_device
from xuannv_embedding.training.frozen_fusion import (
    atomic_save,
    batch_indices,
    source_array,
    verified,
)
from xuannv_embedding.training.resident_features import ResidentFeatures


def specification(path):
    spec = _json(Path(path))
    expected = {
        "schema",
        "index",
        "inputs",
        "worldcover",
        "osm",
        "context_source_shape",
        "model",
        "loss",
        "steps",
        "burnin_steps",
        "warmup_steps",
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip",
        "seed",
        "amp",
        "device",
        "output",
        "reserve_gib",
    }
    if (
        set(spec) != expected
        or spec["schema"] != "semantic-refinement-v1"
        or set(spec["inputs"])
        != {"public", "spectral", "detail", "context", "context_mask", "context_target"}
        or set(spec["model"]) != {"width", "public_residual_scale", "local_residual_scale"}
        or set(spec["loss"])
        != {"worldcover", "area", "geometry", "spectral", "detail", "context", "preservation"}
        or any(
            type(spec[k]) is not int or spec[k] < 1 for k in ["steps", "batch_size", "warmup_steps"]
        )
        or any(
            type(spec[k]) is not int or spec[k] < 0 for k in ["burnin_steps", "reserve_gib", "seed"]
        )
        or not spec["burnin_steps"] < spec["steps"]
        or spec["seed"] >= 2**32
        or type(spec["amp"]) is not bool
        or not isinstance(spec["context_source_shape"], list)
        or len(spec["context_source_shape"]) != 2
        or any(type(v) is not int or v < 1 for v in spec["context_source_shape"])
        or any(
            type(spec[k]) not in {int, float} or not math.isfinite(spec[k]) or spec[k] <= 0
            for k in ["lr", "grad_clip"]
        )
        or type(spec["weight_decay"]) not in {int, float}
        or not math.isfinite(spec["weight_decay"])
        or spec["weight_decay"] < 0
        or any(
            type(v) not in {int, float} or not math.isfinite(v) or v < 0
            for v in spec["loss"].values()
        )
        or not Path(spec["output"]).is_absolute()
        or not isinstance(spec["inputs"]["spectral"], list)
        or len(spec["inputs"]["spectral"]) != 1
        or (spec["inputs"]["context"] is None) != (spec["inputs"]["context_mask"] is None)
        or (spec["inputs"]["context"] is None) != (spec["inputs"]["context_target"] is None)
    ):
        raise ValueError("Invalid self-contained semantic refinement specification")
    # Reuse the established geographic and true-WorldCover validation, not its old model.
    index = _json(verified(spec["index"]))
    if (
        not index["records"]
        or len({r["patch_id"] for r in index["records"]}) != len(index["records"])
        or spec["steps"] * spec["batch_size"] < len(index["records"])
    ):
        raise ValueError("Semantic training budget must cover every geographic tile")
    SemanticResidualFusion(**spec["model"])
    return spec, index


class TrainingData:
    def __init__(self, spec, index):
        legacy = {
            "index": spec["index"],
            "inputs": {
                "public": spec["inputs"]["public"],
                "spectral": spec["inputs"]["spectral"],
                "semantic": spec["inputs"]["detail"],
            },
            "weak_targets": spec["worldcover"],
            "model": {"spectral_channels": 95},
        }
        base = frozen_fusion.TrainingData(legacy, index)
        self.public, self.spectral, self.groups = base.public, base.spectral, base.groups
        self.detail, self.n = base.semantic, base.n
        self.context_shape = tuple(spec["context_source_shape"])
        self.extra = {}
        for name, channels in [("context", 1024), ("context_mask", 1), ("context_target", 64)]:
            ref = spec["inputs"][name]
            a = source_array(ref, spec["index"]["sha256"], self.n) if ref is not None else None
            native = tuple(math.ceil(v / 16) for v in self.context_shape)
            spatial = self.public.shape[-2:] if name == "context_target" else native
            if a is not None and a.shape != (self.n, channels, *spatial):
                raise ValueError("Native context geographic dimensions differ")
            self.extra[name] = a
        if self.extra["context_mask"] is not None:
            m = self.extra["context_mask"]
            if not np.isfinite(m).all() or np.min(m) < 0 or np.max(m) > 1:
                raise ValueError("Native context observed mask must be in [0,1]")
        weak = _json(verified(spec["osm"]))
        if (
            weak.get("schema") != "osm-weak-targets-v1"
            or weak.get("state") != "complete"
            or weak.get("dataset") != "OpenStreetMap"
            or weak.get("index") != spec["index"]
            or weak.get("area_legend") != list(AREA_CLASSES)
            or weak.get("geometry_legend") != list(GEOMETRY_CLASSES)
            or weak.get("unknown_is_background") is not False
            or weak.get("primary_reference_maps_read") is not False
            or weak.get("forest_wood_are_tag_classes") is not True
            or len(weak.get("records", [])) != self.n
        ):
            raise ValueError("Known OSM target source or class schema differs")
        crs = index.get("crs") or index["records"][0].get("crs")
        for original, record in zip(index["records"], weak["records"]):
            if (
                original["patch_id"] != record["patch_id"]
                or original["bounds"] != record["bounds"]
                or CRS.from_user_input(original.get("crs") or crs)
                != CRS.from_user_input(record["crs"])
            ):
                raise ValueError("OSM geographic image order differs")
        self.labels = {"worldcover": base.labels}
        for key, count in [("area", 7), ("geometry", 4)]:
            a = np.load(verified(weak[key]), mmap_mode="r")
            if (
                a.shape != base.labels.shape
                or a.dtype != np.uint8
                or np.max(a) >= count
                or not np.array_equal(
                    np.bincount(a.ravel(), minlength=count), weak[key + "_counts"]
                )
            ):
                raise ValueError("Invalid known OSM encoding or class counts")
            self.labels[key] = a
        domain = np.zeros(base.labels.shape, bool)
        for availability, _ in self.groups:
            domain |= self.spectral[0][:, availability] > 0
        self.domain = domain
        h = hashlib.sha256(domain.tobytes())
        self.class_weights = {}
        for name, count in [("worldcover", 12), ("area", 7), ("geometry", 4)]:
            y = self.labels[name]
            h.update(y.tobytes())
            counts = np.bincount(y[domain], minlength=count).astype(np.float64)
            counts[0] = 0
            active = counts > 0
            weights = np.zeros(count, np.float32)
            weights[active] = np.clip(
                np.sqrt(counts.sum() / (max(1, active.sum()) * counts[active])), 0.25, 10
            )
            self.class_weights[name] = torch.from_numpy(weights)
        self.domain_sha256 = h.hexdigest()

    def batch(self, positions):
        def take(a, *, observed=None):
            if a is None:
                return None
            x = np.array(a[positions], dtype=np.float32, copy=True)
            if observed is not None:
                x = np.where(observed, x, 0)
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite observed semantic refinement input")
            return torch.from_numpy(np.ascontiguousarray(x))

        public = take(self.public)
        spectral = take(self.spectral[0])
        detail = take(self.detail)
        mask = take(self.extra["context_mask"])
        context = take(
            self.extra["context"], observed=mask.numpy() > 0 if mask is not None else None
        )
        target = take(self.extra["context_target"])
        visible = torch.zeros_like(spectral, dtype=torch.bool)
        for availability, group in self.groups:
            visible[:, group] = spectral[:, availability : availability + 1] > 0
        labels = {
            k: torch.from_numpy(np.array(y[positions], np.int64, copy=True))
            for k, y in self.labels.items()
        }
        return public, spectral, detail, context, mask, target, visible, labels

    def moments(self, *, seed):
        result = {}
        arrays = {
            "public": self.public,
            "spectral": self.spectral[0],
            "detail": self.detail,
            "context": self.extra["context"],
            "context_target": self.extra["context_target"],
        }
        for name, array in arrays.items():
            if array is None:
                continue
            # Reset the RNG for every source: local availability cannot change public statistics.
            rng = np.random.default_rng(seed)
            positions = [
                rng.choice(
                    array.shape[-2] * array.shape[-1],
                    min(256, array.shape[-2] * array.shape[-1]),
                    replace=False,
                )
                for _ in range(self.n)
            ]
            values, valid = [], []
            for tile, ids in enumerate(positions):
                x = np.asarray(array[tile], np.float32).reshape(array.shape[1], -1)[:, ids].T
                if name == "spectral":
                    mask = np.zeros(x.shape, bool)
                    for availability, group in self.groups:
                        mask[:, group] = x[:, availability : availability + 1] > 0
                elif name == "context":
                    mask = np.asarray(self.extra["context_mask"][tile]).reshape(-1)[ids, None] > 0
                else:
                    mask = np.abs(x).sum(1, keepdims=True) > 1e-8
                values.append(x)
                valid.append(np.broadcast_to(mask, x.shape))
            x, mask = np.concatenate(values), np.concatenate(valid)
            if len(x) > 65536:
                ids = rng.choice(len(x), 65536, replace=False)
                x, mask = x[ids], mask[ids]
            if not np.isfinite(np.where(mask, x, 0)).all():
                raise ValueError("Nonfinite observed normalization sample")
            count = mask.sum(0)
            mean = np.where(mask, x, 0).sum(0, dtype=np.float64) / np.maximum(count, 1)
            variance = np.where(mask, (x.astype(np.float64) - mean) ** 2, 0).sum(0) / np.maximum(
                count, 1
            )
            std = np.sqrt(variance)
            std[(count < 2) | (std < 1e-6)] = 1
            result[name] = (mean.astype(np.float32), std.astype(np.float32))
        return result


def implementation():
    return {
        "training": sha(__file__),
        "model": sha(semantic_residual.__file__),
        "base_data": sha(frozen_fusion.__file__),
        "resident_features": sha(resident_features.__file__),
    }


def identity(path):
    return {"spec_sha256": sha(path), "implementation": implementation()}


@contextlib.contextmanager
def ownership(root, ident, reserve_gib):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        header = root / "registration.json"
        if header.exists():
            if _json(header) != ident:
                raise ValueError("Semantic refinement implementation or specification changed")
        elif any(p.name != "run.lock" for p in root.iterdir()):
            raise ValueError("Unowned semantic refinement output")
        if shutil.disk_usage(root).free < reserve_gib * 2**30:
            raise RuntimeError("Semantic refinement disk reserve reached")
        dump(header, ident)
        yield


def _device(spec):
    device, _, _ = _setup_device(spec["device"])
    if device.type == "npu":
        torch.npu.config.allow_internal_format = False
    if spec["amp"] and device.type not in {"npu", "cuda"}:
        raise ValueError("BF16 mixed precision requires an accelerator")
    return device


def _autocast(spec, device):
    return (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if spec["amp"]
        else contextlib.nullcontext()
    )


def _move(batch, device):
    return tuple(
        (
            {k: v.to(device) for k, v in x.items()}
            if isinstance(x, dict)
            else x.to(device) if x is not None else None
        )
        for x in batch
    )


def _forward(model, batch, shape, *, burnin=False):
    p, s, d, c, m, _, visible, _ = batch
    return model(p, s, d, c, m, spectral_mask=visible, context_source_shape=shape, burnin=burnin)


def train(path, *, max_updates=None):
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("Partial training limit must be a positive integer")
    spec, index = specification(path)
    root, ident = Path(spec["output"]), identity(path)
    with ownership(root, ident, spec["reserve_gib"]):
        data = TrainingData(spec, index)
        final, latest, status = root / "model.weights.pt", root / "latest.pt", root / "status.json"
        if final.exists():
            saved = torch.load(str(final), weights_only=True, map_location="cpu")
            if (
                saved["identity"] != ident
                or saved["step"] != spec["steps"]
                or saved["summary"]["weak_domain_sha256"] != data.domain_sha256
            ):
                raise ValueError("Semantic refinement final model identity changed")
            digest = sha(final)
            if (
                status.exists()
                and _json(status).get("state") == "complete"
                and _json(status)["model_sha256"] != digest
            ):
                raise ValueError("Semantic refinement final model changed")
            cold = SemanticResidualFusion(**spec["model"])
            cold.load_state_dict(saved["model"], strict=True)
            result = {**saved["summary"], "model_sha256": digest}
            dump(status, result)
            latest.unlink(missing_ok=True)
            return result
        device = _device(spec)
        torch.manual_seed(spec["seed"])
        model = SemanticResidualFusion(**spec["model"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"]
        )
        step, seen = 0, set()
        if latest.exists():
            saved = torch.load(str(latest), weights_only=True, map_location="cpu")
            if saved["identity"] != ident:
                raise ValueError("Semantic refinement recovery identity changed")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            step, seen = saved["step"], set(saved["seen"])
        else:
            for name, stats in data.moments(seed=spec["seed"]).items():
                model.set_statistics(name, *stats)

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
                "sampling_and_label_exposure_unchanged": True,
            },
        )
        weights = {k: v.to(device) for k, v in data.class_weights.items()}
        model.train()
        limit = min(spec["steps"], step + max_updates) if max_updates is not None else spec["steps"]
        for current in range(step, limit):
            ids = batch_indices(data.n, spec["batch_size"], spec["seed"], current)
            batch = _move(resident.batch(ids), device)
            _, s, d, _, _, t, visible, labels = batch
            optimizer.zero_grad(set_to_none=True)
            with _autocast(spec, device):
                out = _forward(
                    model, batch, data.context_shape, burnin=current < spec["burnin_steps"]
                )
                targets = {
                    "spectral": model.standardize(s, "spectral", visible),
                    "detail": (
                        model.standardize(d, "detail", out.detail_mask) if d is not None else None
                    ),
                    "context": (
                        model.standardize(t, "context_target", out.context_mask)
                        if t is not None
                        else None
                    ),
                }
                loss = semantic_loss(out, labels, targets, visible, weights, weights=spec["loss"])
            if not torch.isfinite(loss["loss"]):
                raise FloatingPointError("Nonfinite semantic refinement loss")
            loss["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec["grad_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite semantic refinement gradient")
            warm = min(1.0, (current + 1) / spec["warmup_steps"])
            factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * current / spec["steps"]))
            for group in optimizer.param_groups:
                group["lr"] = spec["lr"] * warm * factor
            optimizer.step()
            step = current + 1
            seen.update(ids)
            if step % 25 == 0 or step == limit:
                row = {
                    "step": step,
                    "burnin": current < spec["burnin_steps"],
                    **{k: float(v.detach().cpu()) for k, v in loss.items()},
                    "gradient_norm": float(norm.detach().cpu()),
                    "local_gates": out.gates.detach().float().mean((0, 2, 3)).cpu().tolist(),
                }
                with (root / "curve.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                dump(
                    status,
                    {
                        "state": "training",
                        "step": step,
                        "steps": spec["steps"],
                        "unique_training_tiles": len(seen),
                        "weak_domain_sha256": data.domain_sha256,
                    },
                )
            if step % 50 == 0 or step == limit:
                if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30:
                    checkpoint()
                    raise RuntimeError("Semantic refinement disk reserve reached")
                checkpoint()
        if step < spec["steps"]:
            result = {"state": "checkpointed", "step": step, "unique_training_tiles": len(seen)}
            dump(status, result)
            return result
        if len(seen) != data.n:
            raise RuntimeError("Not every registered tile was used for training")
        summary = {
            "state": "complete",
            "step": step,
            "unique_training_tiles": len(seen),
            "weak_domain_sha256": data.domain_sha256,
            "class_weights": {k: v.tolist() for k, v in data.class_weights.items()},
            "local_observations_used": data.detail is not None or data.extra["context"] is not None,
            "evaluation_maps_used": False,
        }
        atomic_save(
            final,
            {"identity": ident, "step": step, "model": model.state_dict(), "summary": summary},
        )
        cold = SemanticResidualFusion(**spec["model"])
        cold.load_state_dict(
            torch.load(str(final), weights_only=True, map_location="cpu")["model"], strict=True
        )
        result = {**summary, "model_sha256": sha(final)}
        dump(status, result)
        latest.unlink()
        return result


def export(path):
    spec, index = specification(path)
    root, ident = Path(spec["output"]), identity(path)
    with ownership(root, ident, spec["reserve_gib"]):
        final = root / "model.weights.pt"
        saved = torch.load(str(final), weights_only=True, map_location="cpu")
        if saved["identity"] != ident or saved["step"] != spec["steps"]:
            raise ValueError("Semantic export requires the selected terminal checkpoint")
        data = TrainingData(spec, index)
        if saved["summary"]["weak_domain_sha256"] != data.domain_sha256:
            raise ValueError("Semantic export training domain changed")
        device = _device(spec)
        model = SemanticResidualFusion(**spec["model"]).to(device)
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        output = root / "embedding"
        output.mkdir(exist_ok=True)
        dest, status = output / "annual.npy", output / "status.json"
        header = output / "identity.json"
        export_ident = {"training": ident, "model_sha256": sha(final)}
        if header.exists() and _json(header) != export_ident:
            raise ValueError("Semantic exported model changed")
        dump(header, export_ident)
        if dest.exists() and status.exists():
            summary = _json(status)
            if summary.get("state") != "complete" or sha(dest) != summary["sha256"]:
                raise ValueError("Semantic exported bytes changed")
            return summary
        shape = (data.n, 64, *data.public.shape[-2:])
        if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30 + math.prod(shape) * 2:
            raise RuntimeError("Semantic export disk reserve reached")
        pending = output / "annual.partial.npy"
        array = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float16, shape=shape)
        with torch.inference_mode():
            for start in range(0, data.n, min(spec["batch_size"], 8)):
                ids = list(range(start, min(data.n, start + min(spec["batch_size"], 8))))
                batch = _move(data.batch(ids), device)
                with _autocast(spec, device):
                    out = _forward(model, batch, data.context_shape).embedding
                values = out.float().cpu().numpy().astype(np.float16)
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite or overflowed semantic output")
                array[ids] = values
        array.flush()
        pending.replace(dest)
        summary = {
            "state": "complete",
            "shape": list(shape),
            "sha256": sha(dest),
            "index_sha256": spec["index"]["sha256"],
            "evaluation_maps_used": False,
            "model_sha256": export_ident["model_sha256"],
        }
        dump(status, summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment semantic-refinement")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["train", "export"], required=True)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args(argv)
    if args.phase == "export" and args.max_updates is not None:
        parser.error("--max-updates applies only to training")
    train(args.spec, max_updates=args.max_updates) if args.phase == "train" else export(args.spec)
    return 0
