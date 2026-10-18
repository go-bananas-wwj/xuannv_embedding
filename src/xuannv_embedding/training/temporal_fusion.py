"""Independent regional fitting of one calendar-aware masked feature encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.downstream.worldcover_reference import CLASSES
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS
from xuannv_embedding.models import temporal_feature_fusion as architecture
from xuannv_embedding.models.temporal_feature_fusion import (
    TemporalFeatureFusion,
    temporal_feature_loss,
)
from xuannv_embedding.training.cli import _setup_device
from xuannv_embedding.training.frozen_fusion import atomic_save, batch_indices


def verified(value: dict) -> Path:
    if set(value) != {"path", "sha256"} or sha(value["path"]) != value["sha256"]:
        raise ValueError("Temporal input identity changed")
    return Path(value["path"])


def specification(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text())
    if (
        set(spec)
        != {
            "schema",
            "index",
            "monthly",
            "static",
            "weak_targets",
            "model",
            "loss",
            "steps",
            "batch_size",
            "lr",
            "weight_decay",
            "seed",
            "device",
            "amp",
            "mask_warmup_fraction",
            "output",
        }
        or spec["schema"] != "regional-temporal-fusion-v1"
        or set(spec["static"]) != {"public", "local", "terrain"}
        or set(spec["model"]) != {"width", "residual_scale"}
        or set(spec["loss"])
        != {"worldcover_weight", "observation_weight", "visual_weight", "preservation_weight"}
        or any(type(spec[k]) is not int or spec[k] < 1 for k in ["steps", "batch_size"])
        or type(spec["seed"]) is not int
        or not 0 <= spec["seed"] < 2**32
        or type(spec["amp"]) is not bool
        or not isinstance(spec["device"], str)
        or not re.fullmatch(r"cpu|npu:[0-9]+", spec["device"])
        or any(
            type(spec[k]) not in {float, int} or not math.isfinite(spec[k]) or spec[k] < 0
            for k in ["lr", "weight_decay", "mask_warmup_fraction"]
        )
        or not spec["lr"]
        or spec["mask_warmup_fraction"] > 1
        or any(
            type(v) not in {float, int} or not math.isfinite(v) or v < 0
            for v in spec["loss"].values()
        )
    ):
        raise ValueError("Invalid regional temporal fusion specification")
    return spec


class Data:
    def __init__(self, path: str | Path, *, targets: bool = True):
        self.spec = spec = specification(path)
        index = json.loads(verified(spec["index"]).read_text())
        self.records = index["records"]
        self.n = len(self.records)
        if (
            not self.n
            or len({r["patch_id"] for r in self.records}) != self.n
            or spec["steps"] * spec["batch_size"] < self.n
        ):
            raise ValueError("Temporal training must visit every unique registered tile")
        monthly = json.loads(verified(spec["monthly"]).read_text())
        if (
            monthly.get("schema") != "public-monthly-tensor-v1"
            or monthly["index"] != spec["index"]
            or monthly["bands"] != PUBLIC_BANDS
            or monthly["band_slices"] != {"s2": [0, 11], "s1": [11, 13], "landsat": [13, 19]}
            or monthly.get("map_or_local_image_values_used") is not False
        ):
            raise ValueError("Temporal monthly source schema or geographic index differs")
        self.frames = np.load(verified(monthly["frames"]), mmap_mode="r")
        self.masks = np.load(verified(monthly["masks"]), mmap_mode="r")
        self.months = torch.tensor(monthly["months"], dtype=torch.long)
        if (
            self.frames.ndim != 5
            or self.frames.shape[:3] != (self.n, 12, 19)
            or list(self.frames.shape) != monthly["shape"]
            or self.masks.shape != (self.n, 12, 3, *self.frames.shape[-2:])
            or self.masks.dtype != bool
            or self.months.shape != (12,)
        ):
            raise ValueError("Invalid monthly observation tensor dimensions")
        self.static = {}
        for name, channels in [("public", 64), ("local", 64), ("terrain", 5)]:
            value = spec["static"][name]
            if value is None and name == "local":
                self.static[name] = None
                continue
            if (
                set(value) != {"path", "sha256", "index_sha256", "layout"}
                or value["index_sha256"] != spec["index"]["sha256"]
                or value["layout"] != "nchw"
            ):
                raise ValueError("Temporal static source index or layout differs")
            a = np.load(verified({k: value[k] for k in ["path", "sha256"]}), mmap_mode="r")
            if a.shape != (self.n, channels, *self.frames.shape[-2:]):
                raise ValueError("Temporal static feature footprint differs")
            self.static[name] = a
        self.labels, self.class_weights = None, None
        if targets:
            self.load_targets(spec)

    def load_targets(self, spec: dict) -> None:
        weak = json.loads(verified(spec["weak_targets"]).read_text())
        legend = {
            str(i): {"official_code": code, "name": name}
            for i, (code, name) in enumerate(CLASSES.items(), 1)
        }
        if (
            weak.get("state") != "complete"
            or weak.get("target") != "esa_worldcover"
            or weak.get("classes") != 12
            or weak.get("ignore_index") != 0
            or weak.get("legend") != legend
            or any(
                weak.get("source", {}).get(k) != v
                for k, v in {"dataset": "ESA WorldCover", "year": 2021, "version": "v200"}.items()
            )
            or len(weak["records"]) != self.n
        ):
            raise ValueError("True twelve-class ESA WorldCover2021v200 targets required")
        labels = []
        for original, record in zip(self.records, weak["records"]):
            if any(original[k] != record[k] for k in ["patch_id", "bounds"]):
                raise ValueError("Temporal weak target geographic order differs")
            with np.load(verified({k: record[k] for k in ["path", "sha256"]})) as z:
                y, valid = z["labels"], z["valid"]
                if (
                    y.shape != self.frames.shape[-2:]
                    or valid.dtype != bool
                    or not np.array_equal(valid, y > 0)
                    or not np.isin(y, np.arange(12)).all()
                ):
                    raise ValueError("Invalid temporal weak categorical targets")
                labels.append(y.copy())
        self.labels = np.stack(labels).astype(np.int64)
        if not np.array_equal(np.bincount(self.labels.ravel(), minlength=12), weak["class_counts"]):
            raise ValueError("Temporal weak class counts changed")
        domain = self.masks.any(axis=(1, 2)) & (self.labels > 0)
        self.domain_sha256 = hashlib.sha256(domain.tobytes() + self.labels.tobytes()).hexdigest()
        count = np.bincount(self.labels[domain], minlength=12).astype(float)
        active, weights = count[1:] > 0, np.zeros(12, np.float32)
        weights[1:][active] = np.clip(
            np.sqrt(count.sum() / (max(int(active.sum()), 1) * count[1:][active])), 0.25, 4
        )
        self.class_weights = torch.from_numpy(weights)

    def moments(self) -> dict:
        result = {}
        for j, (name, a) in enumerate(self.static.items()):
            if a is None:
                continue
            rng, sampled = np.random.default_rng(self.spec["seed"] + j), []
            for i in range(self.n):
                tile = np.asarray(a[i], np.float32).reshape(a.shape[1], -1).T
                available = np.flatnonzero(np.linalg.norm(tile, axis=1) > 1e-8)
                sampled.append(tile[rng.choice(available, min(256, len(available)), replace=False)])
            values = np.concatenate(sampled)
            if len(values):
                result[name] = (
                    values.mean(0, dtype=np.float64).astype(np.float32),
                    np.maximum(values.std(0, dtype=np.float64), 1e-4).astype(np.float32),
                )
        return result

    def batch(self, indices: list[int], *, seed: int | None = None) -> tuple:
        flips = (
            np.random.default_rng(seed).integers(0, 2, 2) if seed is not None else [False, False]
        )

        def take(a, *, integer=False):
            if a is None:
                return None
            value = np.array(a[indices], dtype=np.int64 if integer else None)
            if flips[0]:
                value = np.flip(value, -2)
            if flips[1]:
                value = np.flip(value, -1)
            return torch.from_numpy(np.ascontiguousarray(value))

        return (
            take(self.frames),
            take(self.masks),
            self.months,
            take(self.static["public"]).float(),
            take(self.static["terrain"]).float(),
            take(self.static["local"]).float() if self.static["local"] is not None else None,
            take(self.labels, integer=True),
        )


def identity(path: str | Path) -> dict:
    from xuannv_embedding.training import frozen_fusion

    return {
        "spec_sha256": sha(path),
        "implementation": {
            "training": sha(__file__),
            "model": sha(architecture.__file__),
            "checkpoint_helpers": sha(frozen_fusion.__file__),
        },
    }


def cast(device: torch.device, enabled: bool):
    return (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16) if enabled else nullcontext()
    )


def masked_inputs(batch: tuple, *, seed: int, active: bool) -> tuple:
    frames, masks, months, public, terrain, local, _ = batch
    if not active:
        return batch[:6]
    rng = np.random.default_rng(seed)
    mask, public, local = (
        masks.clone(),
        public.clone(),
        local.clone() if local is not None else None,
    )
    for i in range(len(frames)):
        for j in range(3):
            if rng.random() < 0.15:
                mask[i, :, j] = False
        mask[i, rng.choice(12, 2, replace=False)] = False
        if rng.random() < 0.5:
            h, w = frames.shape[-2:]
            side = min(16, h, w)
            y = int(rng.integers(h - side + 1))
            x = int(rng.integers(w - side + 1))
            mask[i, :, :, y : y + side, x : x + side] = False
        if rng.random() < 0.1:
            public[i] = 0
        drop_local = rng.random() < 0.1
        if local is not None and drop_local:
            local[i] = 0
    return frames, mask, months, public, terrain, local


def train(path: str | Path, *, max_updates: int | None = None) -> dict:
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("Positive partial update limit required")
    data = Data(path)
    spec = data.spec
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    ident = identity(path)
    registration = root / "registration.json"
    if any(root.iterdir()) and not registration.exists():
        raise ValueError("Unowned temporal training output directory")
    if registration.exists() and json.loads(registration.read_text()) != ident:
        raise ValueError("Temporal training identity changed")
    dump(registration, ident)
    final = root / "model.weights.pt"
    latest = root / "latest.pt"
    if final.exists():
        saved = torch.load(str(final), map_location="cpu", weights_only=True)
        if (
            saved["identity"] != ident
            or saved["step"] != spec["steps"]
            or saved["training_summary"]["weak_domain_sha256"] != data.domain_sha256
            or saved["training_summary"]["unique_training_tiles"] != data.n
        ):
            raise ValueError("Temporal final checkpoint changed")
        model = TemporalFeatureFusion(**spec["model"])
        model.load_state_dict(saved["model"], strict=True)
        result = {**saved["training_summary"], "model_sha256": sha(final)}
        dump(root / "status.json", result)
        latest.unlink(missing_ok=True)
        return result
    device, _, _ = _setup_device(spec["device"])
    if device.type == "npu":
        torch.npu.config.allow_internal_format = False
    torch.manual_seed(spec["seed"])
    model = TemporalFeatureFusion(**spec["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"]
    )
    step = 0
    seen = set()
    if latest.exists():
        saved = torch.load(str(latest), map_location="cpu", weights_only=True)
        if saved["identity"] != ident:
            raise ValueError("Temporal recovery identity changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step = saved["step"]
        seen = set(saved["seen"])
    else:
        for name, (mean, std) in data.moments().items():
            model.set_statistics(name, mean, std)

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
    model.train()
    limit = min(spec["steps"], step + max_updates) if max_updates is not None else spec["steps"]
    weights = data.class_weights.to(device)
    for current in range(step, limit):
        ids = batch_indices(data.n, spec["batch_size"], spec["seed"], current)
        batch = tuple(
            v.to(device) if v is not None else None
            for v in data.batch(ids, seed=spec["seed"] + current)
        )
        optimizer.zero_grad(set_to_none=True)
        with cast(device, spec["amp"]):
            output = model(
                *masked_inputs(
                    batch,
                    seed=spec["seed"] + current,
                    active=current >= spec["steps"] * spec["mask_warmup_fraction"],
                )
            )
            loss = temporal_feature_loss(
                output,
                batch[0],
                batch[1],
                batch[3],
                batch[5],
                batch[6],
                weights,
                model,
                **spec["loss"],
            )
        if not torch.isfinite(loss["loss"]):
            raise FloatingPointError("Nonfinite temporal training loss")
        loss["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite temporal gradients")
        rate = (
            spec["lr"]
            * min(1, (current + 1) / min(50, spec["steps"]))
            * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * current / spec["steps"])))
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        step = current + 1
        seen.update(ids)
        if step % 25 == 0 or step == limit:
            with (root / "curve.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {"step": step, **{k: float(v.detach().cpu()) for k, v in loss.items()}}
                    )
                    + "\n"
                )
            dump(
                root / "status.json",
                {
                    "state": "training",
                    "step": step,
                    "steps": spec["steps"],
                    "unique_training_tiles": len(seen),
                    "weak_domain_sha256": data.domain_sha256,
                },
            )
        if step % 50 == 0 or step == limit:
            checkpoint()
    if step < spec["steps"]:
        summary = {"state": "checkpointed", "step": step, "unique_training_tiles": len(seen)}
        dump(root / "status.json", summary)
        return summary
    if len(seen) != data.n:
        raise RuntimeError("Not all regional training tiles visited")
    summary = {
        "state": "complete",
        "step": step,
        "unique_training_tiles": len(seen),
        "weak_domain_sha256": data.domain_sha256,
        "high_resolution_used": data.static["local"] is not None,
        "class_weights": data.class_weights.tolist(),
    }
    atomic_save(
        final,
        {"identity": ident, "step": step, "model": model.state_dict(), "training_summary": summary},
    )
    cold = TemporalFeatureFusion(**spec["model"])
    cold.load_state_dict(
        torch.load(str(final), map_location="cpu", weights_only=True)["model"], strict=True
    )
    summary["model_sha256"] = sha(final)
    dump(root / "status.json", summary)
    latest.unlink()
    return summary


def export(path: str | Path) -> Path:
    data = Data(path, targets=False)
    spec = data.spec
    root = Path(spec["output"])
    modelpath = root / "model.weights.pt"
    saved = torch.load(str(modelpath), map_location="cpu", weights_only=True)
    if saved["identity"] != identity(path) or saved["step"] != spec["steps"]:
        raise ValueError("Temporal export model identity differs")
    output = root / "embedding"
    output.mkdir(exist_ok=True)
    dest = output / "annual.npy"
    ident = {"training": saved["identity"], "model_sha256": sha(modelpath)}
    header = output / "identity.json"
    if header.exists() and json.loads(header.read_text()) != ident:
        raise ValueError("Temporal export identity changed")
    dump(header, ident)
    if dest.exists() and (output / "status.json").exists():
        status = json.loads((output / "status.json").read_text())
        if status["state"] == "complete" and status["sha256"] == sha(dest):
            return dest
        raise ValueError("Completed temporal export changed")
    device, _, _ = _setup_device(spec["device"])
    if device.type == "npu":
        torch.npu.config.allow_internal_format = False
    model = TemporalFeatureFusion(**spec["model"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    values = np.lib.format.open_memmap(
        output / "annual.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(data.n, 64, *data.frames.shape[-2:]),
    )
    with torch.inference_mode():
        for i in range(data.n):
            batch = tuple(v.to(device) if v is not None else None for v in data.batch([i]))
            with cast(device, spec["amp"]):
                result = model(*batch[:6]).embedding
            if not torch.isfinite(result).all():
                raise FloatingPointError("Nonfinite temporal export")
            values[i] = result[0].float().cpu().numpy()
    values.flush()
    (output / "annual.partial.npy").replace(dest)
    dump(
        output / "status.json",
        {
            "state": "complete",
            "sha256": sha(dest),
            "shape": list(values.shape),
            "index_sha256": spec["index"]["sha256"],
            "months": data.months.tolist(),
            "map_targets_loaded_for_inference": False,
        },
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv experiment temporal-feature-fusion")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["train", "export"], required=True)
    args = parser.parse_args(argv)
    (train if args.phase == "train" else export)(args.spec)
    return 0
