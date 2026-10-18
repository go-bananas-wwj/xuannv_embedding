"""Independent regional fits of one frozen-feature residual-fusion recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.downstream.worldcover_reference import CLASSES
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.models import feature_fusion, incremental_highres
from xuannv_embedding.models.feature_fusion import FrozenFeatureFusion, feature_fusion_loss
from xuannv_embedding.training.cli import _setup_device


def verified(ref):
    if set(ref) != {"path", "sha256"} or sha(ref["path"]) != ref["sha256"]:
        raise ValueError("Frozen fusion input identity changed")
    return Path(ref["path"])


def implementation():
    return {
        "training": sha(__file__),
        "model": sha(feature_fusion.__file__),
        "residual": sha(incremental_highres.__file__),
    }


def specification(path):
    spec = json.loads(Path(path).read_text())
    if (
        set(spec)
        != {
            "schema",
            "index",
            "inputs",
            "weak_targets",
            "model",
            "loss",
            "steps",
            "batch_size",
            "lr",
            "weight_decay",
            "seed",
            "device",
            "output",
        }
        or spec["schema"] != "frozen-feature-fusion-training-v1"
        or set(spec["inputs"]) != {"public", "spectral", "semantic"}
        or set(spec["model"]) != {"width", "residual_scale", "spectral_channels"}
        or set(spec["loss"]) != {"spectral_weight", "semantic_weight", "preservation_weight"}
        or any(type(spec[k]) is not int or spec[k] < 1 for k in ["steps", "batch_size"])
        or type(spec["seed"]) is not int
        or not 0 <= spec["seed"] < 2**32
        or any(
            type(value) not in {int, float} or not math.isfinite(value) or value < 0
            for value in spec["loss"].values()
        )
        or any(
            type(spec[k]) not in {int, float} or not math.isfinite(spec[k]) or spec[k] < 0
            for k in ["lr", "weight_decay"]
        )
        or spec["lr"] == 0
        or not isinstance(spec["inputs"]["spectral"], list)
        or not spec["inputs"]["spectral"]
    ):
        raise ValueError("Invalid frozen feature fusion training specification")
    index = json.loads(verified(spec["index"]).read_text())
    if (
        not index["records"]
        or len({r["patch_id"] for r in index["records"]}) != len(index["records"])
        or spec["steps"] * spec["batch_size"] < len(index["records"])
    ):
        raise ValueError("Training budget must cover every registered tile")
    return spec, index


def source_array(ref, index_digest, count):
    if (
        not isinstance(ref, dict)
        or set(ref) != {"path", "sha256", "layout", "index_sha256"}
        or ref["index_sha256"] != index_digest
    ):
        raise ValueError("Frozen feature geographic index differs")
    values = np.load(verified({k: ref[k] for k in ["path", "sha256"]}), mmap_mode="r")
    if values.ndim != 4 or len(values) != count or ref["layout"] not in {"nchw", "nhwc"}:
        raise ValueError("Invalid frozen feature layout or tile count")
    return values.transpose(0, 3, 1, 2) if ref["layout"] == "nhwc" else values


class TrainingData:
    def __init__(self, spec, index):
        n, digest = len(index["records"]), spec["index"]["sha256"]
        self.public = source_array(spec["inputs"]["public"], digest, n)
        self.semantic = (
            source_array(spec["inputs"]["semantic"], digest, n)
            if spec["inputs"]["semantic"] is not None
            else None
        )
        self.spectral, self.groups = [], []
        offset = 0
        for block in spec["inputs"]["spectral"]:
            if set(block) != {"array", "columns"}:
                raise ValueError("Invalid spectral block registration")
            array = source_array(block["array"], digest, n)
            columns = json.loads(verified(block["columns"]).read_text())["columns"]
            if len(columns) != array.shape[1] or len(set(columns)) != len(columns):
                raise ValueError("Spectral columns differ from the source channels")
            if any(c.split(":")[0] not in {"s2", "s1", "landsat"} for c in columns):
                raise ValueError("Only public observation statistics enter spectral inputs")
            for source in ["s2", "s1", "landsat"]:
                availability = columns.index(source + ":availability")
                group = [offset + i for i, c in enumerate(columns) if c.startswith(source + ":")]
                self.groups.append((offset + availability, group))
            self.spectral.append(array)
            offset += array.shape[1]
        shape = self.public.shape
        if (
            shape[1] != 64
            or offset != spec["model"]["spectral_channels"]
            or any(a.shape[0] != n or a.shape[-2:] != shape[-2:] for a in self.spectral)
            or (self.semantic is not None and self.semantic.shape != shape)
        ):
            raise ValueError("Frozen source grids or 64D contracts differ")
        self.n = n
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
            or weak.get("source", {}).get("dataset") != "ESA WorldCover"
            or weak["source"].get("year") != 2021
            or weak["source"].get("version") != "v200"
            or len(weak["records"]) != n
        ):
            raise ValueError("Training requires provenance-verified ESA WorldCover targets")
        labels = []
        for original, record in zip(index["records"], weak["records"]):
            if original["patch_id"] != record["patch_id"] or original["bounds"] != record["bounds"]:
                raise ValueError("WorldCover target geographic order differs")
            with np.load(verified({k: record[k] for k in ["path", "sha256"]})) as stored:
                y, valid = stored["labels"], stored["valid"]
                if (
                    y.shape != shape[-2:]
                    or valid.shape != y.shape
                    or valid.dtype != bool
                    or not np.isin(y, np.arange(12)).all()
                    or not np.array_equal(valid, y > 0)
                ):
                    raise ValueError("Invalid WorldCover class encoding or mask")
                labels.append(y.copy())
        self.labels = np.stack(labels).astype(np.int64)
        if not np.array_equal(np.bincount(self.labels.ravel(), minlength=12), weak["class_counts"]):
            raise ValueError("WorldCover target class counts differ")
        domain = np.zeros(self.labels.shape, bool)
        for a, block in zip(self.spectral, spec["inputs"]["spectral"]):
            columns = json.loads(verified(block["columns"]).read_text())["columns"]
            for source in ["s2", "s1", "landsat"]:
                domain |= np.asarray(a[:, columns.index(source + ":availability")]) > 0
        self.domain = domain & (self.labels > 0)
        self.domain_sha256 = hashlib.sha256(
            self.domain.tobytes() + self.labels.tobytes()
        ).hexdigest()
        counts = np.bincount(self.labels[self.domain], minlength=12).astype(np.float64)
        weights = np.zeros(12, np.float32)
        active = counts[1:] > 0
        weights[1:][active] = np.clip(
            np.sqrt(counts.sum() / (max(int(active.sum()), 1) * counts[1:][active])), 0.25, 4
        )
        self.class_weights = torch.from_numpy(weights)

    def batch(self, positions, *, seed):
        rng = np.random.default_rng(seed)
        horizontal, vertical = rng.integers(0, 2, size=2)

        def take(array):
            values = np.asarray(array[positions], np.float32)
            if horizontal:
                values = np.flip(values, axis=-1)
            if vertical:
                values = np.flip(values, axis=-2)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite frozen observation feature")
            return torch.from_numpy(np.ascontiguousarray(values))

        public = take(self.public)
        spectral = torch.cat([take(a) for a in self.spectral], 1)
        semantic = take(self.semantic) if self.semantic is not None else None
        labels = take(self.labels).long()
        return public, spectral, semantic, labels

    def moments(self, *, seed):
        rng = np.random.default_rng(seed)
        positions = []
        h, w = self.public.shape[-2:]
        for i in range(self.n):
            positions.append(rng.choice(h * w, min(256, h * w), replace=False))
        count = sum(len(p) for p in positions)
        chosen = rng.choice(count, 65536, replace=False) if count > 65536 else np.arange(count)

        def sampled(array):
            # Cast each tile once, especially for archived float16 semantic grids.
            return np.concatenate(
                [
                    np.asarray(array[t], np.float32).reshape(array.shape[1], -1)[:, p].T
                    for t, p in enumerate(positions)
                ]
            )[chosen]

        arrays = {
            "public": sampled(self.public),
            "spectral": np.concatenate([sampled(a) for a in self.spectral], 1),
        }
        if self.semantic is not None:
            arrays["semantic"] = sampled(self.semantic)
        result = {}
        for name, a in arrays.items():
            active = a[np.linalg.norm(a, axis=1) > 1e-8]
            if len(active):
                result[name] = (
                    active.mean(0, dtype=np.float64).astype(np.float32),
                    np.maximum(active.std(0, dtype=np.float64), 1e-4).astype(np.float32),
                )
        return result


def batch_indices(n, batch_size, seed, step):
    result = []
    start = step * batch_size
    while len(result) < batch_size:
        epoch, offset = divmod(start, n)
        order = np.random.default_rng(seed + epoch).permutation(n)
        piece = order[offset : offset + batch_size - len(result)]
        result.extend(piece.tolist())
        start += len(piece)
    return result


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def atomic_save(path, value):
    pending = path.with_suffix(".partial.pt")
    torch.save(_cpu(value), pending)
    pending.replace(path)


def identity(path):
    return {"spec_sha256": sha(path), "implementation": implementation()}


def train(path, *, max_updates=None):
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("A partial invocation requires a positive update limit")
    spec, index = specification(path)
    data = TrainingData(spec, index)
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    ident = identity(path)
    registration = root / "registration.json"
    if registration.exists() and json.loads(registration.read_text()) != ident:
        raise ValueError("Frozen fusion training identity changed")
    dump(registration, ident)
    final = root / "model.weights.pt"
    status = root / "status.json"
    latest = root / "latest.pt"
    if final.exists():
        saved = torch.load(str(final), map_location="cpu", weights_only=True)
        if saved["identity"] != ident or saved["step"] != spec["steps"]:
            raise ValueError("Frozen fusion final model identity changed")
        digest = sha(final)
        if status.exists():
            previous = json.loads(status.read_text())
            if previous.get("state") == "complete" and previous["model_sha256"] != digest:
                raise ValueError("Frozen fusion final model changed")
        summary = saved["training_summary"]
        if summary["unique_training_tiles"] != data.n or (
            summary["weak_domain_sha256"] != data.domain_sha256
        ):
            raise ValueError("Frozen fusion completed training domain changed")
        cold = FrozenFeatureFusion(**spec["model"])
        cold.load_state_dict(saved["model"], strict=True)
        result = {**summary, "model_sha256": digest}
        dump(status, result)
        latest.unlink(missing_ok=True)
        return result
    device, _, _ = _setup_device(spec["device"])
    if device.type == "npu":
        torch.npu.config.allow_internal_format = False
    torch.manual_seed(spec["seed"])
    model = FrozenFeatureFusion(**spec["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"]
    )
    step = 0
    seen = set()
    if latest.exists():
        saved = torch.load(str(latest), map_location="cpu", weights_only=True)
        if saved["identity"] != ident:
            raise ValueError("Frozen fusion recovery identity changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step = saved["step"]
        seen = set(saved["seen"])
    else:
        for name, (mean, std) in data.moments(seed=spec["seed"]).items():
            model.set_statistics(name, mean, std)

    def checkpoint():
        atomic_save(
            latest,
            {
                "identity": ident,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "seen": sorted(seen),
            },
        )

    checkpoint()
    model.train()
    weights = data.class_weights.to(device)
    limit = min(spec["steps"], step + max_updates) if max_updates is not None else spec["steps"]
    trace = root / "curve.jsonl"
    for current in range(step, limit):
        ids = batch_indices(data.n, spec["batch_size"], spec["seed"], current)
        p, r, v, y = data.batch(ids, seed=spec["seed"] + current)
        p, r, y = [a.to(device) for a in [p, r, y]]
        v = v.to(device) if v is not None else None
        mask = torch.zeros_like(r, dtype=torch.bool)
        for availability, group in data.groups:
            mask[:, group] = r[:, availability : availability + 1] > 0
        normalized_r = model.standardize(r, "spectral")
        normalized_v = model.standardize(v, "semantic") if v is not None else None
        optimizer.zero_grad(set_to_none=True)
        output = model(p, r, v)
        loss = feature_fusion_loss(
            output,
            p,
            normalized_r,
            normalized_v,
            y,
            spectral_mask=mask,
            class_weights=weights,
            **spec["loss"],
        )
        if not torch.isfinite(loss["loss"]):
            raise FloatingPointError("Nonfinite frozen fusion loss")
        loss["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite frozen fusion gradient")
        warm = min(1.0, (current + 1) / min(30, spec["steps"]))
        cosine = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * current / spec["steps"]))
        for group in optimizer.param_groups:
            group["lr"] = spec["lr"] * warm * cosine
        optimizer.step()
        step = current + 1
        seen.update(ids)
        if step % 25 == 0 or step == limit:
            row = {"step": step, **{k: float(a.detach().cpu()) for k, a in loss.items()}}
            with trace.open("a") as handle:
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
            checkpoint()
    if step < spec["steps"]:
        result = {"state": "checkpointed", "step": step, "unique_training_tiles": len(seen)}
        dump(status, result)
        return result
    if len(seen) != data.n:
        raise RuntimeError("Not all registered tiles participated in training")
    summary = {
        "state": "complete",
        "step": step,
        "unique_training_tiles": len(seen),
        "weak_domain_sha256": data.domain_sha256,
        "class_weights": data.class_weights.tolist(),
        "high_resolution_used": data.semantic is not None,
    }
    atomic_save(
        final,
        {"identity": ident, "step": step, "model": model.state_dict(), "training_summary": summary},
    )
    cold = FrozenFeatureFusion(**spec["model"])
    cold.load_state_dict(
        torch.load(str(final), map_location="cpu", weights_only=True)["model"], strict=True
    )
    result = {**summary, "model_sha256": sha(final)}
    dump(status, result)
    latest.unlink()
    return result


def export(path):
    spec, index = specification(path)
    data = TrainingData(spec, index)
    root = Path(spec["output"])
    modelpath = root / "model.weights.pt"
    saved = torch.load(str(modelpath), map_location="cpu", weights_only=True)
    if saved["identity"] != identity(path):
        raise ValueError("Frozen fusion export identity changed")
    device, _, _ = _setup_device(spec["device"])
    if device.type == "npu":
        torch.npu.config.allow_internal_format = False
    model = FrozenFeatureFusion(**spec["model"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    output = root / "embedding"
    output.mkdir(exist_ok=True)
    dest = output / "annual.npy"
    header = output / "identity.json"
    ident = {"training": identity(path), "model_sha256": sha(modelpath)}
    if header.exists() and json.loads(header.read_text()) != ident:
        raise ValueError("Frozen fusion export changed")
    dump(header, ident)
    if dest.exists() and (output / "status.json").exists():
        old = json.loads((output / "status.json").read_text())
        if old["state"] == "complete" and sha(dest) == old["sha256"]:
            return
        raise ValueError("Frozen fusion completed embedding changed")
    values = np.lib.format.open_memmap(
        output / "annual.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(data.n, 64, *data.public.shape[-2:]),
    )
    with torch.inference_mode():
        for i in range(data.n):
            p = torch.from_numpy(np.array(data.public[i : i + 1], dtype=np.float32)).to(device)
            r = torch.from_numpy(
                np.concatenate([np.asarray(a[i : i + 1], np.float32) for a in data.spectral], 1)
            ).to(device)
            v = (
                torch.from_numpy(np.array(data.semantic[i : i + 1], dtype=np.float32)).to(device)
                if data.semantic is not None
                else None
            )
            result = model(p, r, v).embedding
            if not torch.isfinite(result).all():
                raise FloatingPointError("Nonfinite learned frozen embedding")
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
            "evaluation_maps_used": False,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment frozen-feature-fusion")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["train", "export"], required=True)
    args = parser.parse_args(argv)
    (train if args.phase == "train" else export)(args.spec)
    return 0
