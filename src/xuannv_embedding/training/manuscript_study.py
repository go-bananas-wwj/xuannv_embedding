"""Source-verified, paired regional retraining for the manuscript comparison.

This runner never opens downstream reference maps. Each self-contained spec names
one region's observations, distinct weak targets, architecture and update budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import default_collate

from xuannv_embedding.config import TransformerAdapterSettings
from xuannv_embedding.data_process import subpixel_rgb
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_visual_subspace import read_json
from xuannv_embedding.models.highres_transformer import HighResTransformerModel
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.checkpoint import _atomic_torch_save
from xuannv_embedding.training.cli import _setup_device
from xuannv_embedding.training.frozen_fusion import verified
from xuannv_embedding.training.losses import batch_uniformity_loss, reconstruction_loss
from xuannv_embedding.training.masking import apply_input_masking
from xuannv_embedding.training.runtime import _autocast, _grad_scaler, _move

SCHEMA = "manuscript-regional-retraining-v1"
MASKING = {
    "enabled": True,
    "drop_availability_masks": False,
    "modality_dropout_probs": {"s2": 0.05, "s1": 0.1, "landsat": 0.1},
    "month_dropout_prob": 0.15,
    "max_months_per_sample": 1,
    "spatial_block_prob": 0.15,
    "spatial_block_size": 16,
    "spatial_block_ratio": 0.1,
}


def load_study_checkpoint(path):
    # File-backed mmap storage must not enter the Ascend host-to-device copy path.
    # Materialize CPU storages before load_state_dict/optimizer recovery instead.
    return torch.load(str(path), map_location="cpu", weights_only=True, mmap=False)


def cyclic_batch(total: int, update: int, size: int = 48) -> list[int]:
    if min(total, size) < 1 or update < 0:
        raise ValueError("Invalid cyclic sample domain")
    result = []
    offset = update * size
    while len(result) < size:
        cycle, start = divmod(offset, total)
        order = np.random.default_rng(41 + cycle).permutation(total).tolist()
        part = order[start : start + size - len(result)]
        result.extend(part)
        offset += len(part)
    return result


def phase_settings(phase: str, step: int) -> dict:
    if phase not in {"base", "public", "highres"}:
        raise ValueError("Unregistered phase")
    budget = 2400 if phase == "base" else 1600
    if type(step) is not int or not 1 <= step <= budget:
        raise ValueError("Update outside registered stage budget")
    adapt = phase == "highres" and step <= 800
    return {
        "freeze": adapt,
        "public_lr": 1e-4 if phase == "base" else 1e-5,
        "highres_lr": 3e-4 if adapt else 1e-4,
        "heads_lr": 1e-4,
        "position": (step - 1) % 800 + 1 if phase == "highres" else step,
        "length": 800 if phase == "highres" else budget,
    }


def learning_rates(phase: str, step: int) -> dict:
    s = phase_settings(phase, step)
    scale = min(1.0, s["position"] / math.ceil(0.05 * s["length"]))
    scale *= (1 + math.cos(math.pi * (s["position"] - 1) / s["length"])) / 2
    return {k: s[k + "_lr"] * scale for k in ["public", "highres", "heads"]}


def validate_sources(worldcover: dict, osm: dict) -> None:
    if (
        worldcover.get("target") != "esa_worldcover"
        or worldcover.get("classes") != 12
        or worldcover.get("ignore_index") != 0
        or worldcover.get("source", {}).get("dataset") != "ESA WorldCover"
        or worldcover["source"].get("year") != 2021
        or worldcover["source"].get("version") != "v200"
    ):
        raise ValueError("WorldCover must be the provenance-verified ESA product, not an alias")
    if (
        osm.get("dataset") != "OpenStreetMap"
        or osm.get("unknown_is_background") is not False
        or osm.get("primary_reference_maps_read") is not False
    ):
        raise ValueError("OSM must preserve unknown pixels and exclude evaluation maps")


def specification(path: Path) -> dict:
    d = read_json(path)
    keys = {
        "schema",
        "index",
        "monthly",
        "worldcover",
        "osm",
        "rgb_sources",
        "rgb_cache",
        "architecture",
        "output",
        "reserve_gib",
        "seed",
        "effective_batch",
        "phases",
    }
    if (
        not isinstance(d, dict)
        or not keys.issubset(d)
        or set(d) - keys - {"parent_base", "degraded"}
        or d["schema"] != SCHEMA
    ):
        raise ValueError("Invalid self-contained manuscript training specification")
    if (
        d["seed"] != 41
        or d["effective_batch"] != 48
        or d["reserve_gib"] < 100
        or d["phases"] != {"base": 2400, "public": 1600, "highres": 1600}
        or not Path(d["output"]).is_absolute()
        or set(d["architecture"]) != {"stp", "transformer", "stem_dim", "embed_dim"}
        or d["architecture"]["embed_dim"] != 64
    ):
        raise ValueError("Registered architecture, seed or paired budget changed")
    if "degraded" in d and type(d["degraded"]) is not bool:
        raise ValueError("degraded must be an explicit boolean")
    if "parent_base" in d:
        parent = d["parent_base"]
        if set(parent) != {"checkpoint", "spec"}:
            raise ValueError("Invalid paired ablation parent")
        original = specification(verified(parent["spec"]))
        verified(parent["checkpoint"])
        shared = [
            "index",
            "monthly",
            "worldcover",
            "osm",
            "rgb_sources",
            "rgb_cache",
            "seed",
            "effective_batch",
            "phases",
        ]
        if any(d[k] != original[k] for k in shared) or any(
            d["architecture"][k] != original["architecture"][k]
            for k in ["embed_dim", "stem_dim", "stp"]
        ):
            raise ValueError("Ablation changed shared data or backbone")
    for name in ["index", "monthly", "worldcover", "osm", "rgb_sources"]:
        verified(d[name])
    return d


def prepare_rgb(spec: dict) -> dict:
    """Cache each real optical view once on a nominal three-metre working grid."""
    root = Path(spec["rgb_cache"])
    root.mkdir(parents=True, exist_ok=True)
    raw, index = subpixel_rgb.specification(verified(spec["rgb_sources"]))
    if raw["index"] != spec["index"]:
        raise ValueError("Optical and training indices differ")
    status = root / "status.json"
    identity = {"source": spec["rgb_sources"], "implementation": sha(__file__), "side": 427}
    if status.exists():
        old = json.loads(status.read_text())
        if old.get("identity") != identity or any(
            sha(Path(old[k]["path"])) != old[k]["sha256"] for k in ["images", "coverage"]
        ):
            raise ValueError("Changed optical preparation")
        return old
    data = subpixel_rgb.Sources(raw, index)
    n, views = data.n, len(data.paths)
    shape = (n, views, 3, 427, 427)
    required = math.prod(shape) * 2 * 4 // 3
    if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30 + required:
        raise RuntimeError("Optical preparation would violate disk reserve")
    images = np.lib.format.open_memmap(
        root / "images.partial.npy", mode="w+", dtype="float16", shape=shape
    )
    masks = np.lib.format.open_memmap(
        root / "coverage.partial.npy", mode="w+", dtype="float16", shape=(n, views, 1, 427, 427)
    )
    for i in range(n):
        for j, (image, mask) in enumerate(data.frames(i)):
            image, mask = torch.from_numpy(image)[None], torch.from_numpy(mask)[None]
            coverage = F.interpolate(mask, (427, 427), mode="area")
            pixels = F.interpolate(image * mask, (427, 427), mode="area")
            image = torch.where(coverage > 0, pixels / coverage.clamp_min(1e-6), 0)
            images[i, j] = image[0].numpy().astype("float16")
            masks[i, j] = coverage[0].numpy().astype("float16")
        if i % 20 == 0:
            dump(root / "progress.json", {"state": "preparing", "tiles": i + 1, "total": n})
    images.flush()
    masks.flush()
    result = {
        "state": "complete",
        "identity": identity,
        "actual_views": views,
        "index": spec["index"],
        "time_contract": raw["time_contract"],
        "static": all(v["source_time"]["precision"] == "year" for v in data.contract["views"]),
    }
    for key, stem in [("images", "images"), ("coverage", "coverage")]:
        (root / (stem + ".partial.npy")).replace(root / (stem + ".npy"))
        p = root / (stem + ".npy")
        result[key] = {"path": str(p), "sha256": sha(p)}
    dump(status, result)
    return result


class Observations:
    def __init__(self, spec: dict, *, highres: bool):
        self.spec, self.highres = spec, highres
        self.index = json.loads(verified(spec["index"]).read_text())
        self.meta = json.loads(verified(spec["monthly"]).read_text())
        wc = json.loads(verified(spec["worldcover"]).read_text())
        osm = json.loads(verified(spec["osm"]).read_text())
        validate_sources(wc, osm)
        if self.meta["index"] != spec["index"] or osm["index"] != spec["index"]:
            raise ValueError("Training source geographic identity differs")
        records = self.index["records"]
        self.n = len(records)
        if [r["patch_id"] for r in wc["records"]] != [r["patch_id"] for r in records]:
            raise ValueError("WorldCover tile order differs")
        self.frames = np.load(verified(self.meta["frames"]), mmap_mode="r")
        self.masks = np.load(verified(self.meta["masks"]), mmap_mode="r")
        self.area = np.load(verified(osm["area"]), mmap_mode="r")
        self.geometry = np.load(verified(osm["geometry"]), mmap_mode="r")
        self.wc, self.wc_valid = [], []
        for row in wc["records"]:
            if sha(Path(row["path"])) != row["sha256"]:
                raise ValueError("WorldCover tile changed")
            with np.load(row["path"], allow_pickle=False) as values:
                self.wc.append(values["labels"].copy())
                self.wc_valid.append(values["valid"].copy())
        if highres:
            p = Path(spec["rgb_cache"]) / "status.json"
            status = json.loads(p.read_text())
            if status["state"] != "complete" or status["index"] != spec["index"]:
                raise ValueError("Incomplete optical cache")
            self.images = np.load(verified(status["images"]), mmap_mode="r")
            self.coverage = np.load(verified(status["coverage"]), mmap_mode="r")
            contract = json.loads(verified(status["time_contract"]).read_text())
            if contract["months"] != self.meta["months"]:
                raise ValueError("Optical and public calendars differ")
            self.weights = torch.from_numpy(np.load(verified(contract["weights"]))).float()
            self.static = status["static"]

    def sample(self, i: int) -> dict:
        frames, masks, targets, target_masks = {}, {}, {}, {}
        x = torch.from_numpy(np.array(self.frames[i], dtype="float32"))
        valid = torch.from_numpy(np.array(self.masks[i], dtype="float32"))
        for k, (name, (a, b)) in enumerate(self.meta["band_slices"].items()):
            frames[name] = x[:, a:b]
            masks[name] = valid[:, k].flatten(1).amax(1)
            targets[name + "_recon"] = x[:, a:b].clone()
            target_masks[name + "_recon"] = valid[:, k]
        result = {
            "source_frames": frames,
            "source_masks": masks,
            "timestamps": torch.tensor(self.meta["months"]),
            "targets": targets,
            "target_masks": target_masks,
            "highres_frames": {},
            "highres_masks": {},
            "worldcover": torch.as_tensor(self.wc[i].copy()).long(),
            "worldcover_valid": torch.as_tensor(self.wc_valid[i].copy()).float(),
            "osm_area": torch.tensor(np.array(self.area[i], dtype="int64")),
            "osm_geometry": torch.tensor(np.array(self.geometry[i], dtype="int64")),
        }
        if self.highres:
            image = torch.tensor(np.array(self.images[i], dtype="float32")) / 255
            coverage = torch.tensor(np.array(self.coverage[i], dtype="float32"))
            if self.spec.get("degraded", False):
                low_mask = F.interpolate(coverage, (128, 128), mode="area")
                low = F.interpolate(image * coverage, (128, 128), mode="area") / low_mask.clamp_min(
                    1e-6
                )
                image = F.interpolate(low, (427, 427), mode="bilinear", align_corners=False)
                image = torch.where(coverage > 0, image, 0)
            if self.static:
                count = coverage.sum(0)
                image = (image * coverage).sum(0) / count.clamp_min(1e-6)
                coverage = count / len(coverage)
            else:
                weights = self.weights.T[:, :, None, None, None]
                count = (weights * coverage[None]).sum(1)
                image = (weights * (image * coverage)[None]).sum(1) / count.clamp_min(1e-6)
                coverage = count.clamp(0, 1)
            result["highres_frames"] = {"optical": image}
            result["highres_masks"] = {"optical": coverage}
            a, m = image.reshape(-1, 3, 427, 427), coverage.reshape(-1, 1, 427, 427)
            small_mask = F.interpolate(m, (128, 128), mode="area")
            small = F.interpolate(a * m, (128, 128), mode="area") / small_mask.clamp_min(1e-6)
            result["optical_target"] = small[0] if self.static else small
            result["optical_valid"] = small_mask[0, 0] if self.static else small_mask[:, 0]
        return result


class Objective(nn.Module):
    def __init__(self):
        super().__init__()
        self.worldcover = nn.Conv2d(64, 12, 1)
        self.area = nn.Conv2d(64, 7, 1)
        self.geometry = nn.Conv2d(64, 4, 1)

    def forward(self, out, batch):
        result = {}
        for name, weight in [("s2", 0.4), ("s1", 0.125), ("landsat", 0.225)]:
            key = name + "_recon"
            result[key] = weight * reconstruction_loss(
                out.reconstructions[key].float(), batch["targets"][key], batch["target_masks"][key]
            )
        mean = out.embedding_map.mean(1)
        for name, decoder, key, weight in [
            ("worldcover", self.worldcover, "worldcover", 0.45),
            ("osm_area", self.area, "osm_area", 0.14),
            ("osm_geometry", self.geometry, "osm_geometry", 0.14),
        ]:
            target = batch[key]
            valid = (target > 0).float()
            if name == "worldcover":
                valid *= batch["worldcover_valid"]
            result[name] = weight * reconstruction_loss(decoder(mean).float(), target, valid, "ce")
        result["uniformity"] = 0.06 * batch_uniformity_loss(out.embedding.float())
        if "optical_target" in batch:
            prediction = out.reconstructions["optical_recon"].float()
            if batch["optical_target"].ndim == 4:
                prediction = prediction.mean(1)
            result["optical_recon"] = 0.9 * reconstruction_loss(
                prediction, batch["optical_target"], batch["optical_valid"]
            )
        result["total"] = sum(result.values())
        return result


class StudySystem(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model, self.criterion = model, criterion

    def forward(self, batch):
        output = self.model(
            batch["source_frames"],
            batch["source_masks"],
            batch["timestamps"],
            batch["highres_frames"],
            batch["highres_masks"],
        )
        return self.criterion(output, batch)


def build(spec: dict, metadata: dict, phase: str):
    torch.manual_seed(41)
    arch = spec["architecture"]
    channels = {s: b - a for s, (a, b) in metadata["band_slices"].items()}
    first = metadata["months"][0]
    base = AEFModel(
        sensor_channels=channels,
        target_heads={s + "_recon": ("continuous", c) for s, c in channels.items()},
        source_roles={s: "temporal" for s in channels},
        embed_dim=64,
        stem_dim=arch["stem_dim"],
        num_months=12,
        ref_year=first // 100,
        ref_month=first % 100,
        stp=arch["stp"],
    )
    criterion = Objective()
    if phase == "highres":
        base = HighResTransformerModel(
            base,
            {"optical": 3},
            {"optical_recon": 3},
            settings=TransformerAdapterSettings(
                **{
                    **arch["transformer"],
                    "injection_blocks": tuple(arch["transformer"]["injection_blocks"]),
                }
            ),
            freeze_base=True,
        )
    return base, criterion


def optimizer_for(model, criterion, phase, step):
    if isinstance(model, HighResTransformerModel):
        model.set_freeze_base(phase_settings(phase, step)["freeze"])
    groups = {k: [] for k in ["public", "highres", "heads"]}
    for name, p in model.named_parameters():
        if p.requires_grad:
            group = (
                "heads"
                if "decoders." in name
                else ("public" if phase != "highres" or name.startswith("base.") else "highres")
            )
            groups[group].append(p)
    groups["heads"].extend(criterion.parameters())
    return torch.optim.AdamW(
        [
            {"params": p, "name": k, "lr": learning_rates(phase, step)[k]}
            for k, p in groups.items()
            if p
        ],
        weight_decay=0.05,
    )


def train(
    spec_path,
    phase,
    device,
    micro_batch,
    *,
    gate_updates=None,
    resume=True,
    stop_after_updates=None,
):
    spec = specification(spec_path)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    directory = f"gates-w{world}-b{micro_batch}" if gate_updates else "runs"
    output = Path(spec["output"]) / directory / phase
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "spec_sha256": sha(spec_path),
        "implementation": sha(__file__),
        "model_implementation": sha(Path(__file__).parents[1] / "models/highres_transformer.py"),
        "source_git_sha": os.environ.get("XUANNV_GIT_SHA"),
        "phase": phase,
        "micro_batch": micro_batch,
        "effective_batch": 48,
        "world_size": world,
    }
    if (
        not identity["source_git_sha"]
        or micro_batch not in [1, 2, 3, 4, 6, 8]
        or 48 % (micro_batch * world)
    ):
        raise ValueError("Pinned source and a divisor of effective batch48 are required")
    parent = Path(spec["output"]) / directory / "base/final.pt"
    parent_spec_digest = sha(spec_path)
    if spec.get("parent_base") and phase == "highres" and not gate_updates:
        parent = verified(spec["parent_base"]["checkpoint"])
        parent_spec_digest = spec["parent_base"]["spec"]["sha256"]
    if phase != "base":
        identity["parent_sha256"] = sha(parent)
    if (output / "status.json").exists():
        old = json.loads((output / "status.json").read_text())
        if old.get("identity") != identity:
            raise ValueError("Training source/specification changed")
        if old.get("state") == "complete":
            verified(old["checkpoint"])
            return old
    data = Observations(spec, highres=phase == "highres")
    model, criterion = build(spec, data.meta, phase)
    if phase != "base":
        state = load_study_checkpoint(parent)
        if (
            state["step"] != (gate_updates or 2400)
            or state["identity"]["spec_sha256"] != parent_spec_digest
        ):
            raise ValueError("Paired base is not the registered terminal checkpoint")
        (model.base if phase == "highres" else model).load_state_dict(state["model"], strict=True)
        criterion.load_state_dict(state["criterion"], strict=True)
    dev, distributed, local = _setup_device(device)
    if dev.type == "npu":
        torch.npu.config.allow_internal_format = False
    model.to(dev)
    criterion.to(dev)
    latest = output / "latest.pt"
    state = load_study_checkpoint(latest) if resume and latest.exists() else None
    start = state["step"] + 1 if state else 1
    if state and state["identity"] != identity:
        raise ValueError("Checkpoint recovery identity changed")
    optimizer = optimizer_for(model, criterion, phase, state["step"] if state else start)
    system = StudySystem(model, criterion)

    def wrap():
        if not distributed:
            return system
        return torch.nn.parallel.DistributedDataParallel(
            system, device_ids=[local] if dev.type == "npu" else None, find_unused_parameters=True
        )

    wrapped = wrap()
    scaler = _grad_scaler(dev, dev.type == "npu")
    if state:
        model.load_state_dict(state["model"], strict=True)
        criterion.load_state_dict(state["criterion"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        random = state["rank_states"][rank]
        torch.set_rng_state(random["torch_rng"])
        if dev.type == "npu":
            torch.npu.set_rng_state(random["device_rng"], dev)
        if scaler and random["scaler"] is not None:
            scaler.load_state_dict(random["scaler"])
    total = gate_updates or spec["phases"][phase]
    stop = total if stop_after_updates is None else stop_after_updates
    if type(stop) is not int or not start - 1 <= stop <= total:
        raise ValueError("Stop must lie within the registered remaining budget")
    accumulation = 48 // (world * micro_batch)
    offset = 0 if phase == "base" else 2400
    seen = set(state.get("tiles_seen", []) if state else [])
    elapsed_before = state.get("elapsed", 0.0) if state else 0.0
    started = time.monotonic()
    skipped = 0
    step = start
    while step <= stop:
        if (Path(spec["output"]).parent / "STOP").exists() or shutil.disk_usage(output).free < spec[
            "reserve_gib"
        ] * 2**30:
            raise RuntimeError("STOP or disk reserve reached")
        if phase == "highres" and step == 801:
            optimizer = optimizer_for(model, criterion, phase, step)
            del wrapped
            wrapped = wrap()
        rates = learning_rates(phase, step)
        for group in optimizer.param_groups:
            group["lr"] = rates[group["name"]]
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        ids = cyclic_batch(data.n, offset + step - 1)
        sums = {}
        for j in range(0, 48, world * micro_batch):
            selected = ids[j + rank * micro_batch : j + (rank + 1) * micro_batch]
            batch = _move(default_collate([data.sample(i) for i in selected]), dev)
            torch.manual_seed(41 + (offset + step) * 1009 + j + rank * micro_batch)
            batch = apply_input_masking(batch, MASKING)
            context = (
                wrapped.no_sync() if distributed and j + world * micro_batch < 48 else nullcontext()
            )
            with context:
                with _autocast(dev, dev.type == "npu"):
                    losses = wrapped(batch)
                    loss = losses["total"] / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite manuscript training loss")
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach().float().cpu()) / accumulation
        if scaler:
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], 1.0
        )
        if scaler:
            previous = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous:
                skipped += 1
                if skipped > 20:
                    raise FloatingPointError("Repeated AMP overflows")
                continue
        else:
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite manuscript gradient")
            optimizer.step()
        seen.update(ids)
        elapsed = elapsed_before + time.monotonic() - started
        if distributed and (step == 1 or step % 10 == 0 or step == stop):
            averaged = torch.tensor(list(sums.values()), device=dev)
            torch.distributed.all_reduce(averaged)
            sums = dict(zip(sums, (averaged / world).cpu().tolist()))
        if (step == 1 or step % 10 == 0 or step == stop) and rank == 0:
            row = {
                "step": step,
                "phase": phase,
                "losses": sums,
                "gradient_norm": float(norm.cpu()),
                "rates": rates,
                "elapsed_seconds": elapsed,
                "tiles_seen": len(seen),
            }
            with (output / "curve.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            dump(output / "status.json", {"state": "running", "identity": identity, **row})
        if step % 25 == 0 or step == stop or step in {800, 801}:
            random = {
                "torch_rng": torch.get_rng_state(),
                "device_rng": torch.npu.get_rng_state(dev) if dev.type == "npu" else None,
                "scaler": scaler.state_dict() if scaler else None,
            }
            ranks = [None] * world
            if distributed:
                torch.distributed.all_gather_object(ranks, random)
            else:
                ranks = [random]
            snapshot = {
                "schema": SCHEMA,
                "identity": identity,
                "step": step,
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rank_states": ranks,
                "tiles_seen": sorted(seen),
                "elapsed": elapsed,
                "skipped": skipped,
            }
            if rank == 0:
                _atomic_torch_save(snapshot, latest)
            if distributed:
                torch.distributed.barrier()
        step += 1
    if stop < total:
        result = {
            "state": "checkpointed",
            "identity": identity,
            "step": stop,
            "checkpoint": {"path": str(latest), "sha256": sha(latest)},
        }
        if rank == 0:
            dump(output / "status.json", result)
        if distributed:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return result
    terminal = {
        "schema": SCHEMA,
        "identity": identity,
        "step": total,
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "inference_only": True,
    }
    if rank == 0:
        _atomic_torch_save(terminal, output / "final.pt")
    if distributed:
        torch.distributed.barrier()
    result = {
        "state": "complete",
        "identity": identity,
        "step": total,
        "tiles_seen": len(seen),
        "elapsed_seconds": elapsed_before + time.monotonic() - started,
        "checkpoint": {"path": str(output / "final.pt"), "sha256": sha(output / "final.pt")},
    }
    if rank == 0:
        dump(output / "status.json", result)
        latest.unlink(missing_ok=True)
    if distributed:
        torch.distributed.destroy_process_group()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "base", "public", "highres"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--gate-updates", type=int)
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        prepare_rgb(specification(args.spec))
    else:
        train(
            args.spec,
            args.phase,
            args.device,
            args.micro_batch,
            gate_updates=args.gate_updates,
            stop_after_updates=args.stop_after_updates,
        )
    return 0
