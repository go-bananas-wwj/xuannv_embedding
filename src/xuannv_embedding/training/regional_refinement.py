"""Registered full-region refinement with separate downstream domains and static targets."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import collate_region_batch
from xuannv_embedding.export.context import sha
from xuannv_embedding.training.adaptation import initialize_adaptation
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.losses import reconstruction_loss
from xuannv_embedding.training.runtime import TrainingSystem


def validate_scope(cache, indices):
    if (
        not indices
        or any(type(i) is not int for i in indices)
        or sorted(indices) != list(range(len(cache["records"])))
    ):
        raise ValueError("full-region training must cover every record exactly once")
    domain = [
        i
        for name in ["train", "validation", "test", "buffer"]
        for i in cache["split"].get(name, [])
    ]
    if sorted(domain) != list(range(len(cache["records"]))):
        raise ValueError("downstream partitions must remain disjoint and complete")
    return list(indices)


def cycle_indices(total, *, seed, cycle, rank, world_size):
    if total < 1 or not 0 <= rank < world_size or cycle < 0:
        raise ValueError("invalid distributed sample cycle")
    generator = torch.Generator().manual_seed(seed + cycle)
    indices = torch.randperm(total, generator=generator).tolist()
    size = math.ceil(total / world_size) * world_size
    padded = (indices * math.ceil(size / total))[:size]
    return padded[rank::world_size]


def rename_targets(model, criterion, aliases):
    decoders = model.base.decoders if hasattr(model, "base") else model.decoders
    if len(set(aliases.values())) != len(aliases) or any(
        old not in decoders
        or old not in criterion.target_cfg
        or new in decoders
        or new in criterion.target_cfg
        for old, new in aliases.items()
    ):
        raise ValueError("invalid reconstruction target migration")
    for old, new in aliases.items():
        decoders[new] = decoders.pop(old)
        criterion.target_cfg[new] = criterion.target_cfg.pop(old)


class StaticObjective(nn.Module):
    def __init__(self, base, embed_dim, target, classes):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.static_decoder = nn.Conv2d(embed_dim, classes, 1)
        self.target = target
        self.weight = 1.0
        self.head_only = True

    def forward(self, output, targets, masks, supervised_labels=None, supervised_label_masks=None):
        logits = self.static_decoder(output.embedding_map.mean(1))
        loss = reconstruction_loss(logits.float(), targets[self.target], masks[self.target], "ce")
        if self.head_only:
            results = {"total": loss}
        else:
            results = self.base(output, targets, masks, supervised_labels, supervised_label_masks)
            results["total"] = results["total"] + self.weight * loss
        results["recon_" + self.target] = loss
        results["static_weight"] = loss.new_tensor(self.weight)
        return results


def parameter_groups(system, rates, *, head_only):
    system.model.requires_grad_(not head_only)
    system.criterion.base.requires_grad_(False)
    system.criterion.static_decoder.requires_grad_(True)
    groups = {key: [] for key in rates}
    for name, parameter in system.model.named_parameters():
        if parameter.requires_grad:
            key = (
                "public"
                if name.startswith("base.") and not name.startswith("base.decoders.")
                else "adaptation"
            )
            groups[key].append(parameter)
    groups["static"] = list(system.criterion.static_decoder.parameters())
    return [
        {"params": params, "lr": rates[key], "name": key}
        for key, params in groups.items()
        if params
    ]


def registered(ref):
    if set(ref) != {"path", "sha256"} or sha(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("registered input identity changed")
    return Path(ref["path"])


def read_spec(path):
    spec = json.loads(Path(path).read_text())
    expected = {
        "protocol",
        "parent_config",
        "parent_checkpoint",
        "parent_registration",
        "cache",
        "static_labels",
        "target_aliases",
        "training_indices",
        "training",
        "evaluation_scope",
    }
    if set(spec) != expected or spec["protocol"] != "regional-static-refinement-v1":
        raise ValueError("invalid regional refinement specification")
    for key in [
        "parent_config",
        "parent_checkpoint",
        "parent_registration",
        "cache",
        "static_labels",
    ]:
        registered(spec[key])
    cache = json.loads(Path(spec["cache"]["path"]).read_text())
    validate_scope(cache, spec["training_indices"])
    labels = json.loads(Path(spec["static_labels"]["path"]).read_text())
    if (
        labels["state"] != "complete"
        or labels["classes"] != 12
        or labels["ignore_index"] != 0
        or labels["target"] != "esa_worldcover"
        or labels["image_cache_sha256"] != spec["cache"]["sha256"]
        or len(labels["records"]) != len(cache["records"])
    ):
        raise ValueError("static targets do not match all image records")
    for a, b in zip(labels["records"], cache["records"], strict=True):
        if a["patch_id"] != b["patch_id"] or a["bounds"] != b["bounds"]:
            raise ValueError("static target geography differs")
    settings = spec["training"]
    if set(settings) != {
        "seed",
        "head_steps",
        "joint_steps",
        "static_weight",
        "ramp_steps",
        "learning_rates",
        "weight_decay",
        "micro_batch",
        "accumulation",
        "world_size",
        "amp",
        "save_every",
        "min_free_gib",
    }:
        raise ValueError("unknown or missing refinement training setting")
    for key in [
        "head_steps",
        "joint_steps",
        "ramp_steps",
        "micro_batch",
        "accumulation",
        "world_size",
        "save_every",
    ]:
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError("invalid positive integer training setting")
    if type(settings["seed"]) is not int or type(settings["amp"]) is not bool:
        raise ValueError("invalid seed or precision setting")
    if set(settings["learning_rates"]) != {"public", "adaptation", "static"}:
        raise ValueError("three learning rate groups are required")
    for v in [
        *settings["learning_rates"].values(),
        settings["static_weight"],
        settings["weight_decay"],
        settings["min_free_gib"],
    ]:
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0:
            raise ValueError("invalid finite positive training value")
    if spec["evaluation_scope"] != "full-region-map-supervision; downstream-query-labels-seen":
        raise ValueError("full-region supervision must disclose downstream label exposure")
    return spec, cache, labels


def initialize(spec, cache, labels):
    config = Config.from_yaml(spec["parent_config"]["path"])
    parent = json.loads(Path(spec["parent_registration"]["path"]).read_text())
    if (
        parent["config_sha256"] != spec["parent_config"]["sha256"]
        or parent["cache_sha256"] != spec["cache"]["sha256"]
    ):
        raise ValueError("parent registration differs from cache or configuration")
    if cache["model_inputs"] != {k: asdict(v) for k, v in config.model.input_sources.items()}:
        raise ValueError("cache input schema differs from parent")
    a = parent.get("adaptation")
    if not a or a["highres_encoding"] != "transformer":
        raise ValueError("refinement requires a registered transformer-fusion parent")
    for path, digest in [
        ("base_checkpoint", "base_checkpoint_sha256"),
        ("base_config", "base_config_sha256"),
    ]:
        if sha(Path(a[path])) != a[digest]:
            raise ValueError("parent adaptation lineage changed")
    system = build_training_system(config)
    initialize_adaptation(
        system,
        config,
        argparse.Namespace(
            initialize=Path(a["base_checkpoint"]),
            base_config=Path(a["base_config"]),
            freeze_base=a["freeze_base"],
            highres_encoding="transformer",
            continue_base=False,
            train_semantic_head=a.get("train_semantic_head", False),
        ),
        cache["split"],
    )
    state = load_training_checkpoint(
        spec["parent_checkpoint"]["path"],
        model=system.model,
        criterion=system.criterion,
        expected_config_sha256=spec["parent_config"]["sha256"],
        expected_source_schema={k: asdict(v) for k, v in config.model.input_sources.items()},
        expected_regions=[d.region for d in config.data.datasets],
    )
    if state["git_sha"] != parent["git_sha"]:
        raise ValueError("parent checkpoint source identity differs")
    rename_targets(system.model, system.criterion, spec["target_aliases"])
    system.criterion.set_epoch(config.training.epochs)
    result = TrainingSystem(
        system.model,
        StaticObjective(
            system.criterion, config.model.embed_dim, labels["target"], labels["classes"]
        ),
    )
    return result, config


class RefinementSamples(torch.utils.data.Dataset):
    def __init__(self, cache, labels, aliases, indices):
        self.cache, self.labels, self.aliases, self.indices = cache, labels, aliases, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        sample = torch.load(self.cache["records"][index]["path"], weights_only=True, mmap=True)
        for key in ["targets", "target_masks"]:
            for old, new in self.aliases.items():
                if new in sample[key] or old not in sample[key]:
                    raise ValueError("cached reconstruction target alias mismatch")
                sample[key][new] = sample[key].pop(old)
        with np.load(self.labels["records"][index]["path"], allow_pickle=False) as z:
            values, valid = z["labels"].copy(), z["valid"].copy()
        size = self.cache["data"]["patch_size"]
        if (
            values.shape != (size, size)
            or values.dtype != np.uint8
            or valid.shape != values.shape
            or valid.dtype != np.bool_
            or not np.array_equal(valid, values > 0)
            or values.max() >= self.labels["classes"]
        ):
            raise ValueError("invalid static target tensor")
        sample["targets"][self.labels["target"]] = torch.from_numpy(values.astype(np.int64))
        sample["target_masks"][self.labels["target"]] = torch.from_numpy(valid.astype(np.float32))
        return sample


def batch_at(cache, labels, spec, *, cycle, offset, rank):
    settings = spec["training"]
    indices = cycle_indices(
        len(spec["training_indices"]),
        seed=settings["seed"],
        cycle=cycle,
        rank=rank,
        world_size=settings["world_size"],
    )
    indices = [spec["training_indices"][i] for i in indices]
    # Roll continuously across cycles so each update has the same effective batch.
    selected = []
    for _ in range(settings["micro_batch"]):
        if offset == len(indices):
            cycle, offset = cycle + 1, 0
            indices = cycle_indices(
                len(spec["training_indices"]),
                seed=settings["seed"],
                cycle=cycle,
                rank=rank,
                world_size=settings["world_size"],
            )
            indices = [spec["training_indices"][i] for i in indices]
        selected.append(indices[offset])
        offset += 1
    data = RefinementSamples(cache, labels, spec["target_aliases"], selected)
    return collate_region_batch([data[i] for i in range(len(data))]), cycle, offset, selected
