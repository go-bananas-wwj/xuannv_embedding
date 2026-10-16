"""Checkpoint validation and exact-state recovery for registered regional runs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from xuannv_embedding.export.context import sha
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.distributed_experiment import (
    rank_random_state,
    restore_rank_random_state,
)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def validate_resume_checkpoint(path, *, config_sha256, sources, regions, max_steps) -> int:
    state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not isinstance(state.get("metrics"), dict):
        raise ValueError("Checkpoint must contain a training-state mapping and progress metadata")
    if (
        state.get("format_version") != "1"
        or state.get("config_sha256") != config_sha256
        or state.get("source_schema") != sources
        or state.get("regions") != regions
    ):
        raise ValueError("Resume checkpoint provenance differs from the registered run")
    epoch = state.get("epoch")
    if type(epoch) is not int or not 0 <= epoch < max_steps:
        raise ValueError("Invalid checkpoint training step")
    if state.get("metrics", {}).get("steps") != epoch + 1:
        raise ValueError("Checkpoint progress metadata is inconsistent")
    resume = state["metrics"].get("regional_resume")
    if resume is not None:
        if (
            not isinstance(resume, dict)
            or resume.get("schema") != "regional-resume-v1"
            or not isinstance(resume.get("registration"), dict)
            or not isinstance(resume.get("rank_states"), list)
            or not resume["rank_states"]
        ):
            raise ValueError("Checkpoint has incomplete per-rank recovery state")
        for rank in resume["rank_states"]:
            if not isinstance(rank, dict) or any(
                not isinstance(rank.get(key), torch.Tensor) or rank[key].dtype != torch.uint8
                for key in ["torch_rng_state", "sampler_rng_state"]
            ):
                raise ValueError("Checkpoint has invalid random-generator state")
    optimizer = state.get("optimizer")
    if (
        not isinstance(optimizer, dict)
        or not optimizer.get("param_groups")
        or not isinstance(optimizer.get("state"), dict)
        or not state.get("model")
        or state.get("criterion") is None
    ):
        raise ValueError("Checkpoint lacks model, auxiliary heads or optimizer recovery state")
    for component in ["model", "criterion", "optimizer"]:
        for tensor in _tensors(state[component]):
            if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
                tensor
            ).all():
                raise ValueError("Checkpoint contains nonfinite tensors")
    return epoch + 1


def registration(spec, spec_path: Path, world: int, micro_batch: int) -> dict:
    return {
        "spec_sha256": sha(spec_path),
        "config_sha256": spec["config"]["sha256"],
        "cache_sha256": spec["cache"]["sha256"],
        "world_size": world,
        "micro_batch": micro_batch,
        "effective_batch": spec["training"]["effective_batch"],
    }


def capture_rank(generator, device, scaler) -> dict:
    return {**rank_random_state(device, scaler), "sampler_rng_state": generator.get_state()}


def recover(
    path, system, optimizer, scaler, generator, device, config, spec, identity, cache, *, rank: int
) -> dict:
    state = load_training_checkpoint(
        path,
        model=system.model,
        criterion=system.criterion,
        optimizer=optimizer,
        expected_config_sha256=spec["config"]["sha256"],
        expected_source_schema={k: asdict(v) for k, v in config.model.input_sources.items()},
        expected_regions=[d.region for d in config.data.datasets],
    )
    step = state["epoch"] + 1
    if not 0 < step <= spec["training"]["steps"] or state["metrics"].get("steps") != step:
        raise ValueError("Invalid resumed training progress")
    saved = state["metrics"].get("regional_resume")
    if saved is not None:
        if saved.get("schema") != "regional-resume-v1" or saved.get("registration") != identity:
            raise ValueError("Resume specification, world size or micro batch changed")
        ranks = saved["rank_states"]
        if len(ranks) != identity["world_size"]:
            raise ValueError("Resume random state world size differs")
        generator.set_state(ranks[rank]["sampler_rng_state"].cpu())
        restore_rank_random_state(ranks[rank], device, scaler)
        return {
            "completed_steps": step,
            "elapsed_seconds": saved["elapsed_seconds"],
            "skipped_updates": saved["skipped_updates"],
            "recovery_mode": "exact",
        }
    # Historical checkpoints retain weights/moments, but not AMP/masking RNG.
    # Recreate successful-update sampler draws and disclose the non-exact recovery.
    sizes = (
        [len(d["indices"]) for d in cache["domains"]]
        if "domains" in cache
        else [len(cache["records"])]
    )
    accumulation = 48 // (identity["world_size"] * identity["micro_batch"])
    generator.manual_seed(spec["training"]["seed"])
    for update in range(step):
        for _ in range(accumulation):
            torch.randperm(sizes[update % len(sizes)], generator=generator)
    torch.manual_seed(spec["training"]["seed"] + step * 1009 + rank)
    return {
        "completed_steps": step,
        "elapsed_seconds": 0.0,
        "skipped_updates": 0,
        "recovery_mode": "legacy_rng_reinitialized",
    }
