"""Disposable head-only control: fitting a readout cannot change a frozen embedding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import collate_region_batch
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import _git_sha, _setup_device, build_training_system
from xuannv_embedding.training.experiment import CachedSamples, _json, _sha
from xuannv_embedding.training.runtime import _move


def tensor_digest(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def head_only_step(system, batch: dict, *, lr: float, weight_decay: float) -> dict:
    """Update only a temporary system's semantic probes; never save its weights."""
    if batch.get("highres_frames") or batch.get("highres_masks"):
        raise ValueError("head-only control requires public inputs without highres")
    probe = system.criterion.semantic_probe
    if probe is None or system.criterion.semantic_probe_weight <= 0:
        raise ValueError("head-only control requires active semantic probes")
    system.eval().requires_grad_(False)
    probe.probes.requires_grad_(True)
    system.zero_grad(set_to_none=True)
    named = {n: p for n, p in system.named_parameters() if p.requires_grad}
    model_before = tensor_digest(system.model.state_dict())
    head_before = tensor_digest(probe.probes.state_dict())
    frozen_before = tensor_digest({n: p for n, p in system.state_dict().items() if n not in named})

    def embedding():
        with torch.no_grad():
            return system.model(
                batch["source_frames"], batch["source_masks"], batch["timestamps"]
            ).embedding_map

    before = embedding()
    if not torch.isfinite(before).all():
        raise FloatingPointError("nonfinite control embedding")
    before_digest = tensor_digest({"embedding": before})
    loss, _ = probe(before, batch.get("supervised_labels"), batch.get("supervised_label_masks"))
    loss = loss * system.criterion.semantic_probe_weight
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite control semantic loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(named.values(), 5.0, error_if_nonfinite=True)
    if norm <= 0:
        raise ValueError("control semantic gradient is zero")
    optimizer = torch.optim.AdamW(named.values(), lr=lr, weight_decay=weight_decay)
    optimizer.step()
    after = embedding()
    result = {
        "embedding_unchanged": torch.equal(before, after),
        "embedding_sha256_before": before_digest,
        "embedding_sha256_after": tensor_digest({"embedding": after}),
        "model_unchanged": model_before == tensor_digest(system.model.state_dict()),
        "model_sha256_before": model_before,
        "model_sha256_after": tensor_digest(system.model.state_dict()),
        "frozen_unchanged": frozen_before
        == tensor_digest({n: p for n, p in system.state_dict().items() if n not in named}),
        "head_changed": head_before != tensor_digest(probe.probes.state_dict()),
        "head_sha256_before": head_before,
        "head_sha256_after": tensor_digest(probe.probes.state_dict()),
        "semantic_loss_before": float(loss.detach().cpu()),
        "gradient_norm": float(norm.cpu()),
        "trainable": list(named),
        "trainable_parameters": sum(p.numel() for p in named.values()),
        "disposable_optimizer_updates": 1,
        "saved_optimizer_updates": 0,
    }
    if not all(
        result[k]
        for k in ("embedding_unchanged", "model_unchanged", "frozen_unchanged", "head_changed")
    ):
        raise RuntimeError("head-only structural control failed")
    return result


def run(args) -> None:
    if args.output.exists():
        raise FileExistsError("head-only diagnostic output already exists")
    config = Config.from_yaml(args.config)
    if any(s.role == "highres" for s in config.model.input_sources.values()):
        raise ValueError("head-only control requires a public-only model")
    cache_path = args.cache / "cache.json"
    cache = json.loads(cache_path.read_text())
    registration_path = args.checkpoint.parent / "run.json"
    registration = json.loads(registration_path.read_text())
    if registration["config_sha256"] != _sha(args.config):
        raise ValueError("control configuration differs from registration")
    if registration["cache_sha256"] != _sha(cache_path):
        raise ValueError("control cache differs from registration")
    if registration["train_indices"] != cache["split"]["train"]:
        raise ValueError("control training partition changed")
    device, distributed, _ = _setup_device(args.device)
    if distributed:
        raise ValueError("head-only control requires one process")
    torch.manual_seed(config.experiment.seed)
    system = build_training_system(config)
    checkpoint_sha = _sha(args.checkpoint)
    state = load_training_checkpoint(
        args.checkpoint,
        model=system.model,
        criterion=system.criterion,
        expected_config_sha256=_sha(args.config),
        expected_source_schema={k: asdict(v) for k, v in config.model.input_sources.items()},
        expected_regions=[d.region for d in config.data.datasets],
    )
    if state["git_sha"] != registration["git_sha"]:
        raise ValueError("control parent code identity mismatch")
    system.to(device).float()
    system.criterion.set_epoch(config.training.epochs)
    indices = cache["split"]["train"][:2]
    samples = CachedSamples(cache, indices)
    for record in samples.records:
        if _sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("control training sample changed")
    batch = _move(collate_region_batch([samples[i] for i in range(len(samples))]), device)
    report = head_only_step(
        system, batch, lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    if _sha(args.checkpoint) != checkpoint_sha:
        raise RuntimeError("source checkpoint changed")
    _json(
        args.output,
        {
            "state": "complete",
            "code_commit": _git_sha(),
            "device": str(device),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_unchanged": True,
            "config_sha256": _sha(args.config),
            "cache_sha256": _sha(cache_path),
            "registration_sha256": _sha(registration_path),
            "training_indices": indices,
            "sample_sha256": [r["sha256"] for r in samples.records],
            "test_scored": False,
            "mode": "FP32 eval; unmasked public training inputs; one disposable head-only update",
            "result": report,
        },
    )
    print(
        json.dumps({"state": "complete", "embedding_unchanged": report["embedding_unchanged"]}),
        flush=True,
    )
