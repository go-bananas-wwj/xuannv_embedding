"""Non-persistent loss-gradient diagnostics on a registered training batch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import collate_region_batch
from xuannv_embedding.training.adaptation import initialize_adaptation
from xuannv_embedding.training.cli import _git_sha, _setup_device, build_training_system
from xuannv_embedding.training.experiment import CachedSamples, _json, _sha
from xuannv_embedding.training.masking import apply_input_masking
from xuannv_embedding.training.runtime import _move


def gradient_report(losses, parameters):
    """Measure per-objective gradients without modifying parameter .grad or values."""
    named = [(name, p) for name, p in parameters.items() if p.requires_grad]
    result = {}
    for name, loss in losses.items():
        gradients = torch.autograd.grad(
            loss, [p for _, p in named], retain_graph=True, allow_unused=True
        )
        groups = {}
        for (key, _), gradient in zip(named, gradients, strict=True):
            group = ".".join(key.split(".")[:3])
            if gradient is not None:
                groups.setdefault(group, []).append(gradient.detach().float().square().sum())
        stats = {}
        for group, values in groups.items():
            norm = torch.stack(values).sum().sqrt()
            stats[group] = {"norm": float(norm.cpu()), "finite": bool(torch.isfinite(norm).item())}
        result[name] = {"loss": float(loss.detach().float().cpu()), "groups": stats}
    return result


def _frozen_digest(system):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in system.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            count += parameter.numel()
    return digest.hexdigest(), count


def run(args):
    if args.output.exists():
        raise FileExistsError("diagnostic output exists")
    config = Config.from_yaml(args.config)
    cache = json.loads((args.cache / "cache.json").read_text())
    torch.manual_seed(config.experiment.seed)
    system = build_training_system(config)
    adaptation = initialize_adaptation(system, config, args, cache["split"])
    device, distributed, _ = _setup_device(args.device)
    if distributed:
        raise ValueError("diagnostics require one independent device")
    system.to(device)
    # Evaluation mode disables checkpoint recomputation and vMF noise. All gradient
    # paths remain enabled. This diagnostic trajectory is discarded, never trained on.
    system.eval()
    system.criterion.set_epoch(config.training.epochs)
    indices = cache["split"]["train"][:2]
    samples = CachedSamples(cache, indices)
    for record in samples.records:
        if _sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("cached diagnostic sample changed")
    batch = _move(collate_region_batch([samples[i] for i in range(len(samples))]), device)
    batch = apply_input_masking(batch, asdict(config.training.input_masking))
    before, frozen_count = _frozen_digest(system)
    optimizer = torch.optim.AdamW(
        system.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    traces = []
    for step in range(2):
        values = system(batch)
        terms = {
            "semantic": values["semantic_probe_weighted"],
            "uniformity": values["uniformity_weighted"],
        }
        for name, head in config.model.target_heads.items():
            terms[name] = values[f"recon_{name}"] * head.weight
        traces.append(
            {
                "diagnostic_step": step,
                "terms": gradient_report(terms, dict(system.named_parameters())),
            }
        )
        if step == 0:
            values["total"].backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    after, _ = _frozen_digest(system)
    result = {
        "state": "complete",
        "code_commit": _git_sha(),
        "config_sha256": _sha(args.config),
        "cache_sha256": _sha(args.cache / "cache.json"),
        "adaptation": adaptation,
        "training_indices": indices,
        "test_scored": False,
        "mode": (
            "float32 eval-mode masked training batch; " "one disposable update; no saved checkpoint"
        ),
        "frozen_parameter_count": frozen_count,
        "frozen_digest_before": before,
        "frozen_digest_after": after,
        "frozen_unchanged": before == after,
        "traces": traces,
        "trainable_groups": sorted(
            {".".join(n.split(".")[:3]) for n, p in system.named_parameters() if p.requires_grad}
        ),
    }
    if before != after:
        raise RuntimeError("frozen parameters changed")
    _json(args.output, result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("traces", "adaptation")}),
        flush=True,
    )
