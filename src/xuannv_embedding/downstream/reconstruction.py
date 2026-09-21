"""Validation-only masked reconstruction with explicit input and scoring domains.

Native decoder diagnostic, not a matched-capacity embedding readout. Prefix masking
checks input dependence; it does not undo future observations seen during pretraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import collate_region_batch
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import _git_sha, _setup_device, build_training_system
from xuannv_embedding.training.experiment import CachedSamples, _json, _sha
from xuannv_embedding.training.experiment_export import validate_export_identity
from xuannv_embedding.training.runtime import _move


def hidden_month_inputs(batch, sources, month, *, prefix):
    count = batch["timestamps"].shape[1]
    names = set(batch.get("source_frames", {})) | set(batch.get("highres_frames", {}))
    if not sources or not set(sources) <= names or not 0 <= month < count:
        raise ValueError("unknown hidden source or invalid month")
    result = dict(batch)
    for frames_key, masks_key in [
        ("source_frames", "source_masks"),
        ("highres_frames", "highres_masks"),
    ]:
        result[frames_key] = dict(batch.get(frames_key, {}))
        result[masks_key] = dict(batch.get(masks_key, {}))
        for name, original in batch.get(frames_key, {}).items():
            if name not in sources and not prefix:
                continue
            frame, mask = original.clone(), batch[masks_key][name].clone()
            if frame.ndim == 5:
                if frame.shape[1] != count or mask.shape[:2] != frame.shape[:2]:
                    raise ValueError("monthly frame/mask dimensions differ")
                if name in sources:
                    frame[:, month] = 0
                    mask[:, month] = 0
                if prefix:
                    frame[:, month + 1 :] = 0
                    mask[:, month + 1 :] = 0
            elif frame.ndim == 4:
                # An undated static input cannot certify a prefix time boundary.
                frame.zero_()
                mask.zero_()
            else:
                raise ValueError("unsupported observation dimensions")
            result[frames_key][name], result[masks_key][name] = frame, mask
    return result


def temporal_baseline(values, valid, month, *, prefix):
    values, valid = np.asarray(values, dtype=np.float64), np.asarray(valid, dtype=bool)
    if values.ndim != 4 or valid.shape != (values.shape[0], *values.shape[2:]):
        raise ValueError("temporal baseline requires TCHW values and THW masks")
    if not 0 <= month < len(values):
        raise ValueError("invalid target month")
    left, right = np.zeros_like(values[0]), np.zeros_like(values[0])
    li, ri = np.full(valid.shape[1:], -1), np.full(valid.shape[1:], -1)
    for i in range(month):
        left[:, valid[i]], li[valid[i]] = values[i][:, valid[i]], i
    if not prefix:
        for i in range(len(values) - 1, month, -1):
            right[:, valid[i]], ri[valid[i]] = values[i][:, valid[i]], i
    both = (li >= 0) & (ri >= 0)
    weight = np.divide(month - li, ri - li, out=np.zeros_like(li, dtype=float), where=both)
    interpolated = left * (1 - weight[None]) + right * weight[None]
    prediction = np.where(both[None], interpolated, np.where((li >= 0)[None], left, right))
    return prediction, (li >= 0) | (ri >= 0)


def metric_sums(prediction, truth, valid):
    prediction, truth = np.asarray(prediction, dtype=np.float64), np.asarray(
        truth, dtype=np.float64
    )
    if prediction.shape != truth.shape or truth.ndim != 3:
        raise ValueError("reconstruction metrics require matching CHW arrays")
    valid = np.broadcast_to(np.asarray(valid, dtype=bool), truth.shape)
    rows = []
    for p, y, mask in zip(prediction, truth, valid, strict=True):
        if not np.isfinite(p[mask]).all() or not np.isfinite(y[mask]).all():
            raise ValueError("nonfinite value inside reconstruction evaluation domain")
        error = p[mask] - y[mask]
        rows.append([len(error), np.sum(error**2), np.sum(np.abs(error)), np.sum(error)])
    return np.asarray(rows, dtype=np.float64)


def summarize_sums(sums):
    return [
        {
            "count": int(n),
            "rmse": float(np.sqrt(squared / n)) if n else None,
            "mae": float(absolute / n) if n else None,
            "bias": float(signed / n) if n else None,
        }
        for n, squared, absolute, signed in sums
    ]


def _load_model(config, document, training, checkpoint, config_sha):
    system = build_training_system(config)
    adaptation = training.get("adaptation")
    if adaptation:
        from xuannv_embedding.training.adaptation import initialize_adaptation

        for key in ("base_checkpoint", "base_config"):
            if _sha(Path(adaptation[key])) != adaptation[key + "_sha256"]:
                raise ValueError("parent provenance changed")
        initialize_adaptation(
            system,
            config,
            argparse.Namespace(
                initialize=Path(adaptation["base_checkpoint"]),
                base_config=Path(adaptation["base_config"]),
                freeze_base=adaptation["freeze_base"],
                highres_encoding=adaptation["highres_encoding"],
                continue_base=adaptation.get("mode") == "continue_existing_sources",
            ),
            document["split"],
        )
    state = load_training_checkpoint(
        checkpoint,
        model=system.model,
        expected_config_sha256=config_sha,
        expected_source_schema={k: asdict(v) for k, v in config.model.input_sources.items()},
        expected_regions=[d.region for d in config.data.datasets],
    )
    if state["git_sha"] != training["git_sha"]:
        raise ValueError("checkpoint and training code registration differ")
    return system.model, state["epoch"] + 1


def run(args):
    config = Config.from_yaml(args.config)
    document = json.loads((args.cache / "cache.json").read_text())
    training = json.loads((args.checkpoint.parent / "run.json").read_text())
    validate_export_identity(
        training, config_sha=_sha(args.config), cache_sha=_sha(args.cache / "cache.json")
    )
    head = config.model.target_heads[args.target]
    if head.loss_type != "continuous":
        raise ValueError("masked reconstruction requires a continuous target")
    if args.output.exists():
        raise FileExistsError("never overwrite a reconstruction run")
    if (
        not args.months
        or len(set(args.months)) != len(args.months)
        or any(not 0 <= month < len(config.data.months) for month in args.months)
    ):
        raise ValueError("months must be unique valid zero-based indices")
    sources = [head.source, *args.aliases]
    if len(set(sources)) != len(sources) or not set(sources) <= set(config.model.input_sources):
        raise ValueError("hidden aliases must be distinct registered input sources")
    train, validation = document["split"]["train"], document["split"]["validation"]
    if set(train) & set(validation) or not train or not validation:
        raise ValueError("training and validation splits must be nonempty and disjoint")
    for i in [*train, *validation]:
        record = document["records"][i]
        if _sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("sample checksum changed")
    model, epoch = _load_model(config, document, training, args.checkpoint, _sha(args.config))
    # Estimate the naive baseline from training pixels only, separately by month/band.
    sums = np.zeros((len(config.data.months), head.channels))
    counts = np.zeros(len(config.data.months), dtype=np.int64)
    for sample in CachedSamples(document, train):
        values = sample["targets"][args.target].numpy()
        masks = sample["target_masks"][args.target].numpy() > 0
        for month in args.months:
            selected = values[month][:, masks[month]]
            if not np.isfinite(selected).all():
                raise ValueError("nonfinite training reference in valid domain")
            sums[month] += selected.sum(1, dtype=np.float64)
            counts[month] += selected.shape[1]
    if any(counts[month] == 0 for month in args.months):
        raise ValueError("monthly training-mean baseline lacks valid reference pixels")
    means = sums / np.maximum(counts[:, None], 1)
    device, distributed, _ = _setup_device(args.device)
    if distributed:
        raise ValueError("reconstruction audit requires one independent device")
    model.to(device).eval()
    args.output.mkdir(parents=True)
    _json(
        args.output / "identity.json",
        {
            "code_commit": _git_sha(),
            "checkpoint_sha256": _sha(args.checkpoint),
            "checkpoint_epoch": epoch,
            "config_sha256": _sha(args.config),
            "cache_sha256": _sha(args.cache / "cache.json"),
            "target": args.target,
            "hidden_sources": sources,
            "months": args.months,
            "context": args.context,
            "training_indices": train,
            "validation_indices": validation,
            "test_scored": False,
            "evaluation_type": "native_decoder_diagnostic",
            "units": "archived cache normalization",
            "causal_training_claim": False,
            "training_mean": means.tolist(),
            "training_counts": counts.tolist(),
        },
    )
    totals = {}
    input_keys = ("source_frames", "source_masks", "timestamps", "highres_frames", "highres_masks")
    try:
        with torch.inference_mode():
            for i, sample in zip(validation, CachedSamples(document, validation), strict=True):
                batch = collate_region_batch([sample])
                values = sample["targets"][args.target].numpy()
                valid = sample["target_masks"][args.target].numpy() > 0
                for month in args.months:
                    masked = hidden_month_inputs(
                        batch, sources, month, prefix=args.context == "prefix"
                    )
                    inputs = _move({k: masked[k] for k in input_keys if k in masked}, device)
                    output = model(
                        inputs["source_frames"],
                        inputs["source_masks"],
                        inputs["timestamps"],
                        inputs.get("highres_frames"),
                        inputs.get("highres_masks"),
                    )
                    prediction = output.reconstructions[args.target][0, month].float().cpu().numpy()
                    temporal, available = temporal_baseline(
                        values, valid, month, prefix=args.context == "prefix"
                    )
                    truth, domain = values[month], valid[month]
                    average = np.broadcast_to(means[month, :, None, None], truth.shape)
                    common = domain & available
                    current = {
                        "model_all": metric_sums(prediction, truth, domain),
                        "mean_all": metric_sums(average, truth, domain),
                        "model_common": metric_sums(prediction, truth, common),
                        "mean_common": metric_sums(average, truth, common),
                        "temporal_common": metric_sums(temporal, truth, common),
                    }
                    for name, value in current.items():
                        key = f"{month}:{name}"
                        totals[key] = totals.get(key, np.zeros_like(value)) + value
                    mask_digest = hashlib.sha256()
                    for group in ("source_masks", "highres_masks"):
                        for source, tensor in sorted(masked.get(group, {}).items()):
                            mask_digest.update((group + source).encode())
                            mask_digest.update(tensor.cpu().contiguous().numpy().tobytes())
                    np.savez_compressed(
                        args.output / f"{i}_{month}.npz",
                        prediction=prediction,
                        truth=truth,
                        valid=domain,
                        common=common,
                        temporal=temporal,
                        mean=means[month],
                        mask_sha256=np.asarray(mask_digest.hexdigest()),
                    )
                    del output, inputs
                _json(args.output / "status.json", {"state": "running", "last_validation_index": i})
        _json(
            args.output / "sufficient_statistics.json", {k: v.tolist() for k, v in totals.items()}
        )
        _json(args.output / "results.json", {k: summarize_sums(v) for k, v in totals.items()})
        _json(
            args.output / "status.json",
            {
                "state": "complete",
                "validation_tiles": len(validation),
                "conditions": len(args.months),
                "test_scored": False,
            },
        )
    except BaseException as exc:
        _json(args.output / "status.json", {"state": "failed", "error": repr(exc)})
        raise
