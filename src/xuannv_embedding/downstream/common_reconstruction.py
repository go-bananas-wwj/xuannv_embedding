"""Frozen masked embeddings with an equal-capacity, training-only Ridge readout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import collate_region_batch
from xuannv_embedding.downstream.multitask import check_partition
from xuannv_embedding.downstream.reconstruction import (
    _load_model,
    hidden_month_inputs,
    metric_sums,
    summarize_sums,
    temporal_baseline,
)
from xuannv_embedding.training.cli import _git_sha, _setup_device
from xuannv_embedding.training.experiment import CachedSamples, _json, _sha
from xuannv_embedding.training.experiment_export import validate_export_identity
from xuannv_embedding.training.runtime import _move


def fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float) -> dict:
    """Fit an unpenalized intercept and train-standardized sum-loss Ridge."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (
        not np.isfinite(alpha)
        or alpha <= 0
        or x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or not len(x)
        or not x.shape[1]
        or not y.shape[1]
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Ridge requires positive finite alpha and finite nonempty paired support")
    center, intercept = x.mean(0), y.mean(0)
    scale = x.std(0)
    scale[scale <= 1e-12] = 1
    z = (x - center) / scale
    weights = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - intercept))
    return dict(center=center, scale=scale, weights=weights, intercept=intercept)


def predict_ridge(fitted: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(fitted["center"]) or not np.isfinite(x).all():
        raise ValueError("readout requires finite N by D embeddings")
    return ((x - fitted["center"]) / fitted["scale"]) @ fitted["weights"] + fitted["intercept"]


def grid_support(embedding, truth, valid, *, stride):
    embedding, truth, valid = np.asarray(embedding), np.asarray(truth), np.asarray(valid, bool)
    if (
        type(stride) is not int
        or stride < 1
        or embedding.ndim != 3
        or truth.ndim != 3
        or embedding.shape[1:] != truth.shape[1:]
        or valid.shape != truth.shape[1:]
    ):
        raise ValueError("support requires CHW arrays, HW mask and positive integer stride")
    yy, xx = np.meshgrid(
        np.arange(stride // 2, valid.shape[0], stride),
        np.arange(stride // 2, valid.shape[1], stride),
        indexing="ij",
    )
    yy, xx = yy.ravel(), xx.ravel()
    keep = valid[yy, xx]
    yy, xx = yy[keep], xx[keep]
    return embedding[:, yy, xx].T, truth[:, yy, xx].T, np.column_stack([yy, xx])


def masked_embedding(model, sample, sources, month, *, prefix, device):
    batch = hidden_month_inputs(collate_region_batch([sample]), sources, month, prefix=prefix)
    digest = hashlib.sha256()
    for group in ("source_masks", "highres_masks"):
        for source, tensor in sorted(batch.get(group, {}).items()):
            digest.update((group + source).encode())
            digest.update(tensor.cpu().contiguous().numpy().tobytes())
    keys = ("source_frames", "source_masks", "timestamps", "highres_frames", "highres_masks")
    inputs = _move({key: batch[key] for key in keys if key in batch}, device)
    with torch.inference_mode():
        result = model(
            inputs["source_frames"],
            inputs["source_masks"],
            inputs["timestamps"],
            inputs.get("highres_frames"),
            inputs.get("highres_masks"),
        )
        embedding = result.embedding_map[0, month].float().cpu().numpy()
    if embedding.ndim != 3 or not np.isfinite(embedding).all():
        raise ValueError("masked model output must be a finite DHW embedding")
    return embedding, digest.hexdigest()


def run(args):
    if args.output.exists():
        raise FileExistsError("never overwrite a common reconstruction run")
    if not np.isfinite(args.alpha) or args.alpha <= 0 or args.sample_stride < 1:
        raise ValueError("positive finite alpha and positive sample stride required")
    config = Config.from_yaml(args.config)
    document = json.loads((args.cache / "cache.json").read_text())
    training = json.loads((args.checkpoint.parent / "run.json").read_text())
    validate_export_identity(
        training, config_sha=_sha(args.config), cache_sha=_sha(args.cache / "cache.json")
    )
    check_partition(document["split"], len(document["records"]))
    head = config.model.target_heads[args.target]
    if head.loss_type != "continuous":
        raise ValueError("common reconstruction requires continuous targets")
    if (
        not args.months
        or len(set(args.months)) != len(args.months)
        or any(not 0 <= month < len(config.data.months) for month in args.months)
    ):
        raise ValueError("months must be distinct valid zero-based indices")
    sources = [head.source, *args.aliases]
    if len(set(sources)) != len(sources) or not set(sources) <= set(config.model.input_sources):
        raise ValueError("hidden sources must be distinct registered inputs")
    train, validation = document["split"]["train"], document["split"]["validation"]
    if not train or not validation:
        raise ValueError("training and validation splits must be nonempty")
    for i in [*train, *validation]:
        record = document["records"][i]
        if _sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("sample checksum changed")
    model, epoch = _load_model(config, document, training, args.checkpoint, _sha(args.config))
    device, distributed, _ = _setup_device(args.device)
    if distributed:
        raise ValueError("common reconstruction audit requires one independent device")
    model.to(device).eval().requires_grad_(False)
    args.output.mkdir(parents=True)
    identity = dict(
        code_commit=_git_sha(),
        checkpoint_sha256=_sha(args.checkpoint),
        checkpoint_epoch=epoch,
        config_sha256=_sha(args.config),
        cache_sha256=_sha(args.cache / "cache.json"),
        target=args.target,
        hidden_sources=sources,
        months=args.months,
        context=args.context,
        training_indices=train,
        validation_indices=validation,
        test_scored=False,
        test_records_read=False,
        evaluation_type="common_frozen_ridge_v1",
        units="archived cache normalization",
        causal_training_claim=False,
        alpha=args.alpha,
        sample_stride=args.sample_stride,
        sample_offset=args.sample_stride // 2,
        objective="sum squared residuals + alpha * squared coefficient norm; intercept unpenalized",
        hyperparameters_selected_on_validation=False,
        encoder_optimizer_updates=0,
    )
    _json(args.output / "identity.json", identity)
    try:
        support = {m: {k: [] for k in ("features", "targets", "positions")} for m in args.months}
        sums = np.zeros((len(config.data.months), head.channels))
        counts = np.zeros(len(config.data.months), dtype=np.int64)
        masks = []
        for i, sample in zip(train, CachedSamples(document, train), strict=True):
            values = sample["targets"][args.target].numpy()
            valid = sample["target_masks"][args.target].numpy() > 0
            for month in args.months:
                truth, domain = values[month], valid[month]
                selected = truth[:, domain]
                if not np.isfinite(selected).all():
                    raise ValueError("nonfinite training reference inside valid domain")
                sums[month] += selected.sum(1, dtype=np.float64)
                counts[month] += selected.shape[1]
                embedding, digest = masked_embedding(
                    model, sample, sources, month, prefix=args.context == "prefix", device=device
                )
                x, y, positions = grid_support(embedding, truth, domain, stride=args.sample_stride)
                support[month]["features"].append(x)
                support[month]["targets"].append(y)
                support[month]["positions"].append(
                    np.column_stack([np.full(len(positions), i, dtype=np.int64), positions])
                )
                masks.append(dict(tile=i, month=month, mask_sha256=digest))
            _json(args.output / "status.json", dict(state="fitting_support", last_training_index=i))
        means = sums / np.maximum(counts[:, None], 1)
        fitted, support_identity = {}, {}
        for month in args.months:
            data = {key: np.concatenate(rows) for key, rows in support[month].items()}
            fitted[month] = fit_ridge(data["features"], data["targets"], alpha=args.alpha)
            path = args.output / f"support_{month}.npz"
            np.savez_compressed(path, **data, **fitted[month])
            support_identity[str(month)] = dict(
                count=len(data["features"]),
                dimensions=data["features"].shape[1],
                coefficients=(data["features"].shape[1] + 1) * head.channels,
                positions_sha256=hashlib.sha256(data["positions"].tobytes()).hexdigest(),
                targets_sha256=hashlib.sha256(data["targets"].tobytes()).hexdigest(),
                support_sha256=_sha(path),
            )
        del support
        _json(args.output / "training_masks.json", masks)
        identity.update(
            training_mean=means.tolist(),
            training_counts=counts.tolist(),
            support=support_identity,
            training_masks_sha256=_sha(args.output / "training_masks.json"),
        )
        _json(args.output / "identity.json", identity)
        totals = {}
        for i, sample in zip(validation, CachedSamples(document, validation), strict=True):
            values = sample["targets"][args.target].numpy()
            valid = sample["target_masks"][args.target].numpy() > 0
            for month in args.months:
                embedding, digest = masked_embedding(
                    model, sample, sources, month, prefix=args.context == "prefix", device=device
                )
                truth, domain = values[month], valid[month]
                flat = embedding.reshape(embedding.shape[0], -1).T
                prediction = predict_ridge(fitted[month], flat).T.reshape(truth.shape)
                temporal, available = temporal_baseline(
                    values, valid, month, prefix=args.context == "prefix"
                )
                average = np.broadcast_to(means[month, :, None, None], truth.shape)
                common = domain & available
                for name, pred, mask in (
                    ("model_all", prediction, domain),
                    ("mean_all", average, domain),
                    ("model_common", prediction, common),
                    ("mean_common", average, common),
                    ("temporal_common", temporal, common),
                ):
                    key = f"{month}:{name}"
                    value = metric_sums(pred, truth, mask)
                    totals[key] = totals.get(key, np.zeros_like(value)) + value
                np.savez_compressed(
                    args.output / f"{i}_{month}.npz",
                    embedding=embedding,
                    prediction=prediction,
                    truth=truth,
                    valid=domain,
                    common=common,
                    temporal=temporal,
                    mean=means[month],
                    mask_sha256=np.asarray(digest),
                )
            _json(args.output / "status.json", dict(state="validation", last_validation_index=i))
        _json(
            args.output / "sufficient_statistics.json", {k: v.tolist() for k, v in totals.items()}
        )
        _json(args.output / "results.json", {k: summarize_sums(v) for k, v in totals.items()})
        _json(
            args.output / "status.json",
            dict(
                state="complete",
                validation_tiles=len(validation),
                conditions=len(args.months),
                test_scored=False,
                encoder_optimizer_updates=0,
            ),
        )
    except BaseException as exc:
        _json(args.output / "status.json", dict(state="failed", error=repr(exc)))
        raise
