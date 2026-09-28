"""Bounded neural training checkpoints, selected exclusively with calibration labels."""

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from xuannv_embedding.downstream import neural_readouts as neural
from xuannv_embedding.export.context import dump, sha

FORMAT = "frozen-neural-trajectory-v1"


def _code():
    return {"trajectory": sha(Path(__file__)), "neural": neural._implementation()}


def fit(kind, train_x, train_y, train_valid, val_x, val_y, val_valid, *, checkpoints, device="cpu"):
    if (
        kind not in neural.KINDS
        or not checkpoints
        or any(type(n) is not int or not 1 <= n <= 1000 for n in checkpoints)
        or checkpoints != sorted(set(checkpoints))
    ):
        raise ValueError("invalid bounded neural checkpoints")
    device = neural._device(device)
    preparation_start = time.monotonic()
    x, valid = neural._maps(train_x, train_valid)
    query, qvalid = neural._maps(val_x, val_valid)
    y, labeled = neural._labels(train_y, valid)
    target, qlabeled = neural._labels(val_y, qvalid)
    if x.shape[1:] != query.shape[1:] or not labeled.reshape(len(x), -1).any(1).all():
        raise ValueError("neural support/query dimensions or labels differ")
    scaler = StandardScaler().fit(x.transpose(0, 2, 3, 1)[labeled])
    positive_weight = float(np.clip((y[labeled] == 0).sum() / (y[labeled] == 1).sum(), 1, 50))
    generator = torch.Generator().manual_seed(41)
    batches = np.stack(
        [torch.randperm(len(x), generator=generator)[:2].numpy() for _ in range(max(checkpoints))]
    )
    preparation_seconds = time.monotonic() - preparation_start
    losses, result = [], {}
    tic, checkpoint_seconds = time.monotonic(), 0.0
    with neural._scope(device, rng=True):
        neural._seed(device)
        head = neural.heads.build_head(kind, embed_dim=x.shape[1], num_classes=1).float().to(device)
        initial = neural._weight_digest(head)
        neural._seed(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
        xd = torch.from_numpy(neural._standardized(x, valid, scaler)).to(device)
        yd = torch.from_numpy(y.astype(np.float32)).to(device)
        mask = torch.from_numpy(labeled).to(device)
        weight = torch.tensor(positive_weight, dtype=torch.float32, device=device)
        head.train()
        for step, picks in enumerate(batches, 1):
            positions = torch.from_numpy(picks).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xd[positions])[:, 0]
            chosen = mask[positions]
            loss = F.binary_cross_entropy_with_logits(
                logits[chosen], yd[positions][chosen], pos_weight=weight
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite long-training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step not in checkpoints:
                continue
            fit_seconds = time.monotonic() - tic - checkpoint_seconds
            checkpoint_start = time.monotonic()
            frozen = copy.deepcopy(head).eval().requires_grad_(False)
            if any(not np.isfinite(v).all() for v in neural._weights(frozen).values()):
                raise FloatingPointError("nonfinite long-training weights")
            prefix = batches[:step].copy()
            metadata = {
                "feature_shape": list(x.shape[1:]),
                "head_seed": 41,
                "batch_seed": 41,
                "optimizer_steps": step,
                "batch_size": min(2, len(x)),
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "positive_weight": positive_weight,
                "losses": list(losses),
                "fit_seconds": fit_seconds,
                "preparation_seconds": preparation_seconds,
                "fit_timing_scope": (
                    "head construction, transfer and optimization; "
                    "intermediate checkpoint evaluation excluded"
                ),
                "fitted_pixels": int(labeled.sum()),
                "scaler_fit_split": "labeled_valid_training_support_only",
                "invalid_context": "zero_after_standardization",
                "selection_split": "calibration_AP; shortest exact tie",
                "support_sha256": neural.frozen_readouts._digest(x, y, valid),
                "validation_sha256": neural.frozen_readouts._digest(query, target, qvalid),
                "batch_schedule_sha256": neural.frozen_readouts._digest(prefix),
                "initial_weights_sha256": initial,
                "final_weights_sha256": neural._weight_digest(frozen),
                "registered_checkpoints": checkpoints,
                "test_scored": False,
            }
            model = neural.FrozenNeural(kind, frozen, scaler, prefix, metadata, device)
            val_start = time.monotonic()
            scores = model.predict(query, qvalid)
            metadata.update(
                validation_seconds=time.monotonic() - val_start,
                validation_ap=float(average_precision_score(target[qlabeled], scores[qlabeled])),
                threshold=neural.fixed_audit.threshold(target[qlabeled], scores[qlabeled]),
                validation_predictions_sha256=neural.frozen_readouts._digest(scores),
            )
            result[step] = model
            checkpoint_seconds += time.monotonic() - checkpoint_start
    return result


def choose(checkpoints):
    values = {step: model.metadata["validation_ap"] for step, model in checkpoints.items()}
    if not values or any(not np.isfinite(v) for v in values.values()):
        raise ValueError("calibration metric is undefined")
    return min(values, key=lambda step: (-values[step], step))


def save(model, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "parameters.npz",
        mean=model.scaler.mean_,
        scale=model.scaler.scale_,
        batches=model.batches,
        **{f"weight:{k}": v for k, v in neural._weights(model.head).items()},
    )
    dump(
        output / "identity.json",
        {
            "format": FORMAT,
            "kind": model.kind,
            "metadata": model.metadata,
            "payload_sha256": sha(output / "parameters.npz"),
            "implementation": _code(),
            "runtime": neural._runtime(model.device),
        },
    )


def load(root, expected_identity_sha256, *, device="cpu"):
    root, device = Path(root), neural._device(device)
    if sha(root / "identity.json") != expected_identity_sha256:
        raise ValueError("trajectory identity changed")
    identity = neural.json.loads((root / "identity.json").read_text())
    if (
        identity["format"] != FORMAT
        or identity["kind"] not in neural.KINDS
        or identity["implementation"] != _code()
        or identity["runtime"] != neural._runtime(device)
        or sha(root / "parameters.npz") != identity["payload_sha256"]
    ):
        raise ValueError("trajectory payload, implementation or runtime changed")
    with np.load(root / "parameters.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    metadata = identity["metadata"]
    channels, height, width = metadata["feature_shape"]
    with neural._scope(device, rng=True):
        head = neural.heads.build_head(identity["kind"], embed_dim=channels, num_classes=1).float()
    state = head.state_dict()
    if (
        set(arrays) != {"mean", "scale", "batches"} | {f"weight:{k}" for k in state}
        or arrays["mean"].shape != (channels,)
        or arrays["scale"].shape != (channels,)
        or (arrays["scale"] <= 0).any()
        or arrays["batches"].shape != (metadata["optimizer_steps"], metadata["batch_size"])
        or neural.frozen_readouts._digest(arrays["batches"]) != metadata["batch_schedule_sha256"]
        or any(not np.isfinite(v).all() for v in arrays.values())
    ):
        raise ValueError("invalid trajectory arrays")
    for name, value in state.items():
        if arrays["weight:" + name].shape != tuple(value.shape):
            raise ValueError("trajectory state shape differs")
    head.load_state_dict({k: torch.from_numpy(arrays["weight:" + k]) for k in state}, strict=True)
    head.to(device).eval().requires_grad_(False)
    if neural._weight_digest(head) != metadata["final_weights_sha256"]:
        raise ValueError("trajectory weights differ")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.n_features_in_ = arrays["mean"], arrays["scale"], channels
    return neural.FrozenNeural(identity["kind"], head, scaler, arrays["batches"], metadata, device)
