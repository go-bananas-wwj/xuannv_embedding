"""A compact U-Net baseline, selected on validation and frozen before query scoring."""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from xuannv_embedding.downstream import fixed_audit
from xuannv_embedding.downstream import neural_readouts as neural
from xuannv_embedding.export.context import dump, sha

FORMAT = "frozen-compact-unet-v1"


class UNet(nn.Module):
    """Three pooling levels, skip connections, 32/64/128/256 channels, GroupNorm."""

    def __init__(self, channels):
        super().__init__()

        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1),
                nn.GroupNorm(8, b),
                nn.ReLU(),
                nn.Conv2d(b, b, 3, padding=1),
                nn.GroupNorm(8, b),
                nn.ReLU(),
            )

        self.down = nn.ModuleList([block(channels, 32), block(32, 64), block(64, 128)])
        self.bottom = block(128, 256)
        self.up = nn.ModuleList([block(384, 128), block(192, 64), block(96, 32)])
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for block, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.output(x)


def code():
    return {"unet": sha(Path(__file__)), "neural": neural._implementation()}


def fit(
    train_x,
    train_y,
    train_valid,
    val_x,
    val_y,
    val_valid,
    *,
    device="cpu",
    checkpoints=(100, 300, 1000),
):
    if (
        not checkpoints
        or any(type(s) is not int or s < 1 for s in checkpoints)
        or list(checkpoints) != sorted(set(checkpoints))
    ):
        raise ValueError("checkpoints must be increasing positive integers")
    device = neural._device(device)
    x, valid = neural._maps(train_x, train_valid)
    q, qvalid = neural._maps(val_x, val_valid)
    y, labeled = neural._labels(train_y, valid)
    target, qlabeled = neural._labels(val_y, qvalid)
    if min(x.shape[-2:]) < 8:
        raise ValueError("U-Net maps must be at least eight pixels per side")
    if x.shape[1:] != q.shape[1:] or not labeled.reshape(len(x), -1).any(1).all():
        raise ValueError("incompatible feature maps or empty labeled support tile")
    scaler = StandardScaler().fit(x.transpose(0, 2, 3, 1)[labeled])
    weight = float(np.clip((y[labeled] == 0).sum() / (y[labeled] == 1).sum(), 1, 50))
    generator = torch.Generator().manual_seed(41)
    batches = np.stack(
        [torch.randperm(len(x), generator=generator)[:2].numpy() for _ in range(checkpoints[-1])]
    )
    metadata = {
        "feature_shape": list(x.shape[1:]),
        "fitted_pixels": int(labeled.sum()),
        "architecture": "compact U-Net; widths32,64,128,256; GroupNorm8; bilinear upsampling",
        "checkpoints": list(checkpoints),
        "head_seed": 41,
        "batch_seed": 41,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "positive_weight": weight,
        "selection_split": "validation AP; earliest exact tie",
        "scaler_fit_split": "labeled_valid_training_support_only",
        "invalid_context": "zero_after_standardization",
        "test_scored": False,
        "support_sha256": neural.frozen_readouts._digest(x, y, valid),
        "validation_sha256": neural.frozen_readouts._digest(q, target, qvalid),
    }
    curve, losses, best = [], [], None
    start = time.monotonic()
    with neural._scope(device, rng=True):
        neural._seed(device)
        head = UNet(x.shape[1]).float().to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
        xd = torch.from_numpy(neural._standardized(x, valid, scaler)).to(device)
        yd = torch.from_numpy(y.astype(np.float32)).to(device)
        mask = torch.from_numpy(labeled).to(device)
        pos_weight = torch.tensor(weight, dtype=torch.float32, device=device)
        model = neural.FrozenNeural("unet", head, scaler, batches, metadata, device)
        for step, picks in enumerate(batches, 1):
            head.train()
            picks = torch.from_numpy(picks).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xd[picks])[:, 0]
            selected = mask[picks]
            loss = F.binary_cross_entropy_with_logits(
                logits[selected], yd[picks][selected], pos_weight=pos_weight
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite U-Net training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step in checkpoints:
                scores = model.predict(q, qvalid)
                ap = float(average_precision_score(target[qlabeled], scores[qlabeled]))
                curve.append(
                    {
                        "steps": step,
                        "validation_ap": ap,
                        "elapsed_seconds": time.monotonic() - start,
                    }
                )
                if best is None or ap > best[0]:
                    best = (
                        ap,
                        step,
                        neural._weights(head),
                        fixed_audit.threshold(target[qlabeled], scores[qlabeled]),
                    )
        ap, step, state, threshold = best
        head.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        head.eval().requires_grad_(False)
    metadata.update(
        selected_steps=step,
        validation_ap=ap,
        threshold=threshold,
        curve=curve,
        losses=losses,
        fit_seconds=time.monotonic() - start,
        final_weights_sha256=neural._weight_digest(head),
    )
    return model


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
            "kind": "unet",
            "metadata": model.metadata,
            "payload_sha256": sha(output / "parameters.npz"),
            "implementation": code(),
            "runtime": neural._runtime(model.device),
        },
    )


def load(root, expected_identity_sha256, *, device="cpu"):
    import json

    root, device = Path(root), neural._device(device)
    if sha(root / "identity.json") != expected_identity_sha256:
        raise ValueError("U-Net identity changed")
    identity = json.loads((root / "identity.json").read_text())
    if (
        identity["format"] != FORMAT
        or identity["implementation"] != code()
        or identity["runtime"] != neural._runtime(device)
    ):
        raise ValueError("U-Net implementation or runtime differs")
    if sha(root / "parameters.npz") != identity["payload_sha256"]:
        raise ValueError("U-Net payload changed")
    with np.load(root / "parameters.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    metadata = identity["metadata"]
    channels = metadata["feature_shape"][0]
    with neural._scope(device, rng=True):
        head = UNet(channels).float()
    if (
        set(arrays) != {"mean", "scale", "batches"} | {f"weight:{k}" for k in head.state_dict()}
        or any(not np.isfinite(v).all() for v in arrays.values())
        or (arrays["scale"] <= 0).any()
    ):
        raise ValueError("invalid U-Net arrays")
    head.load_state_dict(
        {k: torch.from_numpy(arrays[f"weight:{k}"]) for k in head.state_dict()}, strict=True
    )
    head.to(device).eval().requires_grad_(False)
    if neural._weight_digest(head) != metadata["final_weights_sha256"]:
        raise ValueError("U-Net trained weights changed")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.n_features_in_ = arrays["mean"], arrays["scale"], channels
    return neural.FrozenNeural("unet", head, scaler, arrays["batches"], metadata, device)
