"""Fixed-budget neural classification readouts with validation-only calibration."""

from __future__ import annotations

import contextlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from xuannv_embedding.downstream import fixed_audit, frozen_readouts, heads
from xuannv_embedding.export.context import dump, sha

FORMAT = "frozen-neural-readout-v1"
KINDS = ("mlp", "conv3x3")


def _device(value):
    if str(value).startswith("npu"):
        import torch_npu  # noqa: F401
    device = torch.device(value)
    if device.type not in ("cpu", "npu") or (device.type == "npu" and device.index is None):
        raise ValueError("neural readout requires cpu or an explicit npu index")
    return device


def _runtime(device):
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "torch": str(torch.__version__),
        "backend": device.type,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn": torch.backends.mkldnn.enabled,
    }
    if device.type == "npu":
        import torch_npu

        result.update(
            torch_npu=str(torch_npu.__version__),
            device_name=torch.npu.get_device_name(device),
            matmul_allow_hf32=torch.npu.matmul.allow_hf32,
            conv_allow_hf32=torch.npu.conv.allow_hf32,
        )
    return result


def _implementation():
    return {
        k: sha(Path(v))
        for k, v in {
            "neural": __file__,
            "architecture": heads.__file__,
            "arrays": frozen_readouts.__file__,
            "threshold": fixed_audit.__file__,
        }.items()
    }


@contextlib.contextmanager
def _scope(device, *, rng=False):
    old_threads = torch.get_num_threads()
    with contextlib.ExitStack() as stack:
        if device.type == "npu":
            stack.enter_context(torch.npu.device(device))
        if rng:
            stack.enter_context(
                torch.random.fork_rng(
                    devices=[] if device.type == "cpu" else [device.index],
                    device_type=device.type,
                )
            )
        try:
            torch.set_num_threads(2)
            yield
        finally:
            torch.set_num_threads(old_threads)


def _seed(device):
    torch.random.default_generator.manual_seed(41)
    if device.type == "npu":
        torch.npu.manual_seed(41)


def _maps(features, valid):
    x = np.asarray(features)
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError("features must be real numeric maps")
    x = np.asarray(x, dtype=np.float32)
    valid = np.asarray(valid)
    if (
        x.ndim != 4
        or min(x.shape[1:]) < 1
        or valid.dtype != np.bool_
        or valid.shape != (len(x), *x.shape[2:])
        or not np.isfinite(x).all()
    ):
        raise ValueError("finite NCHW features and boolean NHW validity required")
    return x, valid


def _labels(target, valid):
    y = np.asarray(target)
    if y.shape != valid.shape or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("labels must match maps and contain only -1, 0, 1")
    y = y.astype(np.int8)
    labeled = valid & (y >= 0)
    if not np.array_equal(np.unique(y[labeled]), [0, 1]):
        raise ValueError("calibration requires both binary classes on the valid domain")
    return y, labeled


def _standardized(x, valid, scaler):
    values = x.transpose(0, 2, 3, 1)
    out = np.zeros(values.shape, dtype=np.float32)
    if valid.any():
        out[valid] = scaler.transform(values[valid])
    if not np.isfinite(out).all():
        raise FloatingPointError("nonfinite standardized neural features")
    return np.ascontiguousarray(out.transpose(0, 3, 1, 2))


def _weights(head):
    return {k: v.detach().cpu().numpy().copy() for k, v in head.state_dict().items()}


def _weight_digest(head):
    return frozen_readouts._digest(*_weights(head).values())


@dataclass
class FrozenNeural:
    kind: str
    head: torch.nn.Module
    scaler: StandardScaler
    batches: np.ndarray
    metadata: dict
    device: torch.device

    def predict(self, features, valid):
        """Return NHW logits; missing-feature positions are NaN and never scored."""
        x, valid = _maps(features, valid)
        if list(x.shape[1:]) != self.metadata["feature_shape"]:
            raise ValueError("neural query feature dimension or spatial shape differs")
        out = np.full(valid.shape, np.nan, np.float32)
        self.head.eval()
        with _scope(self.device), torch.inference_mode():
            # Fixed one-tile inference makes batching independent of query cohort size.
            for i in range(len(x)):
                z = _standardized(x[i : i + 1], valid[i : i + 1], self.scaler)
                scores = self.head(torch.from_numpy(z).to(self.device))[0, 0].cpu().numpy()
                if not np.isfinite(scores).all():
                    raise FloatingPointError("nonfinite neural logits")
                out[i][valid[i]] = scores[valid[i]]
        return out


def fit_neural(kind, train_x, train_y, train_valid, val_x, val_y, val_valid, *, device="cpu"):
    if kind not in KINDS:
        raise ValueError("unknown neural readout kind")
    device = _device(device)
    x, valid = _maps(train_x, train_valid)
    query, qvalid = _maps(val_x, val_valid)
    y, labeled = _labels(train_y, valid)
    target, qlabeled = _labels(val_y, qvalid)
    if x.shape[1:] != query.shape[1:] or not labeled.reshape(len(x), -1).any(1).all():
        raise ValueError("dimensions differ or a support tile has no labeled valid pixels")
    if kind == "conv3x3" and min(2, len(x)) * x.shape[2] * x.shape[3] < 2:
        raise ValueError("convolution training needs at least two batch-spatial observations")
    scaler = StandardScaler().fit(x.transpose(0, 2, 3, 1)[labeled])
    positive_weight = float(np.clip((y[labeled] == 0).sum() / (y[labeled] == 1).sum(), 1, 50))
    # A private CPU generator is independent of architecture construction and dropout.
    generator = torch.Generator().manual_seed(41)
    batches = np.stack(
        [torch.randperm(len(x), generator=generator)[:2].numpy() for _ in range(100)]
    )
    losses = []
    tic = time.monotonic()
    with _scope(device, rng=True):
        _seed(device)
        head = heads.build_head(kind, embed_dim=x.shape[1], num_classes=1).float().to(device)
        initial = _weight_digest(head)
        _seed(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
        xd = torch.from_numpy(_standardized(x, valid, scaler)).to(device)
        yd = torch.from_numpy(y.astype(np.float32)).to(device)
        mask = torch.from_numpy(labeled).to(device)
        weight = torch.tensor(positive_weight, dtype=torch.float32, device=device)
        head.train()
        for picks in batches:
            positions = torch.from_numpy(picks).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xd[positions])[:, 0]
            chosen = mask[positions]
            loss = F.binary_cross_entropy_with_logits(
                logits[chosen], yd[positions][chosen], pos_weight=weight
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite neural training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        head.eval().requires_grad_(False)
        if any(not np.isfinite(v).all() for v in _weights(head).values()):
            raise FloatingPointError("nonfinite trained neural parameters")
    metadata = {
        "feature_shape": list(x.shape[1:]),
        "head_seed": 41,
        "batch_seed": 41,
        "optimizer_steps": 100,
        "batch_size": min(2, len(x)),
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "positive_weight": positive_weight,
        "losses": losses,
        "fit_seconds": time.monotonic() - tic,
        "fitted_pixels": int(labeled.sum()),
        "scaler_fit_split": "labeled_valid_training_support_only",
        "invalid_context": "zero_after_standardization",
        "selection_split": "validation_threshold_only; no epoch selection",
        "support_sha256": frozen_readouts._digest(x, y, valid),
        "validation_sha256": frozen_readouts._digest(query, target, qvalid),
        "batch_schedule_sha256": frozen_readouts._digest(batches),
        "initial_weights_sha256": initial,
        "final_weights_sha256": _weight_digest(head),
        "test_scored": False,
    }
    model = FrozenNeural(kind, head, scaler, batches, metadata, device)
    tic = time.monotonic()
    scores = model.predict(query, qvalid)
    metadata.update(
        validation_seconds=time.monotonic() - tic,
        validation_ap=float(average_precision_score(target[qlabeled], scores[qlabeled])),
        threshold=fixed_audit.threshold(target[qlabeled], scores[qlabeled]),
        validation_predictions_sha256=frozen_readouts._digest(scores),
    )
    return model


def save_neural(model, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "parameters.npz",
        mean=model.scaler.mean_,
        scale=model.scaler.scale_,
        batches=model.batches,
        **{f"weight:{k}": v for k, v in _weights(model.head).items()},
    )
    dump(
        output / "identity.json",
        {
            "format": FORMAT,
            "kind": model.kind,
            "metadata": model.metadata,
            "payload_sha256": sha(output / "parameters.npz"),
            "implementation": _implementation(),
            "runtime": _runtime(model.device),
        },
    )


def load_neural(root, expected_identity_sha256, *, device="cpu"):
    root, device = Path(root), _device(device)
    if sha(root / "identity.json") != expected_identity_sha256:
        raise ValueError("neural identity changed")
    identity = json.loads((root / "identity.json").read_text())
    if identity["format"] != FORMAT or identity["kind"] not in KINDS:
        raise ValueError("unsupported neural format")
    if identity["runtime"] != _runtime(device) or identity["implementation"] != _implementation():
        raise ValueError("neural runtime or implementation changed")
    if sha(root / "parameters.npz") != identity["payload_sha256"]:
        raise ValueError("neural payload changed")
    with np.load(root / "parameters.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    metadata = identity["metadata"]
    channels, height, width = metadata["feature_shape"]
    if any(type(v) is not int or v < 1 for v in (channels, height, width)):
        raise ValueError("invalid neural feature shape")
    with _scope(device, rng=True):
        head = heads.build_head(identity["kind"], embed_dim=channels, num_classes=1).float()
    state = head.state_dict()
    if set(arrays) != {"mean", "scale", "batches"} | {f"weight:{k}" for k in state}:
        raise ValueError("neural parameter fields differ")
    if (
        arrays["mean"].shape != (channels,)
        or arrays["scale"].shape != (channels,)
        or (arrays["scale"] <= 0).any()
        or arrays["batches"].shape != (100, metadata["batch_size"])
        or any(not np.isfinite(v).all() for v in arrays.values())
        or not np.isfinite(metadata["threshold"])
        or frozen_readouts._digest(arrays["batches"]) != metadata["batch_schedule_sha256"]
    ):
        raise ValueError("invalid neural parameters")
    for k, value in state.items():
        a = arrays[f"weight:{k}"]
        if a.shape != tuple(value.shape) or a.dtype != value.numpy().dtype:
            raise ValueError("neural state shape or dtype differs")
    head.load_state_dict({k: torch.from_numpy(arrays[f"weight:{k}"]) for k in state}, strict=True)
    head.to(device).eval().requires_grad_(False)
    if _weight_digest(head) != metadata["final_weights_sha256"]:
        raise ValueError("neural trained weights differ")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.n_features_in_ = arrays["mean"], arrays["scale"], channels
    return FrozenNeural(identity["kind"], head, scaler, arrays["batches"], metadata, device)


def verify_validation(model, features, truth, valid):
    x, valid = _maps(features, valid)
    y, _ = _labels(truth, valid)
    if frozen_readouts._digest(x, y, valid) != model.metadata["validation_sha256"]:
        raise ValueError("neural validation observations changed")
    digest = frozen_readouts._digest(model.predict(x, valid))
    if digest != model.metadata["validation_predictions_sha256"]:
        raise ValueError("neural validation predictions changed")
    return {"state": "verified", "parameters_refitted": False, "test_scored": False}
