"""Calibrate on validation, then persist label-free C/R/Q prediction parameters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from xuannv_embedding.downstream.fixed_audit import threshold
from xuannv_embedding.downstream.multitask import regression_metrics
from xuannv_embedding.export.context import sha

FORMAT = "frozen-multitask-readout-v1"


def _digest(*arrays):
    result = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        result.update(str(array.shape).encode())
        result.update(array.dtype.str.encode())
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _matrix(value, dtype=None):
    array = np.asarray(value)
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
        raise ValueError("features must have a real numeric dtype")
    if dtype is None:
        dtype = np.float32 if array.dtype == np.float32 else np.float64
    array = np.asarray(array, dtype=dtype)
    if array.ndim != 2 or array.shape[1] < 1 or not np.isfinite(array).all():
        raise ValueError("features must be a finite observation-by-channel matrix")
    return array


def _target(value, count, *, classification):
    array = np.asarray(value)
    if array.shape != (count,) or not count or not np.isfinite(array).all():
        raise ValueError("targets must be finite and match nonempty observations")
    if classification:
        if not np.array_equal(np.unique(array), [0, 1]):
            raise ValueError("classification calibration requires both binary classes")
    elif ((array < 0) | (array > 1)).any():
        raise ValueError("coverage targets must be proportions in [0, 1]")
    return array


@dataclass(frozen=True)
class FrozenReadout:
    kind: str
    arrays: dict[str, np.ndarray]
    metadata: dict

    def predict(self, features):
        """Predict without labels, fitting, threshold selection or parameter mutation."""
        x = _matrix(features, self.metadata["feature_dtype"])
        if x.shape[1] != self.metadata["channels"]:
            raise ValueError("prediction dimensions differ from the frozen readout")
        if self.kind == "Q":
            query = x.copy()
            query /= np.maximum(1e-12, np.linalg.norm(query, axis=1, keepdims=True))
            return (query @ self.arrays["prototypes"].T).max(1)
        # Preserve StandardScaler's two in-place casts for float32 calibration parity.
        query = x.copy()
        query -= self.arrays["mean"]
        query /= self.arrays["scale"]
        values = query @ self.arrays["coef"].T + self.arrays["intercept"]
        if self.kind == "C":
            return values.ravel()
        if self.kind == "R":
            return np.clip(values, 0, 1)
        raise ValueError("unknown frozen readout kind")


def _freeze(kind, arrays, metadata):
    arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
    for array in arrays.values():
        if not np.isfinite(array).all():
            raise ValueError("nonfinite frozen parameters")
        array.setflags(write=False)
    return FrozenReadout(kind, arrays, metadata)


def _fit(train_x, train_y, val_x, val_y, alphas, *, classification):
    train_x = _matrix(train_x)
    val_x = _matrix(val_x, train_x.dtype)
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("support and validation dimensions differ")
    train_y = _target(train_y, len(train_x), classification=classification)
    val_y = _target(val_y, len(val_x), classification=classification)
    alphas = tuple(float(v) for v in alphas)
    if (
        not alphas
        or any(not np.isfinite(v) or v <= 0 for v in alphas)
        or tuple(sorted(set(alphas), reverse=True)) != alphas
    ):
        raise ValueError(
            "regularization candidates must be distinct positive values in descending order"
        )
    scaler = StandardScaler()
    support = scaler.fit_transform(train_x)
    query = scaler.transform(val_x)
    trials, best = [], None
    for alpha in alphas:
        if classification:
            fitted = RidgeClassifier(alpha=alpha).fit(support, train_y)
            predictions = fitted.decision_function(query)
            metrics = {"ap": float(average_precision_score(val_y, predictions))}
            objective = metrics["ap"]
        else:
            fitted = Ridge(alpha=alpha).fit(support, train_y)
            predictions = np.clip(fitted.predict(query), 0, 1)
            metrics = regression_metrics(val_y, predictions)
            objective = -metrics["rmse"]
        trials.append({"alpha": alpha, **metrics})
        if best is None or objective > best[0] + 1e-12:
            best = (objective, alpha, fitted, predictions, metrics)
    _, alpha, fitted, predictions, metrics = best
    cut = threshold(val_y, predictions) if classification else None
    if classification:
        positive = predictions >= cut
        tp = int((positive & (val_y == 1)).sum())
        fp = int((positive & (val_y == 0)).sum())
        fn = int((~positive & (val_y == 1)).sum())
        metrics.update(
            f1=2 * tp / max(1, 2 * tp + fp + fn),
            iou=tp / max(1, tp + fp + fn),
            ba=float(balanced_accuracy_score(val_y, positive)),
        )
    metadata = {
        "channels": train_x.shape[1],
        "feature_dtype": str(train_x.dtype),
        "alpha": alpha,
        "threshold": cut,
        "validation_metrics": metrics,
        "trials": trials,
        "support_observations": len(train_x),
        "validation_observations": len(val_x),
        "support_sha256": _digest(train_x, train_y),
        "validation_sha256": _digest(val_x, val_y),
        "validation_predictions_sha256": _digest(predictions),
        "scaler_fit_split": "training_support_only",
        "selection_split": "validation_only",
        "test_scored": False,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }
    readout = _freeze(
        "C" if classification else "R",
        {
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "coef": fitted.coef_,
            "intercept": fitted.intercept_,
        },
        metadata,
    )
    if not np.array_equal(readout.predict(val_x), predictions):
        raise ValueError("frozen parameters do not exactly reproduce calibration predictions")
    return readout


def fit_classification(train_x, train_y, val_x, val_y, *, alphas=(10.0, 1.0, 0.1)):
    return _fit(train_x, train_y, val_x, val_y, alphas, classification=True)


def fit_regression(train_x, train_y, val_x, val_y, *, alphas=(10.0, 1.0, 0.1)):
    return _fit(train_x, train_y, val_x, val_y, alphas, classification=False)


def freeze_retrieval(prototypes, *, normalized=False):
    values = _matrix(prototypes).copy()
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not len(values) or (norms < 1e-12).any():
        raise ValueError("retrieval requires nonzero training prototypes")
    if normalized:
        if not np.allclose(norms, 1, rtol=0, atol=1e-6):
            raise ValueError("registered normalized prototypes must have unit norm")
    else:
        values /= norms
    return _freeze(
        "Q",
        {"prototypes": values},
        {
            "channels": values.shape[1],
            "feature_dtype": str(values.dtype),
            "prototype_count": len(values),
            "prototype_sha256": _digest(values),
            "selection_split": "training_prototypes_only",
            "test_scored": False,
        },
    )


def save_readout(readout: FrozenReadout, output: Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    parameters = output / "parameters.npz"
    np.savez_compressed(parameters, **readout.arrays)
    identity = {
        "format": FORMAT,
        "kind": readout.kind,
        "metadata": readout.metadata,
        "parameters_sha256": sha(parameters),
        "implementation_sha256": sha(Path(__file__)),
    }
    (output / "identity.json").write_text(json.dumps(identity, indent=2, allow_nan=False) + "\n")


def verify_validation(readout: FrozenReadout, features, truth) -> dict:
    """Replay the exact registered validation domain after loading, without fitting."""
    if readout.kind not in ("C", "R"):
        raise ValueError("retrieval has no validation-selected parameters to replay")
    features = _matrix(features, readout.metadata["feature_dtype"])
    truth = _target(truth, len(features), classification=readout.kind == "C")
    digest = _digest(features, truth)
    if digest != readout.metadata["validation_sha256"]:
        raise ValueError("registered validation observations changed")
    predictions = readout.predict(features)
    prediction_digest = _digest(predictions)
    if prediction_digest != readout.metadata["validation_predictions_sha256"]:
        raise ValueError("frozen validation predictions changed")
    return {
        "state": "verified",
        "validation_sha256": digest,
        "validation_predictions_sha256": prediction_digest,
        "observations": len(features),
        "test_scored": False,
        "parameters_refitted": False,
    }


def load_readout(root: Path, expected_identity_sha256: str) -> FrozenReadout:
    root = Path(root)
    if sha(root / "identity.json") != expected_identity_sha256:
        raise ValueError("frozen readout identity changed")
    identity = json.loads((root / "identity.json").read_text())
    if identity["format"] != FORMAT or identity["kind"] not in ("C", "R", "Q"):
        raise ValueError("unsupported frozen readout format or kind")
    if identity["implementation_sha256"] != sha(Path(__file__)):
        raise ValueError("frozen readout implementation differs from its registered producer")
    if sha(root / "parameters.npz") != identity["parameters_sha256"]:
        raise ValueError("frozen readout parameters changed")
    metadata, kind = identity["metadata"], identity["kind"]
    dimensions = metadata["channels"]
    if (
        type(dimensions) is not int
        or dimensions < 1
        or metadata["feature_dtype"]
        not in (
            "float32",
            "float64",
        )
    ):
        raise ValueError("invalid frozen feature dimensions or dtype")
    with np.load(root / "parameters.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if kind == "Q":
        if set(arrays) != {"prototypes"} or arrays["prototypes"].shape != (
            metadata["prototype_count"],
            dimensions,
        ):
            raise ValueError("invalid frozen prototype shape")
        if _digest(arrays["prototypes"]) != metadata["prototype_sha256"]:
            raise ValueError("frozen prototype digest differs")
    else:
        if set(arrays) != {"mean", "scale", "coef", "intercept"}:
            raise ValueError("invalid frozen linear parameters")
        if (
            arrays["mean"].shape != (dimensions,)
            or arrays["scale"].shape != (dimensions,)
            or (arrays["scale"] <= 0).any()
            or arrays["coef"].shape
            not in (((dimensions,), (1, dimensions)) if kind == "C" else ((dimensions,),))
            or arrays["intercept"].shape not in (((), (1,)) if kind == "C" else ((),))
        ):
            raise ValueError("invalid frozen linear parameter shape or scale")
        if kind == "C" and not np.isfinite(metadata["threshold"]):
            raise ValueError("invalid frozen classification threshold")
    return _freeze(kind, arrays, metadata)
