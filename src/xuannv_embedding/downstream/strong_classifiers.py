"""Validation-frozen RF, RBF-SVM and cosine kNN classification readouts."""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numexpr as ne
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from xuannv_embedding.downstream import fixed_audit, frozen_readouts, product_comparison
from xuannv_embedding.export.context import dump, sha

FORMAT = "frozen-strong-classifier-v1"
KINDS = ("rf", "svm", "knn")


def _runtime():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numexpr": ne.__version__,
    }


def _implementation():
    return {
        name: sha(Path(path))
        for name, path in {
            "strong": __file__,
            "arrays": frozen_readouts.__file__,
            "selection": fixed_audit.__file__,
            "svm_kernel": product_comparison.__file__,
        }.items()
    }


def _knn(support, labels, query, k):
    norms = np.maximum(1e-12, np.linalg.norm(query, axis=1, keepdims=True))
    similarity = (query / norms) @ support.T
    k = min(k, len(support))
    boundary = np.partition(similarity, -k, axis=1)[:, -k, None]
    above, tied = similarity > boundary, similarity == boundary
    remaining = k - above.sum(1, keepdims=True)
    chosen = above | (tied & (tied.cumsum(1) <= remaining))
    return (chosen @ labels.astype(np.float64)) / k


@dataclass
class FrozenClassifier:
    kind: str
    scaler: StandardScaler
    estimator: object
    arrays: dict[str, np.ndarray]
    metadata: dict

    def predict(self, features):
        """Use saved training parameters only; no query labels or fitting operation."""
        x = frozen_readouts._matrix(features, np.float64)
        if x.shape[1] != self.metadata["channels"]:
            raise ValueError("query feature dimension differs from classifier")
        if not len(x):
            return np.empty(0, np.float64)
        values = []
        old_threads = ne.get_num_threads()
        try:
            ne.set_num_threads(2)
            with threadpool_limits(limits=2):
                for start in range(0, len(x), 1024):
                    query = self.scaler.transform(x[start : start + 1024])
                    if self.kind == "rf":
                        score = self.estimator.predict_proba(query)[:, 1]
                    elif self.kind == "svm":
                        score = product_comparison.svm_scores(
                            [self.estimator], query, self.arrays["support"]
                        )[:, 0]
                    else:
                        score = _knn(self.arrays["support"], self.arrays["labels"], query, 5)
                    values.append(score)
        finally:
            ne.set_num_threads(old_threads)
        return np.concatenate(values)


def fit_classifier(kind, train_x, train_y, val_x, val_y, *, seed):
    if kind not in KINDS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid classifier or support seed")
    train_x = frozen_readouts._matrix(train_x, np.float64)
    val_x = frozen_readouts._matrix(val_x, np.float64)
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("training and validation feature dimensions differ")
    train_y = frozen_readouts._target(train_y, len(train_x), classification=True)
    val_y = frozen_readouts._target(val_y, len(val_x), classification=True)
    selected = np.arange(len(train_y))
    if kind == "knn":
        # Caller already supplies the paired, randomly ordered class-balanced support.
        selected = np.concatenate([np.flatnonzero(train_y == c)[:1024] for c in (0, 1)])
    elif any((train_y == c).sum() > 4096 for c in (0, 1)):
        raise ValueError("RF/SVM support exceeds the registered 4096 pixels per class")
    scaler = StandardScaler().fit(train_x[selected])
    support = scaler.transform(train_x[selected])
    labels = train_y[selected]
    if kind == "knn":
        support = support / np.maximum(1e-12, np.linalg.norm(support, axis=1, keepdims=True))
    arrays = {"support": support, "labels": labels, "selected_input_positions": selected}
    metadata = {
        "channels": train_x.shape[1],
        "seed": seed,
        "fitted_pixels": len(selected),
        "incoming_pixels": len(train_x),
        "support_sha256": frozen_readouts._digest(train_x, train_y),
        "fitted_support_sha256": frozen_readouts._digest(train_x[selected], labels),
        "validation_sha256": frozen_readouts._digest(val_x, val_y),
        "scaler_fit_split": "actual_training_support_only",
        "selection_split": "validation_only",
        "validation_observations": len(val_x),
        "test_scored": False,
        "feature_dtype": "float64",
        "prediction_chunk": 1024,
        "k": 5 if kind == "knn" else None,
        "knn_ties": "earlier registered support position" if kind == "knn" else None,
    }
    parameters = (0.1, 1.0, 10.0) if kind == "svm" else (200 if kind == "rf" else 5,)
    best, trials = None, []
    for parameter in parameters:
        tic = time.monotonic()
        if kind == "rf":
            estimator = RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )
        elif kind == "svm":
            estimator = SVC(C=parameter, gamma="scale", class_weight="balanced", cache_size=1024)
        else:
            estimator = None
        with threadpool_limits(limits=2):
            if estimator is not None:
                estimator.fit(support, labels)
        fit_seconds = time.monotonic() - tic
        model = FrozenClassifier(kind, scaler, estimator, arrays, metadata)
        tic = time.monotonic()
        predictions = model.predict(val_x)
        ap = float(average_precision_score(val_y, predictions))
        trials.append(
            {
                "parameter": parameter,
                "validation_ap": ap,
                "fit_seconds": fit_seconds,
                "validation_seconds": time.monotonic() - tic,
            }
        )
        if best is None or ap > best[0] + 1e-12:
            best = (ap, parameter, model, predictions)
    ap, parameter, model, predictions = best
    model.metadata = {
        **metadata,
        "selected_parameter": parameter,
        "trials": trials,
        "validation_ap": ap,
        "threshold": fixed_audit.threshold(val_y, predictions),
        "validation_predictions_sha256": frozen_readouts._digest(predictions),
    }
    return model


def save_classifier(model, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "parameters.npz",
        mean=model.scaler.mean_,
        scale=model.scaler.scale_,
        **model.arrays,
    )
    payloads = {"parameters.npz": sha(output / "parameters.npz")}
    if model.kind != "knn":
        joblib.dump(model.estimator, output / "estimator.joblib", compress=3)
        payloads["estimator.joblib"] = sha(output / "estimator.joblib")
    dump(
        output / "identity.json",
        {
            "format": FORMAT,
            "kind": model.kind,
            "metadata": model.metadata,
            "payloads": payloads,
            "implementation": _implementation(),
            "runtime": _runtime(),
        },
    )


def load_classifier(root, expected_identity_sha256):
    """Load only own trusted experiment artifacts, with all hashes checked before joblib."""
    import json

    root = Path(root)
    if sha(root / "identity.json") != expected_identity_sha256:
        raise ValueError("classifier identity changed")
    identity = json.loads((root / "identity.json").read_text())
    kind = identity["kind"]
    if identity["format"] != FORMAT or kind not in KINDS:
        raise ValueError("unsupported classifier format")
    if identity["runtime"] != _runtime() or identity["implementation"] != _implementation():
        raise ValueError("classifier runtime or implementation changed")
    expected = {"parameters.npz"} | ({"estimator.joblib"} if kind != "knn" else set())
    if set(identity["payloads"]) != expected or any(
        sha(root / name) != digest for name, digest in identity["payloads"].items()
    ):
        raise ValueError("classifier payload changed")
    with np.load(root / "parameters.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    if set(arrays) != {"mean", "scale", "support", "labels", "selected_input_positions"}:
        raise ValueError("classifier parameter fields differ")
    metadata = identity["metadata"]
    dimensions, count = metadata["channels"], metadata["fitted_pixels"]
    if (
        type(dimensions) is not int
        or dimensions < 1
        or type(count) is not int
        or count < 2
        or arrays["support"].shape != (count, dimensions)
        or arrays["labels"].shape != (count,)
        or arrays["selected_input_positions"].shape != (count,)
        or arrays["mean"].shape != (dimensions,)
        or arrays["scale"].shape != (dimensions,)
        or (arrays["scale"] <= 0).any()
        or any(not np.isfinite(a).all() for a in arrays.values())
        or not np.isfinite(metadata["threshold"])
        or not np.array_equal(np.unique(arrays["labels"]), [0, 1])
    ):
        raise ValueError("invalid classifier parameter shapes or values")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = arrays.pop("mean"), arrays.pop("scale")
    scaler.n_features_in_ = dimensions
    estimator = None if kind == "knn" else joblib.load(root / "estimator.joblib")
    if estimator is not None:
        expected_type = RandomForestClassifier if kind == "rf" else SVC
        if (
            type(estimator) is not expected_type
            or estimator.n_features_in_ != dimensions
            or not np.array_equal(estimator.classes_, [0, 1])
        ):
            raise ValueError("estimator family or dimensions differ")
    return FrozenClassifier(kind, scaler, estimator, arrays, metadata)


def verify_validation(model, features, truth):
    features = frozen_readouts._matrix(features, np.float64)
    truth = frozen_readouts._target(truth, len(features), classification=True)
    if frozen_readouts._digest(features, truth) != model.metadata["validation_sha256"]:
        raise ValueError("validation observations changed")
    digest = frozen_readouts._digest(model.predict(features))
    if digest != model.metadata["validation_predictions_sha256"]:
        raise ValueError("validation predictions changed")
    return {
        "state": "verified",
        "validation_predictions_sha256": digest,
        "parameters_refitted": False,
        "test_scored": False,
    }
