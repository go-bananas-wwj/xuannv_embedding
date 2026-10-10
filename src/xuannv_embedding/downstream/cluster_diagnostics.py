"""Label-free clustering and covariance diagnostics on fixed support-region positions."""

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from xuannv_embedding.downstream.frozen_readouts import _digest
from xuannv_embedding.export.context import dump, sha


def covariance_summary(samples):
    x = np.asarray(samples, np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("finite covariance samples are required")
    centered = x - x.mean(0)
    covariance = centered.T @ centered / (len(x) - 1)
    values = np.maximum(np.linalg.eigvalsh(covariance), 0)[::-1]
    if values[0] > 0:
        values[values < values[0] * 1e-12] = 0
    total = values.sum()
    ratio = values / total if total > 0 else np.zeros_like(values)
    positive = ratio[ratio > 0]
    return {
        "effective_rank": float(np.exp(-(positive * np.log(positive)).sum())) if total else 0.0,
        "dimensions_for_95_percent": (
            int(np.searchsorted(np.cumsum(ratio), 0.95) + 1) if total else 0
        ),
        "dimensions": x.shape[1],
        "eigenvalues": values.tolist(),
        "variance_ratio": ratio.tolist(),
    }


def fit(values, valid, train_indices, *, clusters=11, step=8, offset=4):
    if (
        values.ndim != 4
        or valid.shape != values.shape[:-1]
        or valid.dtype != np.bool_
        or type(clusters) is not int
        or clusters < 2
        or type(step) is not int
        or step < 1
        or type(offset) is not int
        or not 0 <= offset < step
        or not train_indices
        or len(set(train_indices)) != len(train_indices)
        or any(type(i) is not int or not 0 <= i < len(values) for i in train_indices)
    ):
        raise ValueError("invalid clustering sampling geometry")
    samples, coordinates = [], []
    for i in train_indices:
        mask = valid[i, offset::step, offset::step]
        samples.append(values[i, offset::step, offset::step][mask])
        positions = np.argwhere(mask) * step + offset
        coordinates.append(np.column_stack([np.full(len(positions), i), positions]))
    x = np.concatenate(samples).astype(np.float64)
    coordinates = np.concatenate(coordinates)
    if len(x) < clusters or not np.isfinite(x).all():
        raise ValueError("insufficient finite unlabeled clustering positions")
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    estimator = KMeans(
        n_clusters=clusters, random_state=41, n_init=10, max_iter=300, algorithm="lloyd"
    ).fit(standardized)
    centered = x - x.mean(0)
    covariance = centered.T @ centered / (len(x) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, np.argsort(-eigenvalues)[: min(3, x.shape[1])]]
    for j in range(components.shape[1]):
        if components[np.argmax(np.abs(components[:, j])), j] < 0:
            components[:, j] *= -1
    projection = centered @ components
    limits = np.percentile(projection, [2, 98], axis=0)
    return {
        "centers": estimator.cluster_centers_,
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "pca_center": x.mean(0),
        "pca_components": components,
        "pca_limits": limits,
        "metadata": {
            "clusters": clusters,
            "seed": 41,
            "initializations": 10,
            "max_iterations": 300,
            "iterations": int(estimator.n_iter_),
            "inertia": float(estimator.inertia_),
            "sampled_positions": len(x),
            "train_indices": train_indices,
            "step": step,
            "offset": offset,
            "coordinate_sha256": _digest(coordinates),
            "sample_sha256": _digest(x),
            "labels_used_for_fitting": False,
            "raw": covariance_summary(x),
            "standardized": covariance_summary(standardized),
        },
    }


def predict(model, values, valid):
    if (
        values.shape[:-1] != valid.shape
        or values.shape[-1] != len(model["mean"])
        or valid.dtype != np.bool_
    ):
        raise ValueError("cluster query geometry differs")
    result = np.full(valid.shape, -1, np.int16)
    centers = model["centers"]
    center_norm = (centers * centers).sum(1)
    for i in range(len(values)):
        x = (np.asarray(values[i][valid[i]], np.float64) - model["mean"]) / model["scale"]
        if not np.isfinite(x).all():
            raise ValueError("nonfinite cluster query")
        distance = (x * x).sum(1)[:, None] + center_norm[None] - 2 * x @ centers.T
        result[i][valid[i]] = np.argmin(distance, axis=1)
    return result


def save(model, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "parameters.npz", **{k: v for k, v in model.items() if k != "metadata"}
    )
    dump(
        output / "identity.json",
        {
            "protocol": "unlabeled-cluster-diagnostics-v1",
            "metadata": model["metadata"],
            "payload_sha256": sha(output / "parameters.npz"),
            "implementation_sha256": sha(Path(__file__)),
            "runtime": {"numpy": np.__version__, "sklearn": sklearn.__version__},
        },
    )


def load(root, identity_sha256):
    root = Path(root)
    if sha(root / "identity.json") != identity_sha256:
        raise ValueError("cluster identity changed")
    identity = json.loads((root / "identity.json").read_text())
    if (
        identity["protocol"] != "unlabeled-cluster-diagnostics-v1"
        or sha(root / "parameters.npz") != identity["payload_sha256"]
        or identity["implementation_sha256"] != sha(Path(__file__))
    ):
        raise ValueError("cluster payload or implementation differs")
    with np.load(root / "parameters.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if (
        set(arrays) != {"centers", "mean", "scale", "pca_center", "pca_components", "pca_limits"}
        or any(not np.isfinite(v).all() for v in arrays.values())
        or (arrays["scale"] <= 0).any()
        or arrays["centers"].shape != (identity["metadata"]["clusters"], len(arrays["mean"]))
    ):
        raise ValueError("invalid cluster parameters")
    return {**arrays, "metadata": identity["metadata"]}


def run(spec_path):
    from xuannv_embedding.downstream import paired_multitask as primary
    from xuannv_embedding.downstream import review_readouts as review

    spec = primary._load(spec_path)
    if (
        set(spec)
        != {
            "protocol",
            "cohort",
            "shared",
            "model",
            "reference",
            "clusters",
            "step",
            "offset",
            "output",
        }
        or spec["protocol"] != "review-clustering-v1"
    ):
        raise ValueError("invalid clustering experiment contract")
    primary._registered(spec["cohort"])
    cohort, cache = primary._spec(spec["cohort"]["path"])
    reference = primary._registered(spec["reference"])
    if (
        spec["model"] not in cohort["models"]
        or reference["target"] != "esa_worldcover"
        or len(reference["records"]) != len(cache["records"])
    ):
        raise ValueError("clustering model or map reference differs")
    for a, b in zip(reference["records"], cache["records"], strict=True):
        if a["patch_id"] != b["patch_id"] or a["bounds"] != b["bounds"]:
            raise ValueError("clustering reference geography differs")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=False)
    batch, _, valid = review._shared(spec, "calibration")
    train = list(range(len(cache["split"]["train"])))
    model = fit(
        batch.values,
        valid,
        train,
        clusters=spec["clusters"],
        step=spec["step"],
        offset=spec["offset"],
    )
    model["metadata"]["global_training_indices"] = [batch.indices[i] for i in train]
    save(model, root / "model")
    # The clustering parameters are frozen before opening the query reference maps.
    query, _, qvalid = review._shared(spec, "test")
    predictions = predict(model, query.values, qvalid)
    truth = []
    for index in query.indices:
        r = reference["records"][index]
        if sha(Path(r["path"])) != r["sha256"]:
            raise ValueError("clustering query reference changed")
        with np.load(r["path"], allow_pickle=False) as z:
            truth.append(z["labels"])
    truth = np.stack(truth)
    eligible = qvalid & (truth > 0)
    if not eligible.any():
        raise ValueError("clustering query has no reference labels")
    metrics = {
        "ari": float(adjusted_rand_score(truth[eligible], predictions[eligible])),
        "nmi": float(normalized_mutual_info_score(truth[eligible], predictions[eligible])),
        "positions": int(eligible.sum()),
    }
    np.savez_compressed(
        root / "query_maps.npz",
        predictions=predictions,
        reference=truth,
        valid=eligible,
        indices=query.indices,
    )
    dump(
        root / "summary.json",
        {
            "state": "complete",
            "model": spec["model"],
            "metrics": metrics,
            "diagnostics": model["metadata"],
            "spec_sha256": sha(Path(spec_path)),
            "model_identity_sha256": sha(root / "model/identity.json"),
            "query_maps_sha256": sha(root / "query_maps.npz"),
            "scope": "label-free clustering head; representation training may use maps",
        },
    )
