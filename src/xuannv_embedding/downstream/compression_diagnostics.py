"""Train-only PCA with native-feature controls and validation-only task readouts."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from xuannv_embedding.downstream import frozen_readouts
from xuannv_embedding.downstream import paired_multitask as primary
from xuannv_embedding.export.context import dump, sha

PROTOCOL = "compression-validation-v1"


def fit_pca(samples):
    x = np.asarray(samples, np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("PCA needs at least two finite training samples")
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / (len(x) - 1)
    covariance = (covariance + covariance.T) * 0.5
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0), vectors[:, order]
    for column in range(vectors.shape[1]):
        if vectors[np.argmax(np.abs(vectors[:, column])), column] < 0:
            vectors[:, column] *= -1
    return {"mean": mean, "components": vectors.T, "eigenvalues": values}


def project(features, pca, dimensions):
    x = np.asarray(features)
    if pca is None and dimensions is None:
        return np.asarray(x, np.float32)
    if (
        pca is None
        or type(dimensions) is not int
        or not 1 <= dimensions <= len(pca["components"])
        or x.shape[-1] != len(pca["mean"])
    ):
        raise ValueError("projection dimensions differ from fitted training PCA")
    values = (x.astype(np.float64) - pca["mean"]) @ pca["components"][:dimensions].T
    if not np.isfinite(values).all():
        raise ValueError("nonfinite projected features")
    return values.astype(np.float32)


def _spec(path):
    spec = primary._load(path)
    if (
        set(spec)
        != {
            "protocol",
            "reference_cache",
            "labels",
            "models",
            "dimensions",
            "budgets",
            "retrieval_budgets",
            "support_seeds",
            "sample_step",
            "sample_offset",
            "output",
        }
        or spec["protocol"] != PROTOCOL
        or set(spec["labels"]) != {"train", "validation"}
        or not spec["models"]
        or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in spec["models"])
        or type(spec["sample_step"]) is not int
        or spec["sample_step"] < 1
        or type(spec["sample_offset"]) is not int
        or not 0 <= spec["sample_offset"] < spec["sample_step"]
    ):
        raise ValueError("invalid validation-only compression specification")
    for field in ("dimensions", "budgets", "retrieval_budgets", "support_seeds"):
        values = spec[field]
        if (
            not isinstance(values, list)
            or not values
            or len(set(values)) != len(values)
            or any(type(v) is not int or v < 1 for v in values)
        ):
            raise ValueError("invalid registered compression dimensions or support draws")
    for model in spec["models"].values():
        selection = primary.FeatureSelection(**model["selection"])
        if max(spec["dimensions"]) > selection.channels:
            raise ValueError("compression exceeds a source representation's dimension")
    cache = primary._registered(spec["reference_cache"])
    if max(spec["budgets"]) > len(cache["split"]["train"]):
        raise ValueError("compression support budget exceeds training tiles")
    return spec, cache


def run(spec_path):
    spec, cache = _spec(spec_path)
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    dump(root / "status.json", {"state": "running", "phase": "inputs", "test_scored": False})
    try:
        batches, labels, indices = primary._common(spec, cache, ("train", "validation"), root)
        valid = np.load(root / "common_valid.npy", allow_pickle=False)
        train = list(range(len(cache["split"]["train"])))
        validation = list(range(len(train), len(indices)))
        ids = [cache["records"][i]["patch_id"] for i in indices]
        conditions = [c for c in primary._conditions(spec) if c["family"] in ("C", "Q")]
        step, offset = spec["sample_step"], spec["sample_offset"]
        fit_valid = valid[train, offset::step, offset::step].ravel()
        result = {}
        for name, batch in batches.items():
            directory = root / name
            directory.mkdir()
            values = batch.values
            samples = values[train, offset::step, offset::step].reshape(-1, values.shape[-1])
            pca = fit_pca(samples[fit_valid])
            np.savez_compressed(directory / "pca.npz", **pca)
            model_record = {
                "feature_identity": batch.identity,
                "fit_global_indices": list(cache["split"]["train"]),
                "fit_sample_count": int(fit_valid.sum()),
                "pca_sha256": sha(directory / "pca.npz"),
                "variants": {},
            }
            for dimension in [None, *spec["dimensions"]]:
                variant = "native" if dimension is None else f"pca{dimension}"
                stage = directory / variant
                stage.mkdir()
                projected = values
                temporary = root / "projected.npy"
                if dimension is not None:
                    projected = np.lib.format.open_memmap(
                        temporary, mode="w+", dtype="float32", shape=(*values.shape[:-1], dimension)
                    )
                    for index in range(len(values)):
                        projected[index] = project(values[index], pca, dimension)
                        projected[index][~valid[index]] = 0
                    projected.flush()
                records = []
                for condition in conditions:
                    y = labels[condition["task"]]
                    readout, support = primary._fit(
                        projected, y, train, validation, ids, indices, condition
                    )
                    readout_path = stage / "readouts" / condition["key"]
                    readout_path.parent.mkdir(exist_ok=True)
                    frozen_readouts.save_readout(readout, readout_path)
                    readout_sha = sha(readout_path / "identity.json")
                    readout = frozen_readouts.load_readout(readout_path, readout_sha)
                    query, truth, tiles, positions = primary._query(
                        projected, y, validation, condition["family"]
                    )
                    scores = readout.predict(query)
                    if not np.isfinite(scores).all():
                        raise FloatingPointError("nonfinite compressed readout scores")
                    prediction = stage / (condition["key"] + ".npz")
                    np.savez_compressed(
                        prediction,
                        scores=scores,
                        truth=truth,
                        tiles=np.asarray(indices)[tiles],
                        valid_indices=positions,
                    )
                    ap = (
                        float(average_precision_score(truth, scores))
                        if (truth == 1).any()
                        else None
                    )
                    records.append(
                        {
                            **condition,
                            "metrics": {"ap": ap, "observations": len(truth)},
                            "support": support,
                            "readout_identity_sha256": readout_sha,
                            "prediction_sha256": sha(prediction),
                        }
                    )
                    dump(
                        root / "status.json",
                        {
                            "state": "running",
                            "model": name,
                            "variant": variant,
                            "conditions_complete": len(records),
                            "test_scored": False,
                        },
                    )
                dump(stage / "results.json", records)
                model_record["variants"][variant] = {
                    "dimensions": values.shape[-1] if dimension is None else dimension,
                    "conditions": records,
                    "results_sha256": sha(stage / "results.json"),
                }
                if dimension is not None:
                    del projected
                    temporary.unlink()
            result[name] = model_record
            dump(directory / "summary.json", model_record)
        summary = {
            "state": "complete",
            "protocol": PROTOCOL,
            "spec_sha256": sha(Path(spec_path)),
            "models": result,
            "test_scored": False,
            "pca_fit": "common training positions only",
            "conditions_per_model": len(conditions) * (1 + len(spec["dimensions"])),
            "common_valid_sha256": sha(root / "common_valid.npy"),
            "common_labels_sha256": sha(root / "common_labels.npz"),
            "implementation_sha256": sha(Path(__file__)),
            "readout_implementation": primary._code(),
            "elapsed_seconds": time.monotonic() - started,
            "scope": (
                "validation sensitivity, no test-based dimension or candidate selection; "
                "native differs from full-dimensional PCA"
            ),
        }
        dump(root / "summary.json", summary)
        dump(root / "status.json", {"state": "complete", "test_scored": False})
        return summary
    except BaseException as exc:
        dump(root / "status.json", {"state": "failed", "error": repr(exc), "test_scored": False})
        raise
