"""Label-free monthly embedding diagnostics on a fixed validation grid.

Covariance-spectrum entropy is a diagnostic, not an accuracy or change-detection score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from xuannv_embedding.downstream.multitask import check_partition
from xuannv_embedding.training.cli import _git_sha
from xuannv_embedding.training.experiment import _json, _sha


class Moments:
    """Merge centered second moments in float64 without subtracting large raw moments."""

    def __init__(self, dimensions: int):
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.count = 0
        self.mean = np.zeros(dimensions, dtype=np.float64)
        self.m2 = np.zeros((dimensions, dimensions), dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("expected finite N by D vectors")
        count = len(values)
        if not count:
            return
        # A fixed anchor preserves exact zero variance for repeated nonbinary values.
        center = values[0] + (values - values[0]).mean(0)
        centered = values - center
        delta = center - self.mean
        total = self.count + count
        self.m2 += centered.T @ centered + np.outer(delta, delta) * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def report(self) -> dict:
        if not self.count:
            raise ValueError("empty diagnostic sample domain")
        covariance = self.m2 / self.count
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        trace = float(eigenvalues.sum())
        rank = 0.0
        if trace > 0:
            probabilities = eigenvalues[eigenvalues > 0] / trace
            rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        return {
            "sample_count": self.count,
            "dimensions": len(self.mean),
            "mean_by_dimension": self.mean.tolist(),
            "variance_by_dimension": np.diag(covariance).tolist(),
            "covariance_eigenvalues": eigenvalues.tolist(),
            "total_variance": trace,
            "covariance_effective_rank": rank,
            "largest_variance_fraction": float(eigenvalues[-1] / trace) if trace > 0 else None,
        }


def adjacent_change(left: np.ndarray, right: np.ndarray) -> dict:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or not len(left):
        raise ValueError("adjacent embeddings require a common nonempty N by D domain")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite adjacent embeddings")
    a, b = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
    valid = (a > 1e-12) & (b > 1e-12)
    cosine = np.sum(left[valid] * right[valid], axis=1) / (a[valid] * b[valid])
    distance = np.linalg.norm(right - left, axis=1)
    return {
        "position_count": len(left),
        "cosine_defined_count": int(valid.sum()),
        "mean_cosine_similarity": float(np.clip(cosine, -1, 1).mean()) if valid.any() else None,
        "mean_l2_difference": float(distance.mean()),
        "median_l2_difference": float(np.median(distance)),
        "p90_l2_difference": float(np.quantile(distance, 0.9)),
    }


def run(args) -> None:
    spec = json.loads(args.spec.read_text())
    model = spec["models"][args.model]
    if "array" in model:
        raise ValueError("monthly diagnostics require original monthly NPZ exports")
    manifest_path = Path(model["manifest"])
    manifest = json.loads(manifest_path.read_text())
    cache_path = Path(spec["cache"])
    cache = json.loads(cache_path.read_text())
    check_partition(cache["split"], len(cache["records"]))
    if any(
        manifest["split"][part] != cache["split"][part]
        for part in ("train", "validation", "test", "buffer")
    ):
        raise ValueError("diagnostic spatial partitions differ")
    active = cache["split"]["validation"]
    if not active or len(manifest["records"]) != len(cache["records"]):
        raise ValueError("missing validation grid")
    stride = spec.get("sample_stride", 8)
    if type(stride) is not int or stride < 1:
        raise ValueError("sample stride must be a positive integer")
    months = manifest["months"]
    prefix = manifest.get("input_ablation", {}).get("last_visible_month_index")
    stop = len(months) if prefix is None else prefix + 1
    if not 1 <= stop <= len(months):
        raise ValueError("invalid input time cutoff")
    output = Path(spec["output"]) / args.model
    if output.exists():
        raise FileExistsError("diagnostic output exists; use a new run")
    output.mkdir(parents=True)
    expected_timestamps = np.asarray([int(m.replace("-", "")) for m in months])
    moments, samples, full_norms = None, [[] for _ in months[:stop]], [[] for _ in months[:stop]]
    positions, digests, geometry = [], {}, None
    try:
        for i in active:
            record, exported = cache["records"][i], manifest["records"][i]
            if any(record[k] != exported[k] for k in ("patch_id", "bounds")):
                raise ValueError("diagnostic grid/order mismatch")
            path = Path(exported["path"])
            digest = _sha(path)
            if "sha256" in exported and exported["sha256"] != digest:
                raise ValueError("embedding hash changed")
            digests[record["patch_id"]] = digest
            with np.load(path) as archive:
                values = archive["embedding"]
                if values.ndim != 4 or values.shape[0] != len(months):
                    raise ValueError("embedding month count differs")
                if not np.array_equal(archive["timestamps"], expected_timestamps):
                    raise ValueError("embedding month timestamps differ")
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite embedding export")
                if geometry is None:
                    geometry = values.shape[1:]
                    moments = [Moments(geometry[0]) for _ in months[:stop]]
                elif values.shape[1:] != geometry:
                    raise ValueError("embedding geometry differs across validation tiles")
                dimensions, height, width = geometry
                ys, xs = np.arange(stride // 2, height, stride), np.arange(
                    stride // 2, width, stride
                )
                indices = (ys[:, None] * width + xs[None, :]).ravel()
                if not len(indices):
                    raise ValueError("sample stride leaves an empty sample domain")
                positions.extend([[i, int(p)] for p in indices])
                for month in range(stop):
                    matrix = values[month].reshape(dimensions, -1).T.astype(np.float64)
                    subset = matrix[indices]
                    moments[month].add(subset)
                    samples[month].append(subset)
                    full_norms[month].append(np.linalg.norm(matrix, axis=1))
            _json(output / "status.json", {"state": "running", "last_validation_index": i})
        sample_values = np.stack([np.concatenate(v) for v in samples])
        rows = []
        for month, moment in enumerate(moments):
            norms = np.concatenate(full_norms[month])
            rows.append(
                {
                    "month": months[month],
                    **moment.report(),
                    "full_grid_vector_count": len(norms),
                    "mean_norm": float(norms.mean()),
                    "minimum_norm": float(norms.min()),
                    "maximum_norm": float(norms.max()),
                    "std_norm": float(norms.std()),
                    "zero_norm_vector_count": int((norms <= 1e-12).sum()),
                }
            )
        adjacent = [
            {
                "left_month": months[j - 1],
                "right_month": months[j],
                **adjacent_change(sample_values[j - 1], sample_values[j]),
            }
            for j in range(1, stop)
        ]
        np.savez_compressed(
            output / "sampled.npz",
            embeddings=sample_values,
            positions=np.asarray(positions),
            months=np.asarray(months[:stop]),
        )
        _json(
            output / "identity.json",
            {
                "code_commit": _git_sha(),
                "spec_sha256": _sha(args.spec),
                "cache_sha256": _sha(cache_path),
                "manifest_sha256": _sha(manifest_path),
                "validation_indices": active,
                "embedding_sha256": digests,
                "sample_stride": stride,
                "sample_offset": stride // 2,
                "positions_sha256": hashlib.sha256(
                    np.asarray(positions, dtype=np.int64).tobytes()
                ).hexdigest(),
                "sampled_npz_sha256": _sha(output / "sampled.npz"),
                "domain": (
                    "all finite exported vectors on validation grid; no observation or label mask"
                ),
                "labels_used": False,
                "test_scored": False,
                "input_ablation": manifest.get("input_ablation"),
                "excluded_future_output_months": months[stop:],
                "covariance_definition": (
                    "population covariance of fixed spatial samples, float64 centered moments"
                ),
                "rank_definition": (
                    "exp entropy of normalized covariance eigenvalues; zero for zero variance"
                ),
                "claims": "diagnostics only; neither task accuracy nor change-detection validation",
            },
        )
        _json(output / "results.json", {"months": rows, "adjacent_months": adjacent})
        _json(
            output / "status.json",
            {
                "state": "complete",
                "validation_tiles": len(active),
                "months": stop,
                "labels_used": False,
                "test_scored": False,
            },
        )
    except BaseException as exc:
        _json(output / "status.json", {"state": "failed", "error": repr(exc)})
        raise
