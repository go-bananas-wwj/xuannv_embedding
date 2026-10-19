"""Calendar-associated public observations, without map targets or local imagery."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS

BAND_SLICES = {"s2": (0, 11), "s1": (11, 13), "landsat": (13, 19)}
BASIS_NAMES = ["constant", "annual_sin", "annual_cos", "semiannual_sin", "semiannual_cos"]


def _verified(value: dict) -> Path:
    if set(value) != {"path", "sha256"} or sha(value["path"]) != value["sha256"]:
        raise ValueError("Calendar source identity changed")
    return Path(value["path"])


def _basis(months: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if len(months) != 12 or any(
        isinstance(m, bool) or not isinstance(m, (int, np.integer)) for m in months
    ):
        raise ValueError("Twelve integer calendar months are required")
    dates = [dt.date(int(m) // 100, int(m) % 100, 1) for m in months]
    ordinal = np.asarray([d.year * 12 + d.month for d in dates])
    if (np.diff(np.sort(ordinal)) != 1).any():
        raise ValueError("Calendar observations must cover twelve consecutive months")
    phase = np.asarray([d.month - 1 for d in dates]) * (2 * np.pi / 12)
    basis = np.stack(
        [np.ones(12), np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)], 1
    )
    return basis, ordinal


def _ridge(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not 1e-6 <= float(value) <= 1
    ):
        raise ValueError("Calendar ridge must be finite and in [1e-6, 1]")
    return float(value)


def calendar_coefficients(
    frames: np.ndarray, masks: np.ndarray, months: list[int], *, ridge: float
) -> np.ndarray:
    """Fit mean and two annual harmonics using observed source months only.

    One observation gives its constant value; an absent source gives zeros.
    These describe a period, not a causal forecast or validated change product.
    """
    x, mask = np.asarray(frames), np.asarray(masks)
    basis, _ = _basis(months)
    ridge = _ridge(ridge)
    if (
        x.ndim != 4
        or x.shape[:2] != (12, 19)
        or min(x.shape[-2:]) < 1
        or not np.issubdtype(x.dtype, np.number)
        or np.iscomplexobj(x)
        or mask.dtype != np.bool_
        or mask.shape != (12, 3, *x.shape[-2:])
    ):
        raise ValueError("Calendar fitting requires 12x19 frames and boolean source masks")
    h, w = x.shape[-2:]
    result = np.zeros((5, 19, h, w), np.float64)
    penalty = np.diag([0, ridge, ridge, ridge, ridge])
    for j, (start, stop) in enumerate(BAND_SLICES.values()):
        group = np.asarray(x[:, start:stop], np.float64)
        observed = mask[:, j]
        if not np.isfinite(group.transpose(0, 2, 3, 1)[observed]).all():
            raise ValueError("Nonfinite valid calendar observation")
        clean = np.where(observed[:, None], group, 0)
        weight = observed.reshape(12, -1).T.astype(np.float64)
        weighted_basis = weight[:, :, None] * basis[None]
        gram = np.einsum("pti,tj->pij", weighted_basis, basis, optimize=True) + penalty
        gram[weight.sum(1) == 0, 0, 0] = 1
        values = clean.reshape(12, stop - start, -1).transpose(2, 0, 1)
        rhs = np.einsum("pti,ptc->pic", weighted_basis, values, optimize=True)
        coefficients = np.linalg.solve(gram, rhs)
        result[:, start:stop] = coefficients.transpose(1, 2, 0).reshape(5, stop - start, h, w)
    output = result.reshape(95, h, w).astype(np.float32)
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite calendar coefficient")
    return output


def prepare(path: str | Path) -> Path:
    spec = json.loads(Path(path).read_text())
    if (
        not isinstance(spec, dict)
        or set(spec) != {"schema", "monthly", "index", "ridge", "output", "reserve_gib"}
        or spec["schema"] != "public-calendar-statistics-v1"
        or type(spec["reserve_gib"]) is not int
        or spec["reserve_gib"] < 0
        or not isinstance(spec["output"], str)
        or not spec["output"]
    ):
        raise ValueError("Invalid public calendar specification")
    ridge = _ridge(spec["ridge"])
    monthly = json.loads(_verified(spec["monthly"]).read_text())
    index = json.loads(_verified(spec["index"]).read_text())
    required = {
        "schema",
        "months",
        "index",
        "bands",
        "band_slices",
        "normalization_clip",
        "map_or_local_image_values_used",
        "frames",
        "masks",
        "normalization",
        "shape",
    }
    if (
        not isinstance(monthly, dict)
        or not required.issubset(monthly)
        or monthly["schema"] != "public-monthly-tensor-v1"
        or monthly["index"] != spec["index"]
        or monthly["bands"] != PUBLIC_BANDS
        or monthly["band_slices"] != {k: list(v) for k, v in BAND_SLICES.items()}
        or monthly["normalization_clip"] != [-6, 6]
        or monthly["map_or_local_image_values_used"] is not False
    ):
        raise ValueError("Monthly calendar geography, bands or preprocessing differ")
    _, ordinal = _basis(monthly["months"])
    if (np.diff(ordinal) != 1).any():
        raise ValueError("Monthly cache must be in calendar order")
    frames = np.load(_verified(monthly["frames"]), mmap_mode="r")
    masks = np.load(_verified(monthly["masks"]), mmap_mode="r")
    _verified(monthly["normalization"])
    records = index["records"]
    if (
        frames.ndim != 5
        or frames.shape[1:3] != (12, 19)
        or not len(frames)
        or min(frames.shape[-2:]) < 1
        or list(frames.shape) != monthly["shape"]
        or masks.dtype != np.bool_
        or masks.shape != (len(frames), 12, 3, *frames.shape[-2:])
        or len(records) != len(frames)
        or len({r["patch_id"] for r in records}) != len(records)
    ):
        raise ValueError("Monthly frames, masks and geographic count differ")
    root = Path(spec["output"])
    identity = {"spec_sha256": sha(path), "implementation_sha256": sha(__file__)}
    header = root / "identity.json"
    if root.exists() and any(root.iterdir()) and not header.exists():
        raise ValueError("Unowned calendar output directory")
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError("Calendar preparation identity changed")
    metadata_path, status_path = root / "metadata.json", root / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("stage") == "complete":
        status = json.loads(status_path.read_text())
        if (
            sha(metadata_path) != status["metadata_sha256"]
            or sha(root / "features.npy") != status["features_sha256"]
        ):
            raise ValueError("Completed calendar output changed")
        return metadata_path
    root.mkdir(parents=True, exist_ok=True)
    shape = (len(frames), 95, *frames.shape[-2:])
    if shutil.disk_usage(root).free < int(np.prod(shape)) * 4 + spec["reserve_gib"] * 2**30:
        raise RuntimeError("Calendar preparation would exceed the disk reserve")
    dump(header, identity)
    pending = root / "features.partial.npy"
    output = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float32, shape=shape)
    histograms = {s: np.zeros(13, np.int64) for s in BAND_SLICES}
    for i in range(len(frames)):
        output[i] = calendar_coefficients(frames[i], masks[i], monthly["months"], ridge=ridge)
        for j, s in enumerate(BAND_SLICES):
            histograms[s] += np.bincount(masks[i, :, j].sum(0).ravel(), minlength=13)
        if (i + 1) % 25 == 0:
            dump(status_path, {"stage": "writing", "tiles_done": i + 1, "tiles": len(frames)})
    output.flush()
    pending.replace(root / "features.npy")
    features = {"path": str(root / "features.npy"), "sha256": sha(root / "features.npy")}
    dump(
        metadata_path,
        {
            "schema": "public-calendar-statistics-v1",
            "index": spec["index"],
            "monthly": spec["monthly"],
            "normalization": monthly["normalization"],
            "months": monthly["months"],
            "bands": PUBLIC_BANDS,
            "basis": BASIS_NAMES,
            "feature_order": "basis_then_band",
            "ridge": ridge,
            "shape": list(shape),
            "features": features,
            "observation_count_histograms": {k: v.tolist() for k, v in histograms.items()},
            "map_labels_used": False,
            "missing_source_coefficients": "zero",
            "units": "coefficients of clipped region-normalized public observations",
        },
    )
    dump(
        status_path,
        {
            "stage": "complete",
            "features_sha256": features["sha256"],
            "metadata_sha256": sha(metadata_path),
            "tiles_done": len(frames),
        },
    )
    return metadata_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv experiment public-calendar-statistics")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.spec)
    return 0
