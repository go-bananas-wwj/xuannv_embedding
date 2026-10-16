"""Frozen spatial land-cover and surface-height evaluation for paired regional models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import from_bounds
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

from xuannv_embedding.config import Config
from xuannv_embedding.downstream.heads import build_head
from xuannv_embedding.downstream.unet_readout import UNet
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.cli import _setup_device


def classification_metrics(confusion: np.ndarray, classes: list[int]) -> dict:
    c = np.asarray(confusion, dtype=np.float64)
    if c.shape != (len(classes), len(classes)) or np.any(c < 0) or not np.isfinite(c).all():
        raise ValueError("Invalid confusion matrix")
    true, predicted, correct = c.sum(1), c.sum(0), c.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros_like(correct), where=predicted > 0)
    recall = np.divide(correct, true, out=np.zeros_like(correct), where=true > 0)
    f1 = np.divide(
        2 * correct, true + predicted, out=np.zeros_like(correct), where=(true + predicted) > 0
    )
    union = true + predicted - correct
    iou = np.divide(correct, union, out=np.zeros_like(correct), where=union > 0)
    return {
        "macro_f1": float(f1.mean()),
        "miou": float(iou.mean()),
        "accuracy": float(correct.sum() / max(c.sum(), 1)),
        "confusion": c.astype(np.int64).tolist(),
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "iou": float(iou[i]),
                "support": int(true[i]),
            }
            for i, label in enumerate(classes)
        },
    }


def sample_support(labels, valid, tiles, classes, budget, *, seed):
    """Nested supports, distributed across training tiles and 50 m sampling cells."""
    if budget < 1 or not tiles or not classes:
        raise ValueError("Positive budgets and nonempty support classes/tiles are required")
    labels, valid = np.asarray(labels), np.asarray(valid)
    if labels.shape != valid.shape or labels.ndim != 3:
        raise ValueError("Labels and validity must share [N,H,W] shape")
    _, h, w = labels.shape
    result = []
    for label in classes:
        rng = np.random.default_rng(seed + int(label) * 997)
        candidates = []
        for tile in sorted(tiles):
            yy, xx = np.where(valid[tile] & (labels[tile] == label))
            order = rng.permutation(len(yy))
            seen, positions = set(), []
            for j in order:
                cell = (int(yy[j]) // 5, int(xx[j]) // 5)
                if cell not in seen:
                    positions.append(int(tile * h * w + yy[j] * w + xx[j]))
                    seen.add(cell)
            if positions:
                candidates.append(positions)
        rng.shuffle(candidates)
        ranked = []
        for offset in range(max(map(len, candidates), default=0)):
            ranked.extend(row[offset] for row in candidates if len(row) > offset)
            if len(ranked) >= budget:
                break
        if not ranked:
            raise ValueError(f"Class {label} has no labeled training support")
        result.extend(ranked[:budget])
    return result


def select_classes(labels, valid, folds, bounds, *, min_train=1000, min_query=200):
    b = np.asarray(bounds)
    xy = np.round((b[:, :2] - b[:, :2].min(0)) / (b[0, 2:] - b[0, :2])).astype(int) // 2
    blocks = [tuple(row) for row in xy.tolist()]
    classes, reasons = [], {}
    for label in sorted(int(v) for v in np.unique(labels[valid]) if v > 0):
        rejected = []
        for fold in folds:
            for part in ["train", "validation", "test"]:
                counts = [(i, int(((labels[i] == label) & valid[i]).sum())) for i in fold[part]]
                total = sum(n for _, n in counts)
                groups = len({blocks[i] for i, n in counts if n > 0})
                minimum = min_train if part == "train" else min_query
                if total < minimum or groups < (3 if part == "train" else 2):
                    rejected.append(
                        {
                            "fold": fold["fold"] if "fold" in fold else 0,
                            "part": part,
                            "pixels": total,
                            "blocks": groups,
                        }
                    )
        if rejected:
            reasons[str(label)] = rejected
        else:
            classes.append(label)
    return classes, reasons


def _reduce(values, valid):
    count = valid.sum(0)
    denominator = np.maximum(count, 1)[None]
    clean = np.where(valid[:, None], values, 0)
    mean = clean.sum(0) / denominator
    variance = (clean**2).sum(0) / denominator - mean**2
    minimum = np.where(valid[:, None], values, np.inf).min(0)
    maximum = np.where(valid[:, None], values, -np.inf).max(0)
    minimum[:, count == 0] = maximum[:, count == 0] = 0
    return np.concatenate([mean, np.sqrt(np.maximum(variance, 0)), minimum, maximum])


def raw_features(sample: dict, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, availability = [], []
    for source, tensor in sample["source_frames"].items():
        values = tensor.numpy()
        valid = sample["target_masks"][source + "_recon"].numpy() > 0
        availability.append(valid.any(0))
        channels = values.shape[1] - (source in {"s2", "landsat"})
        values = values[:, :channels]
        features.extend([_reduce(values, valid), valid.mean(0)[None]])
        if source in {"s2", "landsat"}:
            path = config.data.datasets[0].statistics_dir / f"{source}_stats.json"
            stats = json.loads(path.read_text())
            physical = (
                values * np.asarray(stats["std"][:channels])[None, :, None, None]
                + np.asarray(stats["mean"][:channels])[None, :, None, None]
            )
            nir, red, green = (7, 3, 2) if source == "s2" else (3, 2, 1)
            indices = []
            for first, second in [(nir, red), (green, nir)]:
                denominator = physical[:, first] + physical[:, second]
                index = np.divide(
                    physical[:, first] - physical[:, second],
                    denominator,
                    out=np.zeros_like(denominator),
                    where=np.abs(denominator) > 1e-6,
                )
                indices.append(index)
            features.append(_reduce(np.stack(indices, axis=1), valid))
    public = np.concatenate(features).astype(np.float32)
    image = next(iter(sample["highres_frames"].values()))
    mask = next(iter(sample["highres_masks"].values()))
    size = public.shape[-2:]
    coverage = F.interpolate(mask[None], size=size, mode="area")[0]
    mean = F.interpolate((image * mask)[None], size=size, mode="area")[0] / coverage.clamp_min(1e-6)
    second = F.interpolate((image.square() * mask)[None], size=size, mode="area")[0]
    variance = second / coverage.clamp_min(1e-6) - mean.square()
    highres = np.concatenate(
        [public, mean.numpy(), variance.clamp_min(0).sqrt().numpy(), coverage.numpy()]
    ).astype(np.float32)
    return public, highres, np.stack(availability).any(0)


def prepare(spec: dict) -> dict:
    trial, root, output = map(Path, (spec["trial"], spec["data"], spec["output"]))
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "prepared.json"
    if ready.exists():
        return json.loads(ready.read_text())
    config = Config.from_yaml(trial / "configs/highres.yaml")
    cache_path = trial / "cache/highres/cache.json"
    cache = json.loads(cache_path.read_text())
    records = cache["records"]
    folds = json.loads((trial / "folds.json").read_text())["folds"]
    size = config.data.patch_size
    n = len(records)
    labels = np.zeros((n, size, size), np.int16)
    land_valid = np.zeros_like(labels, bool)
    valid = np.zeros_like(labels, bool)
    height_valid = np.zeros_like(labels, bool)
    heights = np.zeros((n, 4, size, size), np.float32)
    aef = np.lib.format.open_memmap(
        output / "aef.partial.npy", mode="w+", dtype=np.float16, shape=(n, 64, size, size)
    )
    public = highres = None
    paths = {
        "labels": root / "evaluation/lcdb_2023_2024_classes.tif",
        "interior": root / "evaluation/lcdb_interior_mask.tif",
        "height": root / "evaluation/lidar_2024_targets.tif",
        "aef": Path(spec["reference"]),
    }
    border = np.zeros((size, size), bool)
    border[10:-10, 10:-10] = True
    for i, record in enumerate(records):
        arrays = {}
        for key, path in paths.items():
            with rasterio.open(path) as raster:
                window = from_bounds(*record["bounds"], transform=raster.transform)
                if str(raster.crs) != record["crs"]:
                    raise ValueError("Benchmark grid CRS differs")
                arrays[key] = (
                    raster.read(window=window, masked=True).astype(np.float32).filled(np.nan)
                    if key in {"height", "aef"}
                    else raster.read(window=window)
                )
                if arrays[key].shape[-2:] != (size, size):
                    raise ValueError("Benchmark grid extent differs")
        sample = torch.load(record["path"], weights_only=True, mmap=True)
        x, h, observed = raw_features(sample, config)
        if public is None:
            public = np.lib.format.open_memmap(
                output / "raw_public.partial.npy", mode="w+", dtype=np.float32, shape=(n, *x.shape)
            )
            highres = np.lib.format.open_memmap(
                output / "raw_highres.partial.npy", mode="w+", dtype=np.float32, shape=(n, *h.shape)
            )
        if not np.isfinite(x).all() or not np.isfinite(h).all():
            raise ValueError("Nonfinite conventional baseline features")
        public[i], highres[i], aef[i] = x, h, arrays["aef"]
        valid[i] = border & observed & np.isfinite(arrays["aef"]).all(0)
        labels[i] = arrays["labels"][0]
        land_valid[i] = valid[i] & (arrays["interior"][0] > 0) & (labels[i] > 0)
        heights[i] = arrays["height"]
        height_valid[i] = valid[i] & np.isfinite(heights[i]).all(0)
        dump(output / "status.json", {"state": "preparing", "completed": i + 1, "total": n})
    classes, excluded = select_classes(labels, land_valid, folds, [r["bounds"] for r in records])
    if len(classes) < 2:
        raise ValueError("Insufficient eligible land-cover classes")
    class_valid = land_valid & np.isin(labels, classes)
    np.savez_compressed(
        output / "targets.npz",
        labels=labels,
        valid=class_valid,
        heights=heights,
        height_valid=height_valid,
    )
    for name, array in [("aef", aef), ("raw_public", public), ("raw_highres", highres)]:
        array.flush()
        (output / f"{name}.partial.npy").replace(output / f"{name}.npy")
    supports = {}
    for fold in folds:
        for budget in [50, 200, 1000]:
            positions = sample_support(labels, class_valid, fold["train"], classes, budget, seed=41)
            supports[f"{fold['fold']}:{budget}"] = positions
    dump(output / "support.json", supports)
    report = {
        "state": "complete",
        "classes": classes,
        "excluded_classes": excluded,
        "class_names": json.loads((root / "evaluation-reference-report.json").read_text())["LCDB"][
            "classes"
        ],
        "eligibility": {
            "minimum_train_pixels": 1000,
            "minimum_validation_test_pixels": 200,
            "minimum_train_blocks": 3,
            "minimum_query_blocks": 2,
        },
        "class_valid_fraction": float(class_valid.mean()),
        "height_valid_fraction": float(height_valid.mean()),
        "folds": folds,
        "reference_sha256": sha(spec["reference"]),
        "cache_sha256": sha(cache_path),
        "folds_sha256": sha(trial / "folds.json"),
        "support_sha256": sha(output / "support.json"),
        "target_sha256": sha(output / "targets.npz"),
        "transductive_encoder": True,
        "test_used_for_tuning": False,
        "classification_budgets": [50, 200, 1000],
    }
    dump(ready, report)
    return report


def _points(features, positions, shape):
    tile, y, x = np.unravel_index(positions, shape)
    return np.asarray(features[tile, :, y, x], np.float32)


def _confusion(head, features, target, valid, tiles, classes, scaler, device, prediction_maps=None):
    matrix = np.zeros((len(classes), len(classes)), np.int64)
    head.eval()
    with torch.inference_mode():
        for tile in tiles:
            x = np.asarray(features[tile], np.float32)
            x = ((x - scaler.mean_[:, None, None]) / scaler.scale_[:, None, None]).astype(
                np.float32
            )
            logits = head(torch.from_numpy(np.nan_to_num(x))[None].to(device))[0]
            prediction = logits.argmax(0).cpu().numpy()
            mask = valid[tile]
            if prediction_maps is not None:
                prediction_maps[tile][mask] = np.asarray(classes, np.uint8)[prediction[mask]]
            matrix += confusion_matrix(
                target[tile][mask], prediction[mask], labels=list(range(len(classes)))
            )
    return matrix


def _fit_neural(kind, features, target, valid, support, fold, classes, device_name):
    device, _, _ = _setup_device(device_name)
    torch.manual_seed(41)
    if device.type == "npu":
        torch.npu.manual_seed_all(41)
    channels = features.shape[1]
    if kind == "unet":
        head = UNet(channels)
        head.output = nn.Conv2d(32, len(classes), 1)
    else:
        head = build_head(kind, embed_dim=channels, num_classes=len(classes))
    head = head.float().to(device)
    scaler = StandardScaler().fit(_points(features, support, target.shape))
    sparse = np.full(target.shape, -1, np.int64)
    sparse.ravel()[support] = target.ravel()[support]
    tiles = [i for i in fold["train"] if (sparse[i] >= 0).any()]
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    counts = np.bincount(target.ravel()[support], minlength=len(classes))
    class_weights = torch.from_numpy(
        (counts.sum() / (len(classes) * counts)).astype(np.float32)
    ).to(device)
    rng = np.random.default_rng(41)
    best, curve = None, []
    for step in range(1, 601):
        picks = rng.choice(tiles, min(4, len(tiles)), replace=False)
        x = np.asarray(features[picks], np.float32)
        x = np.nan_to_num(
            (x - scaler.mean_[None, :, None, None]) / scaler.scale_[None, :, None, None]
        ).astype(np.float32)
        y = torch.from_numpy(sparse[picks]).to(device)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(torch.from_numpy(x).to(device))
        loss = F.cross_entropy(logits, y, ignore_index=-1, weight=class_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite downstream loss")
        loss.backward()
        optimizer.step()
        if step in [100, 300, 600]:
            matrix = _confusion(
                head, features, target, valid, fold["validation"], classes, scaler, device
            )
            score = classification_metrics(matrix, classes)["macro_f1"]
            curve.append({"steps": step, "validation_macro_f1": score})
            if best is None or score > best[0]:
                best = (
                    score,
                    step,
                    {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                )
    head.load_state_dict(best[2])
    return head, scaler, device, {"selected_steps": best[1], "curve": curve}


def feature_path(root: Path, feature: str) -> Path:
    if feature in {"aef", "raw_public", "raw_highres"}:
        return root / f"{feature}.npy"
    base, representation = feature.split("_", 1) if "_" in feature else (feature, "annual")
    if base not in {"public", "highres"} or representation not in {"annual", "april", "mar_may"}:
        raise ValueError("Unregistered feature representation")
    return root / "representations" / base / f"{representation}.npy"


def score_job(spec: dict, job: dict, *, device_name: str) -> dict:
    root = Path(spec["output"])
    name = f"{job['feature']}_{job['head']}_fold{job['fold']}_n{job['budget']}"
    output = root / "scores" / name
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "result.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    metadata = json.loads((root / "prepared.json").read_text())
    feature = job["feature"]
    path = feature_path(root, feature)
    features = np.load(path, mmap_mode="r")
    with np.load(root / "targets.npz") as z:
        labels, valid = z["labels"], z["valid"]
    classes = metadata["classes"]
    target = np.full(labels.shape, -1, np.int64)
    for i, label in enumerate(classes):
        target[(labels == label) & valid] = i
    fold = metadata["folds"][job["fold"]]
    predictions = np.zeros(labels.shape, np.uint8)
    support = json.loads((root / "support.json").read_text())[f"{job['fold']}:{job['budget']}"]
    counts = {str(label): int((labels.ravel()[support] == label).sum()) for label in classes}
    started = time.monotonic()
    if job["head"] == "rf":
        x, y = _points(features, support, target.shape), target.ravel()[support]
        best, trials = None, []
        validation_indices = np.concatenate(
            [
                np.flatnonzero(valid[i]) + i * labels.shape[1] * labels.shape[2]
                for i in fold["validation"]
            ]
        )
        val_x = _points(features, validation_indices, target.shape)
        val_y = target.ravel()[validation_indices]
        with threadpool_limits(limits=2):
            for leaf in [1, 3]:
                head = RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=leaf,
                    class_weight="balanced",
                    random_state=41,
                    n_jobs=2,
                )
                head.fit(x, y)
                matrix = confusion_matrix(val_y, head.predict(val_x), labels=range(len(classes)))
                score = classification_metrics(matrix, classes)["macro_f1"]
                trials.append({"min_samples_leaf": leaf, "validation_macro_f1": score})
                if best is None or score > best[0]:
                    best = score, head
            head = best[1]
            matrix = np.zeros((len(classes), len(classes)), np.int64)
            for tile in fold["test"]:
                positions = np.flatnonzero(valid[tile]) + tile * labels.shape[1] * labels.shape[2]
                prediction = head.predict(_points(features, positions, target.shape))
                predictions.ravel()[positions] = np.asarray(classes, np.uint8)[prediction]
                matrix += confusion_matrix(
                    target.ravel()[positions], prediction, labels=range(len(classes))
                )
        details = {"validation_trials": trials}
    else:
        head, scaler, device, details = _fit_neural(
            job["head"], features, target, valid, support, fold, classes, device_name
        )
        matrix = _confusion(
            head,
            features,
            target,
            valid,
            fold["test"],
            classes,
            scaler,
            device,
            prediction_maps=predictions,
        )
        torch.save(
            {
                "head": head.state_dict(),
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
            output / "head.pt",
        )
    report = {
        "state": "complete",
        **job,
        **details,
        "metrics": classification_metrics(matrix, classes),
        "support_per_class": counts,
        "support_sha256": metadata["support_sha256"],
        "target_sha256": metadata["target_sha256"],
        "selection": "validation_macro_f1_only",
        "elapsed_seconds": time.monotonic() - started,
    }
    np.savez_compressed(
        output / "predictions.npz",
        labels=predictions,
        test_tiles=np.asarray(fold["test"], np.int32),
    )
    dump(report_path, report)
    return report


def score_regression(spec: dict, feature: str) -> dict:
    root = Path(spec["output"])
    path = feature_path(root, feature)
    features = np.load(path, mmap_mode="r")
    metadata = json.loads((root / "prepared.json").read_text())
    with np.load(root / "targets.npz") as z:
        target = z["heights"].transpose(0, 2, 3, 1)
        valid = z["height_valid"]
    names = [
        "terrain_elevation_m",
        "terrain_slope_degrees",
        "surface_height_mean_m",
        "surface_height_p90_m",
    ]
    results = []
    maps = np.full(target.shape, np.nan, np.float16)
    for fold in metadata["folds"]:
        positions = sample_support(
            np.zeros_like(valid, np.uint8), valid, fold["train"], [0], 5000, seed=41
        )
        scaler = StandardScaler().fit(_points(features, positions, valid.shape))
        x = scaler.transform(_points(features, positions, valid.shape))
        y = target.reshape(-1, 4)[positions]
        ymean, yscale = y.mean(0), np.maximum(y.std(0), 1e-6)
        queries = {}
        for part in ["validation", "test"]:
            queries[part] = np.concatenate(
                [np.flatnonzero(valid[i]) + i * valid.shape[1] * valid.shape[2] for i in fold[part]]
            )
        vx = scaler.transform(_points(features, queries["validation"], valid.shape))
        vy = target.reshape(-1, 4)[queries["validation"]]
        best = None
        for alpha in [0.1, 1, 10, 100]:
            model = Ridge(alpha=alpha).fit(x, (y - ymean) / yscale)
            error = np.abs((model.predict(vx) * yscale + ymean - vy) / yscale).mean()
            if best is None or error < best[0]:
                best = error, alpha, model
        tx = scaler.transform(_points(features, queries["test"], valid.shape))
        truth = target.reshape(-1, 4)[queries["test"]]
        prediction = best[2].predict(tx) * yscale + ymean
        maps.reshape(-1, 4)[queries["test"]] = prediction.astype(np.float16)
        metrics = {}
        for i, name in enumerate(names):
            error = prediction[:, i] - truth[:, i]
            denominator = ((truth[:, i] - truth[:, i].mean()) ** 2).sum()
            metrics[name] = {
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt((error**2).mean())),
                "r2": float(1 - (error**2).sum() / denominator) if denominator > 0 else None,
                "pixels": len(error),
                "squared_error_sum": float((error**2).sum()),
                "absolute_error_sum": float(np.abs(error).sum()),
            }
        results.append({"fold": fold["fold"], "alpha": best[1], "metrics": metrics})
    report = {
        "state": "complete",
        "feature": feature,
        "head": "ridge",
        "supervised_points_per_fold": 5000,
        "folds": results,
        "height_definition": "DSM minus DEM includes vegetation and buildings",
    }
    (root / "regression").mkdir(exist_ok=True)
    np.savez_compressed(root / "regression" / f"{feature}_predictions.npz", values=maps)
    dump(root / "regression" / f"{feature}.json", report)
    return report
