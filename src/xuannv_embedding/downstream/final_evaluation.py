"""Held-out evaluation with heads and thresholds fixed by development records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from xuannv_embedding.downstream.development import TASKS, knn_logits, validate_development_split
from xuannv_embedding.downstream.heads import build_head
from xuannv_embedding.downstream.metrics import evaluate_binary
from xuannv_embedding.training.cli import _git_sha
from xuannv_embedding.training.experiment import _json, _sha
from xuannv_embedding.training.probe_followup import cpu_slot


def boundary_counts(prediction, truth, valid, radius):
    counts = np.zeros(4, dtype=np.int64)
    for p, y, v in zip(prediction, truth, valid, strict=True):
        interior = binary_erosion(v, iterations=radius + 1, border_value=0)
        pb = p & ~binary_erosion(p) & interior
        yb = y & ~binary_erosion(y) & interior
        matched_p = int((pb & (distance_transform_edt(~yb) <= radius)).sum()) if yb.any() else 0
        matched_y = int((yb & (distance_transform_edt(~pb) <= radius)).sum()) if pb.any() else 0
        counts += [matched_p, int(pb.sum()), matched_y, int(yb.sum())]
    return counts.tolist()


def boundary_f1(counts):
    mp, npred, my, ntrue = counts
    precision = mp / npred if npred else 0.0
    recall = my / ntrue if ntrue else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def run(args):
    with cpu_slot(getattr(args, "slot_directory", None)):
        _run(args)


def _run(args):
    torch.set_num_threads(2)
    if args.output.exists():
        raise FileExistsError("held-out evaluation already exists")
    cache_path = args.cache / "cache.json"
    cache = json.loads(cache_path.read_text())
    manifest_path = args.embeddings / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    result_path = args.probe / "results.json"
    development = json.loads(result_path.read_text())
    if json.loads((args.probe / "status.json").read_text())["state"] != "complete":
        raise ValueError("development evaluation must finish before test scoring")
    if development["metadata"]["export_manifest_sha256"] != _sha(manifest_path):
        raise ValueError("development features changed")
    if manifest["cache_sha256"] != _sha(cache_path):
        raise ValueError("held-out label cache differs from feature registration")
    split = cache["split"]
    validate_development_split(split)
    by_id = {r["patch_id"]: i for i, r in enumerate(cache["records"])}
    support_ids = {p for row in development["rows"] for p in row["support_patch_ids"]}
    if not {by_id[p] for p in support_ids} <= set(split["train"]):
        raise ValueError("support contains held-out locations")
    indices = sorted({by_id[p] for p in support_ids} | set(split["test"]))
    features, labels = {}, {}
    args.output.mkdir(parents=True)
    for i in indices:
        record = cache["records"][i]
        if _sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("label cache changed")
        sample = torch.load(record["path"], weights_only=True, mmap=True)
        feature_record = manifest["records"][i]
        if feature_record["patch_id"] != record["patch_id"]:
            raise ValueError("feature and label patch order differ")
        with np.load(feature_record["path"]) as archive:
            features[i] = torch.from_numpy(archive["embedding"][-1].astype(np.float32))
        if not torch.isfinite(features[i]).all():
            raise ValueError("nonfinite held-out features")
        labels[i] = {}
        for task, names in TASKS.items():
            y = torch.stack([sample["supervised_labels"][k] for k in names]).amax(dim=0)
            valid = all(bool(sample["supervised_label_masks"][k].all()) for k in names)
            labels[i][task] = y if valid else torch.full_like(y, -1)
    query = torch.stack([features[i] for i in split["test"]])
    meta = {
        "git_sha": _git_sha(),
        "development_results_sha256": _sha(result_path),
        "export_manifest_sha256": _sha(manifest_path),
        "cache_sha256": _sha(cache_path),
        "test_patch_ids": [cache["records"][i]["patch_id"] for i in split["test"]],
        "threshold_source": "saved validation threshold; never optimized on test",
        "scope": "spatially held-out test; OSM-derived reference, not independent human labels",
        "boundary_grid_m": 10,
        "boundary_rule": "one-pixel inner boundary; Euclidean tolerance; tile margin excluded",
    }
    rows = []
    for row in development["rows"]:
        task, head, budget = row["task"], row["head"], row["budget"]
        selected = [by_id[p] for p in row["support_patch_ids"]]
        x = torch.stack([features[i] for i in selected])
        y = torch.stack([labels[i][task] for i in selected]).float()
        if hashlib.sha256(y.numpy().tobytes()).hexdigest() != row["support_label_sha256"]:
            raise ValueError("support labels differ from development")
        if head == "knn":
            logits, _ = knn_logits(x, y, query, torch.device("cpu"))
            head_sha = None
        else:
            model = build_head(head, embed_dim=query.shape[1], num_classes=1)
            path = args.probe / f"{task}_{budget}_{head}.pt"
            model.load_state_dict(torch.load(path, weights_only=True), strict=True)
            model.eval()
            with torch.no_grad():
                logits = torch.cat([model(chunk)[:, 0] for chunk in query.split(2)])
            head_sha = _sha(path)
        target = torch.stack([labels[i][task] for i in split["test"]])
        threshold = row["metrics"]["threshold"]
        metrics = evaluate_binary(logits, target, threshold=threshold)
        metrics["iou"] = metrics["tp"] / max(1, metrics["tp"] + metrics["fp"] + metrics["fn"])
        predicted = torch.sigmoid(logits).numpy() >= threshold
        for radius in (1, 2):
            counts = boundary_counts(predicted, target.numpy() > 0, target.numpy() >= 0, radius)
            metrics[f"boundary_f1_{10 * radius}m"] = boundary_f1(counts)
            metrics[f"boundary_counts_{10 * radius}m"] = counts
        prediction_path = args.output / f"{task}_{budget}_{head}.npz"
        np.savez_compressed(prediction_path, logits=logits.numpy())
        rows.append(
            {
                **row,
                "validation_metrics": row["metrics"],
                "metrics": metrics,
                "head_sha256": head_sha,
                "predictions": str(prediction_path),
                "prediction_sha256": _sha(prediction_path),
                "test_pixels": int((target >= 0).sum()),
            }
        )
        _json(args.output / "results.json", {"metadata": meta, "rows": rows})
        _json(
            args.output / "status.json",
            {"state": "running", "completed": len(rows), "total": len(development["rows"])},
        )
    _json(
        args.output / "status.json",
        {"state": "complete", "completed": len(rows), "total": len(rows)},
    )
