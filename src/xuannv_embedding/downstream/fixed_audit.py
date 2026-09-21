"""Reproducible, paired frozen-embedding evaluation, with no encoder fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier

from xuannv_embedding.export.context import dump, sha


def nested_support(labels, ids, train, budget, seed):
    eligible = [i for i in train if (labels[i] == 1).any() and (labels[i] == 0).any()]
    eligible.sort(key=lambda i: hashlib.sha256(f"{seed}:{ids[i]}".encode()).digest())
    if len(eligible) < budget:
        raise ValueError("insufficient positive-and-negative training tiles")
    return eligible[:budget]


def balanced_positions(y, limit, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(np.flatnonzero(y == c))[:limit] for c in (0, 1)])


def paired_bootstrap(a, b, *, repeats=2000):
    rng = np.random.default_rng(20260921)
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 2:
        a, b = a[None], b[None]
    idx = rng.integers(a.shape[1], size=(repeats, a.shape[1]))

    def f1(x):
        z = x[:, idx].sum(2)
        return (2 * z[..., 0] / np.maximum(1, 2 * z[..., 0] + z[..., 1] + z[..., 2])).mean(0)

    return np.percentile(f1(b) - f1(a), [2.5, 97.5]).tolist()


def threshold(y, scores):
    v = y >= 0
    if np.unique(y[v]).size != 2:
        raise ValueError("validation lacks both classes")
    p, r, t = precision_recall_curve(y[v], scores[v])
    f = 2 * p[:-1] * r[:-1] / np.maximum(1e-15, p[:-1] + r[:-1])
    return float(t[np.argmax(f)])


def boundary_counts(y, pred, radius):
    valid = y >= 0
    # Exclude tile perimeter: clipping an object must not create an artificial boundary.
    valid = ndi.binary_erosion(valid, iterations=radius + 1)
    truth = y == 1
    a = (truth & ~ndi.binary_erosion(truth)) & valid
    b = (pred & ~ndi.binary_erosion(pred)) & valid
    da = ndi.distance_transform_edt(~a) if a.any() else np.full(a.shape, np.inf)
    db = ndi.distance_transform_edt(~b) if b.any() else np.full(b.shape, np.inf)
    return [
        int((b & (da <= radius)).sum()),
        int(b.sum()),
        int((a & (db <= radius)).sum()),
        int(a.sum()),
    ]


def bf1(c):
    a, b, d, e = c
    p, r = a / max(1, b), d / max(1, e)
    return 2 * p * r / max(1e-15, p + r)


def score_tiles(labels, scores, cut, *, small_max=25):
    counts, boundaries, objects, aps = [], [[], []], [], []
    for y, s in zip(labels, scores):
        valid = y >= 0
        truth, pred = y == 1, (s >= cut) & valid
        counts.append(
            [
                int((pred & truth).sum()),
                int((pred & ~truth & valid).sum()),
                int((~pred & truth).sum()),
            ]
        )
        aps.append(float(average_precision_score(truth[valid], s[valid])) if truth.any() else None)
        for r in (1, 2):
            boundaries[r - 1].append(boundary_counts(y, pred, r))
        cc, n = ndi.label(truth, structure=np.ones((3, 3)))
        sizes = np.bincount(cc.ravel())
        edge = set(np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]])))
        eligible = [k for k in range(1, n + 1) if sizes[k] <= small_max and k not in edge]
        hits = np.bincount(cc.ravel(), weights=pred.ravel(), minlength=n + 1)
        detected = sum((hits[k] / sizes[k]) >= 0.5 for k in eligible)
        objects.append([int(detected), len(eligible)])
    c = np.array(counts).sum(0)
    v = labels >= 0
    total_objects = np.array(objects).sum(0)
    return {
        "f1": float(2 * c[0] / max(1, 2 * c[0] + c[1] + c[2])),
        "iou": float(c[0] / max(1, c.sum())),
        "ap": float(average_precision_score(labels[v], scores[v])),
        "boundary_f1_10m": bf1(np.array(boundaries[0]).sum(0)),
        "boundary_f1_20m": bf1(np.array(boundaries[1]).sum(0)),
        "small_object_recall": (
            float(total_objects[0] / total_objects[1]) if total_objects[1] else None
        ),
        "small_objects": int(total_objects[1]),
        "block_counts": counts,
        "block_ap": aps,
        "block_boundaries": boundaries,
        "block_objects": objects,
    }


def read_labels(spec, records, names):
    import rasterio

    out = np.full((len(records), 128, 128), -1, np.int8)
    digests = []
    for i, r in enumerate(records):
        layers = []
        for name in names:
            p = Path(spec["label_root"]) / name / "masks" / f"{r['patch_id']}.tif"
            if not p.exists():
                break
            with rasterio.open(p) as ds:
                if (
                    ds.shape != (128, 128)
                    or str(ds.crs) != spec.get("crs", "EPSG:32650")
                    or not np.allclose(ds.bounds, r["bounds"], rtol=0, atol=1e-5)
                ):
                    raise ValueError("label grid differs")
                a = ds.read(1)
                if not np.isin(a, [0, 1]).all():
                    raise ValueError("unexpected binary label values")
                # Archived binary masks use nodata=0 for background; read raw values.
                layers.append(a)
            digests.append({"path": str(p), "sha256": sha(p)})
        if len(layers) == len(names):
            out[i] = np.maximum.reduce(layers)
    return out, digests


def prepare(args):
    import torch

    spec = json.loads(args.spec.read_text())
    root = Path(spec["output"])
    out = root / "prepared"
    out.mkdir(exist_ok=True)
    cache = json.loads(Path(spec["cache"]).read_text())
    records = cache["records"]
    valid = np.zeros((len(records), 128, 128), bool)
    for i, r in enumerate(records):
        if sha(r["path"]) != r["sha256"]:
            raise ValueError("cache changed")
        sample = torch.load(r["path"], weights_only=True, mmap=True)
        valid[i] = np.stack(
            [
                sample["target_masks"][k][-1].numpy() > 0
                for k in ("s2_recon", "s1_recon", "landsat_recon")
            ]
        ).any(0)
    identity = {
        "spec_sha256": sha(args.spec),
        "cache_sha256": sha(spec["cache"]),
        "labels": {},
        "exports": {},
        "valid_rule": "any valid public-source observation in May",
        "binary_zero": "background despite archived nodata=0",
    }
    for task, names in spec["tasks"].items():
        y, d = read_labels(spec, records, names)
        y[~valid] = -1
        np.save(out / f"label_{task}.npy", y)
        identity["labels"][task] = d
    ref = []
    for name in spec["reference_classes"]:
        y, d = read_labels(spec, records, [name])
        ref.append(y)
        identity["labels"][name] = d
    ref = np.stack(ref)
    count = (ref == 1).sum(0)
    semantic = np.where((count == 1) & (ref >= 0).all(0) & valid, ref.argmax(0), -1).astype(np.int8)
    np.save(out / "semantic.npy", semantic)
    for model in spec["models"]:
        mp = root / "exports" / model / "manifest.json"
        m = json.loads(mp.read_text())
        identity["exports"][model] = sha(mp)
        z = np.lib.format.open_memmap(
            out / f"{model}.npy", mode="w+", dtype="float32", shape=(len(records), 128, 128, 64)
        )
        for i, (r, er) in enumerate(zip(records, m["records"], strict=True)):
            if (
                r["patch_id"] != er["patch_id"]
                or r["bounds"] != er["bounds"]
                or sha(er["path"]) != er["sha256"]
            ):
                raise ValueError("export order, grid or digest differs")
            with np.load(er["path"]) as f:
                a = f["embedding"][-1].transpose(1, 2, 0)
                if not np.isfinite(a).all() or (np.linalg.norm(a, axis=-1)[valid[i]] < 0.9).any():
                    raise ValueError("invalid embedding")
                z[i] = a
        z.flush()
    # Representative panels selected using labels only, before any predictions exist.
    rois = {}
    for task in spec["tasks"]:
        y = np.load(out / f"label_{task}.npy")
        eligible = [i for i in cache["split"]["test"] if 0.02 < np.mean(y[i] == 1) < 0.7]
        eligible.sort(
            key=lambda i: hashlib.sha256(f"roi:{records[i]['patch_id']}".encode()).digest()
        )
        rois[task] = eligible[:1]
    identity["representative_rois"] = rois
    dump(out / "identity.json", identity)
    print("prepared", flush=True)


def accelerated_knn(support, labels, query, device):
    """Exact cosine kNN on a selected accelerator; no training or approximation."""
    import torch
    import torch.nn.functional as F

    from xuannv_embedding.training.cli import _setup_device

    torch.set_num_threads(2)
    target, _, _ = _setup_device(device)
    points = F.normalize(torch.from_numpy(support).to(target), dim=1)
    values = torch.from_numpy(labels.astype("float32")).to(target)
    output = []
    with torch.inference_mode():
        for begin in range(0, len(query), 8192):
            q = F.normalize(torch.from_numpy(query[begin : begin + 8192]).to(target), dim=1)
            indices = (q @ points.T).topk(5, dim=1).indices
            output.append(values[indices].mean(1).cpu().numpy())
    return np.concatenate(output)


def probe(args):
    spec = json.loads(args.spec.read_text())
    root = Path(spec["output"])
    prep = root / "prepared"
    cache = json.loads(Path(spec["cache"]).read_text())
    split = cache["split"]
    rec = cache["records"]
    train, val, test = split["train"], split["validation"], split["test"]
    if set(train) & set(val) or set(val) & set(test) or set(train) & set(test):
        raise ValueError("overlapping split")
    task = args.task
    y = np.load(prep / f"label_{task}.npy")
    ids = [r["patch_id"] for r in rec]
    out = root / "probes" / task
    out.mkdir(parents=True, exist_ok=True)
    x = {m: np.load(prep / f"{m}.npy", mmap_mode="r") for m in spec["models"]}
    query_ids = val + test
    nval = len(val)
    query = {m: np.asarray(a[query_ids]).reshape(-1, 64) for m, a in x.items()}
    rows = []
    for seed in spec["seeds"]:
        for budget in spec["budgets"]:
            selected = nested_support(y, ids, train, budget, seed)
            sy = y[selected].reshape(-1)
            for head in (["ridge", "knn"] if budget == 5 else ["ridge"]):
                picks = balanced_positions(sy, 4096 if head == "ridge" else 1024, seed)
                for model in spec["models"]:
                    key = f"{model}_{seed}_{budget}_{head}"
                    dest = out / f"{key}.json"
                    if dest.exists():
                        rows.append(json.loads(dest.read_text()))
                        continue
                    fit = (
                        RidgeClassifier(alpha=1, class_weight="balanced", solver="cholesky")
                        if head == "ridge"
                        else KNeighborsClassifier(
                            n_neighbors=5, metric="cosine", algorithm="brute", n_jobs=2
                        )
                    )
                    sx = np.asarray(x[model][selected]).reshape(-1, 64)[picks]
                    tic = time.monotonic()
                    fit.fit(sx, sy[picks])
                    fit_s = time.monotonic() - tic
                    if head == "ridge":
                        np.savez(
                            out / f"{key}_head.npz",
                            coef=fit.coef_,
                            intercept=fit.intercept_,
                            classes=fit.classes_,
                            support_indices=selected,
                            pixel_indices=picks,
                        )
                    else:
                        np.savez(
                            out / f"{key}_head.npz",
                            x=sx,
                            y=sy[picks],
                            support_indices=selected,
                            pixel_indices=picks,
                        )
                    tic = time.monotonic()
                    if head == "ridge":
                        scores = fit.decision_function(query[model])
                    elif args.device != "cpu":
                        scores = accelerated_knn(sx, sy[picks], query[model], args.device)
                    else:
                        scores = np.concatenate(
                            [fit.predict_proba(q)[:, 1] for q in np.array_split(query[model], 64)]
                        )
                    inference_s = time.monotonic() - tic
                    scores = scores.reshape(len(query_ids), 128, 128).astype("float32")
                    cut = threshold(y[val].ravel(), scores[:nval].ravel())
                    metrics = score_tiles(
                        y[test], scores[nval:], cut, small_max=spec["small_object_max_pixels"]
                    )
                    np.savez_compressed(
                        out / f"{key}_predictions.npz",
                        scores=scores,
                        patch_indices=query_ids,
                        threshold=cut,
                    )
                    row = {
                        "model": model,
                        "task": task,
                        "seed": seed,
                        "budget": budget,
                        "head": head,
                        "threshold": cut,
                        "support_patch_ids": [ids[i] for i in selected],
                        "support_indices": selected,
                        "fitted_pixels": len(picks),
                        "labeled_pixels": int((sy >= 0).sum()),
                        "support_label_sha256": hashlib.sha256(sy.tobytes()).hexdigest(),
                        "sample_positions_sha256": hashlib.sha256(picks.tobytes()).hexdigest(),
                        "fit_seconds": fit_s,
                        "predict_seconds": inference_s,
                        "predict_device": args.device if head == "knn" else "cpu",
                        "query_pixels": len(query[model]),
                        "metrics": metrics,
                    }
                    dump(dest, row)
                    rows.append(row)
                    print(task, key, "F1", round(metrics["f1"], 4), flush=True)
    dump(out / "results.json", rows)
    retrieval(spec, cache, task, y, x, out)


def retrieval(spec, cache, task, y, features, out):
    test = cache["split"]["test"]
    rows = []
    candidates = []
    for i in cache["split"]["train"]:
        cc, n = ndi.label(y[i] == 1, structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            pix = np.flatnonzero(cc.ravel() == k)
            if len(pix) >= 4:
                candidates.append((i, k, pix))
    for seed in spec["seeds"]:
        ranked = sorted(
            candidates,
            key=lambda o: hashlib.sha256(
                f"{seed}:{cache['records'][o[0]]['patch_id']}:{o[1]}".encode()
            ).digest(),
        )
        for budget in (1, 3, 5):
            if len(ranked) < budget:
                rows.append(
                    {"seed": seed, "budget": budget, "status": "insufficient reference components"}
                )
                continue
            selected = ranked[:budget]
            for model, x in features.items():
                tic = time.monotonic()
                prototypes = np.stack(
                    [np.asarray(x[i]).reshape(-1, 64)[p].mean(0) for i, k, p in selected]
                )
                prototypes /= np.maximum(1e-12, np.linalg.norm(prototypes, axis=1, keepdims=True))
                q = np.asarray(x[test]).reshape(-1, 64)
                scores = (q @ prototypes.T).max(1)
                seconds = time.monotonic() - tic
                valid = y[test].ravel() >= 0
                truth = y[test].ravel()[valid] == 1
                s = scores[valid]
                rank = np.argsort(-s, kind="stable")
                metrics = {"ap": float(average_precision_score(truth, s))}
                for fraction in (0.01, 0.05):
                    k = max(1, int(np.ceil(len(rank) * fraction)))
                    tp = int(truth[rank[:k]].sum())
                    metrics[f"precision_top{fraction}"] = tp / k
                    metrics[f"recall_top{fraction}"] = tp / max(1, truth.sum())
                row = {
                    "task": task,
                    "model": model,
                    "seed": seed,
                    "budget": budget,
                    "seconds": seconds,
                    "samples": [
                        {
                            "patch_id": cache["records"][i]["patch_id"],
                            "component": k,
                            "pixels": len(p),
                        }
                        for i, k, p in selected
                    ],
                    "metrics": metrics,
                }
                rows.append(row)
                np.savez_compressed(
                    out / f"retrieval_{model}_{seed}_{budget}.npz",
                    scores=scores.reshape(len(test), 128, 128),
                )
    dump(out / "retrieval.json", rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=["export", "prepare", "probe", "describe", "report", "reproduce"]
    )
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--model")
    p.add_argument("--task")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if args.stage == "export":
        from xuannv_embedding.export.context import run

        run(args)
    elif args.stage == "prepare":
        prepare(args)
    elif args.stage == "probe":
        probe(args)
    elif args.stage == "describe":
        from xuannv_embedding.downstream.fixed_description import run

        run(args)
    elif args.stage == "reproduce":
        from xuannv_embedding.downstream.fixed_reproduction import run

        run(args)
    else:
        from xuannv_embedding.downstream.fixed_report import run

        run(args)
    return 0
