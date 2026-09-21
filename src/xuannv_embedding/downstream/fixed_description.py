"""Label-space, temporal and seam diagnostics for fixed embedding products."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances

from xuannv_embedding.export.context import dump


def spaced_samples(y, test, maximum=1000):
    # One candidate per 4x4 spatial cell before a seeded, class-balanced selection.
    rng = np.random.default_rng(20260921)
    chosen = []
    classes = []
    for c in range(6):
        pool = []
        for i in test:
            for row in range(0, 128, 4):
                for col in range(0, 128, 4):
                    rr, cc = np.where(y[i, row : row + 4, col : col + 4] == c)
                    if len(rr):
                        k = rng.integers(len(rr))
                        pool.append((i, row + int(rr[k]), col + int(cc[k])))
        selected = rng.permutation(len(pool))[:maximum]
        chosen.extend(pool[j] for j in selected)
        classes.extend([c] * len(selected))
    return np.array(chosen), np.array(classes)


def seam_statistics(features, positions):
    lookup = {p: i for i, p in enumerate(positions)}
    boundary = []
    interior = []
    for i, (x, y) in enumerate(positions):
        a = features[i]
        # Cosine distances are computed separately in each model's native coordinate space.
        if (x + 1, y) in lookup:
            b = features[lookup[(x + 1, y)]]
            boundary.extend((1 - np.sum(a[:, -1] * b[:, 0], axis=-1)).tolist())
            interior.extend((1 - np.sum(a[:, -2] * a[:, -1], axis=-1)).tolist())
            interior.extend((1 - np.sum(b[:, 0] * b[:, 1], axis=-1)).tolist())
        if (x, y + 1) in lookup:
            b = features[lookup[(x, y + 1)]]
            boundary.extend((1 - np.sum(a[-1] * b[0], axis=-1)).tolist())
            interior.extend((1 - np.sum(a[-2] * a[-1], axis=-1)).tolist())
            interior.extend((1 - np.sum(b[0] * b[1], axis=-1)).tolist())
    return {
        "boundary_cosine_distance": float(np.mean(boundary)),
        "adjacent_interior_distance": float(np.mean(interior)),
        "ratio": float(np.mean(boundary) / max(1e-12, np.mean(interior))),
        "edge_pixel_pairs": len(boundary),
    }


def run(args):
    spec = json.loads(args.spec.read_text())
    root = Path(spec["output"])
    prep = root / "prepared"
    out = root / "description"
    out.mkdir(exist_ok=True)
    cache = json.loads(Path(spec["cache"]).read_text())
    records = cache["records"]
    split = cache["split"]
    y = np.load(prep / "semantic.npy")
    loc, classes = spaced_samples(y, split["test"])
    np.savez(out / "semantic_samples.npz", locations=loc, labels=classes)
    models = list(spec["models"])
    maps = {m: np.load(prep / f"{m}.npy", mmap_mode="r") for m in models}
    bounds = np.array([r["bounds"] for r in records])
    xmin, ymin = bounds[:, :2].min(0)
    xmax, ymax = bounds[:, 2:].max(0)
    positions = [(round((b[0] - xmin) / 1280), round((ymax - b[3]) / 1280)) for b in bounds]
    rng = np.random.default_rng(20260921)
    sample_parts = {m: [] for m in models}
    for i in split["train"]:
        picks = rng.choice(128 * 128, 512, replace=False)
        for m in models:
            sample_parts[m].append(np.asarray(maps[m][i]).reshape(-1, 64)[picks])
    samples = {m: np.concatenate(v).astype("float64") for m, v in sample_parts.items()}
    u, _, vt = np.linalg.svd(samples[models[1]].T @ samples[models[0]], full_matrices=False)
    rotation = u @ vt
    joint = np.concatenate([samples[models[0]], samples[models[1]] @ rotation])
    mean = joint.mean(0)
    vals, vec = np.linalg.eigh(np.cov(joint, rowvar=False))
    comp = vec[:, np.argsort(vals)[-3:][::-1]]
    low, high = np.percentile((joint - mean) @ comp, [2, 98], axis=0)
    np.savez(out / "pca_fit.npz", rotation=rotation, mean=mean, components=comp, low=low, high=high)
    report = {
        "semantic": {},
        "temporal": {},
        "seams": {},
        "cost": {},
        "pca": {
            "fit_split": "B0 training tiles only; no labels",
            "display_alignment": "orthogonal Procrustes; statistics always use original embeddings",
            "explained_variance": (np.sort(vals)[-3:][::-1] / vals.sum()).tolist(),
        },
        "class_counts": np.bincount(classes, minlength=6).tolist(),
        "temporal_scope": "same 512 seeded pixels per test tile; no change reference",
    }
    report["small_object_definition"] = (
        "8-connected, <=25 pixels (2500 m2), excluding clipped tile-border objects; "
        "detected at >=50% pixel recall"
    )
    for mi, m in enumerate(models):
        print("describe", m, flush=True)
        z = np.asarray(maps[m][loc[:, 0], loc[:, 1], loc[:, 2]])
        centers = np.stack([z[classes == c].mean(0) for c in np.unique(classes)])
        intra = [
            float(cosine_distances(z[classes == c], centers[j : j + 1]).mean())
            for j, c in enumerate(np.unique(classes))
        ]
        report["semantic"][m] = {
            "silhouette_cosine": float(silhouette_score(z, classes, metric="cosine")),
            "davies_bouldin_euclidean": float(davies_bouldin_score(z, classes)),
            "intra_class_cosine": intra,
            "centroid_cosine_distances": cosine_distances(centers).tolist(),
        }
        np.savez(out / f"semantic_{m}.npz", features=z, labels=classes)
        manifest = json.loads((root / "exports" / m / "manifest.json").read_text())
        without = np.lib.format.open_memmap(
            out / f"{m}_no_context.npy", mode="w+", dtype="float32", shape=maps[m].shape
        )
        sim = np.zeros((6, 6), float)
        sample_count = 0
        roi_arrays = {}
        roi_ids = {
            i
            for v in json.loads((prep / "identity.json").read_text())[
                "representative_rois"
            ].values()
            for i in v
        }
        for i, r in enumerate(manifest["records"]):
            with np.load(r["path"]) as f:
                without[i] = f["without_context"].transpose(1, 2, 0)
                if i in split["test"] or i in roi_ids:
                    a = f["embedding"].transpose(0, 2, 3, 1)
                    if i in split["test"]:
                        picks = np.random.default_rng(20260921 + i).choice(
                            128 * 128, 512, replace=False
                        )
                        small = a.reshape(6, -1, 64)[:, picks]
                        sim += np.einsum("mpd,npd->mn", small, small, optimize=True)
                        sample_count += len(picks)
                    if i in roi_ids:
                        roi_arrays[str(i)] = a
        without.flush()
        sim /= sample_count
        report["temporal"][m] = {"mean_cosine_similarity": sim.tolist(), "pixels": sample_count}
        report["seams"][m] = {
            "with_context": seam_statistics(maps[m], positions),
            "without_context": seam_statistics(without, positions),
        }
        report["cost"][m] = {
            k: manifest[k] for k in ("wall_seconds", "forward_export_seconds", "bytes")
        }
        report["cost"][m]["canonical_float32_6months_bytes"] = len(records) * 6 * 64 * 128 * 128 * 4
        proj = comp if mi == 0 else rotation @ comp
        for suffix, data in [("halo", maps[m]), ("plain", without)]:
            canvas = np.ones((round((ymax - ymin) / 10), round((xmax - xmin) / 10), 3), np.float32)
            for i, (x, y0) in enumerate(positions):
                rgb = (np.asarray(data[i]) @ proj - (mean @ comp) - low) / (high - low)
                canvas[y0 * 128 : (y0 + 1) * 128, x * 128 : (x + 1) * 128] = np.clip(rgb, 0, 1)
            np.save(out / f"{m}_{suffix}_rgb.npy", canvas)
        for i, a in roi_arrays.items():
            rgb = np.clip((a @ proj - (mean @ comp) - low) / (high - low), 0, 1)
            np.save(out / f"{m}_monthly_roi_{i}.npy", rgb)
    dump(out / "results.json", report)
