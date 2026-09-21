"""Paired product comparison with training-only scaling and validation selection."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numexpr as ne
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from xuannv_embedding.downstream.fixed_audit import balanced_positions, nested_support, threshold
from xuannv_embedding.export.context import dump, sha


def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def official_raster(record):
    """Decode native COG samples before bilinear interpolation, then unit-normalize."""
    import math

    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window

    grid = record["reference_grid"]
    dst = np.full((64, *grid["shape"]), np.nan, dtype="float32")
    for asset in record["source_assets"]:
        with rasterio.open(asset["cache_path"]) as ds:
            left, bottom, right, top = transform_bounds(grid["crs"], ds.crs, *grid["bounds"])
            corners = [(~ds.transform) * (x, y) for x in (left, right) for y in (bottom, top)]
            x0 = max(0, math.floor(min(p[0] for p in corners)) - 3)
            y0 = max(0, math.floor(min(p[1] for p in corners)) - 3)
            x1 = min(ds.width, math.ceil(max(p[0] for p in corners)) + 3)
            y1 = min(ds.height, math.ceil(max(p[1] for p in corners)) + 3)
            if x1 <= x0 or y1 <= y0:
                continue
            window = Window(x0, y0, x1 - x0, y1 - y0)
            raw = ds.read(window=window)
            values = raw.astype("float32")
            values = np.sign(values) * (values / 127.5) ** 2
            values[raw == -128] = np.nan
            warped = np.full_like(dst, np.nan)
            reproject(
                values,
                warped,
                src_transform=ds.window_transform(window),
                src_crs=ds.crs,
                dst_transform=Affine(*grid["transform"]),
                dst_crs=grid["crs"],
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            take = np.isfinite(warped).all(0) & ~np.isfinite(dst).all(0)
            dst[:, take] = warped[:, take]
    norm = np.linalg.norm(dst, axis=0)
    valid = np.isfinite(dst).all(0) & (norm > 1e-12)
    dst = np.where(valid[None], dst / np.maximum(norm[None], 1e-12), 0)
    return dst.transpose(1, 2, 0), valid


def choose_candidate(scores, parameters, head):
    best = max(scores)
    tied = [i for i, s in enumerate(scores) if abs(s - best) <= 1e-12]
    return min(tied, key=lambda i: -parameters[i] if head == "ridge" else parameters[i])


def svm_scores(fits, q, support):
    """Exact RBF decision functions, sharing kernel evaluation across C candidates."""
    gamma = fits[0]._gamma
    if any(f._gamma != gamma for f in fits):
        raise ValueError("kernel gamma must be shared")
    indices = np.unique(np.concatenate([f.support_ for f in fits]))
    s = support[indices]
    weights = np.zeros((len(s), len(fits)), dtype=np.float64)
    for j, f in enumerate(fits):
        weights[np.searchsorted(indices, f.support_), j] = f.dual_coef_[0]
    dot = q @ s.T
    qn = np.sum(q * q, axis=1)[:, None]
    sn = np.sum(s * s, axis=1)[None]
    # Preserve the original float64 arithmetic order, fusing only array passes.
    d = ne.evaluate(
        "exp(-gamma * where(((-2 * dot) + qn) + sn > 0, ((-2 * dot) + qn) + sn, 0))",
        local_dict={"dot": dot, "qn": qn, "sn": sn, "gamma": gamma},
        optimization="moderate",
        out=dot,
    )
    return d @ weights + np.array([f.intercept_[0] for f in fits])


def read_spec(path):
    s = json.loads(Path(path).read_text())
    allowed = {"output", "source_audit", "alphaearth", "seeds", "budgets", "workers", "threads"}
    if set(s) != allowed or not 1 <= s["workers"] <= 4 or not 1 <= s["threads"] <= 4:
        raise ValueError("invalid comparison specification")
    return s


def prepare(args):
    import rasterio

    s = read_spec(args.spec)
    root, base, aef = map(Path, (s["output"], s["source_audit"], s["alphaearth"]))
    prep = root / "prepared"
    prep.mkdir(parents=True, exist_ok=False)
    old_spec = json.loads((base / "spec.json").read_text())
    cache = json.loads(Path(old_spec["cache"]).read_text())
    records = cache["records"]
    manifest = json.loads((aef / "aef_source_manifest.json").read_text())
    if (
        manifest["year"] != 2025
        or manifest["dequantization"] != "sign(x) * (x / 127.5)^2; int8 -128 is nodata"
    ):
        raise ValueError("unrecognized official product")
    official = {r["patch_id"]: r for r in manifest["records"]}
    fi = json.loads((aef / "embedding_file_index.json").read_text())
    hashes = {r["path"]: r["sha256"] for r in fi["files"]}
    checked = {}
    sources = {}
    for r in manifest["records"]:
        for asset in r["source_assets"]:
            sources[asset["cache_path"]] = asset["sha256"]
    for p, h in sources.items():
        if sha(p) != h:
            raise ValueError("official COG checksum mismatch")
        with rasterio.open(p) as ds:
            if ds.count != 64 or ds.nodata != -128 or ds.dtypes[0] != "int8":
                raise ValueError("official COG encoding mismatch")
            if ds.descriptions != tuple(f"A{i:02d}" for i in range(64)):
                raise ValueError("official axis order mismatch")
            checked[p] = {"sha256": h, "crs": str(ds.crs), "transform": list(ds.transform)}
    maps = {
        m: np.lib.format.open_memmap(
            prep / f"{m}.npy", mode="w+", dtype="float32", shape=(len(records), 128, 128, d)
        )
        for m, d in (("AlphaEarth", 64), ("raw", 150))
    }
    current = base / "prepared" / "xuannv.npy"
    (prep / "xuannv.npy").symlink_to(current)
    common = np.zeros((len(records), 128, 128), bool)
    identities = []
    for i, r in enumerate(records):
        o = official[r["patch_id"]]
        g = o["reference_grid"]
        if (
            g["shape"] != [128, 128]
            or g["crs"] != "EPSG:32650"
            or not np.allclose(g["bounds"], r["bounds"], rtol=0, atol=1e-5)
        ):
            raise ValueError("official feature grid mismatch")
        p = aef / o["output_path"]
        if (
            sha(p) != hashes[o["output_path"]]
            or sha(aef / o["valid_mask_path"]) != o["valid_mask_sha256"]
        ):
            raise ValueError("official tensor or validity changed")
        a = torch.load(p, weights_only=True, map_location="cpu").numpy().transpose(1, 2, 0)
        av = np.load(aef / o["valid_mask_path"]).astype(bool)
        norm = np.linalg.norm(a, axis=-1)
        if not np.isfinite(a).all() or (norm[av] < 1e-8).any():
            raise ValueError("invalid official vectors")
        corrected, corrected_valid = official_raster(o)
        if not np.array_equal(av, corrected_valid):
            raise ValueError("corrected official coverage differs; revisit shared masks")
        maps["AlphaEarth"][i] = corrected
        if sha(r["path"]) != r["sha256"]:
            raise ValueError("raw source cache changed")
        sample = torch.load(r["path"], weights_only=True, map_location="cpu")
        channels, valids = [], []
        for name in ("s2", "s1", "landsat"):
            x = sample["source_frames"][name].numpy().copy()
            valid = sample["target_masks"][name + "_recon"].numpy() > 0
            x *= valid[:, None]
            channels.extend((x.reshape(-1, 128, 128), valid.astype("float32")))
            valids.append(valid[-1])
        for name in ("highres_optical", "highres_sar"):
            p = base / "exports/xuannv/highres_inputs" / f"{r['patch_id']}_{name}.npz"
            with np.load(p) as z:
                channels.extend((z["x"] * z["m"], z["m"]))
        raw = np.concatenate(channels).transpose(1, 2, 0)
        if raw.shape != (128, 128, 150) or not np.isfinite(raw).all():
            raise ValueError("unexpected raw features")
        maps["raw"][i] = raw
        common[i] = av & np.stack(valids).any(0)
        identities.append(
            {
                "patch_id": r["patch_id"],
                "official_sha256": hashes[o["output_path"]],
                "raw_sha256": r["sha256"],
                "common_pixels": int(common[i].sum()),
            }
        )
    for a in maps.values():
        a.flush()
    tasks = {}
    for task in old_spec["tasks"]:
        y = np.load(base / "prepared" / f"label_{task}.npy")
        y[~common] = -1
        name = "osm_" + task
        np.save(prep / f"label_{name}.npy", y)
        tasks[name] = {"family": "OSM", "name": task}
    semantic = np.load(base / "prepared/semantic.npy")
    for k, name in enumerate(old_spec["reference_classes"]):
        y = np.where((semantic >= 0) & common, (semantic == k).astype("int8"), -1).astype("int8")
        np.save(prep / f"label_{name}.npy", y)
        tasks[name] = {"family": "ESRI", "name": name.removeprefix("esri_")}
    np.save(prep / "common.npy", common)
    rois = {}
    for task in tasks:
        y = np.load(prep / f"label_{task}.npy")
        eligible = [i for i in cache["split"]["test"] if 0.02 < np.mean(y[i] == 1) < 0.7]
        rois[task] = sorted(
            eligible,
            key=lambda i: hashlib.sha256(f"roi:{records[i]['patch_id']}".encode()).digest(),
        )[:1]
    identity = {
        "spec_sha256": sha(args.spec),
        "source_spec_sha256": sha(base / "spec.json"),
        "cache_sha256": sha(old_spec["cache"]),
        "official_manifest_sha256": sha(aef / "aef_source_manifest.json"),
        "official_cogs": checked,
        "records": identities,
        "split": cache["split"],
        "tasks": tasks,
        "rois": rois,
        "feature_dimensions": {"xuannv": 64, "AlphaEarth": 64, "raw": 150},
        "array_sha256": {p.name: sha(p) for p in prep.glob("*.npy")},
        "notes": [
            "official COG decoded before interpolation, then normalized; legacy tensors not used",
            "official 2025 annual; regional December 2025-May 2026 inputs",
            "ESRI 2023 reference, not independent 2026 truth",
            "raw bands incl quality; monthly validity; static HR/availability",
            "all classifier scalers fitted only on sampled support pixels",
        ],
    }
    dump(root / "identity.json", identity)
    print("prepared 320 tiles, 14 tasks, 150-channel raw baseline", flush=True)


def predict(fits, scaler, support, x, ids, head):
    result = np.empty((len(ids), 128, 128, len(fits)), dtype=np.float64)
    for j, i in enumerate(ids):
        tile = np.asarray(x[i]).reshape(-1, x.shape[-1])
        out = result[j].reshape(-1, len(fits))
        chunk = 2048 if head == "svm" else len(tile)
        for start in range(0, len(tile), chunk):
            q = scaler.transform(tile[start : start + chunk].astype("float64"))
            if head == "svm":
                scores = svm_scores(fits, q, support)
            elif head == "rf":
                scores = fits[0].predict_proba(q)[:, 1, None]
            else:
                scores = np.stack([f.decision_function(q) for f in fits], axis=1)
            out[start : start + len(q)] = scores
    return result


def metrics(y, scores, cut):
    valid = y >= 0
    rows = []
    for yy, ss in zip(y, scores):
        v, t, p = yy >= 0, yy == 1, ss >= cut
        rows.append([int((v & t & p).sum()), int((v & ~t & p).sum()), int((v & t & ~p).sum())])
    c = np.array(rows).sum(0)
    return {
        "f1": float(2 * c[0] / max(1, 2 * c[0] + c[1] + c[2])),
        "iou": float(c[0] / max(1, c.sum())),
        "ap": float(average_precision_score(y[valid], scores[valid])),
        "block_counts": rows,
    }


def worker(spec_path, task, head):
    s = read_spec(spec_path)
    root = Path(s["output"])
    ident = json.loads((root / "identity.json").read_text())
    token = sha(root / "identity.json")
    implementation = sha(__file__)
    prep = root / "prepared"
    out = root / "runs" / task / head
    out.mkdir(parents=True, exist_ok=True)
    y = np.load(prep / f"label_{task}.npy")
    split = ident["split"]
    train, val, test = split["train"], split["validation"], split["test"]
    ids = [r["patch_id"] for r in ident["records"]]
    values = {"ridge": [10.0, 1.0, 0.1], "svm": [0.1, 1.0, 10.0], "rf": [200]}[head]
    budgets = s["budgets"] if head == "ridge" else [5]
    ne.set_num_threads(s["threads"])
    with threadpool_limits(limits=s["threads"]):
        for seed in s["seeds"]:
            for budget in budgets:
                try:
                    selected = nested_support(y, ids, train, budget, seed)
                    if any(np.unique(y[ii][y[ii] >= 0]).size != 2 for ii in (val, test)):
                        raise ValueError("validation or test lacks both classes")
                except ValueError as error:
                    dump(
                        out / f"unavailable_{seed}_{budget}.json",
                        {"state": "unavailable", "reason": str(error)},
                    )
                    continue
                sy = y[selected].ravel()
                picks = balanced_positions(sy, 4096, seed)
                for model in ("xuannv", "AlphaEarth", "raw"):
                    key = f"{model}_{seed}_{budget}"
                    dest = out / f"{key}.json"
                    if dest.exists():
                        old = json.loads(dest.read_text())
                        if (
                            old["identity_sha256"] != token
                            or old["spec_sha256"] != sha(spec_path)
                            or old["implementation_sha256"]
                            not in {
                                implementation,
                                # CPU array fusion only; numerical equivalence tested.
                                "edd4166ae25efc65513cd99b3a00154f2bb5c5f7acd19f4318122ae617b3bec0",
                            }
                        ):
                            raise ValueError("resume identity mismatch")
                        continue
                    dump(
                        out / "progress.json",
                        {"model": model, "seed": seed, "budget": budget, "phase": "fit"},
                    )
                    x = np.load(prep / f"{model}.npy", mmap_mode="r")
                    sx = np.asarray(x[selected]).reshape(-1, x.shape[-1])[picks].astype("float64")
                    scaler = StandardScaler().fit(sx)
                    z = scaler.transform(sx)
                    fits, times = [], []
                    for parameter in values:
                        tic = time.monotonic()
                        if head == "ridge":
                            fit = RidgeClassifier(
                                alpha=parameter, class_weight="balanced", solver="cholesky"
                            )
                        elif head == "svm":
                            fit = SVC(
                                C=parameter, gamma="scale", class_weight="balanced", cache_size=1024
                            )
                        else:
                            fit = RandomForestClassifier(
                                n_estimators=200,
                                max_features="sqrt",
                                min_samples_leaf=2,
                                class_weight="balanced",
                                random_state=seed,
                                n_jobs=s["threads"],
                            )
                        fits.append(fit.fit(z, sy[picks]))
                        times.append(time.monotonic() - tic)
                    dump(
                        out / "progress.json",
                        {"model": model, "seed": seed, "budget": budget, "phase": "validation"},
                    )
                    tic = time.monotonic()
                    vs = predict(fits, scaler, z, x, val, head)
                    v = y[val] >= 0
                    aps = [
                        float(average_precision_score(y[val][v], vs[..., j][v]))
                        for j in range(len(fits))
                    ]
                    winner = choose_candidate(aps, values, head)
                    cut = threshold(y[val].ravel(), vs[..., winner].ravel())
                    validation_seconds = time.monotonic() - tic
                    dump(
                        out / "progress.json",
                        {"model": model, "seed": seed, "budget": budget, "phase": "test"},
                    )
                    tic = time.monotonic()
                    ts = predict([fits[winner]], scaler, z, x, test, head)[..., 0]
                    seconds = time.monotonic() - tic
                    m = metrics(y[test], ts, cut)
                    joblib.dump(
                        {
                            "scaler": scaler,
                            "classifier": fits[winner],
                            "support_indices": selected,
                            "pixel_indices": picks,
                            "threshold": cut,
                        },
                        out / f"{key}.joblib",
                    )
                    np.savez_compressed(
                        out / f"{key}_predictions.npz",
                        scores=ts,
                        validation_scores=vs[..., winner],
                        test_indices=test,
                        validation_indices=val,
                        threshold=cut,
                    )
                    row = {
                        "implementation_sha256": implementation,
                        "kernel_backend": "numexpr_float64" if head == "svm" else "sklearn",
                        "head_sha256": sha(out / f"{key}.joblib"),
                        "predictions_sha256": sha(out / f"{key}_predictions.npz"),
                        "task": task,
                        "family": ident["tasks"][task]["family"],
                        "head": head,
                        "model": model,
                        "seed": seed,
                        "budget": budget,
                        "parameters": values,
                        "validation_ap": aps,
                        "selected_parameter": values[winner],
                        "threshold": cut,
                        "support_indices": selected,
                        "support_patch_ids": [ids[i] for i in selected],
                        "sample_positions_sha256": digest(picks),
                        "support_label_sha256": digest(sy),
                        "labeled_pixels": int((sy >= 0).sum()),
                        "fitted_pixels": len(picks),
                        "fit_seconds_candidates": times,
                        "selected_fit_seconds": times[winner],
                        "validation_seconds": validation_seconds,
                        "test_seconds": seconds,
                        "metrics": m,
                        "identity_sha256": token,
                        "spec_sha256": sha(spec_path),
                    }
                    dump(dest, row)
                    print(task, head, key, round(m["f1"], 4), flush=True)
    return task, head


def run(args):
    s = read_spec(args.spec)
    root = Path(s["output"])
    identity = json.loads((root / "identity.json").read_text())
    if identity["spec_sha256"] != sha(args.spec):
        raise ValueError("specification changed after preparation")
    for name, h in identity["array_sha256"].items():
        if sha(root / "prepared" / name) != h:
            raise ValueError("prepared inputs changed")
    split = identity["split"]
    if any(
        set(split[a]) & set(split[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("overlapping spatial splits")
    jobs = [(t, h) for h in ("ridge", "rf", "svm") for t in identity["tasks"]]
    with ProcessPoolExecutor(max_workers=s["workers"]) as pool:
        futures = [pool.submit(worker, str(args.spec), t, h) for t, h in jobs]
        for f in as_completed(futures):
            print("FINISHED", f.result(), flush=True)
    dump(root / "complete.json", {"state": "complete", "jobs": len(jobs)})
