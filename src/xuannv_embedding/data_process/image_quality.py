"""Relative image registration diagnostics and reproducible visual QA panels."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import fftconvolve

from xuannv_embedding.data_process.prepare_observations import atomic_json


def edge_image(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    good = values[valid]
    if good.size < 64 or np.std(good) < 1e-6:
        return np.zeros_like(values, dtype=np.float64)
    values = np.where(valid, values, np.median(good)).astype(np.float64)
    lo, hi = np.percentile(good, [2, 98])
    values = np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)
    smooth = gaussian_filter(values, 0.7)
    gradient = np.hypot(sobel(smooth, axis=0), sobel(smooth, axis=1))
    # Local normalization reduces differences in radiometry and season.
    return gradient / np.maximum(gaussian_filter(gradient, 3), 0.05)


def translation(
    reference: np.ndarray, moving: np.ndarray, valid: np.ndarray, max_shift: int = 8
) -> dict:
    """Masked normalized cross-correlation; unknown is never reported as aligned."""
    if reference.shape != moving.shape or valid.shape != reference.shape:
        raise ValueError("Registration shapes differ")
    if valid.mean() < 0.25:
        return {"status": "insufficient_overlap", "valid_fraction": float(valid.mean())}
    a, b = edge_image(reference, valid), edge_image(moving, valid)
    if min(a[valid].std(), b[valid].std()) < 0.02:
        return {"status": "insufficient_texture"}
    w = valid.astype(np.float64)

    def corr(x, y):
        return fftconvolve(x, y[::-1, ::-1], mode="full")

    count = corr(w, w)
    sa, sb = corr(a * w, w), corr(w, b * w)
    numerator = corr(a * w, b * w) - sa * sb / np.maximum(count, 1)
    denominator = np.sqrt(
        np.maximum(corr(a * a * w, w) - sa * sa / np.maximum(count, 1), 0)
        * np.maximum(corr(w, b * b * w) - sb * sb / np.maximum(count, 1), 0)
    )
    ncc = numerator / np.maximum(denominator, 1e-9)
    center = np.array(reference.shape) - 1
    slices = tuple(slice(c - max_shift, c + max_shift + 1) for c in center)
    window = ncc[slices].copy()
    window[count[slices] < 0.7 * valid.sum()] = -1
    index = np.array(np.unravel_index(np.argmax(window), window.shape))
    delta = index - max_shift
    peak = float(window[tuple(index)])
    competitors = window.copy()
    competitors[max(0, index[0] - 1) : index[0] + 2, max(0, index[1] - 1) : index[1] + 2] = -1
    margin = peak - float(competitors.max())
    confident = peak >= 0.35 and margin >= 0.025 and (np.abs(delta) < max_shift).all()
    return {
        "status": "measured" if confident else "inconclusive",
        "shift_yx_pixels": delta.tolist(),
        "peak_ncc": peak,
        "peak_margin": margin,
        "zero_ncc": float(window[max_shift, max_shift]),
        "valid_fraction": float(valid.mean()),
    }


def read_frame(root: Path, observation: dict, size: int = 128) -> tuple:
    with rasterio.open(root / observation["path"]) as ds:
        values = ds.read(out_shape=(ds.count, size, size), resampling=Resampling.average).astype(
            np.float32
        )
    with rasterio.open(root / observation["mask"]) as ds:
        # GDAL averaging can ignore nodata=0 or round byte masks before output
        # conversion. Reduce the native boolean mask explicitly instead.
        native = ds.read(1) > 0
        height, width = native.shape
        if height % size or width % size:
            raise ValueError("Diagnostic grid must divide native mask dimensions")
        valid = native.reshape(size, height // size, size, width // size).mean(axis=(1, 3)) >= 0.99
    valid &= np.isfinite(values).all(axis=0)
    return values, valid


def evaluate_pair(task: tuple) -> dict:
    candidate, root = task
    observation, reference = candidate["observation"], candidate["reference_s2"]
    result = {
        "path": observation["path"],
        "source": observation["source"],
        "date": observation["date"],
        "parent_key": observation["parent_key"],
        "reference": reference["path"] if reference else None,
    }
    if reference is None or observation["source"] == "s2":
        return {**result, "status": "no_distinct_same_month_reference"}
    try:
        values, valid = read_frame(root, observation)
        ref, refvalid = read_frame(root, reference)
        # NIR for MS/MS; broad visible+NIR for PAN. All diagnostics on 10m grid.
        moving = values[-1] if values.shape[0] > 1 else values[0]
        target = ref[7] if values.shape[0] > 1 else ref[[1, 2, 7]].mean(axis=0)
        mask = valid & refvalid
        measure = translation(target, moving, mask)
        result.update(measure)
        if measure["status"] == "measured":
            tiles = []
            for y, x in [(0, 0), (0, 64), (64, 0), (64, 64)]:
                tiles.append(
                    translation(
                        target[y : y + 64, x : x + 64],
                        moving[y : y + 64, x : x + 64],
                        mask[y : y + 64, x : x + 64],
                    )
                )
            agreed = sum(
                t["status"] == "measured"
                and np.linalg.norm(np.asarray(t["shift_yx_pixels"]) - measure["shift_yx_pixels"])
                <= 1.5
                for t in tiles
            )
            result.update(
                tile_checks=tiles,
                agreeing_tiles=int(agreed),
                offset_m=float(np.linalg.norm(measure["shift_yx_pixels"]) * 10),
            )
            if agreed < 2:
                result["status"] = "inconclusive_tile_disagreement"
            elif result["offset_m"] > 10:
                result["status"] = "large_relative_offset_review"
        return result
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        return {**result, "status": "read_error", "error": str(error)}


def audit_registration(output: Path) -> dict:
    run = json.loads((output / "run.json").read_text())
    root = Path(run["data_root"])
    candidates = json.loads((output / "review-candidates.json").read_text())
    candidates = [c for c in candidates if c["observation"]["source"] != "s2"]
    # Up to 8 per source/year, selected across geographic and seasonal strata.
    buckets = {}
    for c in candidates:
        if c["reference_s2"] is None:
            continue
        key = (c["observation"]["source"], c["year"])
        buckets.setdefault(key, []).append(c)
    selected = []
    for values in buckets.values():
        indices = np.linspace(0, len(values) - 1, min(16, len(values)), dtype=int)
        selected.extend(values[i] for i in indices)
    with ThreadPoolExecutor(max_workers=24) as pool:
        results = list(pool.map(evaluate_pair, ((c, root) for c in selected)))
    from collections import Counter

    report = {
        "method": (
            "masked edge NCC, same-month S2 reference, 10m diagnostic grid, "
            "four-tile consistency"
        ),
        "scope": (
            "relative translation diagnostic; cannot certify absolute or "
            "subpixel 5m accuracy; S2 composite dates are only monthly"
        ),
        "thresholds": {
            "peak_ncc": 0.35,
            "peak_margin": 0.025,
            "agreeing_tiles": 2,
            "review_offset_m": 10,
        },
        "counts": dict(Counter(r["status"] for r in results)),
        "results": results,
    }
    atomic_json(output / "registration-check.json", report)
    return report


def display_rgb(values: np.ndarray, valid: np.ndarray, source: str) -> np.ndarray:
    if source == "s2":
        rgb = values[[2, 1, 0]]
    elif source.startswith("JL1"):
        rgb = values[[4, 3, 2]]
    elif len(values) == 4:
        rgb = values[[2, 1, 0]]
    else:
        rgb = np.repeat(values[:1], 3, axis=0)
    out = []
    for band in rgb:
        good = band[valid]
        lo, hi = np.percentile(good, [2, 98]) if good.size else (0, 1)
        out.append(np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1))
    return np.moveaxis(np.stack(out), 0, -1)


def registration_panels(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import shift

    root = Path(json.loads((output / "run.json").read_text())["data_root"])
    candidates = {
        c["observation"]["path"]: c
        for c in json.loads((output / "review-candidates.json").read_text())
    }
    results = json.loads((output / "registration-check.json").read_text())["results"]
    selected = []
    for source in sorted({r["source"] for r in results}):
        bad = [
            r
            for r in results
            if r["source"] == source and r["status"] == "large_relative_offset_review"
        ]
        good = [r for r in results if r["source"] == source and r["status"] == "measured"]
        if bad:
            selected.append(max(bad, key=lambda r: r["peak_ncc"]))
        if good:
            selected.append(max(good, key=lambda r: r["peak_ncc"]))
    out = output / "review"
    out.mkdir(exist_ok=True)
    atomic_json(out / "registration-selection.json", selected)
    for start in range(0, len(selected), 4):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12), squeeze=False)
        for row, r in enumerate(selected[start : start + 4]):
            c = candidates[r["path"]]
            o = c["observation"]
            ref = c["reference_s2"]
            a, av = read_frame(root, ref)
            b, bv = read_frame(root, o)
            gray_a = a[7] if len(b) > 1 else a[[1, 2, 7]].mean(axis=0)
            gray_b = b[-1] if len(b) > 1 else b[0]
            ea = edge_image(gray_a, av)
            eb = edge_image(gray_b, bv)

            def composite(x, y):
                return np.stack(
                    [np.clip(x / 3, 0, 1), np.clip(y / 3, 0, 1), np.clip(y / 3, 0, 1)], axis=-1
                )

            pictures = [
                display_rgb(a, av, "s2"),
                display_rgb(b, bv, o["source"]),
                composite(ea, eb),
                composite(ea, shift(eb, r["shift_yx_pixels"], mode="constant", order=1)),
            ]
            for col, picture in enumerate(pictures):
                axes[row, col].imshow(picture)
            axes[row, 0].set_ylabel(
                f"R{start+row:02d} {r['source'].split('_')[0]}\n"
                f"{r['offset_m']:.0f}m NCC {r['peak_ncc']:.2f}",
                fontsize=9,
            )
        for row in axes:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])
        for ax, title in zip(
            axes[0], ["S2 RGB", "Highres RGB/PAN", "Edges before", "Edges after diagnostic shift"]
        ):
            ax.set_title(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"registration-{start//4:02d}.png", dpi=120)
        plt.close(fig)


def cloud_panels(output: Path) -> None:
    import hashlib
    import sqlite3

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(json.loads((output / "run.json").read_text())["data_root"])
    candidates = json.loads((output / "review-candidates.json").read_text())
    buckets = {}
    for c in candidates:
        o = c["observation"]
        s = o["source"]
        if s != "s2" and not s.startswith("JL1"):
            continue
        fraction = o["valid_fraction"]
        regime = "clear" if fraction >= 0.9 else "mixed" if fraction >= 0.2 else "low_clear"
        group = (s, regime)
        buckets.setdefault(group, []).append(o)
    selected = []
    for group, items in sorted(buckets.items()):
        ranked = sorted(items, key=lambda o: hashlib.sha256(o["path"].encode()).hexdigest())
        selected.extend({**o, "review_group": list(group)} for o in ranked[:2])
    pan_buckets = {}
    for rank in range(8):
        path = output / f"results-{rank}.sqlite"
        if not path.exists():
            continue
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            for _, payload in db.execute("SELECT path,payload FROM results"):
                o = json.loads(payload)
                if o["status"] == "read_error":
                    continue
                fraction = o["valid_fraction"]
                regime = "clear" if fraction >= 0.9 else "mixed" if fraction >= 0.2 else "low_clear"
                key = (o["source"], regime)
                h = hashlib.sha256(o["path"].encode()).hexdigest()
                if key not in pan_buckets or h < pan_buckets[key][0]:
                    pan_buckets[key] = (h, o)
    selected.extend({**o, "review_group": list(key)} for key, (_, o) in sorted(pan_buckets.items()))
    out = output / "review"
    out.mkdir(exist_ok=True)
    atomic_json(out / "cloud-selection.json", selected)
    colors = np.array([[0, 0, 0], [1, 0, 0], [1, 0.8, 0], [0.2, 0.4, 1]], dtype=float)
    for start in range(0, len(selected), 4):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12), squeeze=False)
        for row, o in enumerate(selected[start : start + 4]):
            pan = "PAN" in o["source"]
            path = o["mux"]["path"] if pan else o["path"]
            with rasterio.open(root / path) as ds:
                values = ds.read()
                valid = (ds.read_masks() > 0).all(axis=0)
            with rasterio.open(root / o["mask"]) as ds:
                qa = ds.read()
            rgb = display_rgb(values, valid, "mux" if pan else o["source"])
            classes = qa[1]
            overlay = rgb.copy()
            bad = (classes > 0) & (classes < 4)
            overlay[bad] = 0.45 * rgb[bad] + 0.55 * colors[classes[bad]]
            axes[row, 0].imshow(rgb)
            axes[row, 1].imshow(overlay)
            nir = values[3 if pan else 7 if o["source"] == "s2" else 5]
            axes[row, 2].imshow(display_rgb(nir[None], valid, "pan"))
            if pan:
                with rasterio.open(root / o["path"]) as ds:
                    pv = ds.read()
                    pm = ds.read_masks(1) > 0
                axes[row, 3].imshow(display_rgb(pv, pm, "pan"))
            else:
                axes[row, 3].imshow(np.where((qa[0] > 0)[..., None], rgb, 0))
            axes[row, 0].set_ylabel(
                f"C{start+row:02d} {o['source'].split('_')[0]}\nclear {o['valid_fraction']:.2f}",
                fontsize=9,
            )
        for row in axes:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])
        for ax, title in zip(
            axes[0],
            [
                "Optical RGB",
                "Red cloud/yellow thin/blue shadow",
                "NIR",
                "Clear RGB or original PAN",
            ],
        ):
            ax.set_title(title, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"cloud-{start//4:02d}.png", dpi=120)
        plt.close(fig)


def full_registration(output: Path) -> None:
    """Expand diagnostics after systematic offsets are found in the stratified sample."""
    import sqlite3
    from collections import Counter

    run = json.loads((output / "run.json").read_text())
    root = Path(run["data_root"])
    result_file = output / "registration-full.sqlite"
    db = sqlite3.connect(result_file)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-524288")
    db.execute("CREATE TABLE IF NOT EXISTS results(path TEXT PRIMARY KEY,payload TEXT)")
    done = {row[0] for row in db.execute("SELECT path FROM results")}
    counts = Counter()
    with ProcessPoolExecutor(max_workers=128) as pool:
        for name in sorted(run["input_manifests"]):
            jobs = []
            with (Path(run["input"]) / name).open() as handle:
                for line in handle:
                    r = json.loads(line)
                    obs = r["provenance"]["observations"]
                    s2 = {o["date"][:7]: o for o in obs.get("s2", [])}
                    for source, items in obs.items():
                        if not (source.startswith("JL1") or "PAN" in source):
                            continue
                        for o in items:
                            if o["path"] not in done:
                                jobs.append(
                                    (
                                        {"observation": o, "reference_s2": s2.get(o["date"][:7])},
                                        root,
                                    )
                                )
            for index, result in enumerate(pool.map(evaluate_pair, jobs, chunksize=16)):
                db.execute("INSERT INTO results VALUES (?,?)", (result["path"], json.dumps(result)))
                counts[result["status"]] += 1
                if index % 1000 == 0:
                    db.commit()
                    print(
                        json.dumps(
                            {
                                "registration_manifest": name,
                                "processed": sum(counts.values()),
                                "counts": dict(counts),
                            }
                        ),
                        flush=True,
                    )
            db.commit()
    counts = Counter()
    for row in db.execute("SELECT payload FROM results"):
        counts[json.loads(row[0])["status"]] += 1
    atomic_json(
        output / "registration-full-summary.json",
        {
            "status": "complete",
            "counts": dict(counts),
            "scope": (
                "all selected highres; same-month reference required, otherwise "
                "explicitly unresolved; conservative quarantine of confident >10m offsets"
            ),
            "method": "same method and thresholds as stratified registration-check.json",
        },
    )
    db.close()
