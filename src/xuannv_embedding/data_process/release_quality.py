"""Final annual-data QA: paired PAN cloud masks, image checks and split audits.

Upstream scripts are not a release requirement. All decisions refer to the supplied
rasters; optical predictions and relative alignment are not ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from xuannv_embedding.data_process.annual_quality import add_moments, relative_file
from xuannv_embedding.data_process.observation_raster import parent_geometry, sha256_file
from xuannv_embedding.data_process.prepare_observations import atomic_json

PRIORITY = {"train": 0, "validation": 1, "test": 2}
VERSION = "annual-observed-quality-v6"


def original_product(observation: dict) -> str | None:
    """Product lineage, without pretending monthly composites retain scene IDs."""
    if "PAN" in observation["source"]:
        return observation.get("scene_id")
    if observation["source"].startswith("JL1"):
        match = re.fullmatch(r"\d{8}_(JL1.+_L3B)_5m_[a-f0-9]+\.tif", Path(observation["path"]).name)
        if match:
            return match[1]
    return None


def mux_identity(record: dict) -> str | None:
    name = Path(record["archive_member"]).name
    match = re.search(
        r"(GF(?:1[B-D]?|6)_PMS\d?_E[-\d.]+_N[-\d.]+_\d{8}_L\w+?)-(?:MUX|MSS)\d?(?:_ORTHO.*)?\.tif$",
        name,
    )
    return match[1] if match else None


def prepare(base: Path, output: Path, catalog: Path, model_dir: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    root = Path(json.loads((base / "summary.json").read_text())["data_root"])
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output outside data_root")
    output.mkdir(parents=True)
    # Build the new index in RAM: remote filesystem random SQLite writes dominate
    # runtime. Publish complete databases only after successful indexing.
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE observations(path TEXT PRIMARY KEY,source "
        "TEXT,split TEXT,priority INTEGER,parent TEXT,year INTEGER,sha TEXT,product TEXT)"
    )
    db.execute("CREATE TABLE selected_highres(path TEXT PRIMARY KEY,payload TEXT)")
    db.execute("CREATE TABLE parents(parent TEXT PRIMARY KEY,split TEXT)")
    tasks = sqlite3.connect(":memory:")
    tasks.execute("CREATE TABLE tasks(path TEXT PRIMARY KEY,shard INTEGER,payload TEXT)")
    mux = {}
    with (catalog / "observations.jsonl").open() as handle:
        for line in handle:
            if '"MUX"' not in line and '"MSS"' not in line:
                continue
            item = json.loads(line)
            if item.get("channels") != 4 or not item.get("materialized_path"):
                continue
            scene = mux_identity(item)
            if scene and item.get("grid_matches_parent") and not item.get("issues"):
                key = (item["parent_key"], scene)
                value = {
                    "path": (catalog / item["materialized_path"]).relative_to(root).as_posix(),
                    "sha256": item["sha256"],
                    "scene_id": scene,
                }
                mux.setdefault(key, []).append(value)
    print(json.dumps({"mux_keys": len(mux)}), flush=True)
    counts, fingerprints = Counter(), {}
    review = {}
    for manifest in sorted(base.glob("*.manifest.jsonl")):
        fingerprints[manifest.name] = sha256_file(manifest)
        with manifest.open() as handle:
            for line in handle:
                record = json.loads(line)
                split = record["provenance"]["split"]
                year = record["provenance"]["year"]
                parent = record["grid"]["parent_key"]
                prev = db.execute("SELECT split FROM parents WHERE parent=?", (parent,)).fetchone()
                if prev and prev[0] != split:
                    raise ValueError("Parent crosses splits")
                db.execute("INSERT OR IGNORE INTO parents VALUES (?,?)", (parent, split))
                obs = record["provenance"]["observations"]
                for source, items in obs.items():
                    for item in items:
                        path = item["path"]
                        db.execute(
                            "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?)",
                            (
                                path,
                                source,
                                split,
                                PRIORITY[split],
                                parent,
                                year,
                                item["sha256"],
                                original_product(item),
                            ),
                        )
                        if source.startswith("JL1") or "PAN" in source:
                            db.execute(
                                "INSERT INTO selected_highres VALUES (?,?)",
                                (path, json.dumps(item)),
                            )
                        if "PAN" in source:
                            matches = mux.get((parent, item["scene_id"]), [])
                            # Multiple names with identical bytes are harmless aliases.
                            unique = {m["sha256"]: m for m in matches}
                            if len(unique) == 1:
                                task = {**item, "mux": next(iter(unique.values()))}
                                rank = int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) % 8
                                tasks.execute(
                                    "INSERT INTO tasks VALUES (?,?,?)",
                                    (path, rank, json.dumps(task)),
                                )
                                counts["pan_with_exact_mux"] += 1
                            else:
                                counts["pan_no_unambiguous_mux"] += 1
                        # Geographic/season/source stratification, independent of labels.
                        if source == "s2" or source.startswith("JL1") or "PAN" in source:
                            group = (
                                source,
                                year,
                                parent.split(":")[0],
                                (int(item["date"][5:7]) - 1) // 3,
                            )
                            rank = hashlib.sha256(path.encode()).hexdigest()
                            prior = review.get(group)
                            if prior is None or rank < prior[0]:
                                s2 = [
                                    v
                                    for v in obs.get("s2", [])
                                    if v["date"][:7] == item["date"][:7]
                                ]
                                review[group] = (
                                    rank,
                                    {
                                        "observation": item,
                                        "reference_s2": s2[0] if s2 else None,
                                        "split": split,
                                        "year": year,
                                    },
                                )
        db.commit()
        tasks.commit()
        print(json.dumps({"indexed": manifest.name, "counts": dict(counts)}), flush=True)
    db.execute("CREATE INDEX sha_lookup ON observations(sha)")
    db.execute("CREATE INDEX product_lookup ON observations(product)")
    db.execute(
        "CREATE TABLE owners AS SELECT sha,max(priority) AS priority FROM observations GROUP BY sha"
    )
    db.execute("CREATE UNIQUE INDEX owner_sha ON owners(sha)")
    db.execute(
        "CREATE TABLE product_owners AS SELECT product,max(priority) AS "
        "priority FROM observations WHERE product IS NOT NULL GROUP BY product"
    )
    db.execute("CREATE UNIQUE INDEX owner_product ON product_owners(product)")
    db.commit()
    tasks.execute("CREATE INDEX shard_lookup ON tasks(shard)")
    tasks.commit()
    provenance = json.loads((model_dir / "provenance.json").read_text())
    for weight in provenance["weights"]:
        if sha256_file(model_dir / weight["name"]) != weight["sha256"]:
            raise ValueError("Cloud weight digest changed")
    contract = {
        "version": VERSION,
        "input": str(base),
        "data_root": str(root),
        "workers": 8,
        "model_dir": str(model_dir),
        "model": provenance,
        "counts": dict(counts),
        "input_manifests": fingerprints,
        "status": "indexed",
        "training_ready": False,
        "pan_spectral_policy": (
            "MUX stored B,G,R,NIR assumed per product guide; R,G,NIR indices "
            "2,1,3; inspected before release"
        ),
        "mask_policy": (
            "same-scene MUX OCM clear; 8m guard around cloud/shadow/invalid; "
            "PAN numeric mask; >=20% clear area"
        ),
        "limitations": [
            "Small 1280m crops limit cloud context",
            "No upstream-script requirement",
            "No inferred physical calibration of PAN",
        ],
    }
    atomic_json(output / "run.json", contract)
    atomic_json(output / "review-candidates.json", [v[1] for _, v in sorted(review.items())])
    shutil.copy2(Path(__file__), output / "release_quality_source.py")
    for connection, name in ((db, "audit.sqlite"), (tasks, "tasks.sqlite")):
        with sqlite3.connect(output / (name + ".partial")) as destination:
            connection.backup(destination)
        (output / (name + ".partial")).replace(output / name)
    db.close()
    tasks.close()


def read_mux_item(item: tuple, root: Path) -> tuple:
    from omnicloudmask.model_utils import channel_norm

    path, payload = item
    observation = json.loads(payload)
    try:
        spec = observation["mux"]
        blob = relative_file(root, spec["path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != spec["sha256"]:
            raise ValueError("MUX payload changed")
        with MemoryFile(blob) as memory, memory.open() as raster:
            epsg, bounds = parent_geometry(observation["parent_key"])
            if (
                raster.count != 4
                or raster.shape != (160, 160)
                or raster.crs.to_epsg() != epsg
                or not np.allclose(raster.bounds, bounds, atol=0.01, rtol=0)
            ):
                raise ValueError("MUX native grid mismatch")
            values = raster.read()
            valid = (
                (raster.read_masks() > 0).all(axis=0)
                & np.isfinite(values).all(axis=0)
                & (values > 0).all(axis=0)
            )
            if np.issubdtype(values.dtype, np.integer):
                valid &= ~(values == np.iinfo(values.dtype).max).any(axis=0)
            profile = raster.profile.copy()
        bands = values[[2, 1, 3]].astype(np.float32)
        bands[:, ~valid] = 0
        normalized = channel_norm(bands, 0)
        return path, observation, (values, valid, profile, normalized), None
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        return path, observation, None, f"{type(error).__name__}: {error}"


def pan_clear_mask(classes: np.ndarray, pan_valid: np.ndarray) -> np.ndarray:
    """Same aligned extent, 8m MUX to 2m PAN; retain uncertainty at cloud edges."""
    from scipy.ndimage import binary_dilation

    if classes.shape != (160, 160) or pan_valid.shape != (640, 640):
        raise ValueError("Expected aligned native MUX/PAN grids")
    blocked = binary_dilation(classes != 0, iterations=1)
    return pan_valid & np.repeat(np.repeat(~blocked, 4, axis=0), 4, axis=1)


def apply_pan(task: tuple) -> dict:
    from xuannv_embedding.data_process.image_quality import translation

    original, prediction, root, output = task
    result = {**original, "status": "excluded"}
    try:
        if prediction["status"] == "read_error":
            return {**result, "exclusion_reason": "mux_read_error"}
        blob = relative_file(root, original["path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != original["sha256"]:
            raise ValueError("PAN bytes changed")
        with MemoryFile(blob) as memory, memory.open() as ds:
            values = ds.read(1)
            valid = (ds.read_masks(1) > 0) & (values > 0) & (values < 65535)
            profile = ds.profile.copy()
            pan_geometry = (ds.crs, ds.bounds)
        with rasterio.open(relative_file(root, original["mask"])) as ds:
            if (
                ds.shape != values.shape
                or ds.crs != pan_geometry[0]
                or not np.allclose(ds.bounds, pan_geometry[1], atol=0.01, rtol=0)
            ):
                raise ValueError("PAN input mask grid mismatch")
            valid &= ds.read(1) > 0
        with rasterio.open(relative_file(root, prediction["mask"])) as ds:
            if ds.crs != pan_geometry[0] or not np.allclose(
                ds.bounds, pan_geometry[1], atol=0.01, rtol=0
            ):
                raise ValueError("MUX/PAN bounds differ")
            classes = ds.read(2)
        clear = pan_clear_mask(classes, valid)
        pixel_sha = hashlib.sha256(values.tobytes()).hexdigest()
        if clear.mean() < 0.2:
            return {
                **result,
                "exclusion_reason": "pan_clear_area_below_20_percent",
                "clear_fraction": float(clear.mean()),
                "pixel_sha256": pixel_sha,
            }
        mux_blob = relative_file(root, prediction["mux"]["path"]).read_bytes()
        if hashlib.sha256(mux_blob).hexdigest() != prediction["mux"]["sha256"]:
            raise ValueError("MUX changed after inference")
        with MemoryFile(mux_blob) as memory, memory.open() as ds:
            mux = ds.read().astype(np.float32)
        pan8 = values.reshape(160, 4, 160, 4).mean(axis=(1, 3))
        clear8 = clear.reshape(160, 4, 160, 4).all(axis=(1, 3))
        registration = translation(mux.mean(axis=0), pan8, clear8, max_shift=5)
        if (
            registration["status"] == "measured"
            and np.linalg.norm(registration["shift_yx_pixels"]) > 1
        ):
            return {
                **result,
                "exclusion_reason": "pan_mux_large_offset",
                "pan_mux_registration": registration,
                "pixel_sha256": pixel_sha,
            }
        accepted = values[clear].astype(np.float64)
        if accepted.var() <= 0:
            return {**result, "exclusion_reason": "constant_clear_pan"}
        destination = (
            output
            / "pan_masks"
            / pixel_sha[:2]
            / (hashlib.sha256(original["path"].encode()).hexdigest() + ".tif")
        )
        destination.parent.mkdir(exist_ok=True, parents=True)
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
        with MemoryFile() as memory:
            with memory.open(**profile) as ds:
                ds.write(clear.astype(np.uint8), 1)
                ds.set_band_description(
                    1, "PAN_valid_and_same_scene_MUX_predicted_clear_with_8m_guard"
                )
            payload = memory.read()
        temporary = destination.with_suffix(".partial.tif")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        return {
            **original,
            "status": "accepted",
            "mask": destination.relative_to(root).as_posix(),
            "mask_sha256": hashlib.sha256(payload).hexdigest(),
            "numeric_mask": original["mask"],
            "pixel_sha256": pixel_sha,
            "band_counts": [int(clear.sum())],
            "band_mean": [float(accepted.mean())],
            "band_variance": [float(accepted.var())],
            "valid_fraction": float(clear.mean()),
            "cloud_shadow_QA": "same_scene_MUX_OCM_v4_with_8m_guard",
            "cloud_class_counts_mux": prediction.get("cloud_class_counts"),
            "mux": prediction["mux"],
            "mux_prediction_mask": prediction["mask"],
            "pan_mux_registration": registration,
        }
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        return {**result, "exclusion_reason": "pan_apply_error", "error": str(error)}


def finalize_pan(output: Path) -> dict:
    run = json.loads((output / "run.json").read_text())
    root = Path(run["data_root"])
    original_db = sqlite3.connect(f"file:{output/'audit.sqlite'}?mode=ro", uri=True)
    originals = dict(original_db.execute("SELECT path,payload FROM selected_highres"))
    results = sqlite3.connect(output / "pan-final.sqlite")
    results.execute("PRAGMA journal_mode=WAL")
    results.execute("PRAGMA synchronous=NORMAL")
    results.execute("PRAGMA cache_size=-524288")
    results.execute("CREATE TABLE IF NOT EXISTS results(path TEXT PRIMARY KEY,payload TEXT)")
    done = {r[0] for r in results.execute("SELECT path FROM results")}
    jobs = []
    for rank in range(8):
        progress = json.loads((output / f"progress-{rank}.json").read_text())
        if progress["status"] != "complete":
            raise ValueError("Cloud worker incomplete")
        with sqlite3.connect(f"file:{output/f'results-{rank}.sqlite'}?mode=ro", uri=True) as db:
            for path, payload in db.execute("SELECT path,payload FROM results"):
                if path in done:
                    continue
                original = json.loads(originals[path])
                jobs.append((original, json.loads(payload), root, output))
    counts = Counter()
    with ProcessPoolExecutor(max_workers=128) as pool:
        for index, result in enumerate(pool.map(apply_pan, jobs, chunksize=16)):
            results.execute(
                "INSERT INTO results VALUES (?,?)", (result["path"], json.dumps(result))
            )
            counts[result.get("exclusion_reason", result["status"])] += 1
            if index % 1000 == 0:
                results.commit()
                print(json.dumps({"pan_applied": index + 1, "counts": dict(counts)}), flush=True)
    results.commit()
    counts = Counter()
    reg = Counter()
    for row in results.execute("SELECT payload FROM results"):
        item = json.loads(row[0])
        counts[item.get("exclusion_reason", item["status"])] += 1
        if "pan_mux_registration" in item:
            reg[item["pan_mux_registration"]["status"]] += 1
    if sum(counts.values()) != run["counts"]["pan_with_exact_mux"]:
        raise ValueError("PAN result count mismatch")
    report = {
        "counts": dict(counts),
        "pan_mux_registration_counts": dict(reg),
        "unmatched_pan_excluded": run["counts"]["pan_no_unambiguous_mux"],
    }
    atomic_json(output / "pan-screening-summary.json", report)
    results.close()
    original_db.close()
    return report


def read_lineage(task: tuple) -> dict:
    observation, root = task
    try:
        blob = relative_file(root, observation["path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != observation["sha256"]:
            raise ValueError("JL1 bytes changed")
        with MemoryFile(blob) as memory, memory.open() as ds:
            values = ds.read()
            tags = ds.tags()
            product = tags.get("source_product")
            if not product or product != original_product(observation):
                raise ValueError("JL1 product identity mismatch")
            if not ds.descriptions or not all(ds.descriptions):
                raise ValueError("Missing spectral descriptions")
            return {
                "path": observation["path"],
                "source": observation["source"],
                "status": "verified",
                "product": product,
                "pixel_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "units_declared": tags.get("units"),
                "pipeline": tags.get("pipeline_version"),
                "resampling": tags.get("resampling"),
                "geometry_method": tags.get("source_geometry_method"),
                "height_mode": tags.get("source_height_mode"),
                "scales": list(ds.scales),
                "band_names": list(ds.descriptions),
            }
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        return {"path": observation["path"], "status": "excluded", "error": str(error)}


def audit_lineage(output: Path) -> dict:
    run = json.loads((output / "run.json").read_text())
    root = Path(run["data_root"])
    source = sqlite3.connect(f"file:{output/'audit.sqlite'}?mode=ro", uri=True)
    db = sqlite3.connect(output / "lineage.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS results(path TEXT PRIMARY KEY,payload TEXT)")
    done = {r[0] for r in db.execute("SELECT path FROM results")}
    jobs = [
        (json.loads(payload), root)
        for path, payload in source.execute(
            "SELECT path,payload FROM selected_highres WHERE path LIKE '%JL1%'"
        )
        if path not in done
    ]
    with ThreadPoolExecutor(max_workers=48) as pool:
        for index, result in enumerate(pool.map(read_lineage, jobs)):
            db.execute("INSERT INTO results VALUES (?,?)", (result["path"], json.dumps(result)))
            if index % 2000 == 0:
                db.commit()
                print(json.dumps({"jl1_lineage_read": index + 1}), flush=True)
    db.commit()
    counts = Counter()
    units = Counter()
    pipeline = Counter()
    resampling = Counter()
    for row in db.execute("SELECT payload FROM results"):
        r = json.loads(row[0])
        counts[r["status"]] += 1
        if r["status"] == "verified":
            units[str(r["units_declared"])] += 1
            pipeline[str(r["pipeline"])] += 1
            resampling[str(r["resampling"])] += 1
    expected = source.execute(
        "SELECT count(*) FROM selected_highres WHERE path LIKE '%JL1%'"
    ).fetchone()[0]
    if sum(counts.values()) != expected:
        raise ValueError("Incomplete lineage audit")
    report = {
        "counts": dict(counts),
        "declared_units": dict(units),
        "pipeline_versions": dict(pipeline),
        "resampling": dict(resampling),
        "scope": (
            "all selected JL1 payload hashes and embedded product IDs; "
            "calibration accuracy not certified"
        ),
    }
    atomic_json(output / "lineage-summary.json", report)
    db.close()
    source.close()
    return report


def merge_release(output: Path) -> dict:
    """Freeze screened observations and refit moments strictly on retained train data."""
    from xuannv_embedding.utils.manifest import ManifestRecord

    run = json.loads((output / "run.json").read_text())
    base = Path(run["input"])
    root = Path(run["data_root"])
    review = json.loads((output / "visual-review.json").read_text())
    if not review.get("completed"):
        raise ValueError("Visual review not complete")
    for name, digest in run["input_manifests"].items():
        if sha256_file(base / name) != digest:
            raise ValueError("Input manifest changed")
    db = sqlite3.connect(":memory:")
    with sqlite3.connect(f"file:{output/'audit.sqlite'}?mode=ro", uri=True) as stored:
        stored.backup(db)
    db.execute("PRAGMA temp_store=MEMORY")
    file_owners = dict(
        db.execute(
            "SELECT sha,max(priority) FROM observations GROUP BY sha HAVING "
            "min(priority)!=max(priority)"
        )
    )
    product_owners = dict(
        db.execute(
            "SELECT product,max(priority) FROM observations WHERE product IS "
            "NOT NULL GROUP BY product"
        )
    )
    highres_identity = {
        p: (s, int(priority))
        for p, s, priority in db.execute(
            "SELECT path,source,priority FROM observations WHERE source NOT "
            "IN ('s1','s2','landsat')"
        )
    }
    pan, lineage = {}, {}
    for name, destination in (("pan-final.sqlite", pan), ("lineage.sqlite", lineage)):
        with sqlite3.connect(f"file:{output/name}?mode=ro", uri=True) as results:
            destination.update(
                (p, json.loads(v)) for p, v in results.execute("SELECT path,payload FROM results")
            )
    pixel_owners = {}
    for path, item in {**pan, **lineage}.items():
        if item.get("pixel_sha256"):
            source, priority = highres_identity[path]
            key = (source, item["pixel_sha256"])
            pixel_owners[key] = max(priority, pixel_owners.get(key, -1))
    registration_summary = json.loads((output / "registration-full-summary.json").read_text())
    if registration_summary["status"] != "complete":
        raise ValueError("Full registration audit incomplete")
    with sqlite3.connect(f"file:{output/'registration-full.sqlite'}?mode=ro", uri=True) as reg_db:
        registration = dict(
            (p, json.loads(v)) for p, v in reg_db.execute("SELECT path,payload FROM results")
        )
    if set(registration) != set(highres_identity):
        raise ValueError("Registration audit observation coverage differs")
    bad_paths = set(review.get("exclude_paths", []))
    bad_paths.update(
        r["path"]
        for r in registration.values()
        if r["status"] in {"large_relative_offset_review", "read_error"}
    )
    bad_products = set(review.get("exclude_products", []))
    states = defaultdict(dict)
    counts = Counter()
    manifests = {}
    supports = {}
    source_counts = Counter()
    retained_products = {}
    retained_files = {}
    retained_pixels = {}
    parents = {}
    exclusions = output / "excluded-observations.jsonl"
    with exclusions.open("w") as rejected:
        for name in sorted(run["input_manifests"]):
            target = output / name
            temporary = target.with_suffix(".partial.jsonl")
            count = 0
            support = Counter()
            digest = hashlib.sha256()
            with (base / name).open() as handle, temporary.open("wb") as destination:
                for line in handle:
                    record = json.loads(line)
                    split = record["provenance"]["split"]
                    priority = PRIORITY[split]
                    observations = record["provenance"]["observations"]
                    for source, items in observations.items():
                        retained = []
                        seen = set()
                        for item in items:
                            reason = None
                            path = item["path"]
                            product = original_product(item)
                            if path in bad_paths or product in bad_products:
                                reason = "image_review_quarantine"
                            elif source == "landsat":
                                reason = "landsat_cloud_qa_unavailable"
                            elif (
                                path in registration and registration[path]["status"] != "measured"
                            ):
                                reason = "highres_relative_registration_unresolved"
                            elif file_owners.get(item["sha256"], priority) != priority:
                                reason = "identical_payload_cross_split"
                            elif product and product_owners[product] != priority:
                                reason = "original_product_cross_split"
                            elif "PAN" in source:
                                updated = pan.get(path)
                                if updated is None:
                                    reason = "pan_no_unambiguous_same_scene_mux"
                                elif updated["status"] != "accepted":
                                    reason = updated.get("exclusion_reason", "pan_rejected")
                                elif updated["pan_mux_registration"]["status"] != "measured":
                                    reason = "pan_mux_alignment_unresolved"
                                else:
                                    item = updated
                            elif source.startswith("JL1"):
                                meta = lineage.get(path)
                                if meta is None or meta["status"] != "verified":
                                    reason = "jl1_identity_or_payload_error"
                                elif item["valid_fraction"] < 0.2:
                                    reason = "ms_clear_area_below_20_percent"
                                else:
                                    item = {
                                        **item,
                                        "original_product": meta["product"],
                                        "pixel_sha256": meta["pixel_sha256"],
                                        "production_metadata": {
                                            k: v
                                            for k, v in meta.items()
                                            if k not in {"path", "pixel_sha256", "status", "source"}
                                        },
                                    }
                            if not reason and item.get("pixel_sha256"):
                                key = (source, item["pixel_sha256"])
                                if pixel_owners[key] != priority:
                                    reason = "identical_pixels_cross_split"
                                elif key in seen:
                                    reason = "repeated_highres_pixels_within_annual_parent"
                                seen.add(key)
                            if reason:
                                counts[reason] += 1
                                rejected.write(
                                    json.dumps(
                                        {
                                            "path": path,
                                            "source": source,
                                            "split": split,
                                            "reason": reason,
                                        }
                                    )
                                    + "\n"
                                )
                            else:
                                if path in registration:
                                    item = {**item, "relative_registration": registration[path]}
                                retained.append(item)
                        observations[source] = retained
                        record["sources"][source] = [o["path"] for o in retained]
                    # Keep unavailable optical QA out of both input and target paths.
                    observations.pop("landsat", None)
                    record["sources"].pop("landsat", None)
                    low = [
                        o
                        for s, items in observations.items()
                        if s in {"s1", "s2", "landsat"}
                        for o in items
                        if o["valid_fraction"] >= 0.2
                    ]
                    months = {o["date"][:7] for o in low}
                    quarters = {(int(o["date"][5:7]) - 1) // 3 for o in low}
                    if len(months) < 6 or len(quarters) < 3:
                        counts["annual_parent_insufficient_temporal_coverage"] += 1
                        rejected.write(
                            json.dumps(
                                {
                                    "patch_id": record["patch_id"],
                                    "year": record["provenance"]["year"],
                                    "reason": "annual_parent_insufficient_temporal_coverage",
                                }
                            )
                            + "\n"
                        )
                        continue
                    has_ms = any(items for s, items in observations.items() if s.startswith("JL1"))
                    has_pan = any(items for s, items in observations.items() if "PAN" in s)
                    record["quality"].update(
                        ms5m_supported=has_ms,
                        pan2m_supported=has_pan,
                        highres_supported=has_ms or has_pan,
                        cloud_shadow="S2_JL1_OCM_and_PAN_same_scene_MUX_OCM; Landsat_withheld",
                        pan_cloud_shadow="same_scene_MUX_OCM_v4_8m_guard",
                        relative_registration="all_highres_diagnostic_and_confident_offset_quarantine",
                        upstream_scripts_required=False,
                    )
                    support[
                        (
                            "both"
                            if has_ms and has_pan
                            else "ms_only" if has_ms else "pan_only" if has_pan else "lowres_only"
                        )
                    ] += 1
                    for source, items in observations.items():
                        source_counts[source] += len(items)
                        for item in items:
                            if split == "train":
                                add_moments(states[source], item)
                            product = original_product(item)
                            for key, groups in (
                                (item["sha256"], retained_files),
                                (product, retained_products),
                                (
                                    (
                                        (source, item["pixel_sha256"])
                                        if item.get("pixel_sha256")
                                        else None
                                    ),
                                    retained_pixels,
                                ),
                            ):
                                if key is not None:
                                    if key in groups and groups[key] != split:
                                        raise ValueError("Retained cross-split duplicate")
                                    groups[key] = split
                    parent = record["grid"]["parent_key"]
                    if parent in parents and parents[parent] != split:
                        raise ValueError("Retained parent split conflict")
                    parents[parent] = split
                    ManifestRecord.from_dict(record, f"{name}:{count}")
                    payload = (
                        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)
                        + "\n"
                    ).encode()
                    destination.write(payload)
                    digest.update(payload)
                    count += 1
            temporary.replace(target)
            meta = json.loads((base / (name + ".meta.json")).read_text())
            meta.update(record_count=count, sha256=digest.hexdigest(), generator_version=VERSION)
            atomic_json(output / (name + ".meta.json"), meta)
            manifests[name] = count
            supports[name] = dict(support)
            print(
                json.dumps({"merged": name, "records": count, "support": dict(support)}), flush=True
            )
    schemas = json.loads((base / "sources.json").read_text())
    schemas.pop("landsat", None)
    # A high-resolution source that has no surviving train observation cannot
    # have a train-only normalization statistic. Remove it consistently from
    # every split instead of silently fitting statistics on validation/test.
    missing_sources = set(schemas) - set(states)
    if missing_sources:
        source_counts = Counter()
        supports = {}
        for name in sorted(manifests):
            target = output / name
            temporary = target.with_suffix(".source-filter.partial.jsonl")
            digest = hashlib.sha256()
            support = Counter()
            with target.open() as source_handle, temporary.open("wb") as destination:
                for line in source_handle:
                    record = json.loads(line)
                    observations = record["provenance"]["observations"]
                    for source in missing_sources:
                        observations.pop(source, None)
                        record["sources"].pop(source, None)
                    has_ms = any(
                        items for source, items in observations.items() if source.startswith("JL1")
                    )
                    has_pan = any(
                        items for source, items in observations.items() if "PAN" in source
                    )
                    record["quality"].update(
                        ms5m_supported=has_ms,
                        pan2m_supported=has_pan,
                        highres_supported=has_ms or has_pan,
                    )
                    support[
                        (
                            "both"
                            if has_ms and has_pan
                            else "ms_only" if has_ms else "pan_only" if has_pan else "lowres_only"
                        )
                    ] += 1
                    for source, items in observations.items():
                        source_counts[source] += len(items)
                    payload = (
                        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)
                        + "\n"
                    ).encode()
                    destination.write(payload)
                    digest.update(payload)
            temporary.replace(target)
            meta = json.loads((base / (name + ".meta.json")).read_text())
            meta.update(
                record_count=manifests[name], sha256=digest.hexdigest(), generator_version=VERSION
            )
            atomic_json(output / (name + ".meta.json"), meta)
            supports[name] = dict(support)
        counts["sources_without_train_statistics_withheld"] += sum(1 for _ in missing_sources)
        for source in missing_sources:
            schemas.pop(source, None)
    statistics = output / "statistics"
    statistics.mkdir(exist_ok=True)
    for source, state in states.items():
        std = np.sqrt(state["moment"] / state["counts"])
        if not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid retained training statistics")
        atomic_json(
            statistics / f"{source}_stats.json",
            {
                "mean": state["mean"].tolist(),
                "std": std.tolist(),
                "band_counts": state["counts"].astype(np.int64).tolist(),
                "num_files": state["files"],
                "fit_split": "train",
                "fit_years": [2020, 2021],
                "units": "stored_values",
                "basis": "final_selected_valid_pixels",
            },
        )
    if set(schemas) - set(states):
        raise ValueError("Source without training statistics")
    atomic_json(output / "sources.json", schemas)
    split_report = {
        "file_digest_cross_split_conflicts_before": len(file_owners),
        "retained_file_digest_cross_split_conflicts": 0,
        "retained_known_product_cross_split_conflicts": 0,
        "retained_highres_pixel_digest_cross_split_conflicts": 0,
        "retained_parent_cross_split_conflicts": 0,
        "counts": dict(counts),
        "original_scene_scope": (
            "PAN and JL1; monthly lowres source scenes unavailable; inherited "
            "spatial buffer plus exact file-digest checks retained"
        ),
        "known_products_retained": len(retained_products),
    }
    atomic_json(output / "split-audit.json", split_report)
    report = {
        "version": VERSION,
        "data_root": str(root),
        "previous_version": str(base),
        "status": "screened_waiting_final_io_validation",
        "training_ready": False,
        "scientific_quality_validated": False,
        "annual_model_ready": False,
        "manifests": manifests,
        "branch_support_samples": supports,
        "source_observation_counts": dict(source_counts),
        "exclusions": dict(counts),
        "upstream_scripts_required": False,
        "quality_scope": (
            "training-oriented screening and sampled diagnostic review; no "
            "independent cloud accuracy or absolute geolocation certification"
        ),
        "known_limits": [
            "Cloud inference uses only 1280m spatial context",
            "Visual review finds snow and terrain-shadow overmasking; "
            "conservative coverage loss retained",
            "PAN MUX band order follows product convention and visual "
            "diagnostics, not recovered production code",
            "Landsat withheld from this release because cloud QA is unavailable",
            "Lowres composite original scenes unknown; no all-sensor scene-disjoint guarantee",
            "All highres diagnosed; unresolved alignment withheld; "
            "no true 5m accuracy certification",
        ],
        "remaining_gates": ["final_real_sample_loader_check"],
        "remaining_model_work": "annual three-branch runtime and real-data 8-GPU smoke",
    }
    atomic_json(output / "summary.json", report)
    atomic_json(
        output / "checksums.json",
        {
            p.relative_to(output).as_posix(): sha256_file(p)
            for p in [
                *output.glob("*.manifest.jsonl*"),
                output / "sources.json",
                *statistics.glob("*.json"),
            ]
        },
    )
    shutil.copy2(Path(__file__), output / "release_merge_source.py")
    db.close()
    return report


def validate_release(output: Path) -> dict:
    """Verify the frozen contract, statistics and sampled I/O before data release."""
    summary = json.loads((output / "summary.json").read_text())
    loader = json.loads((output / "loader-check.json").read_text())
    if not loader.get("loader_passed") or set(loader["manifests"]) != set(summary["manifests"]):
        raise ValueError("Final loader coverage incomplete")
    for name, count in summary["manifests"].items():
        if loader["manifests"][name]["samples_read"] < min(100, count):
            raise ValueError("Too few final loader samples")
    checksums = json.loads((output / "checksums.json").read_text())
    for name, expected in checksums.items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"Release artifact digest mismatch: {name}")
    states = defaultdict(dict)
    counts, source_counts = Counter(), Counter()
    owners = {key: {} for key in ("parent", "file", "product", "pixels")}
    for name, expected_count in summary["manifests"].items():
        with (output / name).open() as handle:
            for line in handle:
                record = json.loads(line)
                provenance = record["provenance"]
                observations = provenance["observations"]
                split = provenance["split"]
                parent = record["grid"]["parent_key"]
                if "landsat" in observations or "landsat" in record["sources"]:
                    raise ValueError("Unscreened Landsat in formal release")
                keys = [("parent", parent)]
                months = set()
                for source, items in observations.items():
                    if record["sources"][source] != [o["path"] for o in items]:
                        raise ValueError("Release source/observation mismatch")
                    source_counts[source] += len(items)
                    for item in items:
                        if (
                            item["parent_key"] != parent
                            or int(item["date"][:4]) != provenance["year"]
                        ):
                            raise ValueError("Observation parent/year mismatch")
                        if source in {"s1", "s2"} and item["valid_fraction"] >= 0.2:
                            months.add(int(item["date"][5:7]))
                        if source != "s1" and source != "s2":
                            reg = item["relative_registration"]
                            if reg["status"] != "measured" or reg["offset_m"] > 10:
                                raise ValueError("Unresolved highres alignment in formal release")
                            if item["valid_fraction"] < 0.2:
                                raise ValueError("Insufficient highres clear area")
                        if source == "s2" or source.startswith("JL1"):
                            if item.get("cloud_QA") != "OCM_v4_prediction_unvalidated":
                                raise ValueError("Optical observation without cloud screening")
                        if "PAN" in source:
                            reg = item["pan_mux_registration"]
                            if (
                                reg["status"] != "measured"
                                or np.linalg.norm(reg["shift_yx_pixels"]) > 1
                            ):
                                raise ValueError("Unresolved PAN/MUX alignment")
                            if item.get("cloud_shadow_QA") != "same_scene_MUX_OCM_v4_with_8m_guard":
                                raise ValueError("PAN without cloud screening")
                        keys.extend([("file", item["sha256"]), ("product", original_product(item))])
                        if item.get("pixel_sha256"):
                            keys.append(("pixels", (source, item["pixel_sha256"])))
                        if split == "train":
                            add_moments(states[source], item)
                if len(months) < 6 or len({(m - 1) // 3 for m in months}) < 3:
                    raise ValueError("Insufficient retained annual temporal coverage")
                for group, key in keys:
                    if key is not None:
                        if key in owners[group] and owners[group][key] != split:
                            raise ValueError(f"Cross-split conflict: {group}")
                        owners[group][key] = split
                counts[name] += 1
        if counts[name] != expected_count:
            raise ValueError("Final record count mismatch")
    if dict(source_counts) != summary["source_observation_counts"]:
        raise ValueError("Final source counts mismatch")
    for source, state in states.items():
        stats = json.loads((output / "statistics" / f"{source}_stats.json").read_text())
        if stats["fit_split"] != "train" or stats["num_files"] != state["files"]:
            raise ValueError("Statistics not fit exclusively on retained train data")
        for key, values in (
            ("mean", state["mean"]),
            ("std", np.sqrt(state["moment"] / state["counts"])),
            ("band_counts", state["counts"]),
        ):
            if not np.allclose(stats[key], values, rtol=1e-10, atol=1e-10):
                raise ValueError(f"Statistics mismatch: {source}/{key}")
    result = {
        "contract_passed": True,
        "manifest_records": dict(counts),
        "source_observation_counts": dict(source_counts),
        "cross_split_conflicts": {key: 0 for key in owners},
        "statistics_recomputed_from_final_train_metadata": True,
        "loader_samples_read": sum(r["samples_read"] for r in loader["manifests"].values()),
        "loader_check_sha256": sha256_file(output / "loader-check.json"),
        "artifact_checksums_sha256": sha256_file(output / "checksums.json"),
        "limitations": summary["known_limits"],
    }
    atomic_json(output / "contract-check.json", result)
    summary.update(
        status="released_for_training_with_documented_limits",
        training_ready=True,
        remaining_gates=[],
        release_scope=(
            "data contract and conservative training QA; " "not independent accuracy certification"
        ),
        excluded_optional_sources=["landsat"],
    )
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "release-readiness.json",
        {
            "training_data_ready": True,
            "scientific_quality_validated": False,
            "annual_model_ready": False,
            "contract_check_sha256": sha256_file(output / "contract-check.json"),
            "summary_sha256": sha256_file(output / "summary.json"),
            "meaning": (
                "Screened inputs and masks ready for annual runtime integration; "
                "no trained model produced"
            ),
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
    parser = argparse.ArgumentParser(prog="xuannv data release-quality")
    parser.add_argument(
        "--phase",
        choices=(
            "prepare",
            "worker",
            "pan-finalize",
            "lineage",
            "registration",
            "full-registration",
            "merge",
            "validate-release",
        ),
        required=True,
    )
    parser.add_argument("--base", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        prepare(args.base, args.output, args.catalog, args.model_dir)
    elif args.phase == "pan-finalize":
        print(json.dumps(finalize_pan(args.output)))
    elif args.phase == "lineage":
        print(json.dumps(audit_lineage(args.output)))
    elif args.phase == "merge":
        print(json.dumps(merge_release(args.output)))
    elif args.phase == "validate-release":
        print(json.dumps(validate_release(args.output)))
    elif args.phase == "full-registration":
        from xuannv_embedding.data_process.image_quality import full_registration

        full_registration(args.output)
    elif args.phase == "registration":
        from xuannv_embedding.data_process.image_quality import audit_registration

        print(json.dumps(audit_registration(args.output)["counts"]))
    else:
        from xuannv_embedding.data_process.cloud_quality import worker

        worker(
            args.output,
            args.rank,
            f"cuda:{args.rank}",
            args.batch_size,
            write_threads=16,
            read_function=read_mux_item,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
