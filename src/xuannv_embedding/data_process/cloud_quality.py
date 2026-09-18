"""Resumable S2 cloud-screened candidates; never grants scientific quality approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from xuannv_embedding.data_process.annual_quality import LOWRES_CHANNELS, add_moments, relative_file
from xuannv_embedding.data_process.observation_raster import parent_geometry, sha256_file
from xuannv_embedding.data_process.prepare_observations import atomic_json
from xuannv_embedding.utils.manifest import load_manifest, write_manifest

VERSION = "annual-s2-cloud-candidate-v3"


def screened_pixels(values: np.ndarray, numeric: np.ndarray, scores: np.ndarray) -> tuple:
    """Retain predicted clear pixels; preserve unknown/invalid pixels as invalid."""
    if scores.shape != (4, *numeric.shape) or not np.isfinite(scores).all():
        raise ValueError("Invalid cloud model output")
    valid = numeric & np.isfinite(values).all(axis=0) & (values > 0).all(axis=0)
    classes = scores.argmax(axis=0).astype(np.uint8)
    clear = valid & (classes == 0)
    confidence = np.rint(scores.max(axis=0).clip(0, 1) * 100).astype(np.uint8)
    classes[~valid], confidence[~valid] = 255, 255
    accepted = values[:, clear].astype(np.float64)
    count = int(clear.sum())
    moments = {
        "valid_fraction": float(clear.mean()),
        "band_counts": [count] * values.shape[0],
        "band_mean": accepted.mean(axis=1).tolist() if count else [],
        "band_variance": accepted.var(axis=1).tolist() if count else [],
        "cloud_class_counts": [int((classes == k).sum()) for k in range(4)],
        "numeric_valid_pixels": int(valid.sum()),
    }
    return np.stack([clear.astype(np.uint8), classes, confidence]), moments


def load_s2(root: Path, observation: dict) -> tuple:
    path = relative_file(root, observation["path"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != observation["sha256"]:
        raise ValueError("Observation checksum changed")
    with MemoryFile(payload) as memory, memory.open() as raster:
        values = raster.read()
        numeric = (raster.read_masks() > 0).all(axis=0)
        profile = raster.profile.copy()
        geometry = (raster.crs, raster.transform, raster.shape)
        epsg, bounds = parent_geometry(observation["parent_key"])
        if (
            raster.crs is None
            or raster.crs.to_epsg() != epsg
            or not np.allclose(raster.bounds, bounds, rtol=0, atol=0.01)
            or values.shape != (10, 128, 128)
        ):
            raise ValueError("S2 parent geometry or channels changed")
    mask_bytes = relative_file(root, observation["mask"]).read_bytes()
    with MemoryFile(mask_bytes) as memory, memory.open() as raster:
        if (raster.crs, raster.transform, raster.shape) != geometry:
            raise ValueError("Numeric mask geometry changed")
        numeric &= raster.read(1) > 0
    numeric &= np.isfinite(values).all(axis=0) & (values > 0).all(axis=0)
    if np.issubdtype(values.dtype, np.integer):
        numeric &= ~(values == np.iinfo(values.dtype).max).any(axis=0)
    return values, numeric, profile


def prepare(dataset: Path, output: Path, model_dir: Path, workers: int, limit: int) -> dict:
    if output.exists():
        contract = json.loads((output / "run.json").read_text())
        if (
            contract["input"] != str(dataset.resolve())
            or contract["workers"] != workers
            or contract["limit"] != limit
            or contract["model_dir"] != str(model_dir.resolve())
        ):
            raise ValueError("Resume contract mismatch")
        return contract
    summary = json.loads((dataset / "summary.json").read_text())
    root = Path(summary["data_root"]).resolve()
    if not output.resolve().is_relative_to(root):
        raise ValueError("Output must be below the dataset data_root")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_version"] != 4.0 or provenance["package_version"] != "1.7.1":
        raise ValueError("This pipeline requires the audited OmniCloudMask v4 / 1.7.1")
    for weight in provenance["weights"]:
        if sha256_file(model_dir / weight["name"]) != weight["sha256"]:
            raise ValueError("Cloud model weight checksum mismatch")
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "cloud_quality_source.py")
    database = sqlite3.connect(output / "tasks.sqlite")
    database.execute("CREATE TABLE tasks(path TEXT PRIMARY KEY, shard INTEGER, payload TEXT)")
    fingerprints = {}
    for manifest in sorted(dataset.glob("*.manifest.jsonl")):
        document = load_manifest(manifest)
        fingerprints[manifest.name] = sha256_file(manifest)
        count = 0
        for record in document.records:
            for observation in record.provenance["observations"].get("s2", []):
                key = observation["path"]
                shard = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % workers
                database.execute(
                    "INSERT OR IGNORE INTO tasks VALUES (?,?,?)",
                    (key, shard, json.dumps(observation)),
                )
                count += 1
                if limit and count >= limit:
                    break
            if limit and count >= limit:
                break
        database.commit()
        print(json.dumps({"indexed": manifest.name, "s2": count}), flush=True)
        del document
    database.execute("CREATE INDEX shard_lookup ON tasks(shard)")
    total = database.execute("SELECT count(*) FROM tasks").fetchone()[0]
    database.commit()
    database.close()
    contract = {
        "version": VERSION,
        "processor_sha256": sha256_file(Path(__file__)),
        "input": str(dataset.resolve()),
        "data_root": str(root),
        "workers": workers,
        "limit": limit,
        "model_dir": str(model_dir.resolve()),
        "model": provenance,
        "input_manifests": fingerprints,
        "total": total,
        "policy": "numeric_valid AND OCM_argmax_clear; retain any nonempty clear area",
        "annual_coverage_policy": "6 months / 3 quarters with >=20% valid area per observation",
        "status": "processing",
        "scientific_quality_validated": False,
        "training_ready": False,
    }
    atomic_json(output / "run.json", contract)
    return contract


def read_cloud_item(item: tuple, root: Path) -> tuple:
    from omnicloudmask.model_utils import channel_norm

    path, payload = item
    observation = json.loads(payload)
    try:
        values, numeric, profile = load_s2(root, observation)
        bands = values[[2, 1, 7]].astype(np.float32)
        bands[:, ~numeric] = 0
        normalized = channel_norm(bands, 0)
        return path, observation, (values, numeric, profile, normalized), None
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        return path, observation, None, f"{type(error).__name__}: {error}"


def write_cloud_item(item: tuple, root: Path, output: Path) -> tuple:
    (path, observation, (values, numeric, profile, _), _), scores = item
    qa, moments = screened_pixels(values, numeric, scores)
    key = hashlib.sha256(path.encode()).hexdigest()
    destination = output / "masks" / key[:2] / f"{key}.tif"
    temporary = destination.with_suffix(".partial.tif")
    profile.update(count=3, dtype="uint8", nodata=None, compress="deflate")
    with MemoryFile() as memory:
        with memory.open(**profile) as raster:
            raster.write(qa)
            for band, name in enumerate(
                ("candidate_valid", "ocm_class", "uncalibrated_score_percent"), 1
            ):
                raster.set_band_description(band, name)
        payload = memory.read()
    temporary.write_bytes(payload)
    temporary.replace(destination)
    updated = {
        **observation,
        **moments,
        "numeric_mask": observation["mask"],
        "mask": destination.relative_to(root).as_posix(),
        "mask_sha256": hashlib.sha256(payload).hexdigest(),
        "cloud_QA": "OCM_v4_prediction_unvalidated",
        "status": "accepted" if moments["band_counts"][0] else "no_clear_pixels",
    }
    return path, json.dumps(updated)


def worker(
    output: Path,
    rank: int,
    device: str,
    batch_size: int,
    io_processes: int = 0,
    write_threads: int = 64,
    read_function=None,
) -> None:
    import torch
    from omnicloudmask.cloud_mask import collect_models

    torch.set_num_threads(2)
    read_function = read_function or read_cloud_item
    contract = json.loads((output / "run.json").read_text())
    root = Path(contract["data_root"])
    torch_device = torch.device(device)
    models = collect_models(
        custom_models=None,
        source="hugging_face",
        inference_device=torch_device,
        inference_dtype=torch.float32,
        destination_model_dir=Path(contract["model_dir"]),
        model_version=4.0,
    )
    for model in models:
        model.eval()
    tasks = sqlite3.connect(f"file:{output / 'tasks.sqlite'}?mode=ro", uri=True)
    result = sqlite3.connect(output / f"results-{rank}.sqlite")
    result.execute("CREATE TABLE IF NOT EXISTS results(path TEXT PRIMARY KEY, payload TEXT)")
    done = {row[0] for row in result.execute("SELECT path FROM results")}
    iterator = tasks.execute("SELECT path,payload FROM tasks WHERE shard=? ORDER BY path", (rank,))
    started, processed = time.monotonic(), len(done)
    status = {"rank": rank, "processed": processed, "elapsed_seconds": 0}

    for prefix in range(256):
        (output / "masks" / f"{prefix:02x}").mkdir(exist_ok=True, parents=True)

    def commit(writes: list, errors: list, count: int) -> None:
        nonlocal processed, status
        for future in writes:
            result.execute("INSERT INTO results VALUES (?,?)", future.result())
        for path, observation, _, error in errors:
            result.execute(
                "INSERT INTO results VALUES (?,?)",
                (path, json.dumps({**observation, "status": "read_error", "error": error})),
            )
        result.commit()
        processed += count
        status = {
            "rank": rank,
            "processed": processed,
            "elapsed_seconds": time.monotonic() - started,
            "processed_this_run": processed - len(done),
            "batch_size": batch_size,
            "io_processes_per_stage": io_processes,
            "io_workers_per_stage": io_processes or 16,
            "read_workers": io_processes or 16,
            "write_workers": io_processes or write_threads,
            "status": "running",
        }
        atomic_json(output / f"progress-{rank}.json", status)
        if (processed - len(done)) % (batch_size * 20) == 0:
            print(json.dumps(status), flush=True)

    def executor(threads: int):
        if io_processes:
            return ProcessPoolExecutor(
                max_workers=io_processes, mp_context=multiprocessing.get_context("spawn")
            )
        return ThreadPoolExecutor(max_workers=threads)

    with executor(16) as readers, executor(write_threads) as writers:

        def prefetch() -> list:
            batch = []
            while len(batch) < batch_size:
                item = next(iterator, None)
                if item is None:
                    break
                if item[0] not in done:
                    batch.append(readers.submit(read_function, item, root))
            return batch

        pending_reads = prefetch()
        pending_writes, pending_errors, pending_count = [], [], 0
        while pending_reads:
            loaded = [future.result() for future in pending_reads]
            pending_reads = prefetch()
            good = [item for item in loaded if item[3] is None]
            if good:
                tensor = torch.from_numpy(np.stack([item[2][3] for item in good]))
                if torch_device.type == "cuda":
                    tensor = tensor.pin_memory().to(torch_device, non_blocking=True)
                else:
                    tensor = tensor.to(torch_device)
                with torch.inference_mode():
                    logits = torch.stack([model(tensor) for model in models]).mean(dim=0)
                    probabilities = logits.softmax(dim=1).cpu().numpy()
            if pending_count:
                commit(pending_writes, pending_errors, pending_count)
            pending_writes = (
                [
                    writers.submit(write_cloud_item, item, root, output)
                    for item in zip(good, probabilities)
                ]
                if good
                else []
            )
            pending_errors = [item for item in loaded if item[3] is not None]
            pending_count = len(loaded)
        if pending_count:
            commit(pending_writes, pending_errors, pending_count)

    status["status"] = "complete"
    atomic_json(output / f"progress-{rank}.json", status)
    result.close()
    tasks.close()


def finalize(output: Path) -> dict:
    contract = json.loads((output / "run.json").read_text())
    if contract["limit"]:
        raise ValueError("A limited pilot cannot publish annual manifests")
    dataset = Path(contract["input"])
    database = sqlite3.connect(output / "combined.sqlite")
    database.execute("DROP TABLE IF EXISTS results")
    database.execute("CREATE TABLE results(path TEXT PRIMARY KEY, payload TEXT)")
    counts = Counter()
    for rank in range(contract["workers"]):
        progress = json.loads((output / f"progress-{rank}.json").read_text())
        if progress["status"] != "complete":
            raise ValueError("Cloud workers incomplete")
        with sqlite3.connect(
            f"file:{output / f'results-{rank}.sqlite'}?mode=ro", uri=True
        ) as shard:
            for path, payload in shard.execute("SELECT path,payload FROM results"):
                item = json.loads(payload)
                counts[item["status"]] += 1
                database.execute("INSERT INTO results VALUES (?,?)", (path, payload))
        database.commit()
    if sum(counts.values()) != contract["total"]:
        raise ValueError("Cloud result count mismatch")
    stats = defaultdict(dict)
    manifest_counts, highres_counts, excluded = {}, {}, Counter()
    with (output / "excluded_samples.jsonl").open("w") as reject:
        for name, fingerprint in contract["input_manifests"].items():
            manifest = dataset / name
            if sha256_file(manifest) != fingerprint:
                raise ValueError("Input manifest changed during screening")
            document = load_manifest(manifest)
            selected = []
            for record in document.records:
                observations = record.provenance["observations"]
                updated = []
                for original in observations.get("s2", []):
                    found = database.execute(
                        "SELECT payload FROM results WHERE path=?", (original["path"],)
                    ).fetchone()
                    if found is None:
                        raise ValueError("Missing S2 cloud result")
                    item = json.loads(found[0])
                    if item["status"] == "accepted":
                        updated.append(item)
                observations["s2"] = updated
                record.sources["s2"] = [item["path"] for item in updated]
                low = [
                    o
                    for s, items in observations.items()
                    if s in LOWRES_CHANNELS
                    for o in items
                    if o["valid_fraction"] >= 0.2
                ]
                months = {o["date"] for o in low}
                quarters = {(int(o["date"][5:7]) - 1) // 3 for o in low}
                if len(months) < 6 or len(quarters) < 3:
                    excluded["insufficient_annual_coverage"] += 1
                    reject.write(
                        json.dumps(
                            {
                                "patch_id": record.patch_id,
                                "year": record.provenance["year"],
                                "reason": "insufficient_annual_coverage",
                            }
                        )
                        + "\n"
                    )
                    continue
                record.quality.update(
                    cloud_shadow="s2_model_screened_other_optical_unverified",
                    s2_cloud_model="OCM_v4_unvalidated",
                    s2_months_min_20pct=sum(o["valid_fraction"] >= 0.2 for o in updated),
                    s2_retained_clear_fraction_sum=sum(o["valid_fraction"] for o in updated),
                    highres_supported=any(
                        s not in LOWRES_CHANNELS and v for s, v in observations.items()
                    ),
                )
                if record.provenance["split"] == "train":
                    for source, items in observations.items():
                        for item in items:
                            add_moments(stats[source], item)
                selected.append(record)
            write_manifest(
                output / name, selected, months=document.meta.months, generator_version=VERSION
            )
            load_manifest(output / name)
            manifest_counts[name] = len(selected)
            highres_counts[name] = sum(r.quality["highres_supported"] for r in selected)
            print(json.dumps({"finalized": name, "samples": len(selected)}), flush=True)
            del document, selected
    (output / "statistics").mkdir(exist_ok=True)
    for source, state in stats.items():
        std = np.sqrt(state["moment"] / state["counts"])
        if not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError(f"Invalid post-screening statistics: {source}")
        atomic_json(
            output / "statistics" / f"{source}_stats.json",
            {
                "mean": state["mean"].tolist(),
                "std": std.tolist(),
                "band_counts": state["counts"].astype(np.int64).tolist(),
                "num_files": state["files"],
                "fit_split": "train",
                "fit_years": [2020, 2021],
                "units": "stored_values",
                "basis": "s2_predicted_clear_pixels_other_sources_numeric",
                "cloud_QA": "unvalidated_model_candidate" if source == "s2" else "unverified",
            },
        )
    shutil.copy2(dataset / "sources.json", output / "sources.json")
    summary = {
        **json.loads((dataset / "summary.json").read_text()),
        "version": VERSION,
        "status": "s2_cloud_screened_candidate_awaiting_validation",
        "training_ready": False,
        "scientific_quality_validated": False,
        "manifests": manifest_counts,
        "highres_supported_samples": highres_counts,
        "cloud_screening_counts": dict(counts),
        "cloud_sample_exclusions": dict(excluded),
        "previous_version": str(dataset),
        "cloud_run": "run.json",
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "checksums.json",
        {
            p.relative_to(output).as_posix(): sha256_file(p)
            for p in sorted(output.glob("*.manifest.jsonl*"))
        },
    )
    contract["status"] = "complete_candidate"
    atomic_json(output / "run.json", contract)
    database.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv data cloud-annual")
    parser.add_argument("--phase", choices=("prepare", "worker", "finalize", "run"), default="run")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--io-processes", type=int, default=0)
    parser.add_argument("--write-threads", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-per-manifest", type=int, default=0)
    args = parser.parse_args(argv)
    if (
        args.workers < 1
        or args.batch_size < 1
        or args.limit_per_manifest < 0
        or args.io_processes < 0
        or args.write_threads < 1
    ):
        parser.error("workers/batch-size must be positive; limit must be nonnegative")
    if args.phase == "worker":
        worker(
            args.output,
            args.rank,
            args.device,
            args.batch_size,
            args.io_processes,
            args.write_threads,
        )
    elif args.phase == "finalize":
        print(json.dumps(finalize(args.output)))
    else:
        if args.dataset is None or args.model_dir is None:
            parser.error("dataset and model-dir are required")
        prepare(args.dataset, args.output, args.model_dir, args.workers, args.limit_per_manifest)
        if args.phase == "run":
            jobs, logs = [], []
            for rank in range(args.workers):
                log = (args.output / f"worker-{rank}.log").open("a")
                logs.append(log)
                jobs.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "xuannv_embedding.cli",
                            "data",
                            "cloud-annual",
                            "--phase",
                            "worker",
                            "--output",
                            str(args.output),
                            "--rank",
                            str(rank),
                            "--device",
                            f"cuda:{rank}",
                            "--batch-size",
                            str(args.batch_size),
                            "--io-processes",
                            str(args.io_processes),
                            "--write-threads",
                            str(args.write_threads),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "HF_HUB_OFFLINE": "1"},
                    )
                )
            codes = [job.wait() for job in jobs]
            for log in logs:
                log.close()
            if any(codes):
                raise RuntimeError(f"Cloud workers failed: {codes}")
            if not args.limit_per_manifest:
                print(json.dumps(finalize(args.output)))
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "xuannv_embedding.cli",
                        "data",
                        "check-annual",
                        "--dataset",
                        str(args.output),
                        "--samples-per-manifest",
                        "100",
                        "--output",
                        str(args.output / "loader-check.json"),
                    ],
                    check=True,
                )
    return 0
