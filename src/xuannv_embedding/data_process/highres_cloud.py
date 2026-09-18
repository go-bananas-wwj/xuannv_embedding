"""Cloud-screen wavelength-described 5m observations; PAN still requires separate QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from xuannv_embedding.data_process.annual_quality import add_moments, relative_file
from xuannv_embedding.data_process.cloud_quality import worker
from xuannv_embedding.data_process.observation_raster import parent_geometry, sha256_file
from xuannv_embedding.data_process.prepare_observations import atomic_json
from xuannv_embedding.utils.manifest import load_manifest, write_manifest

VERSION = "annual-three-branch-cloud-candidate-v5"


def spectral_indices(schema: dict) -> list[int]:
    wavelengths = []
    for name in schema.get("band_names", []):
        match = re.fullmatch(r"B\d+\((0\.\d+)\)", name or "")
        if not match:
            raise ValueError("Missing wavelength description")
        wavelengths.append(float(match[1]))
    indices = []
    for low, high in ((0.62, 0.69), (0.52, 0.60), (0.78, 0.90)):
        found = [i for i, value in enumerate(wavelengths) if low <= value <= high]
        if len(found) != 1:
            raise ValueError("Unique red/green/NIR wavelengths required")
        indices.append(found[0])
    return indices


def read_highres_item(item: tuple, root: Path) -> tuple:
    from omnicloudmask.model_utils import channel_norm

    path, payload = item
    observation = json.loads(payload)
    try:
        blob = relative_file(root, observation["path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != observation["sha256"]:
            raise ValueError("Highres payload checksum changed")
        with MemoryFile(blob) as memory, memory.open() as raster:
            epsg, bounds = parent_geometry(observation["parent_key"])
            if (
                raster.crs is None
                or raster.crs.to_epsg() != epsg
                or raster.shape != (256, 256)
                or raster.count != observation["channels"]
                or not np.allclose(
                    list(raster.transform)[:6],
                    [5, 0, bounds[0], 0, -5, bounds[3]],
                    rtol=0,
                    atol=1e-8,
                )
            ):
                raise ValueError("Highres native grid changed")
            indices = spectral_indices({"band_names": list(raster.descriptions)})
            if indices != observation["cloud_rgbnir_indices"]:
                raise ValueError("Wavelength mapping changed")
            if (
                list(raster.scales) != observation["stored_scales"]
                or list(raster.offsets) != observation["stored_offsets"]
            ):
                raise ValueError("Highres scaling metadata changed")
            values = raster.read()
            valid = (raster.read_masks() > 0).all(axis=0) & np.isfinite(values).all(axis=0)
            profile = raster.profile.copy()
            geometry = raster.crs, raster.transform, raster.shape
        mask_blob = relative_file(root, observation["mask"]).read_bytes()
        if (
            observation.get("mask_sha256")
            and hashlib.sha256(mask_blob).hexdigest() != observation["mask_sha256"]
        ):
            raise ValueError("Highres numeric mask checksum changed")
        with MemoryFile(mask_blob) as memory, memory.open() as raster:
            if (raster.crs, raster.transform, raster.shape) != geometry:
                raise ValueError("Highres mask grid changed")
            valid &= raster.read(1) > 0
        valid &= (values > 0).all(axis=0)
        if np.issubdtype(values.dtype, np.integer):
            valid &= ~(values == np.iinfo(values.dtype).max).any(axis=0)
        bands = values[indices].astype(np.float32)
        bands *= np.asarray(observation["stored_scales"], dtype=np.float32)[indices, None, None]
        bands += np.asarray(observation["stored_offsets"], dtype=np.float32)[indices, None, None]
        bands[:, ~valid] = 0
        normalized = channel_norm(bands, 0)
        return path, observation, (values, valid, profile, normalized), None
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        return path, observation, None, f"{type(error).__name__}: {error}"


def prepare(dataset: Path, output: Path, model_dir: Path, workers: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((dataset / "summary.json").read_text())
    root = Path(summary["data_root"])
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output must be within data_root")
    schemas = json.loads((dataset / "sources.json").read_text())
    supported, rejected = {}, {}
    for source, schema in schemas.items():
        if schema.get("product_group") != "5m":
            continue
        try:
            supported[source] = spectral_indices(schema)
        except ValueError as error:
            rejected[source] = str(error)
    if not supported:
        raise ValueError("No wavelength-described highres sources")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    if provenance["model_version"] != 4.0 or provenance["package_version"] != "1.7.1":
        raise ValueError("Requires audited OmniCloudMask v4 / 1.7.1")
    for weights in provenance["weights"]:
        if sha256_file(model_dir / weights["name"]) != weights["sha256"]:
            raise ValueError("Cloud model checksum mismatch")
    output.mkdir(parents=True)
    database = sqlite3.connect(output / "tasks.sqlite")
    database.execute("CREATE TABLE tasks(path TEXT PRIMARY KEY, shard INTEGER, payload TEXT)")
    fingerprints = {}
    for manifest in sorted(dataset.glob("*.manifest.jsonl")):
        document = load_manifest(manifest)
        fingerprints[manifest.name] = sha256_file(manifest)
        for record in document.records:
            for source, indices in supported.items():
                for original in record.provenance["observations"].get(source, []):
                    item = {
                        **original,
                        "cloud_rgbnir_indices": indices,
                        "stored_scales": schemas[source]["stored_scales"],
                        "stored_offsets": schemas[source]["stored_offsets"],
                    }
                    path = item["path"]
                    rank = int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) % workers
                    database.execute(
                        "INSERT OR IGNORE INTO tasks VALUES (?,?,?)", (path, rank, json.dumps(item))
                    )
        database.commit()
        print(json.dumps({"indexed_highres": manifest.name}), flush=True)
        del document
    database.execute("CREATE INDEX shard_lookup ON tasks(shard)")
    count = database.execute("SELECT count(*) FROM tasks").fetchone()[0]
    database.commit()
    database.close()
    contract = {
        "version": VERSION,
        "input": str(dataset),
        "data_root": str(root),
        "model_dir": str(model_dir),
        "workers": workers,
        "total": count,
        "supported_sources": supported,
        "rejected_sources": rejected,
        "input_manifests": fingerprints,
        "model": provenance,
        "processor_sha256": sha256_file(Path(__file__)),
        "worker_sha256": sha256_file(Path(__file__).with_name("cloud_quality.py")),
        "status": "processing",
        "training_ready": False,
        "scientific_quality_validated": False,
        "spectral_policy": "red_green_NIR_from_stored_center_wavelength_descriptions",
    }
    atomic_json(output / "run.json", contract)
    shutil.copy2(Path(__file__), output / "highres_cloud_source.py")
    shutil.copy2(Path(__file__).with_name("cloud_quality.py"), output / "cloud_worker_source.py")
    return contract


def finalize(output: Path, base: Path) -> dict:
    contract = json.loads((output / "run.json").read_text())
    base_fingerprints = {
        path.name: sha256_file(path)
        for path in [
            base / "summary.json",
            base / "sources.json",
            *sorted(base.glob("*.manifest.jsonl")),
        ]
    }
    database = sqlite3.connect(output / "combined.sqlite")
    database.execute("PRAGMA cache_size=-524288")
    database.execute("PRAGMA temp_store=MEMORY")
    database.execute("DROP TABLE IF EXISTS results")
    database.execute("CREATE TABLE results(path TEXT PRIMARY KEY,payload TEXT)")
    counts = Counter()
    for rank in range(contract["workers"]):
        if json.loads((output / f"progress-{rank}.json").read_text())["status"] != "complete":
            raise ValueError("Incomplete highres cloud workers")
        with sqlite3.connect(
            f"file:{output / f'results-{rank}.sqlite'}?mode=ro", uri=True
        ) as shard:
            for path, payload in shard.execute("SELECT path,payload FROM results"):
                counts[json.loads(payload)["status"]] += 1
                database.execute("INSERT INTO results VALUES (?,?)", (path, payload))
        database.commit()
    if sum(counts.values()) != contract["total"]:
        raise ValueError("Incomplete highres cloud results")
    # New base may add PAN; each inherited highres observation must still match its audited input.
    for name, digest in contract["input_manifests"].items():
        if sha256_file(Path(contract["input"]) / name) != digest:
            raise ValueError("Cloud input manifest changed")
    schemas = json.loads((base / "sources.json").read_text())
    states, manifest_counts, support_counts = defaultdict(dict), {}, {}
    branch_support = {}
    for manifest in sorted(base.glob("*.manifest.jsonl")):
        document = load_manifest(manifest)
        supported_samples = 0
        support = Counter()
        for record in document.records:
            observations = record.provenance["observations"]
            for source in contract["rejected_sources"]:
                observations.pop(source, None)
                record.sources.pop(source, None)
            has_ms = False
            for source in contract["supported_sources"]:
                retained = []
                for original in observations.get(source, []):
                    row = database.execute(
                        "SELECT payload FROM results WHERE path=?", (original["path"],)
                    ).fetchone()
                    if row is None:
                        raise ValueError("Unscreened highres in merge base")
                    item = json.loads(row[0])
                    if any(
                        item[key] != original[key]
                        for key in ("sha256", "parent_key", "source", "date")
                    ):
                        raise ValueError("Changed highres identity in merge base")
                    if item["status"] == "accepted":
                        retained.append(item)
                observations[source] = retained
                record.sources[source] = [item["path"] for item in retained]
                has_ms |= bool(retained)
            record.quality["ms5m_supported"] = has_ms
            record.quality["ms5m_cloud_shadow"] = "OCM_v4_prediction_unvalidated"
            record.quality["cloud_shadow"] = "S2_and_MS5m_screened_PAN_and_Landsat_unverified"
            record.quality["highres_supported"] = has_ms or record.quality.get(
                "pan2m_supported", False
            )
            has_pan = record.quality.get("pan2m_supported", False)
            support[
                (
                    "both"
                    if has_ms and has_pan
                    else "ms_only" if has_ms else "pan_only" if has_pan else "lowres_only"
                )
            ] += 1
            supported_samples += has_ms
            if record.provenance["split"] == "train":
                for source, items in observations.items():
                    for item in items:
                        add_moments(states[source], item)
        write_manifest(
            output / manifest.name,
            document.records,
            months=document.meta.months,
            generator_version=VERSION,
        )
        manifest_counts[manifest.name] = len(document.records)
        support_counts[manifest.name] = supported_samples
        branch_support[manifest.name] = dict(support)
        print(
            json.dumps({"finalized": manifest.name, "ms5m_supported": supported_samples}),
            flush=True,
        )
        del document
    statistics = output / "statistics"
    statistics.mkdir(exist_ok=True)
    for source, state in states.items():
        std = np.sqrt(state["moment"] / state["counts"])
        if not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid post-screening statistics")
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
                "basis": "post_screening_selected_observations",
                "cloud_QA": (
                    "OCM_prediction_unvalidated"
                    if source in contract["supported_sources"] or source == "s2"
                    else "unverified"
                ),
            },
        )
    for source in contract["rejected_sources"]:
        schemas.pop(source, None)
    if set(schemas) - set(states):
        raise ValueError("Sources without train-only statistics")
    atomic_json(output / "sources.json", schemas)
    report = {
        **json.loads((base / "summary.json").read_text()),
        "version": VERSION,
        "previous_version": str(base),
        "status": "three_branch_candidate_PAN_quality_pending",
        "training_ready": False,
        "scientific_quality_validated": False,
        "manifests": manifest_counts,
        "ms5m_supported_samples": support_counts,
        "branch_support_samples": branch_support,
        "ms5m_cloud_counts": dict(counts),
        "ms5m_rejected_sources": contract["rejected_sources"],
        "cloud_run": str(output / "run.json"),
    }
    atomic_json(output / "summary.json", report)
    atomic_json(
        output / "checksums.json",
        {
            str(p.relative_to(output)): sha256_file(p)
            for p in [
                *sorted(output.glob("*.manifest.jsonl*")),
                output / "sources.json",
                *sorted(statistics.glob("*.json")),
            ]
        },
    )
    contract["status"] = "complete_candidate"
    contract["finalization"] = {
        "processor_sha256": sha256_file(Path(__file__)),
        "merge_base": str(base),
        "base_fingerprints": base_fingerprints,
    }
    shutil.copy2(Path(__file__), output / "highres_finalize_source.py")
    atomic_json(output / "run.json", contract)
    database.close()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv data cloud-highres")
    parser.add_argument("--phase", choices=("infer", "worker", "finalize"), default="infer")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 8 or args.batch_size < 1:
        parser.error("workers must be 1..8 and batch-size must be positive")
    if args.phase == "worker":
        worker(
            args.output,
            args.rank,
            f"cuda:{args.rank}",
            args.batch_size,
            write_threads=16,
            read_function=read_highres_item,
        )
    elif args.phase == "finalize":
        if args.base is None:
            parser.error("finalize requires --base")
        print(json.dumps(finalize(args.output, args.base)))
    else:
        if args.dataset is None or args.model_dir is None:
            parser.error("infer requires --dataset and --model-dir")
        prepare(args.dataset, args.output, args.model_dir, args.workers)
        jobs, logs = [], []
        for rank in range(args.workers):
            log = (args.output / f"worker-{rank}.log").open("a")
            logs.append(log)
            jobs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "xuannv_embedding.cli",
                        "data",
                        "cloud-highres",
                        "--phase",
                        "worker",
                        "--output",
                        str(args.output),
                        "--rank",
                        str(rank),
                        "--batch-size",
                        str(args.batch_size),
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
            raise RuntimeError(f"Highres cloud workers failed: {codes}")
    return 0
