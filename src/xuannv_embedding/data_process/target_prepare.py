"""Freeze spatial experiments and derive coarse DSM targets from locked native sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from xuannv_embedding.data_process.target_audit import digest, summarize, write_json


def native_cache(index: list[dict], integrity: dict, output: Path, *, workers: int) -> dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rasterio
    import zarr
    from numcodecs import Blosc

    from xuannv_embedding.data_process.native_dem import NativeDemArchive

    output.mkdir(parents=True, exist_ok=False)
    if not integrity.get("all_zip_member_crc_passed"):
        raise ValueError("Native source must first pass complete archive CRC verification")
    parts = [Path(item["path"]) for item in integrity["files"]]
    for path, receipt in zip(parts, integrity["files"]):
        status = path.stat()
        if (status.st_size, status.st_mtime_ns) != (receipt["bytes"], receipt["mtime_ns"]):
            raise ValueError("Native source changed since full SHA-256 computation")
    count = len(index)
    if [row["target_row"] for row in index] != list(range(count)):
        raise ValueError("Native cache requires sorted contiguous target_row identities")
    group = zarr.open_group(str(output / "dem40m.zarr"), mode="w")
    group.attrs.update(
        schema="xuannv.native-dsm-cache.v1",
        complete=False,
        spacing_m=40,
        patch_ids=[row["patch_id"] for row in index],
    )
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    values = group.create_dataset(
        "values", shape=(count, 32, 32), chunks=(32, 32, 32), dtype="float32", compressor=compressor
    )
    masks = group.create_dataset(
        "valid", shape=values.shape, chunks=values.chunks, dtype=bool, compressor=compressor
    )
    stats = {key: np.zeros(count, dtype=np.float64) for key in ("sum", "sumsq")}
    for key in ("valid", "nonfinite", "illegal"):
        stats[key] = np.zeros(count, dtype=np.int64)
    stats["min"], stats["max"] = np.full(count, np.inf), np.full(count, -np.inf)
    stats["codes"] = np.array([], dtype=np.int16)
    local, archives = threading.local(), []
    start_time = time.monotonic()

    def process(start):
        if not hasattr(local, "archive"):
            local.archive = NativeDemArchive(parts, cache_size=32)
            archives.append(local.archive)
        stop = min(count, start + 32)
        block = np.zeros((stop - start, 32, 32), dtype=np.float32)
        valid = np.zeros_like(block, dtype=bool)
        receipts = []
        for offset, row in enumerate(index[start:stop]):
            patch = local.archive.read_patch(row)
            block[offset], valid[offset] = patch["values"], patch["valid"]
            row_sha = hashlib.sha256(block[offset].tobytes() + valid[offset].tobytes()).hexdigest()
            receipts.append(
                {
                    "patch_id": row["patch_id"],
                    "target_row": row["target_row"],
                    "row_sha256": row_sha,
                    "source_members": patch["members"],
                    "native_x_spacings_degree": patch["native_x_spacings_degree"],
                }
            )
        # Each task owns complete, disjoint Zarr chunks, including the final short chunk.
        values[start:stop], masks[start:stop] = block, valid
        clean = np.where(valid, block.astype(np.float64), 0)
        stats["valid"][start:stop] = valid.sum(axis=(1, 2))
        stats["sum"][start:stop] = clean.sum(axis=(1, 2))
        stats["sumsq"][start:stop] = (clean * clean).sum(axis=(1, 2))
        stats["min"][start:stop] = np.where(valid, block, np.inf).min(axis=(1, 2))
        stats["max"][start:stop] = np.where(valid, block, -np.inf).max(axis=(1, 2))
        return receipts

    receipts = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for block in executor.map(process, range(0, count, 32)):
                receipts.extend(block)
                if len(receipts) % 2048 == 0 or len(receipts) == count:
                    progress = {
                        "positions": len(receipts),
                        "total": count,
                        "seconds": time.monotonic() - start_time,
                    }
                    write_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
    finally:
        for archive in archives:
            archive.close()
    for path, receipt in zip(parts, integrity["files"]):
        status = path.stat()
        if (status.st_size, status.st_mtime_ns) != (receipt["bytes"], receipt["mtime_ns"]):
            raise ValueError("Native source changed during derivation")
    pq.write_table(pa.Table.from_pylist(receipts), output / "index.parquet")
    np.savez_compressed(output / "row_statistics.npz", **stats)
    group.attrs["complete"] = True
    checksum = {
        str(path.relative_to(output)): digest(path)
        for path in sorted((output / "dem40m.zarr").rglob("*"))
        if path.is_file()
    }
    write_json(output / "checksums.json", checksum)
    manifest = {
        "schema": "xuannv.native-dsm-cache.v1",
        "version": "native-dem40m-20261016-v1",
        "complete": True,
        "positions": count,
        "shape": [count, 32, 32],
        "spacing_m": 40,
        "crs": "Per-position canonical UTM CRS from the accepted target index",
        "pixel_centers": "xmin+20+40*col, ymax-20-40*row",
        "vertical_reference": "EGM2008 (EPSG:3855), inherited product definition",
        "units": "m",
        "surface_type": "DSM, not DTM",
        "native_sources": integrity,
        "derivation": {
            "mosaic": "Native geographic cells, first valid value; smallest source spacing; "
            "nearest resampling only where native cell lattices differ",
            "horizontal_resampling": "GDAL average to exact32x32 canonical UTM cells",
            "vertical_conversion": "none",
            "value_clipping": "none",
            "source_point_semantics": "GDAL affine footprints around RasterPixelIsPoint samples",
            "native_information": "GLO-30 nominal1arcsec, latitude-dependent horizontal spacing; "
            "40m is a supervision grid, not a new measurement accuracy claim",
            "source_epoch": "Product acquisition epoch is not asserted to be2020or2021",
        },
        "runtime": {
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "gdal": rasterio.__gdal_version__,
        },
        "index_sha256": digest(output / "index.parquet"),
        "checksums_sha256": digest(output / "checksums.json"),
        "row_statistics_sha256": digest(output / "row_statistics.npz"),
        "official_product_reference": "https://dataspace.copernicus.eu/explore-data/"
        "data-collections/copernicus-contributing-missions/collections-description/COP-DEM",
        "source_code_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("native_dem.py", "target_prepare.py")
        },
        "native_dem_ready": True,
        "full_training_acceptance": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def training_statistics(
    audit: Path, native: Path, split: dict, index: list[dict], scope: str
) -> dict:
    import numpy as np

    key = {
        "allowed": "allowed_train_patch_ids",
        "screen": "screen_patch_ids",
        "expanded": "expanded_patch_ids",
    }[scope]
    lookup = {row["patch_id"]: row["target_row"] for row in index}
    rows = [lookup[patch] for patch in split[key]]
    statistics = {}
    for name, path in [
        ("dem_elevation", native / "row_statistics.npz"),
        *[
            (f"worldcover_{year}", audit / f"worldcover_{year}.row_statistics.npz")
            for year in (2020, 2021)
        ],
    ]:
        with np.load(path) as loaded:
            stats = dict(loaded)
        summary = summarize(stats, rows)
        if name == "dem_elevation":
            summary["valid_fraction"] = summary["valid_pixels"] / (len(rows) * 32 * 32)
        else:
            for field in ("mean", "population_std", "min", "max"):
                summary.pop(field)
        statistics[name] = summary
    return {
        "scope": scope,
        "unique_positions": len(rows),
        "static_positions_counted_once": True,
        "target_rows_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        "normalization_scope": "Only this experiment's actual training positions "
        "and valid DSM 40m pixels",
        "targets": statistics,
        "native_manifest_sha256": digest(native / "manifest.json"),
        "ready_for_engineering": True,
        "formal_training_released": False,
    }


def prepare(args) -> dict:
    import pyarrow.parquet as pq
    import yaml

    from xuannv_embedding.data_process.target_splits import freeze_splits, read_seen_patches

    if args.output.exists():
        raise FileExistsError("Preparation output must be a new directory")
    contract = json.loads((args.audit_root / "target_contract.json").read_text())
    if not contract["data_checks_passed"]:
        raise ValueError("Initial target audit failed")
    target_root = Path(contract["target_root"])
    for name, expected in contract["input_sha256"].items():
        if digest(target_root / name) != expected:
            raise ValueError(f"Published target metadata changed: {name}")
    for source in (
        target_root,
        Path(contract["annual_root"]),
        *[
            Path(row["path"]).parent
            for row in json.loads(args.source_integrity.read_text())["files"]
        ],
    ):
        if args.output.resolve().is_relative_to(source.resolve()):
            raise ValueError("Preparation must not write into source data roots")
    index = sorted(
        pq.read_table(args.audit_root / "target_index.snapshot.parquet").to_pylist(),
        key=lambda row: row["target_row"],
    )
    memberships = pq.read_table(args.audit_root / "memberships.parquet").to_pylist()
    config = yaml.safe_load(args.initial_config.read_text())
    audited = {row["path"]: row["sha256"] for row in contract["annual_manifests"]}
    original = []
    for dataset in config["data"]["datasets"]:
        path = Path(dataset["manifest_path"])
        if ".train." not in path.name or digest(path) != audited.get(str(path)):
            raise ValueError("Original pretraining split does not match the accepted annual data")
        original.append({"path": str(path), "sha256": audited[str(path)]})
    if len(original) != 2:
        raise ValueError("Expected two original pretraining annual datasets")
    seen = read_seen_patches(args.seen_patches)
    split = freeze_splits(index, memberships, seen)
    args.output.mkdir(parents=True)
    (args.output / "source").mkdir()
    for name in ("native_dem.py", "target_splits.py", "target_prepare.py"):
        source = Path(__file__).with_name(name)
        (args.output / "source" / name).write_bytes(source.read_bytes())
    write_json(args.output / "experiment_splits.json", split)
    write_json(
        args.output / "pretraining_split_binding.json",
        {
            "config_path": str(args.initial_config),
            "config_sha256": digest(args.initial_config),
            "original_pretraining_manifests": original,
            "prior_candidate_and_evaluation_pools": [
                {"path": str(path), "sha256": digest(path)} for path in args.seen_patches
            ],
            "seen_pool_unique_positions": len(seen),
        },
    )
    print("Spatial identities frozen; deriving native DSM40m targets", flush=True)
    integrity = json.loads(args.source_integrity.read_text())
    native = native_cache(index, integrity, args.output / "native_dem", workers=args.workers)
    for scope in ("allowed", "screen", "expanded"):
        write_json(
            args.output / f"train_statistics.{scope}.json",
            training_statistics(args.audit_root, args.output / "native_dem", split, index, scope),
        )
    result = {
        "schema": "xuannv.target-preparation.v1",
        "status": "engineering_data_prepared",
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "initial_audit_contract_sha256": digest(args.audit_root / "target_contract.json"),
        "target_manifest_sha256": contract["input_sha256"]["manifest.json"],
        "target_index_sha256": contract["input_sha256"]["index.parquet"],
        "native_manifest_sha256": digest(args.output / "native_dem/manifest.json"),
        "splits_sha256": digest(args.output / "experiment_splits.json"),
        "train_statistics_sha256": {
            scope: digest(args.output / f"train_statistics.{scope}.json")
            for scope in ("allowed", "screen", "expanded")
        },
        "native_positions": native["positions"],
        "paired_train_positions": split["eligible_paired_train_positions"],
        "roles": {key: len(rows) for key, rows in split["roles"].items()},
        "training_started": False,
        "formal_training_released": False,
        "remaining_conditions": [
            "Archived S2 processing baseline/physical offset and compositing provenance",
            "Independent feature registration and genuinely manual5mevaluationtruth",
        ],
    }
    write_json(args.output / "preparation.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--source-integrity", type=Path, required=True)
    parser.add_argument("--initial-config", type=Path, required=True)
    parser.add_argument("--seen-patches", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    print(json.dumps(prepare(args), ensure_ascii=False))
    return 0
