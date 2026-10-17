"""Derive local multi-source supervision on frozen training positions only."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from xuannv_embedding.data.local_target_cache import target_row_sha
from xuannv_embedding.data.target_contract import (
    LOCAL_TASKS,
    MULTISOURCE_TARGET_SCHEMA,
    OSM_CHANNELS,
    STRUCTURE_CHANNELS,
    TARGET_TASKS,
    target_shape,
    training_positions,
)
from xuannv_embedding.data.target_transforms import (
    aggregate_continuous,
    aggregate_structure,
    clcd_fractions,
    dsm_slope,
    osm_evidence,
)
from xuannv_embedding.data_process.target_audit import digest, verify_archive, write_json


def read_rows(array, rows):
    """Read each touched source chunk once, retaining requested position order."""
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty((len(rows), *array.shape[1:]), dtype=array.dtype)
    chunk = array.chunks[0]
    for index in np.unique(rows // chunk):
        selected = rows // chunk == index
        start = int(index * chunk)
        out[selected] = array[start : min(start + chunk, array.shape[0])][rows[selected] - start]
    return out


def summarize_batch(data, task):
    values, valid, weight = data["values"], data["valid"], data["weight"]
    result = {
        "valid_units": int(valid.sum()),
        "weighted_support": float(weight.sum(dtype=np.float64)),
        "positions_with_valid": int(valid.reshape(len(valid), -1).any(1).sum()),
    }
    kind = TARGET_TASKS[task][2]
    if kind == "binary":
        axes = (0, 2, 3)
        result["positive_mass"] = (values * weight).sum(axes, dtype=np.float64).tolist()
        result["negative_mass"] = ((1 - values) * weight).sum(axes, dtype=np.float64).tolist()
    elif kind == "soft_categorical":
        result["class_mass"] = (values * weight[:, None]).sum((0, 2, 3), dtype=np.float64).tolist()
    else:
        selected = values[valid].astype(np.float64)
        result.update(sum=float(selected.sum()), sumsq=float((selected**2).sum()))
    return result


def prepare_local_targets(
    parent: Path, target_root: Path, output: Path, *, workers=4, scope="screen"
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zarr
    from numcodecs import Blosc

    from xuannv_embedding.data.targets import AnnualTargetReader

    if output.exists():
        raise FileExistsError("Multi-source preparation must use a new directory")
    if not 1 <= workers <= 8:
        raise ValueError("Preparation workers must be between1and8")
    if output.resolve().is_relative_to(target_root.resolve()) or output.resolve().is_relative_to(
        parent.resolve()
    ):
        raise ValueError("Derived supervision must not overwrite its source roots")
    parent_sha = digest(parent / "preparation.json")
    with AnnualTargetReader(
        target_root, parent, preparation_sha256=parent_sha, scope=scope
    ) as reader:
        source_rows, native_hashes = dict(reader.rows), dict(reader.row_hashes)
    split = json.loads((parent / "experiment_splits.json").read_text())
    ids = sorted(training_positions(split, scope), key=source_rows.__getitem__)
    source_index = np.array([source_rows[p] for p in ids])
    manifest = json.loads((target_root / "manifest.json").read_text())
    names = {
        f"static/{task}_{year}.zarr.zip"
        for task in ("clcd", "nightlights")
        for year in (2020, 2021)
    }
    names |= {"osm/2020.zarr.zip", "osm/2021.zarr.zip", "osm/reliable_negative.zarr.zip"}
    entries = [r for r in manifest["files"] if r["path"] in names]
    if len(entries) != 7:
        raise ValueError("Multi-source preparation requires all7annual/OSM archives")

    def verify(entry):
        before = (target_root / entry["path"]).stat()
        result = verify_archive(target_root, entry)
        after = (target_root / entry["path"]).stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source archive changed during integrity verification")
        result["mtime_ns"] = before.st_mtime_ns
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(verify, entries))
    if not all(r["passed"] for r in receipts):
        raise ValueError("Multi-source source archive integrity mismatch")
    output.mkdir(parents=True)
    cache = output / "local_targets"
    cache.mkdir()
    (output / "native_dem").symlink_to((parent / "native_dem").resolve(), target_is_directory=True)
    for name in ("experiment_splits.json", "pretraining_split_binding.json"):
        (output / name).write_bytes((parent / name).read_bytes())
    for other_scope in {"screen", "expanded", "allowed"} - {scope}:
        (output / f"train_statistics.{other_scope}.json").symlink_to(
            (parent / f"train_statistics.{other_scope}.json").resolve()
        )
    group = zarr.open_group(str(cache / "targets.zarr"), mode="w")
    group.attrs.update(complete=False, schema=MULTISOURCE_TARGET_SCHEMA)
    arrays = {}
    for year in (2020, 2021):
        for task in LOCAL_TASKS:
            for field in ("values", "valid", "weight"):
                shape = target_shape(task)
                if field != "values" and TARGET_TASKS[task][2] == "soft_categorical":
                    shape = shape[1:]
                arrays[year, task, field] = group.create_dataset(
                    f"{year}/{task}/{field}",
                    shape=(len(ids), *shape),
                    chunks=(16, *shape),
                    dtype=bool if field == "valid" else np.float32,
                    compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
                )
    local, opened = threading.local(), []

    def source_array(filename, key):
        if not hasattr(local, "groups"):
            local.groups, local.arrays = {}, {}
        if filename not in local.groups:
            store = zarr.ZipStore(str(target_root / filename), mode="r")
            opened.append(store)
            local.groups[filename] = zarr.open_group(store, mode="r")
        if (filename, key) not in local.arrays:
            array = local.groups[filename][key]
            if array.shape[0] != len(source_rows):
                raise ValueError("Source row count changed")
            local.arrays[filename, key] = array
        return local.arrays[filename, key]

    def process(job):
        year, start = job
        stop = min(len(ids), start + 16)
        rows = source_index[start:stop]
        if not hasattr(local, "native"):
            local.native = zarr.open_group(str(parent / "native_dem/dem40m.zarr"), mode="r")
        dem = read_rows(local.native["values"], rows)
        valid = read_rows(local.native["valid"], rows)
        import hashlib

        for i, patch in enumerate(ids[start:stop]):
            if (
                hashlib.sha256(dem[i].tobytes() + valid[i].tobytes()).hexdigest()
                != native_hashes[patch]
            ):
                raise ValueError("Native DSM row changed before slope derivation")
        derived = {"dem_slope": dsm_slope(dem, valid)}
        for name, factor in (("clcd", 4), ("nightlights", 64)):
            file = f"static/{name}_{year}.zarr.zip"
            x = read_rows(source_array(file, f"targets/{name}_{year}"), rows)
            mask = read_rows(source_array(file, f"valid_masks/{name}_{year}"), rows)
            derived[name] = (
                clcd_fractions(x, mask) if name == "clcd" else aggregate_continuous(x, mask, factor)
            )
        file, neg = f"osm/{year}.zarr.zip", "osm/reliable_negative.zarr.zip"

        def fields(channels, structure=False):
            prefix = f"{year}/" + ("structure_2p5m/" if structure else "")
            data = [
                np.stack(
                    [
                        read_rows(source_array(file, prefix + field + "/" + c), rows)
                        for c in channels
                    ],
                    1,
                )
                for field in ("targets", "states", "confidence", "source_bits")
            ]
            negative = [
                np.stack(
                    [read_rows(source_array(neg, f"{year}/{field}/{c}"), rows) for c in channels], 1
                )
                for field in ("states", "confidence")
            ]
            if structure:
                negative = [a.repeat(4, -2).repeat(4, -1) for a in negative]
            return osm_evidence(*data, *negative)

        derived["osm_semantic"] = fields(OSM_CHANNELS)
        derived["osm_structure"] = aggregate_structure(fields(STRUCTURE_CHANNELS, True))
        derived = {
            task: {
                field: np.ascontiguousarray(value, dtype=bool if field == "valid" else np.float32)
                for field, value in data.items()
            }
            for task, data in derived.items()
        }
        row_receipts = []
        for i, patch in enumerate(ids[start:stop]):
            record = {
                "patch_id": patch,
                "target_row": source_rows[patch],
                "cache_row": start + i,
                "year": year,
            }
            for task, data in derived.items():
                single = {key: np.ascontiguousarray(value[i]) for key, value in data.items()}
                record[f"{task}_sha256"] = target_row_sha(single)
                record[f"{task}_valid_units"] = int(single["valid"].sum())
                if TARGET_TASKS[task][2] == "binary":
                    record[f"{task}_positive_mass"] = (
                        (single["values"] * single["weight"]).sum((1, 2), dtype=np.float64).tolist()
                    )
                    record[f"{task}_negative_mass"] = (
                        ((1 - single["values"]) * single["weight"])
                        .sum((1, 2), dtype=np.float64)
                        .tolist()
                    )
            row_receipts.append(record)
        for task, data in derived.items():
            for field, values in data.items():
                arrays[year, task, field][start:stop] = values
        return (
            year,
            row_receipts,
            {task: summarize_batch(data, task) for task, data in derived.items()},
        )

    row_receipts, statistics = [], {year: {} for year in (2020, 2021)}
    try:
        jobs = [(year, start) for year in (2020, 2021) for start in range(0, len(ids), 16)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (year, rows, summary) in enumerate(pool.map(process, jobs)):
                row_receipts.extend(rows)
                for task, counts in summary.items():
                    total = statistics[year].setdefault(task, {})
                    for key, value in counts.items():
                        total[key] = (np.asarray(total.get(key, 0)) + value).tolist()
                if (index + 1) % 64 == 0 or index + 1 == len(jobs):
                    print(
                        json.dumps({"local_blocks": index + 1, "total_blocks": len(jobs)}),
                        flush=True,
                    )
    finally:
        for store in opened:
            store.close()
    for entry in receipts:
        status = (target_root / entry["path"]).stat()
        if (status.st_size, status.st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
            raise ValueError("Source archive changed during target derivation")
    if digest(parent / "preparation.json") != parent_sha:
        raise ValueError("Parent preparation changed during target derivation")
    pq.write_table(pa.Table.from_pylist(row_receipts), cache / "index.parquet")
    group.attrs["complete"] = True
    checks = {
        str(p.relative_to(cache)): digest(p)
        for p in sorted((cache / "targets.zarr").rglob("*"))
        if p.is_file()
    }
    write_json(cache / "checksums.json", checks)
    derivation = {
        "dem_slope": "Degrees from central40m DSM differences; center and4neighbors valid; "
        "patch edges omitted; static source counted once.",
        "clcd": "Published nearest-resampled10m masks/classes aggregated into9class fractions "
        "on40m cells; coverage>=0.5; confidence=valid fraction.",
        "nightlights": "Published continuous values averaged on640m cells; "
        "coverage>=0.5; no clipping, log or restandardization.",
        "osm_semantic": "4historical channels at10m; unknown ignored; confidence/255; "
        "rule negatives fill unknown only, positives take precedence.",
        "osm_structure": "3historical2.5m channels confidence-aggregated to5m; "
        "same10m rule-negative interiors inherited; unknowns never filled with background.",
        "binary_loss": "Per-channel balanced positive/negative soft target mass; "
        "global valid support normalization across ranks.",
    }
    local_manifest = {
        "schema": MULTISOURCE_TARGET_SCHEMA,
        "complete": True,
        "scope": scope,
        "positions": len(ids),
        "years": [2020, 2021],
        "tasks": list(LOCAL_TASKS),
        "source_archives": receipts,
        "source_parent_preparation_sha256": parent_sha,
        "index_sha256": digest(cache / "index.parquet"),
        "checksums_sha256": digest(cache / "checksums.json"),
        "derivation": derivation,
        "osm_channels": list(OSM_CHANNELS),
        "structure_channels": list(STRUCTURE_CHANNELS),
        "provenance_receipts": [
            {"path": str(p), "sha256": digest(p)}
            for p in sorted((target_root / "provenance").glob("target_geometry_*full.json"))
        ],
        "source_code_sha256": {
            str(p): digest(p)
            for p in [Path(__file__), Path(__file__).parents[1] / "data/target_transforms.py"]
        },
        "formal_training_released": False,
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "runtime": {"numpy": np.__version__, "zarr": zarr.__version__},
    }
    write_json(cache / "manifest.json", local_manifest)
    stats = json.loads((parent / f"train_statistics.{scope}.json").read_text())
    stats["local_target_statistics_by_year"] = statistics
    stats["local_target_statistics_note"] = (
        "Descriptive training-scope counts; static slope is duplicated by year in cache, "
        "unique statistics are year2020 only. Existing DSM normalization retained; "
        "nightlights preserved in stored units."
    )
    write_json(output / f"train_statistics.{scope}.json", stats)
    preparation = json.loads((parent / "preparation.json").read_text())
    preparation.update(
        schema="xuannv.target-preparation.v2",
        target_schema=MULTISOURCE_TARGET_SCHEMA,
        parent_preparation={"path": str(parent.resolve()), "sha256": parent_sha},
        local_manifest_sha256=digest(cache / "manifest.json"),
        training_started=False,
        formal_training_released=False,
        completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    )
    preparation["train_statistics_sha256"][scope] = digest(
        output / f"train_statistics.{scope}.json"
    )
    write_json(output / "preparation.json", preparation)
    return preparation


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="xuannv data targets-local")
    for name in ("prepared-root", "target-root", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--scope", choices=["screen", "expanded", "allowed"], default="screen")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    result = prepare_local_targets(
        args.prepared_root, args.target_root, args.output, workers=args.workers, scope=args.scope
    )
    print(
        json.dumps(
            {
                "prepared_scope": args.scope,
                "formal_training_released": result["formal_training_released"],
            }
        )
    )
    return 0
