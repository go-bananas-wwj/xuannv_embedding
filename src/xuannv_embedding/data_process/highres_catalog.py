"""Build a spatially sampled archive catalog and a native-grid pilot cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tarfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

VERSION = "observation-pilot-v1"
_PARENT = re.compile(r"(?:candidate|preview)_utm(\d+)n_c(\d+)_r(\d+)_c(\d+)_r(\d+)")
_OWNER = re.compile(r"ownerfix_epsg(\d+)_c(\d+)_r(\d+)")
_DATES = re.compile(r"(?<!\d)((?:19|20)\d{6})(?!\d)")


@dataclass(frozen=True)
class ArchiveRow:
    archive: str
    patch_id: str
    parent_key: str


def parent_key(patch_id: str) -> str:
    match = _PARENT.fullmatch(patch_id)
    if match:
        zone, column, row, subcolumn, subrow = map(int, match.groups())
        if not 1 <= zone <= 60 or not 0 <= subcolumn < 10 or not 0 <= subrow < 10:
            raise ValueError(f"Invalid parent grid: {patch_id}")
        return f"{32600 + zone}:{column * 10 + subcolumn}:{row * 10 + subrow}"
    match = _OWNER.fullmatch(patch_id)
    if match:
        return ":".join(str(int(value)) for value in match.groups())
    raise ValueError(f"Unknown patch identity: {patch_id}")


def safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or "://" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return path.as_posix()


def read_archive_index(path: Path) -> list[ArchiveRow]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["archive", "patchid"]:
            raise ValueError("Expected archive/patchid TSV header")
        for record in reader:
            if record["patchid"] in {"manifest.csv", "manifest_generic.csv"}:
                continue
            archive = safe_member(record["archive"])
            patch_id = record["patchid"]
            rows.append(ArchiveRow(archive, patch_id, parent_key(patch_id)))
    if not rows or len({row.parent_key for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate parent index")
    return rows


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_parents(
    rows: list[ArchiveRow], count: int, materialize_count: int, seed: int
) -> list[dict]:
    from pyproj import Transformer

    from xuannv_embedding.data_process.observation_raster import parent_geometry

    cells: dict[str, list[dict]] = defaultdict(list)
    transforms = {}
    for row in rows:
        epsg, bounds = parent_geometry(row.parent_key)
        if epsg not in transforms:
            transforms[epsg] = Transformer.from_crs(epsg, 4326, always_xy=True)
        longitude, latitude = transforms[epsg].transform(bounds[0] + 640, bounds[1] + 640)
        cell = f"{math.floor(longitude / 2)}:{math.floor(latitude / 2)}"
        split_bucket = int(_rank(seed, "split:" + cell)[:8], 16) % 10
        split = "train" if split_bucket < 8 else "validation" if split_bucket == 8 else "test"
        cells[cell].append(
            {
                **asdict(row),
                "longitude": longitude,
                "latitude": latitude,
                "spatial_block": cell,
                "split": split,
            }
        )
    queues = {
        cell: deque(sorted(records, key=lambda item: _rank(seed, item["parent_key"])))
        for cell, records in cells.items()
    }
    active = deque(sorted(queues, key=lambda cell: _rank(seed, "cell:" + cell)))
    selected = []
    while active and len(selected) < count:
        cell = active.popleft()
        record = queues[cell].popleft()
        record["selection_rank"] = len(selected)
        record["materialize"] = len(selected) < materialize_count
        selected.append(record)
        if queues[cell]:
            active.append(cell)
    return selected


def observation_record(parent: dict, name: str) -> dict[str, Any]:
    parts = PurePosixPath(safe_member(name)).parts
    filename = parts[-1]
    tokens = sorted(set(_DATES.findall(filename)))
    dates, issues = [], []
    for token in tokens:
        try:
            dates.append(datetime.strptime(token, "%Y%m%d").date().isoformat())
        except ValueError:
            issues.append("invalid_filename_date")
    date = dates[0] if len(dates) == 1 and not issues else None
    if len(dates) != 1:
        issues.append("ambiguous_or_missing_date")
    declared = filename[:4] if re.match(r"^20\d{2}_", filename) else None
    if declared and any(value[:4] != declared for value in dates):
        issues.append("declared_year_conflict")
    if date and date[:4] not in {"2020", "2021"}:
        issues.append("outside_pilot_years")
    product = "UNKNOWN"
    for candidate in ("PAN", "MUX", "WFV", "CCD", "NAD", "FWD", "BWD", "IMG"):
        if re.search(rf"(?:^|[_-]){candidate}(?:[_\-.]|$)", filename.upper()):
            product = candidate
            break
    if product == "UNKNOWN":
        issues.append("unknown_product")
    spacing = re.search(r"(?:^|_)(\d+(?:\.\d+)?)m(?:_|\.)", filename)
    return {
        "observation_id": hashlib.sha256(f"{parent['archive']}:{name}".encode()).hexdigest(),
        "parent_key": parent["parent_key"],
        "patch_id": parent["patch_id"],
        "archive": parent["archive"],
        "archive_member": name,
        "split": parent["split"],
        "platform": parts[1] if len(parts) >= 3 else "UNKNOWN",
        "product_type": product,
        "acquisition_time": date,
        "date_candidates": dates,
        "time_precision": "day" if date else "unknown",
        "date_source": "filename_not_manifest_verified",
        "declared_year": declared,
        "declared_grid_spacing_m": float(spacing.group(1)) if spacing else None,
        "issues": issues,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _scan_archive(task: tuple) -> dict:
    from rasterio.errors import RasterioError

    from xuannv_embedding.data_process.observation_raster import inspect_raster, sha256_file

    archive_root, stage, archive_name, parents = task
    stage = Path(stage)
    output = stage / "catalog_parts" / (Path(archive_name).name + ".jsonl")
    summary_path = output.with_suffix(".summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if sha256_file(output) != summary["sha256"]:
            raise ValueError(f"Changed catalog part: {output}")
        for relative, digest in summary["artifacts"].items():
            if sha256_file(stage / relative) != digest:
                raise ValueError(f"Changed cache artifact: {relative}")
        return summary
    selected = {parent["patch_id"]: parent for parent in parents}
    records = []
    seen = set()
    with tarfile.open(Path(archive_root) / archive_name, mode="r|gz") as bundle:
        for member in bundle:
            name = safe_member(member.name)
            if not name.lower().endswith((".tif", ".tiff")):
                continue
            parent = selected.get(name.split("/")[0])
            if parent is None:
                continue
            if not member.isfile() or name in seen:
                raise ValueError(f"Non-regular or duplicate TIFF member: {name}")
            seen.add(name)
            record = observation_record(parent, name)
            eligible = parent["materialize"] and not record["issues"]
            relative = (
                Path("highres")
                / ("parent_" + parent["parent_key"])
                / "/".join(PurePosixPath(name).parts[1:])
            )
            destination = stage / relative if eligible else None
            try:
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("Unreadable TIFF member")
                payload = stream.read()
                if len(payload) != member.size:
                    raise ValueError("Truncated TIFF payload")
                record.update(
                    inspect_raster(
                        payload, parent["parent_key"], pixels=eligible, destination=destination
                    )
                )
                if not record["grid_matches_parent"]:
                    record["issues"].append("grid_mismatch")
                if eligible and record.get("valid_fraction") == 0:
                    record["issues"].append("all_pixels_invalid")
                if record.pop("materialized", False):
                    record["materialized_path"] = relative.as_posix()
                    record["mask_path"] = relative.with_name(relative.stem + "_mask.tif").as_posix()
                record["source_signature"] = (
                    f"{record['platform']}_{record['product_type']}_c{record['channels']}"
                )
                record["status"] = "quarantine" if record["issues"] else "indexed_semantics_pending"
            except RasterioError as error:
                record.update(status="quarantine", read_error=type(error).__name__)
                record["issues"].append("raster_read_error")
            records.append(record)
    write_jsonl(output, records)
    summary = {
        "archive": archive_name,
        "observations": len(records),
        "materialized": sum("materialized_path" in record for record in records),
        "sha256": sha256_file(output),
        "artifacts": {
            record[field]: sha256_file(stage / record[field])
            for record in records
            if "materialized_path" in record
            for field in ("materialized_path", "mask_path")
        },
    }
    write_json(summary_path, summary)
    return summary


def build_catalog(
    *,
    archive_root: Path,
    index_path: Path,
    output_root: Path,
    parent_limit: int = 1024,
    materialize_parents: int = 128,
    seed: int = 42,
    workers: int = 2,
    resume: bool = False,
    lowres_root: Path | None = None,
) -> dict:
    from xuannv_embedding.data_process.observation_raster import sha256_file

    if parent_limit < 1 or not 0 <= materialize_parents <= parent_limit or workers < 1:
        raise ValueError("Invalid parent/materialization/worker counts")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    rows = read_archive_index(index_path)
    if parent_limit > len(rows):
        raise ValueError("Requested more parents than the archive index contains")
    parents = select_parents(rows, parent_limit, materialize_parents, seed)
    archives = sorted({parent["archive"] for parent in parents})
    inputs = []
    for name in archives:
        path = archive_root / name
        inputs.append(
            {"name": name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        )
    lowres_files = []
    if lowres_root is not None:
        for source in ("pc-s2", "pc-s1", "pc-ls"):
            for path in sorted((lowres_root / source).rglob("*.zip")):
                lowres_files.append(
                    {
                        "name": path.relative_to(lowres_root).as_posix(),
                        "bytes": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                    }
                )
    fingerprint = {
        "version": VERSION,
        "index_sha256": sha256_file(index_path),
        "archives": inputs,
        "archive_root": str(archive_root.resolve()),
        "parent_limit": parent_limit,
        "materialize_parents": materialize_parents,
        "seed": seed,
        "lowres_root": str(lowres_root.resolve()) if lowres_root else None,
        "lowres_files": lowres_files,
        "code": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("highres_catalog.py", "observation_raster.py", "pilot_cache.py")
        },
    }
    stage = output_root.with_name(output_root.name + ".partial")
    if stage.exists():
        if not resume or json.loads((stage / "fingerprint.json").read_text()) != fingerprint:
            raise ValueError("Existing partial output requires --resume with unchanged inputs/code")
    else:
        stage.mkdir(parents=True)
        (stage / "catalog_parts").mkdir()
        write_json(stage / "fingerprint.json", fingerprint)
        write_jsonl(stage / "selected_parents.jsonl", parents)
    groups = defaultdict(list)
    for parent in parents:
        groups[parent["archive"]].append(parent)
    tasks = [(str(archive_root), str(stage), name, groups[name]) for name in archives]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, future in enumerate(
            as_completed([pool.submit(_scan_archive, task) for task in tasks]), 1
        ):
            result = future.result()
            print(
                f"Highres {index}/{len(tasks)}: {result['archive']}, "
                f"observations={result['observations']}, cached={result['materialized']}",
                flush=True,
            )
    records = []
    for path in sorted((stage / "catalog_parts").glob("*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    records.sort(key=lambda record: (record["parent_key"], record["archive_member"]))
    write_jsonl(stage / "observations.jsonl", records)
    summary = {
        "version": VERSION,
        "parent_count": len(parents),
        "observation_count": len(records),
        "parents_without_observations": sorted(
            {parent["parent_key"] for parent in parents}
            - {record["parent_key"] for record in records}
        ),
        "materialized_tiffs": sum("materialized_path" in record for record in records),
        "cached_parents": len(
            {record["parent_key"] for record in records if "materialized_path" in record}
        ),
        "platforms": dict(Counter(record["platform"] for record in records)),
        "products": dict(Counter(record["product_type"] for record in records)),
        "issues": dict(Counter(issue for record in records for issue in record["issues"])),
        "selection": "seeded_round_robin_2_degree_cells; no_ecology_or_landcover_stratification",
        "split_policy": "2_degree_block_hash_80_10_10; no_boundary_buffer_or_scene_grouping",
        "p0_split_counts": dict(
            Counter(parent["split"] for parent in parents if parent["materialize"])
        ),
        "scientific_training_ready": False,
    }
    if lowres_root is not None:
        from xuannv_embedding.data_process.pilot_cache import prepare_lowres

        summary["lowres"] = prepare_lowres(stage, lowres_root, parents, records)
    write_json(stage / "summary.json", summary)
    outputs = {
        path.relative_to(stage).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    }
    write_json(stage / "checksums.json", outputs)
    stage.rename(output_root)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv data catalog")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lowres-root", type=Path)
    parser.add_argument("--parent-limit", type=int, default=1024)
    parser.add_argument("--materialize-parents", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    summary = build_catalog(
        archive_root=args.archive_root,
        index_path=args.index,
        output_root=args.output_root,
        parent_limit=args.parent_limit,
        materialize_parents=args.materialize_parents,
        seed=args.seed,
        workers=args.workers,
        resume=args.resume,
        lowres_root=args.lowres_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
