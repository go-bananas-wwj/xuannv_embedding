"""Read-only acceptance of published target sidecars against annual manifests.

This command creates evidence, not a training dataset or a training authorization.
All scans follow storage chunks; no image or target source is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

WC_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not relative or Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes input root: {relative}")
    return path


def verify_archive(root: Path, entry: dict) -> dict:
    path = local_path(root, entry["path"])
    actual = digest(path)
    return {
        **entry,
        "actual_bytes": path.stat().st_size,
        "actual_sha256": actual,
        "passed": actual == entry["sha256"] and path.stat().st_size == entry["bytes"],
    }


def scan_static(root: Path, filename: str, declarations: list[dict], patch_ids: list[str]) -> dict:
    """Read every static value/mask chunk, retaining sufficient per-position statistics."""
    import numpy as np
    import zarr

    name = Path(filename).name.removesuffix(".zarr.zip")
    codes = WC_CODES if name.startswith("worldcover_") else tuple(range(1, 10))
    categorical = name.startswith(("worldcover_", "clcd_"))
    count = len(patch_ids)
    result = {key: np.zeros(count, dtype=np.float64) for key in ("sum", "sumsq")}
    for key in ("valid", "nonfinite", "illegal"):
        result[key] = np.zeros(count, dtype=np.int64)
    result["min"] = np.full(count, np.inf)
    result["max"] = np.full(count, -np.inf)
    if categorical:
        result["histogram"] = np.zeros((count, len(codes)), dtype=np.int64)
    with zarr.ZipStore(str(local_path(root, filename)), mode="r") as store:
        group = zarr.open_group(store, mode="r")
        if group.attrs.get("patch_ids") != patch_ids:
            raise ValueError(f"Archive row identities differ from target index: {filename}")
        for declaration in declarations:
            array = group[declaration["array"]]
            if (
                list(array.shape) != declaration["shape"]
                or list(array.chunks) != declaration["chunks"]
                or str(array.dtype) != declaration["dtype"]
            ):
                raise ValueError(f"Array contract mismatch: {filename}/{declaration['array']}")
        values, masks = group[f"targets/{name}"], group[f"valid_masks/{name}"]
        if values.shape != (count, 128, 128) or masks.shape != values.shape:
            raise ValueError(f"Unsupported static shape: {filename}")
        if masks.dtype != np.dtype(bool):
            raise ValueError(f"Validity mask must be boolean: {filename}")
        for start in range(0, count, values.chunks[0]):
            stop = min(count, start + values.chunks[0])
            value = values[start:stop].astype(np.float64)
            mask = masks[start:stop]
            finite = np.isfinite(value)
            valid = mask & finite
            result["valid"][start:stop] = valid.sum(axis=(1, 2))
            result["nonfinite"][start:stop] = (mask & ~finite).sum(axis=(1, 2))
            clean = np.where(valid, value, 0)
            result["sum"][start:stop] = clean.sum(axis=(1, 2))
            result["sumsq"][start:stop] = (clean * clean).sum(axis=(1, 2))
            result["min"][start:stop] = np.where(valid, value, np.inf).min(axis=(1, 2))
            result["max"][start:stop] = np.where(valid, value, -np.inf).max(axis=(1, 2))
            if categorical:
                legal = np.isin(value, codes)
                result["illegal"][start:stop] = (valid & ~legal).sum(axis=(1, 2))
                for column, code in enumerate(codes):
                    result["histogram"][start:stop, column] = ((value == code) & valid).sum(
                        axis=(1, 2)
                    )
            elif name == "dem_slope":
                result["illegal"][start:stop] = (valid & ((value < 0) | (value > 90))).sum(
                    axis=(1, 2)
                )
    result["codes"] = np.asarray(codes if categorical else (), dtype=np.int16)
    return result


def summarize(stats: dict, rows: Any) -> dict:
    import numpy as np

    rows = np.asarray(rows, dtype=np.int64)
    counts = stats["valid"][rows]
    count = int(counts.sum())
    total = float(stats["sum"][rows].sum())
    mean = total / count if count else None
    variance = max(0.0, float(stats["sumsq"][rows].sum()) / count - mean**2) if count else None
    result = {
        "positions": len(rows),
        "positions_with_valid_pixels": int((counts > 0).sum()),
        "positions_without_valid_pixels": int((counts == 0).sum()),
        "valid_pixels": count,
        "valid_fraction": count / (len(rows) * 128 * 128) if len(rows) else None,
        "nonfinite_valid_pixels": int(stats["nonfinite"][rows].sum()),
        "illegal_valid_pixels": int(stats["illegal"][rows].sum()),
        "mean": mean,
        "population_std": variance**0.5 if variance is not None else None,
        "min": float(stats["min"][rows].min()) if count else None,
        "max": float(stats["max"][rows].max()) if count else None,
    }
    if "histogram" in stats:
        histogram = stats["histogram"][rows]
        result["class_pixels"] = dict(zip(map(str, stats["codes"]), histogram.sum(axis=0).tolist()))
        result["class_positions"] = dict(
            zip(map(str, stats["codes"]), (histogram > 0).sum(axis=0).tolist())
        )
    return result


def read_memberships(annual_root: Path, index_rows: dict[str, int]) -> tuple:
    """Verify raw manifest bytes and retain compact joins plus header audit samples."""
    records, receipts, samples = [], [], defaultdict(list)
    for path in sorted(annual_root.glob("*.manifest.jsonl")):
        year_text, split, *_ = path.name.split(".")
        year = int(year_text)
        meta_path = Path(str(path) + ".meta.json")
        meta = json.loads(meta_path.read_text())
        if meta["months"] != [f"{year}-{month:02d}" for month in range(1, 13)]:
            raise ValueError(f"Year/month mismatch: {path}")
        hasher, seen = hashlib.sha256(), set()
        with path.open("rb") as handle:
            for line in handle:
                hasher.update(line)
                if not line.strip():
                    continue
                record = json.loads(line)
                patch = record["patch_id"]
                if patch in seen or patch not in index_rows:
                    raise ValueError(f"Duplicate or unmatched patch: {path.name}/{patch}")
                seen.add(patch)
                if record["grid"]["parent_key"] != patch.removeprefix("parent_"):
                    raise ValueError(f"Manifest parent key mismatch: {patch}")
                if record.get("provenance", {}).get("year", year) != year:
                    raise ValueError(f"Manifest year mismatch: {patch}")
                quality = record["quality"]
                records.append(
                    {
                        "year": year,
                        "split": split,
                        "patch_id": patch,
                        "target_row": index_rows[patch],
                        "highres_supported": bool(quality["highres_supported"]),
                        "ms5m_supported": bool(quality["ms5m_supported"]),
                        "pan2m_supported": bool(quality["pan2m_supported"]),
                    }
                )
                for source, paths in record["sources"].items():
                    if not paths:
                        continue
                    rank = hashlib.sha256(f"{patch}/{source}".encode()).hexdigest()
                    key = (year, split, source)
                    candidates = samples[key]
                    if len(candidates) < 4 or rank < candidates[-1]["rank"]:
                        candidates.append(
                            {"rank": rank, "patch_id": patch, "source": source, "path": paths[0]}
                        )
                        candidates.sort(key=lambda item: item["rank"])
                        del candidates[4:]
        if hasher.hexdigest() != meta["sha256"] or len(seen) != meta["record_count"]:
            raise ValueError(f"Manifest checksum/count mismatch: {path}")
        receipts.append(
            {
                "path": str(path),
                "sha256": hasher.hexdigest(),
                "records": len(seen),
                "sidecar_sha256": digest(meta_path),
            }
        )
    if not records:
        raise ValueError("No annual manifests found")
    return records, receipts, samples


def audit_geometry(index: list[dict], samples: dict, data_root: Path) -> dict:
    import numpy as np
    import rasterio

    errors, headers = [], []
    for row in index:
        epsg, col, grid_row = map(int, row["patch_id"].removeprefix("parent_").split(":"))
        expected = np.array([col, grid_row, col + 1, grid_row + 1]) * 1280
        if row["grid_epsg"] != epsg or not np.allclose(
            row["utm_bounds"], expected, rtol=0, atol=0.001
        ):
            errors.append(row["patch_id"])
    lookup = {row["patch_id"]: row for row in index}
    for key, candidates in sorted(samples.items()):
        for sample in candidates:
            row = lookup[sample["patch_id"]]
            with rasterio.open(local_path(data_root, sample["path"])) as raster:
                error = float(np.max(np.abs(np.array(raster.bounds) - row["utm_bounds"])))
                passed = (
                    raster.crs is not None
                    and raster.crs.to_epsg() == row["grid_epsg"]
                    and error <= 0.001
                    and raster.transform.a > 0
                    and raster.transform.e < 0
                    and abs(raster.transform.b) < 1e-9
                    and abs(raster.transform.d) < 1e-9
                )
                headers.append(
                    {
                        **sample,
                        "year": key[0],
                        "split": key[1],
                        "shape": list(raster.shape),
                        "crs": str(raster.crs),
                        "transform": list(raster.transform)[:6],
                        "bounds_max_error_m": error,
                        "passed": passed,
                        "bands": raster.count,
                        "descriptions": list(raster.descriptions),
                        "scales": list(raster.scales),
                        "offsets": list(raster.offsets),
                    }
                )
    return {
        "index_positions_checked": len(index),
        "index_geometry_errors": errors,
        "header_samples": headers,
        "passed": not errors and all(row["passed"] for row in headers),
        "scope": "Full canonical index bounds; deterministic sampled image headers. "
        "Header agreement does not prove feature registration or native target resampling.",
        "target_grid": {
            "shape": [128, 128],
            "spacing_m": 10,
            "transform": "[10,0,xmin,0,-10,ymax]; centers at half-pixel offsets",
        },
    }


def audit_spatial_splits(index: list[dict], memberships: list[dict]) -> dict:
    """Distinguish position separation from geographic block separation across years."""
    by_row = {row["target_row"]: row for row in index}
    positions = defaultdict(set)
    for record in memberships:
        positions[record["split"]].add(record["target_row"])
    report = {}
    for field in ("macro_id", "spatial_component_id"):
        blocks = {split: {by_row[row][field] for row in rows} for split, rows in positions.items()}
        pairs = {}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            shared = blocks.get(left, set()) & blocks.get(right, set())
            pairs[f"{left}/{right}"] = {
                "shared_blocks": sorted(shared),
                "right_positions_in_shared_blocks": sorted(
                    by_row[row]["patch_id"]
                    for row in positions[right]
                    if by_row[row][field] in shared
                ),
            }
        report[field] = pairs
    return {
        "grouping": "Same position across both years; two independent block definitions",
        "blocks": report,
        "independent_spatial_confirmation_ready": False,
        "required_action": "Exclude train-overlapping blocks and previously scored positions; "
        "freeze new development/confirmation identities and nearest train distances "
        "before training.",
    }


def run_audit(
    target_root: Path, annual_root: Path, data_root: Path, output: Path, workers: int
) -> dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zarr

    for root in (target_root, annual_root, data_root):
        if output.resolve().is_relative_to(root.resolve()):
            raise ValueError("Evidence output must be outside read-only data roots")
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    source = Path(__file__)
    (output / "audit_source.snapshot.py").write_bytes(source.read_bytes())
    write_json(
        output / "code_provenance.json",
        {
            "module": "xuannv_embedding.data_process.target_audit",
            "module_sha256": digest(source),
            "execution_source_snapshot": "audit_source.snapshot.py",
            "scope": "This audit module only; not model training or historic experiment provenance",
        },
    )
    manifest = json.loads((target_root / "manifest.json").read_text())
    if manifest["schema"] != "xuannv.static-target-publication.v1":
        raise ValueError("Unsupported target publication schema")
    table = pq.read_table(local_path(target_root, manifest["index"]))
    index = sorted(table.to_pylist(), key=lambda row: row["target_row"])
    patch_ids = [row["patch_id"] for row in index]
    if (
        len(index) != manifest["patches"]
        or len(set(patch_ids)) != len(index)
        or [row["target_row"] for row in index] != list(range(len(index)))
    ):
        raise ValueError("Target index identities or row permutation invalid")
    (output / "target_manifest.snapshot.json").write_bytes(
        (target_root / "manifest.json").read_bytes()
    )
    pq.write_table(table, output / "target_index.snapshot.parquet")
    print("Verifying all published archive SHA-256 values", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        receipts = list(
            executor.map(lambda entry: verify_archive(target_root, entry), manifest["files"])
        )
    write_json(output / "archive_integrity.json", receipts)
    if not all(row["passed"] for row in receipts):
        raise ValueError("Archive integrity failed; see archive_integrity.json")
    print("Joining and hashing annual manifests", flush=True)
    memberships, inputs, samples = read_memberships(
        annual_root, {patch: row for row, patch in enumerate(patch_ids)}
    )
    expected = {
        (year, split) for year in manifest["years"] for split in ("train", "validation", "test")
    }
    if {(row["year"], row["split"]) for row in memberships} != expected:
        raise ValueError("Annual manifests do not cover all declared years and three splits")
    pq.write_table(pa.Table.from_pylist(memberships), output / "memberships.parquet")
    write_json(output / "annual_manifest_integrity.json", inputs)
    write_json(output / "spatial_split_audit.json", audit_spatial_splits(index, memberships))
    geometry = audit_geometry(index, samples, data_root)
    write_json(output / "geometry_audit.json", geometry)
    declarations = defaultdict(list)
    for declaration in manifest["arrays"]:
        declarations[declaration["file"]].append(declaration)
    # OSM is not a first-wave target: verify all declared shapes/dtypes, not semantic values.
    osm_arrays = 0
    for filename in sorted(declarations):
        if filename.startswith("osm/"):
            with zarr.ZipStore(str(local_path(target_root, filename)), mode="r") as store:
                group = zarr.open_group(store, mode="r")
                for spec in declarations[filename]:
                    array = group[spec["array"]]
                    if (list(array.shape), list(array.chunks), str(array.dtype)) != (
                        spec["shape"],
                        spec["chunks"],
                        spec["dtype"],
                    ):
                        raise ValueError(f"OSM array contract mismatch: {filename}")
                    osm_arrays += 1
    groups, geographic_groups = defaultdict(list), defaultdict(list)
    splits = defaultdict(set)
    for record in memberships:
        row = record["target_row"]
        splits[record["split"]].add(row)
        key = (record["year"], record["split"])
        groups[key + ("all",)].append(row)
        support = (
            "ms_and_pan"
            if record["ms5m_supported"] and record["pan2m_supported"]
            else (
                "ms_only"
                if record["ms5m_supported"]
                else ("pan_only" if record["pan2m_supported"] else "no_highres")
            )
        )
        groups[key + (support,)].append(row)
        geographic_groups[key + (index[row]["macro_id"],)].append(row)
    overlaps = {
        f"{left}/{right}": sorted(splits[left] & splits[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    train = sorted(splits["train"] - splits["validation"] - splits["test"])
    train_set = set(train)
    split_report = {
        "unique_positions": {name: len(rows) for name, rows in splits.items()},
        "cross_year_split_overlaps_target_rows": overlaps,
        "statistics_candidate_train_positions": len(train),
        "statistics_scope": "Current v6 train union minus all validation/test positions. "
        "Candidate only: final development/confirmation exclusions are not yet frozen.",
        "old_target_split_is_not_used_for_training": True,
    }
    write_json(output / "split_audit.json", split_report)
    coverage, geographic, train_statistics, global_values = [], [], {}, {}
    print("Scanning all eight static target arrays and masks", flush=True)

    def scan(filename: str) -> tuple:
        stats = scan_static(target_root, filename, declarations[filename], patch_ids)
        name = Path(filename).name.removesuffix(".zarr.zip")
        np.savez_compressed(output / f"{name}.row_statistics.npz", **stats)
        print(f"Completed full value/mask scan: {name}", flush=True)
        return name, stats

    with ThreadPoolExecutor(max_workers=workers) as executor:
        filenames = sorted(name for name in declarations if name.startswith("static/"))
        for name, stats in executor.map(scan, filenames):
            global_values[name] = summarize(stats, range(len(index)))
            for key, rows in sorted(groups.items()):
                year, split, support = key
                if name.endswith(("_2020", "_2021")) and not name.endswith(str(year)):
                    continue
                coverage.append(
                    {
                        "target": name,
                        "year": year,
                        "split": split,
                        "highres_support": support,
                        **summarize(stats, rows),
                    }
                )
            for key, rows in sorted(geographic_groups.items()):
                year, split, macro = key
                if name.endswith(("_2020", "_2021")) and not name.endswith(str(year)):
                    continue
                summary = summarize(stats, rows)
                summary.pop("class_positions", None)
                summary.pop("class_pixels", None)
                geographic.append(
                    {"target": name, "year": year, "split": split, "macro_id": macro, **summary}
                )
            if name.endswith(("_2020", "_2021")):
                year = int(name[-4:])
                rows = sorted(
                    {
                        record["target_row"]
                        for record in memberships
                        if record["year"] == year
                        and record["split"] == "train"
                        and record["target_row"] in train_set
                    }
                )
            else:
                rows = train
            train_statistics[name] = summarize(stats, rows)
    write_json(output / "coverage_by_split.json", coverage)
    write_json(output / "global_value_audit.json", global_values)
    write_json(
        output / "train_statistics.json",
        {
            "scope": split_report["statistics_scope"],
            "static_unique_positions_counted_once": True,
            "normalization_ready_for_training": False,
            "targets": train_statistics,
        },
    )
    pq.write_table(pa.Table.from_pylist(geographic), output / "coverage_by_macro.parquet")
    contract = {
        "schema": "xuannv.target-acceptance.v1",
        "started_at": started,
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "target_version": manifest["version"],
        "image_version": manifest["image_version"],
        "target_root": str(target_root),
        "annual_root": str(annual_root),
        "input_sha256": {
            name: digest(target_root / name)
            for name in ("manifest.json", "index.parquet", "README.md", "validation.json")
        },
        "annual_manifests": inputs,
        "join": ["patch_id", "year_for_annual_targets"],
        "row_identity_verified_for_all_static_archives": True,
        "worldcover_raw_to_class": {str(code): row for row, code in enumerate(WC_CODES)},
        "validity": "valid_masks AND finite values; raw WorldCover 0 is invalid, never background",
        "units": {
            "dem_elevation": "m (DSM)",
            "dem_slope": "degree",
            "nightlights": "published preprocessing values; transform not re-derived",
        },
        "osm": {
            "array_metadata_verified": osm_arrays,
            "all_archive_hashes_verified": True,
            "semantic_value_scan_performed": False,
            "first_wave_training_target": False,
        },
        "data_checks_passed": geometry["passed"]
        and not any(overlaps.values())
        and all(
            value["nonfinite_valid_pixels"] == value["illegal_valid_pixels"] == 0
            for value in global_values.values()
        ),
        "training_ready": False,
        "open_gates": [
            "Recover native DEM footprint, vertical datum and target resampling contract",
            "Prove archived S2 ten-band order, physical units and processing level",
            "Validate high-resolution feature registration, beyond raster header alignment",
            "Freeze new spatial evaluation/confirmation sets and exclusion-aware train statistics",
            "Implement target-only training reader/losses after data acceptance",
        ],
        "status": "completed_with_open_gates",
        "training_started": False,
    }
    write_json(output / "target_contract.json", contract)
    artifacts = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "artifacts.sha256.json", artifacts)
    return contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--annual-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    result = run_audit(
        args.target_root, args.annual_root, args.data_root, args.output, args.workers
    )
    print(
        json.dumps({key: result[key] for key in ("status", "data_checks_passed", "training_ready")})
    )
    return 0 if result["data_checks_passed"] else 1
