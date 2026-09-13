"""Full local ZIP pixel/grid audit, independent of unresolved physical product contracts."""

from __future__ import annotations

import hashlib
import json
import stat
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rasterio.io import MemoryFile

from xuannv_embedding.data_process.v5_catalog import grid_lookup
from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

BAND_COUNTS = {"s2_local": 10, "s1_local": 2, "landsat_local": 6}
SCHEMA = pa.schema(
    [
        ("member_name", pa.string()),
        ("patch_id", pa.string()),
        ("split", pa.string()),
        ("file_sha256", pa.string()),
        ("bytes", pa.int64()),
        ("status", pa.string()),
        ("nonfinite_pixels", pa.int64()),
        ("metadata_json", pa.string()),
        ("reason", pa.string()),
    ]
)


def inspect_dense_archive(
    path: Path, registry: pd.DataFrame, output_root: Path, *, product: str, year: int, month: int
) -> dict:
    if product not in BAND_COUNTS or year not in (2020, 2021) or not 1 <= month <= 12:
        raise ValueError("unsupported dense archive interval/product")
    key = f"{product}_{year}_{month:02d}"
    output_root.mkdir(parents=True, exist_ok=True)
    initial_stat = path.stat()
    fingerprint = {
        "source_sha256": sha256(path),
        "source_bytes": path.stat().st_size,
        "code_sha256": sha256(Path(__file__)),
        "registry_sha256": hashlib.sha256(
            registry[["patch_id", "split", "grid_epsg", "utm_bounds"]].to_json().encode()
        ).hexdigest(),
    }
    receipt_path = output_root / (key + ".json")
    inventory = output_root / (key + ".parquet")
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError("dense source or audit changed; choose a new audit version")
        if (
            previous["status"] != "failed"
            and inventory.exists()
            and sha256(inventory) == previous["inventory_sha256"]
        ):
            return previous
    lookup = grid_lookup(registry)
    seen = {}
    conflicts = set()
    auxiliary = []
    decoded = failed = tiffs = 0
    temporary = inventory.with_name(inventory.name + ".partial")
    rows = []
    with (
        pq.ParquetWriter(temporary, SCHEMA, compression="zstd") as writer,
        ZipFile(path) as archive,
    ):
        for member in archive.infolist():
            if member.is_dir():
                continue
            if not member.filename.lower().endswith((".tif", ".tiff")):
                auxiliary.append(
                    {"member_name": member.filename, "bytes": member.file_size, "crc32": member.CRC}
                )
                continue
            tiffs += 1
            row = {
                "member_name": member.filename,
                "bytes": member.file_size,
                "patch_id": None,
                "split": None,
                "file_sha256": None,
                "nonfinite_pixels": None,
                "metadata_json": None,
                "reason": None,
                "status": "failed",
            }
            try:
                name = PurePosixPath(member.filename)
                if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                    raise ValueError("unsafe archive member identity")
                if stat.S_ISLNK(member.external_attr >> 16) or member.file_size > 512 * 1024**2:
                    raise ValueError("unsupported link or oversized raster member")
                blob = archive.read(member)  # ZIP CRC checked before raster decoding.
                row["file_sha256"] = hashlib.sha256(blob).hexdigest()
                with MemoryFile(blob) as memory, memory.open() as raster:
                    epsg = raster.crs.to_epsg() if raster.crs else None
                    geometry = (epsg, *(round(float(v), 3) for v in raster.bounds))
                    if geometry not in lookup:
                        raise ValueError("raster outside unique national grid")
                    row["patch_id"], row["split"] = lookup[geometry]
                    if (
                        raster.count != BAND_COUNTS[product]
                        or raster.shape != (128, 128)
                        or raster.transform.b != 0
                        or raster.transform.d != 0
                        or not np.allclose(
                            [raster.transform.a, raster.transform.e], [10, -10], rtol=0, atol=1e-9
                        )
                        or not np.allclose(raster.res, [10, 10], rtol=0, atol=1e-9)
                    ):
                        raise ValueError("dense shape/band/grid structure differs")
                    nonfinite = 0
                    for _, window in raster.block_windows(1):
                        values = raster.read(window=window)
                        raster.read_masks(window=window)
                        nonfinite += int((~np.isfinite(values)).sum())
                    row["nonfinite_pixels"] = nonfinite
                    row["metadata_json"] = json.dumps(
                        {
                            "crs": str(raster.crs),
                            "transform": list(raster.transform)[:6],
                            "descriptions": raster.descriptions,
                            "scales": raster.scales,
                            "offsets": raster.offsets,
                            "nodata": raster.nodata,
                            "dtype": raster.dtypes,
                            "radiometry_status": "unverified",
                        }
                    )
                old_hash = seen.get(row["patch_id"])
                if old_hash and old_hash != row["file_sha256"]:
                    conflicts.add(row["patch_id"])
                    raise ValueError("conflicting pixels for the same monthly location")
                row["status"] = "duplicate_equal" if old_hash else "decoded_contract_pending"
                seen[row["patch_id"]] = row["file_sha256"]
                decoded += 1
            except Exception as exc:
                row["reason"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            rows.append(row)
            if len(rows) >= 1024:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                rows.clear()
                write_json(
                    output_root / (key + ".progress.json"),
                    {
                        "status": "running",
                        "scanned_tiffs": tiffs,
                        "decoded_tiffs": decoded,
                        "failed_tiffs": failed,
                        "updated_at": now(),
                    },
                )
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    temporary.replace(inventory)
    final_stat = path.stat()
    source_changed = (initial_stat.st_size, initial_stat.st_mtime_ns) != (
        final_stat.st_size,
        final_stat.st_mtime_ns,
    )
    missing = sorted(set(registry.patch_id) - (set(seen) - conflicts))
    atomic_parquet(pd.DataFrame({"patch_id": missing}), output_root / (key + ".missing.parquet"))
    result = {
        "status": "failed" if failed or source_changed else "integrity_checked_contract_pending",
        "source_changed_during_audit": source_changed,
        **fingerprint,
        "fingerprint": fingerprint,
        "product_id": product,
        "year": year,
        "month": month,
        "source_path": str(path.resolve()),
        "tiffs": tiffs,
        "decoded_tiffs": decoded,
        "failed_tiffs": failed,
        "missing_patches": len(missing),
        "conflicting_patch_ids": sorted(conflicts),
        "auxiliary_files": len(auxiliary),
        "auxiliary_inventory": auxiliary,
        "inventory_path": str(inventory.resolve()),
        "inventory_sha256": sha256(inventory),
        "finished_at": now(),
        "physical_contract_verified": False,
    }
    write_json(receipt_path, result)
    return result


def audit_dense_integrity(dense_root: Path, dataset_root: Path, report_root: Path) -> dict:
    registry = pd.read_parquet(dataset_root / "registry/national_62000.parquet")
    selected = []
    for product, folder in [
        ("s2_local", "pc-s2"),
        ("s1_local", "pc-s1"),
        ("landsat_local", "pc-ls"),
    ]:
        for year in (2020, 2021):
            for month in range(1, 13):
                path = (
                    dense_root
                    / folder
                    / str(year)
                    / f"{month:02d}"
                    / (f"{folder}_{year}_{month:02d}.zip")
                )
                selected.append((product, year, month, path))
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {
            pool.submit(
                inspect_dense_archive,
                path,
                registry,
                report_root / "dense_integrity_shards",
                product=product,
                year=year,
                month=month,
            ): (product, year, month)
            for product, year, month, path in selected
        }
        for future in as_completed(jobs):
            product, year, month = jobs[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "product_id": product,
                    "year": year,
                    "month": month,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            results.append(result)
            write_json(
                report_root / "dense_integrity_summary.json",
                {
                    "status": "running",
                    "processed_archives": len(results),
                    "selected_archives": len(selected),
                    "failed_archives": sum(row["status"] == "failed" for row in results),
                    "results": results,
                    "updated_at": now(),
                },
            )
    summary = {
        "status": "integrity_audit_finished_contract_pending",
        "processed_archives": len(results),
        "selected_archives": len(selected),
        "failed_archives": sum(row["status"] == "failed" for row in results),
        "results": results,
        "finished_at": now(),
    }
    write_json(report_root / "dense_integrity_summary.json", summary)
    return summary
