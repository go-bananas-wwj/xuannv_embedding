"""Reconstruct annual target pixels from locked source rasters for spatial/year audits."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import rasterio
import zarr
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

FAMILIES = {"worldcover", "clcd", "nightlights"}
POLICY = {
    "pixels": 128,
    "extent_m": 1280,
    "categorical_resampling": "nearest",
    "continuous_resampling": "bilinear",
    "categorical_boundary": "both adjacent pixels unknown at valid 4-neighbor class changes",
    "continuous_absolute_tolerance": 1e-5,
    "continuous_relative_tolerance": 1e-6,
    "chunk_positions": 32,
}


def member_matches_year(family: str, year: int, member: str) -> bool:
    if family not in FAMILIES or year not in {2020, 2021}:
        raise ValueError("unsupported annual target family or year")
    name = Path(member).name
    patterns = {
        "worldcover": rf"ESA_WorldCover_10m_{year}_v[0-9]+_[NS][0-9]+[EW][0-9]+_Map\.tif",
        "clcd": rf"CLCD_v[0-9]+_{year}_albert\.tif",
        "nightlights": rf"VNL_.*npp_{year}_.*\.average\.tif",
    }
    return re.fullmatch(patterns[family], name) is not None


def categorical_validity(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape != valid.shape:
        raise ValueError("categorical boundary requires corresponding 2D arrays")
    boundary = np.zeros(values.shape, bool)
    for axis in [0, 1]:
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        different = valid[a] & valid[b] & (values[a] != values[b])
        boundary[a] |= different
        boundary[b] |= different
    return valid & ~boundary


def compare_target(
    expected: np.ndarray,
    expected_valid: np.ndarray,
    actual: np.ndarray,
    actual_valid: np.ndarray,
    *,
    categorical: bool,
) -> dict:
    if not (expected.shape == expected_valid.shape == actual.shape == actual_valid.shape):
        raise ValueError("reconstructed and stored target grids disagree")
    both = expected_valid & actual_valid
    finite = np.isfinite(expected) & np.isfinite(actual)
    match = (
        expected == actual if categorical else np.isclose(expected, actual, atol=1e-5, rtol=1e-6)
    )
    mask_mismatch = int(np.count_nonzero(expected_valid != actual_valid))
    value_mismatch = int(np.count_nonzero(both & (~finite | ~match)))
    invalid_nonzero = int(np.count_nonzero(~actual_valid & (actual != 0)))
    maximum = float(np.abs(expected[both & finite] - actual[both & finite]).max(initial=0))
    return {
        "status": "failed" if mask_mismatch or value_mismatch or invalid_nonzero else "passed",
        "expected_valid_pixels": int(expected_valid.sum()),
        "stored_valid_pixels": int(actual_valid.sum()),
        "mask_mismatch_pixels": mask_mismatch,
        "value_mismatch_pixels": value_mismatch,
        "nonzero_invalid_pixels": invalid_nonzero,
        "maximum_absolute_error": maximum,
    }


class AnnualRasterArchive:
    """Read real GeoTIFF bounds and masks; names select product/year, never spatial bounds."""

    def __init__(self, path: Path, family: str, year: int):
        self.path, self.family, self.year = path, family, year
        self.readers = OrderedDict()
        with ZipFile(path) as archive:
            names = archive.namelist()
        members = sorted(name for name in names if member_matches_year(family, year, name))
        if not members or len(set(members)) != len(members):
            raise ValueError("missing or duplicate year-specific source members")
        if family != "worldcover" and len(members) != 1:
            raise ValueError("annual source member is ambiguous")
        self.entries = []
        for member in members:
            with rasterio.open(self.uri(member)) as source:
                if source.count != 1 or source.crs is None or source.transform.is_identity:
                    raise ValueError("source georeferencing or single-band contract is missing")
                if source.scales != (1.0,) or source.offsets != (0.0,):
                    raise ValueError("unverified static source scale/offset")
                self.entries.append(
                    {
                        "member": member,
                        "crs": source.crs.to_string(),
                        "transform": list(source.transform)[:6],
                        "shape": [source.height, source.width],
                        "bounds": list(source.bounds),
                        "dtype": source.dtypes[0],
                        "nodata": source.nodata,
                        "wgs84_bounds": list(
                            transform_bounds(
                                source.crs, "EPSG:4326", *source.bounds, densify_pts=21
                            )
                        ),
                    }
                )

    def uri(self, member: str) -> str:
        return f"/vsizip/{self.path.resolve()}/{member}"

    def reader(self, member: str):
        source = self.readers.pop(member, None)
        if source is None:
            source = rasterio.open(self.uri(member))
        self.readers[member] = source
        while len(self.readers) > 8:
            self.readers.popitem(last=False)[1].close()
        return source

    def close(self) -> None:
        for source in self.readers.values():
            source.close()
        self.readers.clear()

    def reconstruct(self, epsg: int, bounds) -> tuple[np.ndarray, np.ndarray, list[str]]:
        bounds = tuple(float(v) for v in bounds)
        if len(bounds) != 4 or not np.allclose(
            [bounds[2] - bounds[0], bounds[3] - bounds[1]], 1280
        ):
            raise ValueError("target must cover the frozen 1280 m grid")
        shape = (128, 128)
        output = np.zeros(shape, "f4")
        valid = np.zeros(shape, bool)
        destination_crs = rasterio.crs.CRS.from_epsg(int(epsg))
        destination_transform = from_bounds(*bounds, *shape)
        geographic = transform_bounds(destination_crs, "EPSG:4326", *bounds, densify_pts=21)
        categorical = self.family != "nightlights"
        used = []
        for entry in self.entries:
            candidate = entry["wgs84_bounds"]
            if not (
                geographic[0] < candidate[2]
                and geographic[2] > candidate[0]
                and geographic[1] < candidate[3]
                and geographic[3] > candidate[1]
            ):
                continue
            source = self.reader(entry["member"])
            transformed = transform_bounds(destination_crs, source.crs, *bounds, densify_pts=21)
            requested = window_from_bounds(*transformed, transform=source.transform)
            pad = 1 if categorical else 2
            left, top = math.floor(requested.col_off) - pad, math.floor(requested.row_off) - pad
            right = math.ceil(requested.col_off + requested.width) + pad
            bottom = math.ceil(requested.row_off + requested.height) + pad
            try:
                window = Window(left, top, right - left, bottom - top).intersection(
                    Window(0, 0, source.width, source.height)
                )
            except rasterio.errors.WindowError:
                continue
            raw = source.read(1, window=window).astype("f4")
            raw_valid = (source.read_masks(1, window=window) > 0) & np.isfinite(raw)
            raw[~raw_valid] = np.nan
            temporary = np.full(shape, np.nan, "f4")
            reproject(
                raw,
                temporary,
                src_transform=source.window_transform(window),
                src_crs=source.crs,
                src_nodata=np.nan,
                dst_transform=destination_transform,
                dst_crs=destination_crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest if categorical else Resampling.bilinear,
            )
            current = np.isfinite(temporary)
            overlap = valid & current
            if categorical and np.any(overlap & (output != temporary)):
                raise ValueError("overlapping source tiles disagree about categorical labels")
            output[current] = temporary[current]
            valid |= current
            used.append(entry["member"])
        if categorical:
            allowed = (
                [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
                if self.family == "worldcover"
                else list(range(1, 10))
            )
            if not np.isin(output[valid], allowed).all():
                raise ValueError("unexpected class or valid NoData in source reconstruction")
            valid = categorical_validity(output, valid)
        return np.where(valid, output, 0), valid, used


def _source_reference(report_root: Path, family: str, year: int) -> dict:
    audit = json.loads((report_root / "target_source_audit.json").read_text())
    if audit.get("status") != "source_audit_finished" or audit.get("failed_sources") != 0:
        raise ValueError("completed source provenance audit is required")
    fragment = {
        "worldcover": f"esa_worldcover_{year}",
        "clcd": f"clcd_china_{year}",
        "nightlights": "baidu_vnpp_ntl_2012_2023",
    }[family]
    selected = [
        r for r in audit["sources"] if r["family"] == "static" and fragment in Path(r["path"]).stem
    ]
    if len(selected) != 1:
        raise ValueError("static source reference is missing or ambiguous")
    return selected[0]


def audit_target_geometry(
    dataset_root: Path, report_root: Path, family: str, *, max_patches: int | None = None
) -> dict:
    if family not in FAMILIES or (max_patches is not None and max_patches <= 0):
        raise ValueError("invalid target family or patch limit")
    registry_path = dataset_root / "registry/national_62000.parquet"
    registry = pd.read_parquet(registry_path)
    if registry.empty or registry.patch_id.duplicated().any():
        raise ValueError("empty or duplicate national registry")
    manifest_path = dataset_root / "targets/manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    value_audit = pd.read_parquet(report_root / "target_value_audit.parquet")
    selected_registry = registry if max_patches is None else registry.iloc[:max_patches]
    scope = "full" if max_patches is None else f"pilot_{max_patches}"
    directory = dataset_root / "quality/targets/geometry" / family / scope
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "code_sha256": sha256(Path(__file__)),
        "registry_sha256": sha256(registry_path),
        "manifest_sha256": sha256(manifest_path),
        "policy": POLICY,
        "runtime": {
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "gdal": rasterio.__gdal_version__,
        },
    }
    progress = report_root / f"target_geometry_{family}_{scope}.json"
    rows, source_locks = [], []
    reused = 0
    for year in [2020, 2021]:
        name = f"{family}_{year}"
        metadata = manifest.loc[
            (manifest.family == "static") & (manifest.array == f"targets/{name}")
        ]
        if (
            len(metadata) != 1
            or not bool(metadata.iloc[0].registry_order_verified)
            or metadata.iloc[0].year != year
        ):
            raise ValueError("annual target metadata is missing or disagrees with requested year")
        target = zarr.open_group(str(metadata.iloc[0].path), mode="r")
        if list(target.attrs["patch_ids"]) != registry.patch_id.tolist():
            raise ValueError("target patch order changed")
        reference = _source_reference(report_root, family, year)
        source_path = Path(reference["path"])
        before = source_path.stat()
        source_hash = sha256(source_path)
        if source_hash != reference["actual_sha256"]:
            raise ValueError("source archive changed after provenance audit")
        source = AnnualRasterArchive(source_path, family, year)
        source_lock = {
            "year": year,
            "path": str(source_path),
            "sha256": source_hash,
            "members": source.entries,
        }
        source_locks.append(source_lock)
        write_json(directory / f"source_{year}.json", source_lock)
        source_fingerprint = hashlib.sha256(
            json.dumps(source_lock, sort_keys=True).encode()
        ).hexdigest()
        try:
            actual = target[f"targets/{name}"]
            masks = target[f"valid_masks/{name}"]
            if actual.shape != (len(registry), 128, 128) or masks.shape != actual.shape:
                raise ValueError("target shape differs from frozen grid")
            values_digest, mask_digest = hashlib.sha256(), hashlib.sha256()
            for start in range(0, len(selected_registry), POLICY["chunk_positions"]):
                stop = min(len(selected_registry), start + POLICY["chunk_positions"])
                values, valid = np.asarray(actual[start:stop]), np.asarray(masks[start:stop], bool)
                values_digest.update(values.tobytes())
                mask_digest.update(valid.tobytes())
                digest = hashlib.sha256(values.tobytes() + valid.tobytes()).hexdigest()
                expected = {
                    **fingerprint,
                    "year": year,
                    "source_index_sha256": source_fingerprint,
                    "target_chunk_sha256": digest,
                    "start": start,
                    "stop": stop,
                }
                receipt = directory / "chunks" / f"{year}_{start:06d}.json"
                old = json.loads(receipt.read_text()) if receipt.exists() else {}
                if old.get("fingerprint") == expected:
                    chunk_rows = old["rows"]
                    reused += len(chunk_rows)
                else:
                    chunk_rows = []
                    for index, row in enumerate(selected_registry.iloc[start:stop].itertuples()):
                        result = {
                            "patch_id": row.patch_id,
                            "year": year,
                            "target": name,
                            "split": row.split,
                        }
                        try:
                            fresh, fresh_valid, members = source.reconstruct(
                                row.grid_epsg, row.utm_bounds
                            )
                            result.update(
                                compare_target(
                                    fresh,
                                    fresh_valid,
                                    values[index],
                                    valid[index],
                                    categorical=family != "nightlights",
                                )
                            )
                            result["source_members"] = members
                        except (ValueError, OSError) as exc:
                            result.update(status="failed", reason=str(exc), source_members=[])
                        chunk_rows.append(result)
                    write_json(receipt, {"fingerprint": expected, "rows": chunk_rows})
                rows.extend(chunk_rows)
                write_json(
                    progress,
                    {
                        "status": "running",
                        "family": family,
                        "scope": scope,
                        "processed_targets": len(rows),
                        "selected_targets": len(selected_registry) * 2,
                        "failed_targets": sum(r["status"] == "failed" for r in rows),
                        "reused_targets": reused,
                        "updated_at": now(),
                    },
                )
            if max_patches is None:
                for array_name, digest in [
                    (f"targets/{name}", values_digest),
                    (f"valid_masks/{name}", mask_digest),
                ]:
                    checked = value_audit.loc[
                        (value_audit.family == "static") & (value_audit.array == array_name)
                    ]
                    if (
                        len(checked) != 1
                        or checked.iloc[0].decoded_values_sha256 != digest.hexdigest()
                    ):
                        raise ValueError("stored target bytes changed after completed value audit")
        finally:
            source.close()
        after = source_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source archive changed during reconstruction")
    table = pd.DataFrame(
        [{**row, "source_members": json.dumps(row["source_members"])} for row in rows]
    )
    atomic_parquet(table, directory / "observations.parquet")
    result = {
        "status": "target_geometry_audit_finished",
        "family": family,
        "scope": scope,
        "processed_targets": len(rows),
        "selected_targets": len(selected_registry) * 2,
        "failed_targets": sum(r["status"] == "failed" for r in rows),
        "reused_targets": reused,
        "source_locks": source_locks,
        "fingerprint": fingerprint,
        "output": str(directory / "observations.parquet"),
        "training_authorized": False,
        "limitation": "Reconstruction proves consistency with current locked annual inputs; "
        "it does not supply absent historical SHA or independent label accuracy.",
        "finished_at": now(),
    }
    write_json(progress, result)
    return result
