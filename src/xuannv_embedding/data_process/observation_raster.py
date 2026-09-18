"""Inspect stored rasters without guessing spectral semantics or radiometric units."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.io import MemoryFile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parent_geometry(key: str) -> tuple[int, tuple[float, float, float, float]]:
    epsg, column, row = map(int, key.removeprefix("parent_").split(":"))
    left, bottom = column * 1280.0, row * 1280.0
    return epsg, (left, bottom, left + 1280.0, bottom + 1280.0)


def inspect_raster(
    payload: bytes, key: str, *, pixels: bool = False, destination: Path | None = None
) -> dict[str, Any]:
    expected_epsg, expected_bounds = parent_geometry(key)
    with MemoryFile(payload) as memory, memory.open() as raster:
        error = max(
            abs(actual - expected) for actual, expected in zip(raster.bounds, expected_bounds)
        )
        aligned = (
            raster.crs is not None
            and raster.crs.to_epsg() == expected_epsg
            and error <= 0.01
            and raster.transform.a > 0
            and raster.transform.e < 0
            and abs(raster.transform.b) < 1e-9
            and abs(raster.transform.d) < 1e-9
        )
        nodata = raster.nodata
        result: dict[str, Any] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "width": raster.width,
            "height": raster.height,
            "channels": raster.count,
            "dtypes": list(raster.dtypes),
            "crs": str(raster.crs),
            "transform": list(raster.transform)[:6],
            "bounds": list(raster.bounds),
            "grid_spacing_m": list(raster.res),
            "native_gsd_m": None,
            "band_names": list(raster.descriptions),
            "radiometric_units": list(raster.units),
            "stored_scales": list(raster.scales),
            "stored_offsets": list(raster.offsets),
            "nodata": nodata if nodata is not None and math.isfinite(nodata) else None,
            "nodata_is_nonfinite": nodata is not None and not math.isfinite(nodata),
            "mask_flags": [[flag.name for flag in flags] for flags in raster.mask_flag_enums],
            "grid_matches_parent": aligned,
            "bounds_error_m": error,
            "georegistration_status": "grid_only_not_landmark_verified",
            "band_semantics_status": "unverified",
            "radiometry_status": "stored_values_no_conversion",
            "quality_status": "cloud_shadow_saturation_unverified",
        }
        if not pixels:
            return result
        values = raster.read()
        valid = (raster.read_masks() > 0) & np.isfinite(values)
        shared_mask = valid.all(axis=0)
        counts, means, variances = [], [], []
        extrema, zero_fractions, maximum_fractions = [], [], []
        for band in values:
            accepted = band[shared_mask].astype(np.float64)
            counts.append(int(accepted.size))
            means.append(float(accepted.mean()) if accepted.size else None)
            variances.append(float(accepted.var()) if accepted.size else None)
            extrema.append(
                [float(accepted.min()), float(accepted.max())] if accepted.size else None
            )
            zero_fractions.append(float((band == 0).mean()))
            maximum_fractions.append(
                float((band == np.iinfo(band.dtype).max).mean())
                if np.issubdtype(band.dtype, np.integer)
                else None
            )
        result.update(
            pixel_check="decoded",
            valid_fraction=float(shared_mask.mean()),
            band_counts=counts,
            band_mean=means,
            band_variance=variances,
            band_min_max=extrema,
            zero_fractions=zero_fractions,
            dtype_max_fractions=maximum_fractions,
            mask_policy="all_bands_finite_and_declared_gdal_masks; not_clear_sky_QA",
        )
        if destination is not None and aligned and shared_mask.any():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256_file(destination) != result["sha256"]:
                    raise ValueError(f"Changed cached TIFF: {destination}")
            else:
                descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent)
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(payload)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            mask_path = destination.with_name(destination.stem + "_mask.tif")
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=raster.height,
                width=raster.width,
                count=1,
                dtype="uint8",
                crs=raster.crs,
                transform=raster.transform,
                compress="deflate",
                nodata=0,
            ) as target:
                target.write(shared_mask.astype(np.uint8), 1)
            result["materialized"] = True
        return result


def merge_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    channels = records[0]["channels"]
    counts = np.zeros(channels, dtype=np.int64)
    means = np.zeros(channels, dtype=np.float64)
    moments = np.zeros(channels, dtype=np.float64)
    for record in records:
        if record["channels"] != channels:
            raise ValueError("Cannot combine statistics with different channel counts")
        current_counts = np.asarray(record["band_counts"])
        current_means = np.asarray([value or 0.0 for value in record["band_mean"]])
        current_variances = np.asarray([value or 0.0 for value in record["band_variance"]])
        total = counts + current_counts
        delta = current_means - means
        means += delta * current_counts / np.maximum(total, 1)
        moments += (
            current_variances * current_counts
            + delta**2 * counts * current_counts / np.maximum(total, 1)
        )
        counts = total
    raw_std = np.sqrt(moments / np.maximum(counts, 1))
    return {
        "mean": means.tolist(),
        "std": np.maximum(raw_std, 1e-6).tolist(),
        "raw_std": raw_std.tolist(),
        "band_counts": counts.tolist(),
        "num_files": len(records),
        "fit_split": "train",
        "units": "stored_values_unverified",
        "purpose": "loader_smoke_only_not_calibrated_training",
    }
