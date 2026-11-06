"""Native-grid raster contracts and calendar-year scene selection for V5."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from xuannv_embedding.data_process.v5_sources import sha256

BAND = re.compile(r"^(B\d+)\((\d+(?:\.\d+)?)\)$")
BRANCH_BANDS = {
    "jilin1_ms_5m": tuple(f"B{i}" for i in range(1, 7)),
    "jilin1_b0_5m": ("B0",),
    "jilin1_extra_10m": tuple(f"B{i}" for i in range(7, 13)),
    "jilin1_extra_20m": tuple(f"B{i}" for i in range(13, 20)),
}


def band_metadata(descriptions: tuple) -> tuple[tuple[str, ...], tuple[float, ...]]:
    matches = [BAND.fullmatch(name or "") for name in descriptions]
    if not all(matches):
        raise ValueError("missing or unsupported band descriptions")
    names = tuple(match[1] for match in matches)
    if len(set(names)) != len(names):
        raise ValueError("duplicate band identities")
    return names, tuple(float(match[2]) for match in matches)


@dataclass(frozen=True)
class NativeRaster:
    values: np.ndarray
    valid: np.ndarray
    band_ids: tuple[str, ...]
    transform: tuple[float, ...]
    crs: str


def read_native(
    path: Path,
    bands: list[str] | tuple[str, ...],
    *,
    quality=None,
    mean=None,
    std=None,
    contract: dict | None = None,
) -> NativeRaster:
    """Apply metadata scaling exactly once; retain zero/negative valid physical values.

    Sources without named physical-band metadata require an explicitly verified contract
    containing ordered band_ids, scales and offsets; no sensor defaults are guessed.
    """
    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError("missing raster CRS")
        if contract is None:
            names, _ = band_metadata(source.descriptions)
            scales, offsets = source.scales, source.offsets
        else:
            if not contract.get("verified"):
                raise ValueError("unverified raster contract")
            names = tuple(contract["band_ids"])
            scales, offsets = contract["scales"], contract["offsets"]
            if len(names) != source.count or len(set(names)) != len(names):
                raise ValueError("contract band count or identities disagree")
        if len(scales) != source.count or len(offsets) != source.count:
            raise ValueError("scale/offset count disagreement")
        if len(set(bands)) != len(bands) or not set(bands).issubset(names):
            raise ValueError("missing or repeated requested band")
        indices = [names.index(band) + 1 for band in bands]
        raw = source.read(indices)
        valid = (source.read_masks(indices) > 0) & np.isfinite(raw)
        scales = np.asarray([scales[i - 1] for i in indices], dtype=np.float32)
        offsets = np.asarray([offsets[i - 1] for i in indices], dtype=np.float32)
        if not np.isfinite(scales).all() or not np.isfinite(offsets).all() or (scales == 0).any():
            raise ValueError("invalid scale/offset")
        values = raw.astype(np.float32) * scales[:, None, None] + offsets[:, None, None]
        valid &= np.isfinite(values)
        if quality is not None:
            if quality.shape not in {values.shape, values.shape[1:]}:
                raise ValueError("quality mask grid differs from raster")
            valid &= np.asarray(quality, dtype=bool)
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be specified together")
        if mean is not None:
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            if (
                mean.shape != (len(bands),)
                or std.shape != mean.shape
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or (std <= 0).any()
            ):
                raise ValueError("invalid normalization statistics")
            values = (values - mean[:, None, None]) / std[:, None, None]
        values[~valid] = 0
        return NativeRaster(
            values, valid, tuple(bands), tuple(source.transform)[:6], str(source.crs)
        )


def inspect_jilin(path: Path) -> dict:
    with rasterio.open(path) as source:
        names, wavelengths = band_metadata(source.descriptions)
        tags = source.tags()
        gsd = float(source.res[0])
        if (
            source.crs is None
            or gsd not in {5, 10, 20}
            or not np.isclose(source.res[1], gsd)
            or source.width * gsd != 1280
            or source.height * gsd != 1280
            or source.transform.b != 0
            or source.transform.d != 0
        ):
            raise ValueError("unverified Jilin grid contract")
        expected = {5: 6, 10: 12, 20: 19}[int(gsd)]
        b0 = names == ("B0",) and gsd == 5
        if not b0 and (
            len(names) != expected or set(names) != {f"B{i}" for i in range(1, expected + 1)}
        ):
            raise ValueError("unverified Jilin band contract")
        if (
            set(source.dtypes) != {"int16"}
            or source.nodata != -28672
            or not np.allclose(source.scales, 0.0001, rtol=0, atol=1e-12)
            or not np.allclose(source.offsets, 0, rtol=0, atol=1e-12)
            or tags.get("units") != "reflectance"
        ):
            raise ValueError("unverified Jilin radiometry contract")
        raw_time = tags.get("acquisition_time", "")
        acquired = datetime.fromisoformat(raw_time)
        scene = tags.get("source_product")
        signature = tags.get("source_signature")
        if not scene or not signature or not tags.get("patch_id"):
            raise ValueError("missing Jilin scene provenance")
        product = (
            "jilin1_b0_5m"
            if b0
            else {5: "jilin1_ms_5m", 10: "jilin1_extra_10m", 20: "jilin1_extra_20m"}[int(gsd)]
        )
        return {
            "source_patch_id": tags["patch_id"],
            "sensor": path.parent.name,
            "scene_id": scene,
            "source_signature": signature,
            "acquired_at": acquired.isoformat(),
            "acquired_at_raw": raw_time,
            "time_precision": "second",
            "time_zone": "unspecified" if acquired.tzinfo is None else str(acquired.tzinfo),
            "year": acquired.year,
            "product_id": product,
            "path": str(path.resolve()),
            "file_sha256": sha256(path),
            "crs": str(source.crs),
            "epsg": source.crs.to_epsg(),
            "transform": list(source.transform)[:6],
            "bounds": list(source.bounds),
            "band_ids": list(names),
            "wavelengths": list(wavelengths),
            "selected_band_ids": list(BRANCH_BANDS[product]),
            "shape": [source.count, source.height, source.width],
            "dtype": source.dtypes[0],
            "native_gsd": gsd,
            "stored_gsd": gsd,
            "nodata": source.nodata,
            "scale": list(source.scales),
            "offset": list(source.offsets),
            "metadata_status": "verified",
        }


def select_annual(rows: list[dict], *, year: int, limit: int = 4) -> list[dict]:
    if limit <= 0:
        raise ValueError("annual limit must be positive")
    eligible = [row for row in rows if int(row["year"]) == year]
    eligible.sort(
        key=lambda row: (-float(row["clear_fraction"]), row["acquired_at"], row["scene_group_id"])
    )
    chosen = []
    quarters = set()
    for row in eligible:
        quarter = (datetime.fromisoformat(row["acquired_at"]).month - 1) // 3
        if quarter not in quarters:
            quarters.add(quarter)
            chosen.append(row)
    chosen = chosen[:limit]
    identities = {row["scene_group_id"] for row in chosen}
    for row in eligible:
        if len(chosen) == limit:
            break
        if row["scene_group_id"] not in identities:
            chosen.append(row)
            identities.add(row["scene_group_id"])
    return sorted(chosen, key=lambda row: (row["acquired_at"], row["scene_group_id"]))
