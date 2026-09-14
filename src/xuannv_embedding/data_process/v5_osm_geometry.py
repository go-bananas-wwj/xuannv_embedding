"""Reconstruct dated OSM labels and measure features missed by footprint-only queries."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
import zarr
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from shapely.ops import transform as transform_geometry

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json

CHANNELS = (
    "building",
    "road_all",
    "road_major",
    "road_secondary",
    "road_local",
    "path_track",
    "railway",
    "water_area",
    "waterway",
    "agriculture",
    "green_space",
    "residential",
    "commercial",
    "industrial",
    "construction",
    "settlement",
    "parking",
    "aerodrome",
    "runway",
    "taxiway",
    "helipad",
    "transport_facility",
    "traffic_facility",
    "place_of_worship",
    "amenity_fuel",
    "power_infrastructure",
    "tower_mast",
    "storage_tank_silo",
    "leisure",
    "works_poi",
)
STRUCTURE = ("building", "road_all", "railway", "waterway")
FIELDS = ("targets", "states", "confidence", "source_bits")


def rasterize_records(records, bounds, pixels, oversample, channels):
    bounds = tuple(float(v) for v in bounds)
    if (
        pixels <= 0
        or oversample <= 0
        or len(bounds) != 4
        or not np.isfinite(bounds).all()
        or not np.allclose([bounds[2] - bounds[0], bounds[3] - bounds[1]], 1280, rtol=0, atol=1e-6)
        or len(set(channels)) != len(channels)
        or not set(channels).issubset(CHANNELS)
    ):
        raise ValueError("invalid frozen OSM raster grid or channels")
    polygons = {c: [] for c in channels}
    points = {c: [] for c in channels}
    for record in records:
        code = record["channels"]
        geometry = record["geometry"]
        width = record["width_m"]
        if type(code) is not int or not 0 < code < (1 << len(CHANNELS)):
            raise ValueError("unregistered OSM channel bits")
        if not np.isfinite(width) or width < 0:
            raise ValueError("invalid indexed physical feature width")
        if (
            geometry is None
            or geometry.is_empty
            or not geometry.is_valid
            or not np.isfinite(geometry.bounds).all()
        ):
            raise ValueError("invalid indexed feature geometry")
        selected = [c for c in channels if code & (1 << CHANNELS.index(c))]
        if geometry.geom_type in ["Point", "MultiPoint"]:
            centers = list(geometry.geoms) if geometry.geom_type == "MultiPoint" else [geometry]
            for c in selected:
                points[c].extend((float(p.x), float(p.y)) for p in centers)
        else:
            if geometry.geom_type in ["LineString", "MultiLineString"]:
                if width <= 0:
                    raise ValueError("line requires positive indexed width")
                geometry = geometry.buffer(width / 2, cap_style=2, join_style=2)
            elif geometry.geom_type not in ["Polygon", "MultiPolygon"]:
                raise ValueError("unsupported indexed geometry type")
            for c in selected:
                polygons[c].append((geometry, 1.0))
    size = pixels * oversample
    transform = from_bounds(*bounds, size, size)
    result = np.zeros((len(channels), pixels, pixels), "f4")
    for i, c in enumerate(channels):
        if not polygons[c] and not points[c]:
            continue
        high = np.zeros((size, size), "f4")
        if polygons[c]:
            high = rasterize(
                polygons[c],
                out_shape=(size, size),
                transform=transform,
                fill=0,
                dtype="float32",
                all_touched=True,
                skip_invalid=False,
            )
        for x, y in points[c]:
            column = (x - bounds[0]) / (bounds[2] - bounds[0]) * size
            row = (bounds[3] - y) / (bounds[3] - bounds[1]) * size
            radius = max(2, oversample * 2)
            row0, row1 = max(0, int(row) - radius), min(size, int(row) + radius + 1)
            col0, col1 = max(0, int(column) - radius), min(size, int(column) + radius + 1)
            if row1 <= row0 or col1 <= col0:
                continue
            yy, xx = np.mgrid[row0:row1, col0:col1]
            heat = np.exp(-((yy - row) ** 2 + (xx - column) ** 2) / (2.0 * radius**2)).astype("f4")
            high[row0:row1, col0:col1] = np.maximum(high[row0:row1, col0:col1], heat)
        result[i] = high.reshape(pixels, oversample, pixels, oversample).mean(axis=(1, 3))
    return result


def encode_coverage(historical, current):
    if (
        historical.shape != current.shape
        or historical.ndim != 3
        or any(not np.isfinite(v).all() or np.any((v < 0) | (v > 1)) for v in [historical, current])
    ):
        raise ValueError("invalid historical/current geometric coverage")
    positive = historical > 0
    return {
        "targets": np.where(positive, np.clip(np.rint(historical * 255), 1, 255), 0).astype("u1"),
        "states": positive.astype("u1"),
        "confidence": np.where(positive, 255, 0).astype("u1"),
        "source_bits": positive.astype("u1") + 2 * (current > 0).astype("u1"),
    }


class SpatialIndex:
    def __init__(self, reference: dict, *, year: int | None):
        self.path = Path(reference["path"])
        if sha256(self.path) != reference["sha256"]:
            raise ValueError("OSM index changed")
        self.connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            self.metadata = {
                k: json.loads(v)
                for k, v in self.connection.execute("SELECT key,value FROM metadata")
            }
            if self.metadata.get("schema") != "xuannv.osm30-spatial-index.v1" or self.metadata.get(
                "channel_order"
            ) != list(CHANNELS):
                raise ValueError("unknown indexed OSM product contract")
            if year is not None and self.metadata.get("snapshot_date") != f"{year}-01-01":
                raise ValueError("historical geometry index has wrong snapshot date")
            self.raw_source = Path(self.metadata["source_path"])
            if sha256(self.raw_source) != self.metadata["source_sha256"]:
                raise ValueError("original OSM source changed after indexing")
            quick = self.connection.execute("PRAGMA quick_check").fetchall()
            if quick != [("ok",)]:
                raise ValueError("invalid OSM SQLite source")
            count, minimum, maximum, bits = self.connection.execute(
                "SELECT COUNT(*),MIN(width_m),MAX(width_m),MAX(channels) FROM features"
            ).fetchone()
            bounds_count = self.connection.execute(
                "SELECT COUNT(*) FROM feature_bounds"
            ).fetchone()[0]
            if (
                count != bounds_count
                or count != sum(self.metadata["feature_counts"].values())
                or (count and (minimum < 0 or not np.isfinite(maximum) or bits >= (1 << 30)))
            ):
                raise ValueError("OSM feature/RTree inventory mismatch")
            self.maximum_width = float(maximum or 0)
            self.fingerprint = {
                "path": str(self.path),
                "sha256": reference["sha256"],
                "metadata": self.metadata,
                "features": count,
                "maximum_width_m": self.maximum_width,
            }
            self.stamps = [
                (p.stat().st_size, p.stat().st_mtime_ns) for p in [self.path, self.raw_source]
            ]
            self.transformers = {}
        except BaseException:
            self.connection.close()
            raise

    def unchanged(self):
        return self.stamps == [
            (p.stat().st_size, p.stat().st_mtime_ns) for p in [self.path, self.raw_source]
        ]

    def close(self):
        self.connection.close()

    def query(self, epsg, bounds, padding_m=0):
        if padding_m < 0 or not np.isfinite(padding_m):
            raise ValueError("invalid query padding")
        b = tuple(float(v) for v in bounds)
        b = (b[0] - padding_m, b[1] - padding_m, b[2] + padding_m, b[3] + padding_m)
        xmin, ymin, xmax, ymax = transform_bounds(
            f"EPSG:{int(epsg)}", "EPSG:4326", *b, densify_pts=21
        )
        rows = self.connection.execute(
            "SELECT f.feature_id,f.channels,f.width_m,f.wkb "
            "FROM feature_bounds b JOIN features f USING(feature_id) "
            "WHERE b.maxx>=? AND b.minx<=? AND b.maxy>=? AND b.miny<=?",
            (xmin, xmax, ymin, ymax),
        ).fetchall()
        if epsg not in self.transformers:
            self.transformers[epsg] = Transformer.from_crs("EPSG:4326", int(epsg), always_xy=True)
        transformer = self.transformers[epsg]
        return [
            {
                "feature_id": i,
                "channels": int(c),
                "width_m": float(w),
                "geometry": transform_geometry(
                    transformer.transform, shapely.from_wkb(bytes(blob))
                ),
            }
            for i, c, w, blob in rows
        ]

    def coverages(self, epsg, bounds, pixels, oversample, channels):
        original = self.query(epsg, bounds)
        # Gaussian integer-index truncation includes one extra fine pixel outside its radius.
        padding = max(
            self.maximum_width / 2, (max(2, oversample * 2) + 1) * 1280 / (pixels * oversample)
        )
        expanded = self.query(epsg, bounds, padding)
        legacy = rasterize_records(original, bounds, pixels, oversample, channels)
        changed = {r["feature_id"] for r in expanded} != {r["feature_id"] for r in original}
        candidate = (
            rasterize_records(expanded, bounds, pixels, oversample, channels) if changed else legacy
        )
        return (
            legacy,
            candidate,
            {
                "legacy_features": len(original),
                "expanded_features": len(expanded),
                "query_padding_m": padding,
            },
        )


def _patch_coverages(indexes, row):
    low = {
        key: source.coverages(row.grid_epsg, row.utm_bounds, 128, 4, CHANNELS)
        for key, source in indexes.items()
    }
    structure_indices = [CHANNELS.index(c) for c in STRUCTURE]
    current_fine = None
    results = []
    for year in [2020, 2021]:
        historical, current = low[str(year)], low["current"]
        results.append(
            (
                year,
                "10m",
                CHANNELS,
                encode_coverage(historical[0], current[0]),
                encode_coverage(historical[1], current[1]),
                [historical[2], current[2]],
            )
        )
        legacy_has = bool(
            historical[0][structure_indices].any() or current[0][structure_indices].any()
        )
        candidate_has = bool(
            historical[1][structure_indices].any() or current[1][structure_indices].any()
        )
        empty = np.zeros((4, 512, 512), "f4")
        if legacy_has or candidate_has:
            if current_fine is None:
                current_fine = indexes["current"].coverages(
                    row.grid_epsg, row.utm_bounds, 512, 1, STRUCTURE
                )
            hist = indexes[str(year)].coverages(row.grid_epsg, row.utm_bounds, 512, 1, STRUCTURE)
            legacy = (
                encode_coverage(hist[0], current_fine[0])
                if legacy_has
                else encode_coverage(empty, empty)
            )
            candidate = (
                encode_coverage(hist[1], current_fine[1])
                if candidate_has
                else encode_coverage(empty, empty)
            )
            query = [hist[2], current_fine[2]]
        else:
            legacy = candidate = encode_coverage(empty, empty)
            query = []
        results.append((year, "2p5m", STRUCTURE, legacy, candidate, query))
    return results


def audit_osm_geometry(
    dataset_root: Path, report_root: Path, *, max_patches: int | None = None
) -> dict:
    if max_patches is not None and max_patches <= 0:
        raise ValueError("invalid OSM geometry patch limit")
    regpath = dataset_root / "registry/national_62000.parquet"
    manifest_path = dataset_root / "targets/manifest.parquet"
    valuepath = report_root / "target_value_audit.parquet"
    temporalpath = report_root / "osm_temporal_progress.json"
    registry = pd.read_parquet(regpath)
    manifest = pd.read_parquet(manifest_path)
    entries = manifest.loc[manifest.family == "osm"]
    temporal = json.loads(temporalpath.read_text())
    if (
        temporal.get("status") != "temporal_cross_checks_finished"
        or temporal.get("failed_groups") != 0
        or temporal.get("manifest_sha256") != sha256(manifest_path)
        or temporal.get("value_audit_sha256") != sha256(valuepath)
    ):
        raise ValueError("matching complete OSM temporal audit required")
    if (
        entries.path.nunique() != 1
        or not entries.registry_order_verified.all()
        or registry.empty
        or registry.patch_id.duplicated().any()
    ):
        raise ValueError("one complete OSM registry contract required")
    base = Path(entries.path.iloc[0])
    metadata_sha = sha256(base / ".zattrs")
    root = zarr.open_group(str(base), mode="r")
    attrs = dict(root.attrs)
    if (
        set(entries.source_metadata_sha256) != {metadata_sha}
        or temporal.get("metadata_sha256") != metadata_sha
        or attrs.get("patch_ids") != registry.patch_id.tolist()
        or attrs.get("target_names") != list(CHANNELS)
        or attrs.get("structure_names") != list(STRUCTURE)
        or attrs.get("external_evidence", "missing") is not None
        or attrs.get("target_encoding") != "uint8_coverage_or_soft_evidence_confidence"
        or attrs.get("years") != [2020, 2021]
        or attrs.get("unlabeled_is_unknown") is not True
    ):
        raise ValueError("unverified historical OSM raster contract")
    references = attrs.get("indexes", {})
    if set(references) != {"2020", "2021"}:
        raise ValueError("historical geometry dates missing")
    references = {**references, "current": attrs["current_index"]}
    indexes = {}
    scope = "full" if max_patches is None else f"pilot_{max_patches}"
    directory = dataset_root / "quality/targets/geometry/osm/v1" / scope
    directory.mkdir(parents=True, exist_ok=True)
    progress = report_root / f"osm_geometry_{scope}.json"
    write_json(
        progress,
        {
            "status": "checking_index_sources",
            "scope": scope,
            "processed_positions": 0,
            "selected_positions": min(len(registry), max_patches or len(registry)),
            "updated_at": now(),
        },
    )
    try:
        for key, ref in references.items():
            indexes[key] = SpatialIndex(ref, year=None if key == "current" else int(key))
        fingerprint = {
            "registry_sha256": sha256(regpath),
            "manifest_sha256": sha256(manifest_path),
            "value_audit_sha256": sha256(valuepath),
            "temporal_audit_sha256": sha256(temporalpath),
            "metadata_sha256": metadata_sha,
            "index_sources": {k: s.fingerprint for k, s in indexes.items()},
            "code_sha256": sha256(Path(__file__)),
            "parameters": {
                "10m_oversample": 4,
                "2p5m_oversample": 1,
                "all_touched": True,
                "line_cap_style": 2,
                "line_join_style": 2,
            },
            "runtime": {
                "numpy": np.__version__,
                "shapely": shapely.__version__,
                "rasterio": rasterio.__version__,
                "gdal": rasterio.__gdal_version__,
                "pyproj": pyproj.__version__,
                "proj": pyproj.proj_version_str,
            },
        }
        lock = directory / "input.lock.json"
        if lock.exists() and json.loads(lock.read_text()) != fingerprint:
            raise ValueError("OSM geometry inputs changed; use a new version")
        if not lock.exists():
            write_json(lock, fingerprint)
        values = pd.read_parquet(valuepath)
        values = values.loc[(values.family == "osm") & (values.path == str(base))]
        arrays = {}
        receipts = {}
        for year in [2020, 2021]:
            for branch, channels, pixels in [
                ("", CHANNELS, 128),
                ("structure_2p5m/", STRUCTURE, 512),
            ]:
                for field in FIELDS:
                    for channel in channels:
                        name = f"{year}/{branch}{field}/{channel}"
                        array = root[name]
                        record = values.loc[values.array == name]
                        if (
                            array.shape != (len(registry), pixels, pixels)
                            or array.dtype != np.dtype("u1")
                            or len(record) != 1
                            or record.iloc[0].status != "values_checked_provenance_pending"
                        ):
                            raise ValueError("OSM array shape or value receipt missing")
                        arrays[name] = array
                        receipts[name] = record.iloc[0].decoded_values_sha256
        digests = {name: hashlib.sha256() for name in arrays}
        results = []
        reused = 0
        selected = registry if max_patches is None else registry.iloc[:max_patches]
        for start in range(0, len(selected), 16):
            stop = min(start + 16, len(selected))
            block = {name: np.asarray(a[start:stop]) for name, a in arrays.items()}
            digest = hashlib.sha256()
            for name, a in block.items():
                digests[name].update(a.tobytes())
                digest.update(a.tobytes())
            expected = {
                **fingerprint,
                "start": start,
                "stop": stop,
                "decoded_chunk_sha256": digest.hexdigest(),
            }
            receipt = directory / "chunks" / f"{start:06d}.json"
            cached = json.loads(receipt.read_text()) if receipt.exists() else {}
            cache_valid = cached.get("fingerprint") == expected
            if cache_valid:
                for item in cached["rows"]:
                    if (
                        item.get("candidate_file")
                        and sha256(directory / item["candidate_file"]) != item["candidate_sha256"]
                    ):
                        raise ValueError("OSM boundary candidate changed")
                chunk = cached["rows"]
                reused += len(chunk)
            else:
                chunk = []
                for offset, row in enumerate(selected.iloc[start:stop].itertuples(index=False)):
                    try:
                        reconstructed = _patch_coverages(indexes, row)
                        for year, resolution, channels, legacy, candidate, query in reconstructed:
                            branch = "" if resolution == "10m" else "structure_2p5m/"
                            stored = {
                                field: np.stack(
                                    [block[f"{year}/{branch}{field}/{c}"][offset] for c in channels]
                                )
                                for field in FIELDS
                            }
                            differences = {
                                field: int((legacy[field] != stored[field]).sum())
                                for field in FIELDS
                            }
                            delta = {
                                field: int((candidate[field] != legacy[field]).sum())
                                for field in FIELDS
                            }
                            item = {
                                "patch_id": row.patch_id,
                                "index": start + offset,
                                "split": row.split,
                                "year": year,
                                "resolution": resolution,
                                "status": "failed" if any(differences.values()) else "passed",
                                "checked_channels": len(channels),
                                "mismatch_pixels": differences,
                                "boundary_change_pixels": delta,
                                "boundary_historical_added_pixels": int(
                                    ((candidate["states"] == 1) & (legacy["states"] == 0)).sum()
                                ),
                                "query": query,
                                "candidate_file": "",
                                "candidate_sha256": "",
                            }
                            if any(delta.values()):
                                name = (
                                    f"boundary_candidates/{start+offset:06d}"
                                    f"_{year}_{resolution}.npz"
                                )
                                path = directory / name
                                path.parent.mkdir(parents=True, exist_ok=True)
                                temp = path.with_suffix(".partial")
                                with temp.open("wb") as handle:
                                    np.savez_compressed(handle, **candidate)
                                temp.replace(path)
                                item.update(candidate_file=name, candidate_sha256=sha256(path))
                            chunk.append(item)
                    except (ValueError, OSError, shapely.errors.GEOSException) as exc:
                        chunk = [x for x in chunk if x["patch_id"] != row.patch_id]
                        chunk.extend(
                            {
                                "patch_id": row.patch_id,
                                "index": start + offset,
                                "split": row.split,
                                "year": year,
                                "resolution": res,
                                "status": "failed",
                                "reason": str(exc),
                                "candidate_file": "",
                                "boundary_historical_added_pixels": 0,
                            }
                            for year in [2020, 2021]
                            for res in ["10m", "2p5m"]
                        )
                write_json(receipt, {"fingerprint": expected, "rows": chunk})
            results.extend(chunk)
            write_json(
                progress,
                {
                    "status": "running",
                    "scope": scope,
                    "processed_positions": stop,
                    "selected_positions": len(selected),
                    "compared_views": len(results),
                    "failed_views": sum(x["status"] == "failed" for x in results),
                    "boundary_changed_views": sum(bool(x.get("candidate_file")) for x in results),
                    "reused_views": reused,
                    "updated_at": now(),
                },
            )
        if max_patches is None and any(
            digests[name].hexdigest() != receipts[name] for name in arrays
        ):
            raise ValueError("OSM labels changed after full value audit")
        if (
            not all(s.unchanged() for s in indexes.values())
            or sha256(base / ".zattrs") != metadata_sha
        ):
            raise ValueError("OSM geometry sources changed during audit")
        serial = [
            {
                **row,
                **{
                    k: json.dumps(row[k])
                    for k in ["mismatch_pixels", "boundary_change_pixels", "query"]
                    if k in row
                },
            }
            for row in results
        ]
        atomic_parquet(pd.DataFrame(serial), directory / "observations.parquet")
        summary = {
            "status": "osm_geometry_audit_finished",
            "scope": scope,
            "processed_positions": len(selected),
            "selected_positions": len(selected),
            "compared_views": len(results),
            "failed_views": sum(x["status"] == "failed" for x in results),
            "boundary_changed_views": sum(bool(x.get("candidate_file")) for x in results),
            "boundary_historical_added_pixels": sum(
                x["boundary_historical_added_pixels"] for x in results
            ),
            "reused_views": reused,
            "fingerprint": fingerprint,
            "output": str(directory / "observations.parquet"),
            "training_authorized": False,
            "boundary_candidates_authorized_as_labels": False,
            "limitation": "Index-to-raster equivalence and boundary-query diagnostics; "
            "source feature classification and label accuracy need separate evidence.",
            "finished_at": now(),
        }
        write_json(progress, summary)
        return summary
    finally:
        for source in indexes.values():
            source.close()
