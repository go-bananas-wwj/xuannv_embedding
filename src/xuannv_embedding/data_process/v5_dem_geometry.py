"""Read split DEM sources in place and audit elevation, slope, and derivative support."""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import math
import re
import struct
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

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
from xuannv_embedding.data_process.v5_target_geometry import compare_target


class ConcatenatedReader(io.RawIOBase):
    """Seek across raw byte parts without constructing another copy of the archive."""

    def __init__(self, parts: list[Path]):
        super().__init__()
        if not parts or len(set(parts)) != len(parts):
            raise ValueError("ordered unique source parts are required")
        self.ends = np.cumsum([p.stat().st_size for p in parts]).tolist()
        if any(p.stat().st_size <= 0 for p in parts):
            raise ValueError("empty source part")
        self.handles = []
        try:
            for part in parts:
                self.handles.append(part.open("rb"))
        except BaseException:
            self.close()
            raise
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if whence not in {io.SEEK_SET, io.SEEK_CUR, io.SEEK_END}:
            raise ValueError("invalid seek mode")
        position = (
            offset
            + {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.ends[-1]}[whence]
        )
        if position < 0:
            raise ValueError("seek before start of concatenated source")
        self.position = position
        return position

    def readinto(self, buffer) -> int:
        self._checkClosed()
        view, total = memoryview(buffer), 0
        while total < len(view) and self.position < self.ends[-1]:
            index = bisect.bisect_right(self.ends, self.position)
            start = 0 if index == 0 else self.ends[index - 1]
            handle = self.handles[index]
            handle.seek(self.position - start)
            count = handle.readinto(
                view[total : total + min(len(view) - total, self.ends[index] - self.position)]
            )
            if not count:
                raise OSError("source part shortened during read")
            total += count
            self.position += count
        return total

    def close(self) -> None:
        for handle in getattr(self, "handles", []):
            handle.close()
        super().close()


def write_sparse_descriptor(
    parts: list[Path], output: Path, *, offset: int = 0, length: int | None = None
) -> str:
    """Map a stored member's exact bytes with GDAL /vsisparse/, without copying pixels."""
    if not parts or len(set(parts)) != len(parts) or any(p.stat().st_size <= 0 for p in parts):
        raise ValueError("ordered, nonempty, unique source parts are required")
    total = sum(p.stat().st_size for p in parts)
    length = total - offset if length is None else length
    if offset < 0 or length <= 0 or offset + length > total:
        raise ValueError("member range outside source parts")
    root = ET.Element("VSISparseFile")
    ET.SubElement(root, "Length").text = str(length)
    position = 0
    for part in parts:
        size = part.stat().st_size
        start, stop = max(offset, position), min(offset + length, position + size)
        if start < stop:
            region = ET.SubElement(root, "SubfileRegion")
            ET.SubElement(region, "Filename", relative="0").text = str(part.resolve())
            for key, value in [
                ("DestinationOffset", start - offset),
                ("SourceOffset", start - position),
                ("RegionLength", stop - start),
            ]:
                ET.SubElement(region, key).text = str(value)
        position += size
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != payload:
        raise ValueError("split source descriptor changed")
    if not output.exists():
        temporary = output.with_suffix(".partial")
        temporary.write_bytes(payload)
        temporary.replace(output)
    return f"/vsisparse/{output.resolve()}"


def stored_member_range(stream: ConcatenatedReader, info: ZipInfo) -> tuple[int, int]:
    if info.compress_type != ZIP_STORED or info.flag_bits & 1:
        raise ValueError("DEM source requires unencrypted stored ZIP members")
    stream.seek(info.header_offset)
    header = stream.read(30)
    if (
        len(header) != 30
        or header[:4] != b"PK\x03\x04"
        or struct.unpack("<H", header[8:10])[0] != ZIP_STORED
    ):
        raise ValueError("invalid stored ZIP local header")
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    return info.header_offset + 30 + name_length + extra_length, info.file_size


def slope_support(valid: np.ndarray) -> np.ndarray:
    if valid.ndim != 2 or min(valid.shape) < 2:
        raise ValueError("slope support requires a 2D grid with neighbors")
    supported = np.asarray(valid, bool).copy()
    supported[1:-1] &= valid[:-2] & valid[2:]
    supported[0] &= valid[1]
    supported[-1] &= valid[-2]
    supported[:, 1:-1] &= valid[:, :-2] & valid[:, 2:]
    supported[:, 0] &= valid[:, 1]
    supported[:, -1] &= valid[:, -2]
    return supported


def legacy_slope(elevation: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Reconstruct the recorded old policy; unsupported derivatives remain separately flagged."""
    if elevation.shape != valid.shape or elevation.ndim != 2:
        raise ValueError("elevation and mask differ")
    filled = elevation.copy()
    if not valid.any():
        return np.zeros_like(elevation, dtype="f4")
    if not np.isfinite(elevation[valid]).all():
        raise ValueError("valid nonfinite elevation")
    filled[~valid] = float(np.median(elevation[valid]))
    dy, dx = np.gradient(filled, 10.0, 10.0)
    return np.degrees(np.arctan(np.hypot(dx, dy))).astype("f4")


class DEMSource:
    def __init__(self, parts: list[Path], descriptor: Path):
        self.readers = OrderedDict()
        self.uris = {}
        with ConcatenatedReader(parts) as stream, ZipFile(stream) as archive:
            infos = sorted(
                (i for i in archive.infolist() if i.filename.lower().endswith(".tif")),
                key=lambda i: i.filename,
            )
            names = [i.filename for i in infos]
            if not names or len(set(names)) != len(names):
                raise ValueError("empty or duplicate DEM member list")
            for info in infos:
                offset, size = stored_member_range(stream, info)
                name = hashlib.sha256(info.filename.encode()).hexdigest() + ".xml"
                self.uris[info.filename] = write_sparse_descriptor(
                    parts, descriptor.parent / "members" / name, offset=offset, length=size
                )
        self.entries = []
        for name in names:
            if not re.fullmatch(
                r"Copernicus_DSM_COG_10_[NS][0-9]+_00_[EW][0-9]+_00_DEM\.tif", Path(name).name
            ):
                raise ValueError("unexpected DEM product member")
            with rasterio.open(self.uri(name)) as src:
                if (
                    src.count != 1
                    or src.crs is None
                    or src.scales != (1.0,)
                    or src.offsets != (0.0,)
                ):
                    raise ValueError("unverified DEM grid or numeric contract")
                self.entries.append(
                    {
                        "member": name,
                        "crs": src.crs.to_string(),
                        "transform": list(src.transform)[:6],
                        "bounds": list(src.bounds),
                        "shape": [src.height, src.width],
                        "nodata": src.nodata,
                        "dtype": src.dtypes[0],
                        "wgs84_bounds": list(
                            transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
                        ),
                    }
                )

    def uri(self, member: str) -> str:
        return self.uris[member]

    def reader(self, name: str):
        src = self.readers.pop(name, None)
        if src is None:
            src = rasterio.open(self.uri(name))
        self.readers[name] = src
        while len(self.readers) > 4:
            self.readers.popitem(last=False)[1].close()
        return src

    def close(self) -> None:
        for src in self.readers.values():
            src.close()
        self.readers.clear()

    def reconstruct(self, epsg: int, bounds) -> tuple[np.ndarray, np.ndarray, list[str]]:
        bounds = tuple(float(v) for v in bounds)
        if len(bounds) != 4 or not np.allclose(
            [bounds[2] - bounds[0], bounds[3] - bounds[1]], 1280, rtol=0, atol=1e-6
        ):
            raise ValueError("DEM target differs from frozen grid")
        crs = rasterio.crs.CRS.from_epsg(int(epsg))
        transform = from_bounds(*bounds, 128, 128)
        geographic = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
        output, valid, used = np.zeros((128, 128), "f4"), np.zeros((128, 128), bool), []
        for entry in self.entries:
            b = entry["wgs84_bounds"]
            if not (
                geographic[0] < b[2]
                and geographic[2] > b[0]
                and geographic[1] < b[3]
                and geographic[3] > b[1]
            ):
                continue
            src = self.reader(entry["member"])
            box = transform_bounds(crs, src.crs, *bounds, densify_pts=21)
            request = window_from_bounds(*box, transform=src.transform)
            left, top = math.floor(request.col_off) - 2, math.floor(request.row_off) - 2
            right, bottom = (
                math.ceil(request.col_off + request.width) + 2,
                math.ceil(request.row_off + request.height) + 2,
            )
            try:
                window = Window(left, top, right - left, bottom - top).intersection(
                    Window(0, 0, src.width, src.height)
                )
            except rasterio.errors.WindowError:
                continue
            raw = src.read(1, window=window).astype("f4")
            raw[(src.read_masks(1, window=window) == 0) | ~np.isfinite(raw)] = np.nan
            tile = np.full((128, 128), np.nan, "f4")
            reproject(
                raw,
                tile,
                src_transform=src.window_transform(window),
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=transform,
                dst_crs=crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            current = np.isfinite(tile)
            output[current] = tile[current]
            valid |= current
            used.append(entry["member"])
        return output, valid, used


def audit_dem_geometry(
    dataset_root: Path, report_root: Path, *, max_patches: int | None = None
) -> dict:
    if max_patches is not None and max_patches <= 0:
        raise ValueError("invalid DEM patch limit")
    provenance = json.loads((report_root / "target_source_audit.json").read_text())
    if provenance.get("status") != "source_audit_finished" or provenance.get("failed_sources") != 0:
        raise ValueError("completed source audit required")
    references = [
        r
        for r in provenance["sources"]
        if r["family"] == "static"
        and re.search(r"copernicus_dem_glo30\.zip\.part[0-9]+$", r["path"])
    ]
    references.sort(key=lambda r: int(r["path"].rsplit("part", 1)[1]))
    if not references or [int(r["path"].rsplit("part", 1)[1]) for r in references] != list(
        range(1, len(references) + 1)
    ):
        raise ValueError("DEM split archive has missing or duplicate parts")
    parts = [Path(r["path"]) for r in references]
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in parts]
    for reference, part in zip(references, parts, strict=True):
        if sha256(part) != reference["actual_sha256"]:
            raise ValueError("DEM source changed after provenance audit")
    registry_path = dataset_root / "registry/national_62000.parquet"
    manifest_path = dataset_root / "targets/manifest.parquet"
    registry, manifest = pd.read_parquet(registry_path), pd.read_parquet(manifest_path)
    names = ["dem_elevation", "dem_slope"]
    rows = manifest.loc[
        (manifest.family == "static") & manifest.array.isin([f"targets/{n}" for n in names])
    ]
    if (
        len(rows) != 2
        or rows.path.nunique() != 1
        or not rows.registry_order_verified.all()
        or not (rows.temporal_mode == "static").all()
    ):
        raise ValueError("DEM static label contract missing")
    target = zarr.open_group(str(rows.path.iloc[0]), mode="r")
    if (
        list(target.attrs["patch_ids"]) != registry.patch_id.tolist()
        or registry.patch_id.duplicated().any()
    ):
        raise ValueError("DEM label order differs from frozen registry")
    scope = "full" if max_patches is None else f"pilot_{max_patches}"
    directory = dataset_root / "quality/targets/geometry/dem" / scope
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ["v5_dem_geometry.py", "v5_target_geometry.py"]
        },
        "registry_sha256": sha256(registry_path),
        "manifest_sha256": sha256(manifest_path),
        "sources": [
            {"path": str(p), "sha256": r["actual_sha256"]}
            for p, r in zip(parts, references, strict=True)
        ],
        "runtime": {
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "gdal": rasterio.__gdal_version__,
        },
    }
    source = DEMSource(parts, directory / "source.xml")
    write_json(
        directory / "source_index.json", {"fingerprint": fingerprint, "members": source.entries}
    )
    selected = registry if max_patches is None else registry.iloc[:max_patches]
    arrays = {
        f"{group}/{name}": target[f"{group}/{name}"]
        for group in ["targets", "valid_masks"]
        for name in names
    }
    if any(a.shape != (len(registry), 128, 128) for a in arrays.values()):
        source.close()
        raise ValueError("DEM array shape disagrees with frozen grid")
    digests = {name: hashlib.sha256() for name in arrays}
    results, reused = [], 0
    progress = report_root / f"target_geometry_dem_{scope}.json"
    try:
        for start in range(0, len(selected), 32):
            stop = min(len(selected), start + 32)
            blocks = {name: np.asarray(a[start:stop]) for name, a in arrays.items()}
            block_hash = hashlib.sha256()
            for name, values in blocks.items():
                digests[name].update(values.tobytes())
                block_hash.update(values.tobytes())
            expected = {
                **fingerprint,
                "start": start,
                "stop": stop,
                "target_chunk_sha256": block_hash.hexdigest(),
            }
            path = directory / "chunks" / f"{start:06d}.json"
            cached = json.loads(path.read_text()) if path.exists() else {}
            if cached.get("fingerprint") == expected:
                chunk = cached["rows"]
                reused += len(chunk)
            else:
                chunk = []
                for index, row in enumerate(selected.iloc[start:stop].itertuples()):
                    common = {
                        "patch_id": row.patch_id,
                        "split": row.split,
                        "temporal_mode": "static",
                    }
                    try:
                        elevation, valid, members = source.reconstruct(
                            row.grid_epsg, row.utm_bounds
                        )
                        slope = legacy_slope(elevation, valid)
                        supported = slope_support(valid)
                        for name, fresh in [("dem_elevation", elevation), ("dem_slope", slope)]:
                            actual_valid = blocks[f"valid_masks/{name}"][index].astype(bool)
                            result = compare_target(
                                np.where(valid, fresh, 0),
                                valid,
                                blocks[f"targets/{name}"][index],
                                actual_valid,
                                categorical=False,
                            )
                            unsupported = (
                                int((actual_valid & ~supported).sum()) if name == "dem_slope" else 0
                            )
                            chunk.append(
                                {
                                    **common,
                                    "target": name,
                                    **result,
                                    "source_members": members,
                                    "unsupported_slope_pixels": unsupported,
                                }
                            )
                    except (ValueError, OSError) as exc:
                        for name in names:
                            chunk.append(
                                {
                                    **common,
                                    "target": name,
                                    "status": "failed",
                                    "reason": str(exc),
                                    "source_members": [],
                                    "unsupported_slope_pixels": 0,
                                }
                            )
                write_json(path, {"fingerprint": expected, "rows": chunk})
            results.extend(chunk)
            write_json(
                progress,
                {
                    "status": "running",
                    "scope": scope,
                    "processed_targets": len(results),
                    "selected_targets": len(selected) * 2,
                    "failed_targets": sum(r["status"] == "failed" for r in results),
                    "unsupported_slope_pixels": sum(r["unsupported_slope_pixels"] for r in results),
                    "reused_targets": reused,
                    "updated_at": now(),
                },
            )
        if max_patches is None:
            audit = pd.read_parquet(report_root / "target_value_audit.parquet")
            for name, digest in digests.items():
                item = audit.loc[(audit.family == "static") & (audit.array == name)]
                if len(item) != 1 or item.iloc[0].decoded_values_sha256 != digest.hexdigest():
                    raise ValueError("DEM labels changed after completed value audit")
    finally:
        source.close()
    if before != [(p.stat().st_size, p.stat().st_mtime_ns) for p in parts]:
        raise ValueError("DEM source changed during audit")
    atomic_parquet(
        pd.DataFrame([{**r, "source_members": json.dumps(r["source_members"])} for r in results]),
        directory / "observations.parquet",
    )
    summary = {
        "status": "dem_geometry_audit_finished",
        "scope": scope,
        "processed_targets": len(results),
        "selected_targets": len(selected) * 2,
        "failed_targets": sum(r["status"] == "failed" for r in results),
        "unsupported_slope_pixels": sum(r["unsupported_slope_pixels"] for r in results),
        "reused_targets": reused,
        "fingerprint": fingerprint,
        "output": str(directory / "observations.parquet"),
        "training_authorized": False,
        "limitation": "Reconstructs recorded median-fill slope policy; "
        "unsupported derivatives are flagged separately, not silently approved.",
        "finished_at": now(),
    }
    write_json(progress, summary)
    return summary
