"""Read native DSM tiles from split ZIPs without concatenating or extracting them."""

from __future__ import annotations

import io
import math
import re
import struct
import zipfile
from collections import OrderedDict
from pathlib import Path


class JoinedReader(io.RawIOBase):
    """A read-only seekable view over an explicitly ordered set of file parts."""

    def __init__(self, paths: list[Path]):
        super().__init__()
        self.paths = [path.resolve() for path in paths]
        self.files = [path.open("rb") for path in self.paths]
        self.sizes = [path.stat().st_size for path in self.paths]
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError("Unknown seek origin")
        base = (
            0
            if whence == io.SEEK_SET
            else (self.position if whence == io.SEEK_CUR else sum(self.sizes))
        )
        if base + offset < 0:
            raise ValueError("Negative seek position")
        self.position = base + offset
        return self.position

    def read(self, size=-1):
        remaining = max(0, sum(self.sizes) - self.position)
        if size >= 0:
            remaining = min(size, remaining)
        result, start = [], 0
        for handle, length in zip(self.files, self.sizes):
            if remaining and start <= self.position < start + length:
                handle.seek(self.position - start)
                payload = handle.read(min(remaining, start + length - self.position))
                if not payload:
                    raise OSError("Archive part was truncated while reading")
                result.append(payload)
                self.position += len(payload)
                remaining -= len(payload)
            start += length
        return b"".join(result)

    def close(self):
        for handle in self.files:
            handle.close()
        super().close()


class NativeDemArchive:
    """Cache independent raster handles; instances must not be shared between threads."""

    def __init__(self, parts: list[Path], *, cache_size: int = 16):
        self.reader = JoinedReader(parts)
        self.archive = zipfile.ZipFile(self.reader)
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.tiles = {}
        for entry in self.archive.infolist():
            if not entry.filename.lower().endswith(".tif"):
                continue
            match = re.search(r"_([NS])(\d+)_00_([EW])(\d+)_00_DEM\.tif$", entry.filename)
            if not match:
                raise ValueError(f"Unrecognized DEM tile identity: {entry.filename}")
            latitude = int(match[2]) * (1 if match[1] == "N" else -1)
            longitude = int(match[4]) * (1 if match[3] == "E" else -1)
            key = (latitude, longitude)
            if key in self.tiles:
                raise ValueError(f"Duplicate native DEM tile: {key}")
            self.tiles[key] = entry
        if not self.tiles:
            raise ValueError("Native DEM archive contains no tiles")

    def open_tile(self, entry):
        import rasterio
        from rasterio.io import MemoryFile

        if entry.filename in self.cache:
            self.cache.move_to_end(entry.filename)
            return self.cache[entry.filename][0]
        self.reader.seek(entry.header_offset)
        header = self.reader.read(30)
        if header[:4] != b"PK\x03\x04" or entry.flag_bits & 1:
            raise ValueError("Invalid or encrypted native ZIP member")
        filename_length, extra_length = struct.unpack("<HH", header[26:30])
        offset = entry.header_offset + 30 + filename_length + extra_length
        path, memory, start = None, None, 0
        if entry.compress_type == zipfile.ZIP_STORED:
            for part, size in zip(self.reader.paths, self.reader.sizes):
                if start <= offset and offset + entry.file_size <= start + size:
                    path = f"/vsisubfile/{offset - start}_{entry.file_size},{part}"
                    break
                start += size
        if path is None:
            memory = MemoryFile(self.archive.read(entry))
            raster = memory.open()
        else:
            raster = rasterio.open(path)
        if raster.crs is None or raster.crs.to_epsg() != 4326 or raster.count != 1:
            raster.close()
            if memory:
                memory.close()
            raise ValueError("Native DEM must be a single-band EPSG:4326 raster")
        self.cache[entry.filename] = raster, memory
        if len(self.cache) > self.cache_size:
            _, (old, old_memory) = self.cache.popitem(last=False)
            old.close()
            if old_memory:
                old_memory.close()
        return raster

    def read_patch(self, row: dict, *, spacing_m: int = 40, resampling: str = "average") -> dict:
        """Reproject a local native-grid mosaic; no vertical conversion or value clipping."""
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.merge import merge
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject, transform_bounds
        from rasterio.windows import Window, bounds
        from rasterio.windows import from_bounds as window_from_bounds

        if spacing_m < 10 or 1280 % spacing_m:
            raise ValueError("Spacing must divide the parent width and be at least 10 m")
        method = Resampling[resampling]
        shape = (1280 // spacing_m,) * 2
        output = np.full(shape, np.nan, dtype=np.float32)
        target_transform = from_bounds(*row["utm_bounds"], shape[1], shape[0])
        geographic = transform_bounds(row["grid_epsg"], 4326, *row["utm_bounds"], densify_pts=21)
        # Include a native-cell halo for interpolation and geocell seams.
        west, south, east, north = geographic
        entries = [
            self.tiles[(lat, lon)]
            for lat in range(math.floor(south - 0.001), math.floor(north + 0.001) + 1)
            for lon in range(math.floor(west - 0.001), math.floor(east + 0.001) + 1)
            if (lat, lon) in self.tiles
        ]
        if entries:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_PAM_ENABLED="NO"):
                sources = [self.open_tile(entry) for entry in entries]
                source = sources[0]
                window = window_from_bounds(*geographic, transform=source.transform)
                left, top = math.floor(window.col_off) - 2, math.floor(window.row_off) - 2
                right = math.ceil(window.col_off + window.width) + 2
                bottom = math.ceil(window.row_off + window.height) + 2
                mosaic_bounds = bounds(
                    Window(left, top, right - left, bottom - top), source.transform
                )
                resolution = (
                    min(src.res[0] for src in sources),
                    min(src.res[1] for src in sources),
                )
                mosaic, transform = merge(
                    sources,
                    bounds=mosaic_bounds,
                    res=resolution,
                    nodata=np.nan,
                    dtype="float32",
                    method="first",
                    resampling=Resampling.nearest,
                )
                reproject(
                    source=mosaic[0],
                    destination=output,
                    src_transform=transform,
                    src_crs=4326,
                    src_nodata=np.nan,
                    dst_transform=target_transform,
                    dst_crs=row["grid_epsg"],
                    dst_nodata=np.nan,
                    resampling=method,
                    num_threads=1,
                )
        return {
            "values": np.where(np.isfinite(output), output, 0),
            "valid": np.isfinite(output),
            "members": [entry.filename for entry in entries],
            "transform": list(target_transform)[:6],
            "native_x_spacings_degree": sorted({self.open_tile(entry).res[0] for entry in entries}),
        }

    def close(self):
        for raster, memory in self.cache.values():
            raster.close()
            if memory:
                memory.close()
        self.cache.clear()
        self.archive.close()
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
