"""Stream real RGB working-grid observations into geographic subpixel caches."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS

from xuannv_embedding.data_process import observation_time
from xuannv_embedding.data_process.observation_time import ObservationInterval, month_support
from xuannv_embedding.export import public_visual_subspace as shared
from xuannv_embedding.export import semantic_features
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.frozen_fusion import source_array, verified

read_json = shared.read_json


def _clean_frame(image, coverage):
    image, coverage = np.asarray(image, np.float32), np.asarray(coverage, np.float32)
    if (
        image.ndim != 3
        or image.shape[0] != 3
        or coverage.shape != (1, *image.shape[-2:])
        or not np.isfinite(coverage).all()
        or (coverage < 0).any()
        or (coverage > 1).any()
    ):
        raise ValueError("Invalid RGB coverage or channel geometry")
    clean = np.where(coverage > 0, image, 0)
    if not np.isfinite(clean).all() or (clean < 0).any() or (clean > 255).any():
        raise ValueError("Invalid observed prepared RGB values; expected digital values in [0,255]")
    return clean, coverage


def pool_rgb(frames, shape):
    total = np.zeros((3, *shape), np.float64)
    count = np.zeros((1, *shape), np.float64)
    n = 0
    for image, coverage in frames:
        image, coverage = _clean_frame(image, coverage)
        if image.shape[-2:] != tuple(shape):
            raise ValueError("RGB frame geometry differs")
        total += image.astype(np.float64) * coverage
        count += coverage
        n += 1
    return (total / np.maximum(count, 1e-12)).astype(np.float32), (count / max(1, n)).astype(
        np.float32
    )


def pack_subpixels(image, coverage, *, factor):
    image, coverage = _clean_frame(image, coverage)
    h, w = image.shape[-2:]
    if type(factor) is not int or factor < 1 or h % factor or w % factor:
        raise ValueError("Subpixel packing geometry must divide into complete cells")

    def pack(values):
        c = len(values)
        return (
            values.reshape(c, h // factor, factor, w // factor, factor)
            .transpose(0, 2, 4, 1, 3)
            .reshape(c * factor**2, h // factor, w // factor)
            .copy()
        )

    return pack(image), pack(coverage)


def validate_specification(spec):
    keys = {
        "schema",
        "index",
        "public",
        "views",
        "time_contract",
        "working_pixel_m",
        "output_pixel_m",
        "factor",
        "source_shape",
        "output_shape",
        "output",
        "reserve_gib",
    }
    if (
        not isinstance(spec, dict)
        or set(spec) != keys
        or spec["schema"] != "subpixel-rgb-cache-v1"
        or type(spec["factor"]) is not int
        or spec["factor"] != 5
        or any(
            type(spec[k]) not in {int, float} or not math.isfinite(spec[k]) or spec[k] <= 0
            for k in ["working_pixel_m", "output_pixel_m"]
        )
        or spec["output_pixel_m"] != spec["factor"] * spec["working_pixel_m"]
        or any(
            not isinstance(spec[k], list)
            or len(spec[k]) != 2
            or any(type(v) is not int or v < 1 for v in spec[k])
            for k in ["source_shape", "output_shape"]
        )
        or spec["source_shape"] != [v * spec["factor"] for v in spec["output_shape"]]
        or type(spec["reserve_gib"]) is not int
        or spec["reserve_gib"] < 0
        or not isinstance(spec["output"], str)
        or not Path(spec["output"]).is_absolute()
        or not isinstance(spec["public"], dict)
        or set(spec["public"]) != {"array", "domain"}
        or not isinstance(spec["public"]["domain"], dict)
        or set(spec["public"]["domain"]) != {"array", "columns"}
        or not isinstance(spec["views"], list)
    ):
        raise ValueError("Invalid self-contained subpixel RGB specification")
    refs = [spec["index"], spec["time_contract"], spec["public"]["domain"]["columns"]]
    for i, view in enumerate(spec["views"]):
        if (
            not isinstance(view, dict)
            or set(view) != {"slot", "producer_spec", "rasters", "geometry"}
            or type(view["slot"]) is not int
            or view["slot"] != i
        ):
            raise ValueError("Invalid actual-view specification inventory")
        refs.extend(view[k] for k in ["producer_spec", "rasters", "geometry"])
    for ref in refs:
        if (
            not isinstance(ref, dict)
            or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str)
            or not Path(ref["path"]).is_absolute()
            or not isinstance(ref["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None
        ):
            raise ValueError("Invalid source reference in subpixel RGB specification")
    for value in [spec["public"]["array"], spec["public"]["domain"]["array"]]:
        if (
            not isinstance(value, dict)
            or set(value) != {"path", "sha256", "layout", "index_sha256"}
            or not isinstance(value["path"], str)
            or not Path(value["path"]).is_absolute()
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or value["index_sha256"] != spec["index"]["sha256"]
            or value["layout"] not in {"nchw", "nhwc"}
        ):
            raise ValueError("Invalid public array in subpixel RGB specification")
    return spec


def specification(path):
    spec = validate_specification(read_json(path))
    index = read_json(verified(spec["index"]))
    if not index.get("records") or len({r["patch_id"] for r in index["records"]}) != len(
        index["records"]
    ):
        raise ValueError("Invalid subpixel geographic index")
    return spec, index


def _same_geography(originals, records, default_crs):
    if len(originals) != len(records):
        raise ValueError("RGB source geographic tile count differs")
    for a, b in zip(originals, records):
        if (
            a["patch_id"] != b["patch_id"]
            or a["bounds"] != b["bounds"]
            or CRS.from_user_input(a.get("crs", default_crs)) != CRS.from_user_input(b["crs"])
        ):
            raise ValueError("RGB source geographic order differs")


class Sources:
    def __init__(self, spec, index):
        self.spec, self.index, self.n = spec, index, len(index["records"])
        p = source_array(spec["public"]["array"], spec["index"]["sha256"], self.n)
        if p.shape != (self.n, 64, *spec["output_shape"]):
            raise ValueError("Public feature grid differs from RGB output geometry")
        self.contract = read_json(verified(spec["time_contract"]))
        contract = self.contract
        if (
            contract.get("state") != "complete"
            or contract.get("one_row_per_actual_view_not_month_copy") is not True
            or len(contract.get("views", [])) != len(spec["views"])
        ):
            raise ValueError("RGB actual-view calendar differs")
        if spec["views"]:
            expected = month_support(
                [ObservationInterval.from_mapping(v["source_time"]) for v in contract["views"]],
                contract["months"],
            )
        else:
            expected = np.zeros((0, len(contract["months"])), np.float32)
        if not np.array_equal(expected, np.load(verified(contract["weights"]))):
            raise ValueError("RGB physical time support changed")
        self.records = [{**r, "crs": r.get("crs", index.get("crs"))} for r in index["records"]]
        self.paths = []
        seen = set()
        for i, view in enumerate(spec["views"]):
            if contract["views"][i]["slot"] != i:
                raise ValueError("RGB observation order differs")
            producer = read_json(verified(view["producer_spec"]))
            if (
                producer.get("rasters") != view["rasters"]
                or producer.get("cache") != view["geometry"]
            ):
                raise ValueError("RGB producer identity differs")
            geometry = read_json(verified(view["geometry"]))
            if geometry.get("state") != "complete" or geometry.get("months") != contract["months"]:
                raise ValueError("RGB source calendar/geometry incomplete")
            _same_geography(index["records"], geometry["records"], index.get("crs"))
            paths = semantic_features.registered_rasters(producer, geometry)
            key = tuple(paths)
            if key in seen:
                raise ValueError("Duplicate actual RGB observation")
            seen.add(key)
            self.paths.append(paths)

    def tile(self, position):
        record = self.records[position]
        spec = self.spec

        def frames():
            for paths in self.paths:
                path = paths[position]
                with rasterio.open(path) as source:
                    crs = CRS.from_user_input(source.crs)
                    if (
                        source.count != 3
                        or not crs.is_projected
                        or any(abs(v - spec["working_pixel_m"]) > 1e-6 for v in source.res)
                        or source.transform.b != 0
                        or source.transform.d != 0
                        or source.transform.a <= 0
                        or source.transform.e >= 0
                        or any(
                            abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in crs.axis_info
                        )
                    ):
                        raise ValueError(
                            "RGB raster resolution or north-up metric geometry differs"
                        )
                image, mask = semantic_features.read_aerial_window(path, record)
                if list(image.shape[-2:]) != spec["source_shape"]:
                    raise ValueError("RGB raster physical window geometry differs")
                yield image.numpy(), mask.numpy()

        image, coverage = pool_rgb(frames(), spec["source_shape"])
        return pack_subpixels(image, coverage, factor=spec["factor"])


def identity(path):
    return {
        "spec_sha256": sha(path),
        "implementation": {
            "subpixel_rgb": sha(__file__),
            "raster_reader": sha(semantic_features.__file__),
            "physical_time": sha(observation_time.__file__),
        },
    }


def _array_ref(path, digest):
    return {"path": str(path), "sha256": sha(path), "layout": "nchw", "index_sha256": digest}


def load_manifest(path):
    record = read_json(path)
    if record.get("state") != "complete" or record.get("schema") != "subpixel-rgb-cache-data-v1":
        raise ValueError("Incomplete subpixel cache")
    n, digest = record["registered_tiles"], record["index"]["sha256"]
    for name, channels in [("rgb", 75), ("coverage", 25)]:
        values = source_array(record[name], digest, n)
        if values.shape != (n, channels, *record["output_shape"]) or values.dtype != np.float16:
            raise ValueError("Subpixel cache dimensions or dtype changed")
    return record


def prepare(path, *, max_tiles=None):
    if max_tiles is not None and (type(max_tiles) is not int or max_tiles < 1):
        raise ValueError("Partial cache limit must be positive")
    spec, index = specification(path)
    root, ident = Path(spec["output"]), identity(path)
    with shared.ownership(root, ident):
        manifest = root / "manifest.json"
        if manifest.exists():
            result = load_manifest(manifest)
            if result["identity"] != ident:
                raise ValueError("Subpixel cache identity changed")
            return result
        sources = Sources(spec, index)
        shapes = {
            "rgb": (sources.n, 75, *spec["output_shape"]),
            "coverage": (sources.n, 25, *spec["output_shape"]),
        }
        progress_path = root / "progress.json"
        progress = (
            read_json(progress_path)
            if progress_path.exists()
            else {"identity": ident, "next_tile": 0, "observed_tiles": 0, "chunks": []}
        )
        if (
            set(progress) != {"identity", "next_tile", "observed_tiles", "chunks"}
            or progress["identity"] != ident
            or type(progress["next_tile"]) is not int
            or not 0 <= progress["next_tile"] <= sources.n
            or type(progress["observed_tiles"]) is not int
            or not 0 <= progress["observed_tiles"] <= progress["next_tile"]
            or not isinstance(progress["chunks"], list)
        ):
            raise ValueError("Subpixel recovery identity changed")
        arrays, paths = {}, {}
        needed = sum(
            math.prod(shape) * 2
            for name, shape in shapes.items()
            if not (root / (name + ".npy")).exists()
            and not (root / (name + ".partial.npy")).exists()
        )
        if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30 + needed:
            raise RuntimeError("Subpixel cache disk reserve")
        for name, shape in shapes.items():
            final = root / (name + ".npy")
            pending = root / (name + ".partial.npy")
            active = final if final.exists() else pending
            if active.exists():
                array = np.lib.format.open_memmap(active, mode="r+")
                if array.shape != shape or array.dtype != np.float16:
                    raise ValueError("Subpixel recovery array geometry changed")
            else:
                if progress["next_tile"] != 0:
                    raise ValueError("Missing partial subpixel cache")
                array = np.lib.format.open_memmap(active, mode="w+", dtype=np.float16, shape=shape)
            arrays[name], paths[name] = array, active

        def chunk_digest(name, first, last):
            return hashlib.sha256(arrays[name][first:last].tobytes()).hexdigest()

        checked, observed = 0, 0
        for chunk in progress["chunks"]:
            if (
                set(chunk) != {"start", "stop", "rgb", "coverage", "observed_tiles"}
                or type(chunk["start"]) is not int
                or chunk["start"] != checked
                or type(chunk["stop"]) is not int
                or not checked < chunk["stop"] <= progress["next_tile"]
                or any(chunk[name] != chunk_digest(name, checked, chunk["stop"]) for name in arrays)
            ):
                raise ValueError("Subpixel recovery prefix changed")
            count = int(
                np.any(arrays["coverage"][checked : chunk["stop"]] > 0, axis=(1, 2, 3)).sum()
            )
            if type(chunk["observed_tiles"]) is not int or chunk["observed_tiles"] != count:
                raise ValueError("Subpixel recovery observation count changed")
            observed += count
            checked = chunk["stop"]
        if checked != progress["next_tile"] or observed != progress["observed_tiles"]:
            raise ValueError("Subpixel recovery prefix is incomplete")
        start = progress["next_tile"]
        chunk_start = start
        stop = min(sources.n, start + max_tiles) if max_tiles is not None else sources.n
        for i in range(start, stop):
            if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30:
                raise RuntimeError("Subpixel cache disk reserve")
            values, coverage = sources.tile(i)
            if not np.isfinite(values).all() or not np.isfinite(coverage).all():
                raise ValueError("Nonfinite subpixel cache values")
            arrays["rgb"][i], arrays["coverage"][i] = values, coverage
            progress["observed_tiles"] += int((coverage > 0).any())
            progress["next_tile"] = i + 1
            if (i + 1) % 16 == 0 or i + 1 == stop:
                for a in arrays.values():
                    a.flush()
                chunk = {
                    "start": chunk_start,
                    "stop": i + 1,
                    **{name: chunk_digest(name, chunk_start, i + 1) for name in arrays},
                    "observed_tiles": int(
                        np.any(arrays["coverage"][chunk_start : i + 1] > 0, axis=(1, 2, 3)).sum()
                    ),
                }
                progress["chunks"].append(chunk)
                dump(progress_path, progress)
                chunk_start = i + 1
        if stop < sources.n:
            return {"state": "checkpointed", "processed_tiles": stop, "registered_tiles": sources.n}
        for name, a in arrays.items():
            a.flush()
            if paths[name] != root / (name + ".npy"):
                paths[name].replace(root / (name + ".npy"))
        result = {
            "state": "complete",
            "schema": "subpixel-rgb-cache-data-v1",
            "identity": ident,
            "index": spec["index"],
            "public": spec["public"],
            "registered_tiles": sources.n,
            "observed_tiles": progress["observed_tiles"],
            "actual_observations": len(spec["views"]),
            "output_shape": spec["output_shape"],
            "working_pixel_m": spec["working_pixel_m"],
            "output_pixel_m": spec["output_pixel_m"],
            "factor": spec["factor"],
            "rgb": _array_ref(root / "rgb.npy", spec["index"]["sha256"]),
            "coverage": _array_ref(root / "coverage.npy", spec["index"]["sha256"]),
            "source_producers": [v["producer_spec"] for v in spec["views"]],
            "source_time_contract": spec["time_contract"],
            "primary_or_weak_maps_loaded": False,
            "working_grid_is_not_effective_sensor_resolution": True,
        }
        dump(manifest, result)
        return load_manifest(manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment subpixel-rgb-cache")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--max-tiles", type=int)
    args = parser.parse_args(argv)
    prepare(args.spec, max_tiles=args.max_tiles)
    return 0
