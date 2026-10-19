"""Label-free public/observed-visual subspaces with explicit missing-data geometry."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from pyproj import CRS
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from xuannv_embedding.data_process import observation_time
from xuannv_embedding.data_process.observation_time import ObservationInterval, month_support
from xuannv_embedding.export import semantic_features
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.semantic_features import align_patch_features
from xuannv_embedding.training.frozen_fusion import source_array, verified


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def pool_views(views, masks):
    views, masks = np.asarray(views, np.float32), np.asarray(masks, np.float32)
    if (
        views.ndim != 4
        or not len(views)
        or masks.shape != (len(views), 1, *views.shape[-2:])
        or not np.isfinite(masks).all()
        or (masks < 0).any()
        or (masks > 1).any()
    ):
        raise ValueError("Invalid actual-view coverage or geometry")
    clean = np.where(masks > 0, views, 0)
    if not np.isfinite(clean).all():
        raise ValueError("Nonfinite observed visual values")
    denominator = masks.sum(0)
    values = (clean * masks).sum(0) / np.maximum(denominator, 1e-12)
    return values.astype(np.float32), masks.max(0)


def _sample_tiles(tiles, count, *, limit, seed):
    if type(limit) is not int or limit < count or type(seed) is not int:
        raise ValueError("Invalid per-tile sample budget or seed")
    rng = np.random.default_rng(seed)
    rows, positions = [], []
    channels = None
    for i, (values, mask) in enumerate(tiles):
        values, mask = np.asarray(values), np.asarray(mask)
        if (
            values.ndim != 3
            or mask.shape != (1, *values.shape[-2:])
            or not np.isfinite(mask).all()
            or (mask < 0).any()
            or (mask > 1).any()
        ):
            raise ValueError("Sample mask and source geometry differ")
        channels = values.shape[0] if channels is None else channels
        if channels != values.shape[0]:
            raise ValueError("Sample channel count differs")
        valid = np.flatnonzero(mask.reshape(-1) > 0)
        quota = limit // count + (i < limit % count)
        selected = rng.choice(valid, min(quota, len(valid)), replace=False)
        x = values.reshape(channels, -1)[:, selected].T.astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite observed samples")
        rows.append(x)
        positions.extend((i, int(p)) for p in selected)
    if len(rows) != count or channels is None:
        raise ValueError("Tile sample inventory differs")
    return np.concatenate(rows), np.asarray(positions, np.int64).reshape(-1, 2)


def sample_rows(arrays, masks, *, limit, seed):
    if len(arrays) != len(masks) or not len(arrays):
        raise ValueError("Missing tile sample arrays")
    return _sample_tiles(zip(arrays, masks), len(arrays), limit=limit, seed=seed)


def fit_projection(samples, *, dimensions, seed, clip, solver, allow_empty=False):
    x = np.asarray(samples, np.float64)
    if (
        x.ndim != 2
        or type(dimensions) is not int
        or not 1 <= dimensions <= x.shape[1]
        or not np.isfinite(x).all()
        or (len(x) < dimensions and not (allow_empty and len(x) == 0))
        or solver not in {"eigh", "randomized"}
        or not math.isfinite(clip)
        or clip <= 0
    ):
        raise ValueError("Invalid projection samples, dimensions or solver")
    c = x.shape[1]
    if not len(x):
        return {
            "mean": np.zeros(c, np.float32),
            "scale": np.ones(c, np.float32),
            "center": np.zeros(c, np.float32),
            "components": np.zeros((dimensions, c), np.float32),
            "explained_variance_ratio": np.zeros(dimensions, np.float32),
        }
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1
    # Persist and use the same float32 scaler during fit and cold inference.
    mean, scale = mean.astype(np.float32), scale.astype(np.float32)
    z = np.clip((x.astype(np.float32) - mean) / scale, -clip, clip).astype(np.float64)
    center = z.mean(0).astype(np.float32)
    z -= center
    if not np.any(z):
        return {
            "mean": mean,
            "scale": scale,
            "center": center,
            "components": np.zeros((dimensions, c), np.float32),
            "explained_variance_ratio": np.zeros(dimensions, np.float32),
        }
    if solver == "eigh":
        eigenvalues, basis = np.linalg.eigh(z.T @ z)
        order = np.argsort(eigenvalues)[::-1]
        components = basis[:, order[:dimensions]].T
        pivots = np.abs(components).argmax(1)
        components *= np.where(components[np.arange(dimensions), pivots] < 0, -1, 1)[:, None]
        ratios = np.maximum(eigenvalues[order[:dimensions]], 0) / max(eigenvalues.sum(), 1e-12)
    else:
        pca = PCA(
            n_components=dimensions,
            svd_solver="randomized",
            random_state=seed,
            n_oversamples=16,
            iterated_power=4,
            whiten=False,
        ).fit(z)
        # z was already centered; retain sklearn's additional numerical correction.
        center += pca.mean_.astype(np.float32)
        components, ratios = pca.components_, pca.explained_variance_ratio_
    return {
        "mean": mean,
        "scale": scale,
        "center": center,
        "components": components.astype(np.float32),
        "explained_variance_ratio": ratios.astype(np.float32),
    }


def project(values, model, *, dimensions, clip):
    x = np.asarray(values, np.float32)
    basis = model["components"]
    if (
        x.ndim != 3
        or basis.ndim != 2
        or basis.shape[1] != x.shape[0]
        or not 1 <= dimensions <= len(basis)
        or not np.isfinite(x).all()
        or any(not np.isfinite(v).all() for v in model.values())
        or (model["scale"] <= 0).any()
    ):
        raise ValueError("Invalid subspace inference values or parameters")
    z = np.clip((x.reshape(len(x), -1).T - model["mean"]) / model["scale"], -clip, clip)
    # Always evaluate the full saved basis; both budgets have identical leading coordinates.
    output = (z - model["center"]) @ basis.T
    return output.T.reshape(len(basis), *x.shape[-2:])[:dimensions]


def masked_align(values, coverage, source_shape, output_shape, *, threshold):
    values, coverage = np.asarray(values, np.float32), np.asarray(coverage, np.float32)
    if coverage.shape != (1, *values.shape[-2:]) or not np.isfinite(coverage).all():
        raise ValueError("Invalid alignment coverage")
    if (coverage < 0).any() or (coverage > 1).any():
        raise ValueError("Invalid alignment coverage")
    clean = np.where(coverage > 0, values, 0)
    if not np.isfinite(clean).all():
        raise ValueError("Nonfinite observed alignment values")
    v = torch.from_numpy(np.ascontiguousarray(clean * coverage))[None]
    m = torch.from_numpy(np.ascontiguousarray(coverage))[None]
    numerator = align_patch_features(v, source_shape, output_shape, patch=16)
    denominator = align_patch_features(m, source_shape, output_shape, patch=16)
    return torch.where(denominator > threshold, numerator / denominator.clamp_min(1e-12), 0)[
        0
    ].numpy()


def transform_blocks(
    public,
    visual,
    coverage,
    domain,
    bundle,
    *,
    public_dimensions,
    visual_dimensions,
    source_shape,
    threshold,
    clip,
):
    if public_dimensions + visual_dimensions != 64:
        raise ValueError("Output subspaces must total 64 coordinates")
    p = project(public, bundle["public"], dimensions=public_dimensions, clip=clip)
    clean = np.where(np.asarray(coverage) > 0, visual, 0)
    v = project(clean, bundle["visual"], dimensions=visual_dimensions, clip=clip)
    v = masked_align(v, coverage, source_shape, p.shape[-2:], threshold=threshold)
    if np.shape(domain) != (1, *p.shape[-2:]):
        raise ValueError("Public observation domain differs")
    return np.where(domain, np.concatenate([p, v]), 0).astype(np.float32)


def validate_specification(spec):
    expected = {
        "schema",
        "index",
        "public",
        "views",
        "time_contract",
        "source_shape",
        "seed",
        "fit_samples",
        "clip",
        "mask_threshold",
        "output",
        "reserve_gib",
        "budgets",
    }
    if (
        not isinstance(spec, dict)
        or set(spec) != expected
        or spec["schema"] != "public-visual-subspace-v1"
        or not isinstance(spec["public"], dict)
        or set(spec["public"]) != {"array", "domain"}
        or not isinstance(spec["public"]["domain"], dict)
        or set(spec["public"]["domain"]) != {"array", "columns"}
        or type(spec["seed"]) is not int
        or not 0 <= spec["seed"] < 2**32
        or type(spec["fit_samples"]) is not int
        or spec["fit_samples"] < 48
        or type(spec["reserve_gib"]) is not int
        or spec["reserve_gib"] < 0
        or type(spec["clip"]) not in {int, float}
        or not math.isfinite(spec["clip"])
        or spec["clip"] <= 0
        or type(spec["mask_threshold"]) not in {int, float}
        or not 0 <= spec["mask_threshold"] < 1
        or not isinstance(spec["source_shape"], list)
        or len(spec["source_shape"]) != 2
        or any(type(x) is not int or x < 1 for x in spec["source_shape"])
        or not isinstance(spec["output"], str)
        or not Path(spec["output"]).is_absolute()
        or not isinstance(spec["views"], list)
        or not spec["views"]
        or not isinstance(spec["budgets"], list)
        or not spec["budgets"]
    ):
        raise ValueError("Invalid self-contained public/visual specification")
    names = []
    for budget in spec["budgets"]:
        if (
            not isinstance(budget, dict)
            or set(budget) != {"name", "public_dimensions", "visual_dimensions"}
            or not isinstance(budget["name"], str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", budget["name"]) is None
            or any(type(budget[k]) is not int for k in ["public_dimensions", "visual_dimensions"])
            or not 1 <= budget["public_dimensions"] <= 48
            or not 1 <= budget["visual_dimensions"] <= 32
            or budget["public_dimensions"] + budget["visual_dimensions"] != 64
        ):
            raise ValueError("Invalid public/visual specification budget")
        names.append(budget["name"])
    if len(set(names)) != len(names):
        raise ValueError("Duplicate specification budget")
    for i, view in enumerate(spec["views"]):
        if (
            not isinstance(view, dict)
            or set(view) != {"slot", "features", "coverage", "geometry"}
            or type(view["slot"]) is not int
            or view["slot"] != i
        ):
            raise ValueError("Actual-view specification inventory differs")

    def check_reference(value, *, array=False):
        keys = {"path", "sha256", "layout", "index_sha256"} if array else {"path", "sha256"}
        if (
            not isinstance(value, dict)
            or set(value) != keys
            or not isinstance(value["path"], str)
            or not Path(value["path"]).is_absolute()
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or (
                array
                and (
                    value["layout"] not in {"nchw", "nhwc"}
                    or value["index_sha256"] != spec["index"]["sha256"]
                )
            )
        ):
            raise ValueError("Invalid source reference in subspace specification")

    check_reference(spec["index"])
    check_reference(spec["time_contract"])
    check_reference(spec["public"]["array"], array=True)
    check_reference(spec["public"]["domain"]["array"], array=True)
    check_reference(spec["public"]["domain"]["columns"])
    for view in spec["views"]:
        check_reference(view["features"], array=True)
        check_reference(view["coverage"], array=True)
        check_reference(view["geometry"])
    return spec


def specification(path):
    spec = validate_specification(read_json(path))
    index = read_json(verified(spec["index"]))
    records = index.get("records")
    if (
        not isinstance(records, list)
        or not records
        or len({r["patch_id"] for r in records}) != len(records)
    ):
        raise ValueError("Invalid subspace geographic index")
    if spec["fit_samples"] < len(records):
        raise ValueError("Sample budget cannot cover the registered tiles")
    return spec, index


class SourceData:
    """Only frozen public/visual observations and geography; no map labels are opened."""

    def __init__(self, spec, index):
        self.spec, self.index, self.n = spec, index, len(index["records"])
        digest = spec["index"]["sha256"]
        self.public = source_array(spec["public"]["array"], digest, self.n)
        domain = spec["public"]["domain"]
        self.physical = source_array(domain["array"], digest, self.n)
        columns = read_json(verified(domain["columns"]))["columns"]
        self.availability = [i for i, c in enumerate(columns) if c.endswith(":availability")]
        if (
            self.public.shape[1] != 64
            or self.physical.shape[1] != len(columns)
            or self.physical.shape[-2:] != self.public.shape[-2:]
            or not self.availability
        ):
            raise ValueError("Public/physical source dimensions differ")
        contract = read_json(verified(spec["time_contract"]))
        if (
            contract.get("state") != "complete"
            or contract.get("one_row_per_actual_view_not_month_copy") is not True
            or len(contract.get("views", [])) != len(spec["views"])
        ):
            raise ValueError("Actual-view calendar contract differs")
        expected = month_support(
            [ObservationInterval.from_mapping(v["source_time"]) for v in contract["views"]],
            contract["months"],
        )
        if not np.array_equal(expected, np.load(verified(contract["weights"]))):
            raise ValueError("Physical observation-time support changed")
        native = tuple(math.ceil(v / 16) for v in spec["source_shape"])
        self.views, self.masks = [], []
        used = set()
        for i, view in enumerate(spec["views"]):
            if contract["views"][i]["slot"] != i:
                raise ValueError("Physical view order changed")
            geometry = read_json(verified(view["geometry"]))
            if geometry.get("state") != "complete" or geometry.get("months") != contract["months"]:
                raise ValueError("Visual geography/calendar is incomplete")
            records = geometry.get("records", [])
            if len(records) != self.n:
                raise ValueError("Visual geographic tile count differs")
            for original, record in zip(index["records"], records):
                crs = original.get("crs", index.get("crs"))
                if (
                    original["patch_id"] != record["patch_id"]
                    or original["bounds"] != record["bounds"]
                    or CRS.from_user_input(crs) != CRS.from_user_input(record["crs"])
                ):
                    raise ValueError("Visual geographic tile order differs")
            values = source_array(view["features"], digest, self.n)
            masks = source_array(view["coverage"], digest, self.n)
            if values.shape != (self.n, 1024, *native) or masks.shape != (self.n, 1, *native):
                raise ValueError("Native visual source dimensions differ")
            key = (view["features"]["path"], view["coverage"]["path"])
            if key in used:
                raise ValueError("Duplicated actual visual view")
            used.add(key)
            self.views.append(values)
            self.masks.append(masks)

    def domain(self, i):
        availability = np.asarray(self.physical[i, self.availability], np.float32)
        if not np.isfinite(availability).all() or (availability < 0).any():
            raise ValueError("Invalid observed public availability")
        return (availability > 0).any(0, keepdims=True)

    def visual(self, i):
        return pool_views(
            np.stack([v[i] for v in self.views]), np.stack([m[i] for m in self.masks])
        )

    def samples(self, branch):
        def tiles():
            for i in range(self.n):
                if branch == "public":
                    yield self.public[i], self.domain(i)
                else:
                    values, mask = self.visual(i)
                    domain = F.interpolate(
                        torch.from_numpy(self.domain(i).astype(np.float32))[None],
                        size=values.shape[-2:],
                        mode="area",
                    )[0].numpy()
                    yield values, (mask > self.spec["mask_threshold"]) & (
                        domain > self.spec["mask_threshold"]
                    )

        return _sample_tiles(
            tiles(), self.n, limit=self.spec["fit_samples"], seed=self.spec["seed"]
        )


def identity(path):
    return {
        "spec_sha256": sha(path),
        "implementation": {
            "subspace": sha(__file__),
            "alignment": sha(semantic_features.__file__),
            "physical_time": sha(observation_time.__file__),
        },
    }


@contextlib.contextmanager
def ownership(root, ident):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = root / "identity.json"
        if p.exists() and read_json(p) != ident:
            raise ValueError("Subspace implementation or specification changed")
        dump(p, ident)
        yield


def load_bundle(root):
    record = read_json(root / "status.json")
    if (
        record.get("state") != "complete"
        or record.get("identity") != read_json(root / "identity.json")
        or sha(root / "projection.npz") != record["model_sha256"]
    ):
        raise ValueError("Incomplete or changed subspace parameters")
    with np.load(root / "projection.npz", allow_pickle=False) as saved:
        bundle = {
            branch: {
                key.split("__", 1)[1]: saved[key]
                for key in saved.files
                if key.startswith(branch + "__")
            }
            for branch in ["public", "visual"]
        }
    for branch, channels, dimensions in [("public", 64, 48), ("visual", 1024, 32)]:
        model = bundle[branch]
        shapes = {
            "mean": (channels,),
            "scale": (channels,),
            "center": (channels,),
            "components": (dimensions, channels),
            "explained_variance_ratio": (dimensions,),
        }
        if (
            set(model) != set(shapes)
            or any(
                model[k].shape != shape or not np.isfinite(model[k]).all()
                for k, shape in shapes.items()
            )
            or (model["scale"] <= 0).any()
        ):
            raise ValueError("Invalid subspace parameters")
    return bundle


def fit(path):
    spec, index = specification(path)
    root, ident = Path(spec["output"]), identity(path)
    with ownership(root, ident), threadpool_limits(limits=2):
        if (root / "status.json").exists():
            load_bundle(root)
            return read_json(root / "status.json")
        if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30:
            raise RuntimeError("Subspace fit disk reserve")
        data = SourceData(spec, index)
        bundle, samples = {}, {}
        for branch, dimensions, solver in [("public", 48, "eigh"), ("visual", 32, "randomized")]:
            values, positions = data.samples(branch)
            bundle[branch] = fit_projection(
                values,
                dimensions=dimensions,
                seed=spec["seed"],
                clip=spec["clip"],
                solver=solver,
                allow_empty=branch == "visual",
            )
            with (root / (branch + "-samples.npz")).open("wb") as f:
                np.savez(f, positions=positions)
            samples[branch] = {
                "rows": len(values),
                "observed_tiles": len(np.unique(positions[:, 0])),
                "positions_sha256": sha(root / (branch + "-samples.npz")),
            }
        pending = root / "projection.partial.npz"
        with pending.open("wb") as f:
            np.savez(
                f,
                **{
                    branch + "__" + k: v
                    for branch, model in bundle.items()
                    for k, v in model.items()
                },
            )
        pending.replace(root / "projection.npz")
        result = {
            "state": "complete",
            "unique_input_tiles": data.n,
            "samples": samples,
            "model_sha256": sha(root / "projection.npz"),
            "identity": ident,
            "primary_or_weak_labels_loaded": False,
            "neural_encoder_updated": False,
        }
        dump(root / "status.json", result)
        load_bundle(root)
        return result


def export(path):
    spec, index = specification(path)
    root, ident = Path(spec["output"]), identity(path)
    with ownership(root, ident), threadpool_limits(limits=2):
        bundle = load_bundle(root)
        data = SourceData(spec, index)
        shape = (data.n, 64, *data.public.shape[-2:])
        missing = [
            b
            for b in spec["budgets"]
            if not (root / "representations" / b["name"] / "status.json").exists()
        ]
        if (
            shutil.disk_usage(root).free
            < spec["reserve_gib"] * 2**30 + len(missing) * math.prod(shape) * 2
        ):
            raise RuntimeError("Subspace export disk reserve")
        arrays, records = {}, []
        for budget in spec["budgets"]:
            dest = root / "representations" / budget["name"]
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "status.json").exists():
                status = read_json(dest / "status.json")
                if (
                    status.get("state") != "complete"
                    or status.get("identity") != ident
                    or status.get("model_sha256") != sha(root / "projection.npz")
                    or status.get("index_sha256") != spec["index"]["sha256"]
                    or status.get("shape") != list(shape)
                    or sha(dest / "annual.npy") != status["sha256"]
                ):
                    raise ValueError("Cached subspace export changed")
                records.append(status)
            else:
                arrays[budget["name"]] = np.lib.format.open_memmap(
                    dest / "annual.partial.npy", mode="w+", dtype=np.float16, shape=shape
                )
        for i in range(data.n):
            if not arrays:
                break
            if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30:
                raise RuntimeError("Subspace export disk reserve")
            visual, mask = data.visual(i)
            for budget in spec["budgets"]:
                if budget["name"] not in arrays:
                    continue
                out = transform_blocks(
                    data.public[i],
                    visual,
                    mask,
                    data.domain(i),
                    bundle,
                    public_dimensions=budget["public_dimensions"],
                    visual_dimensions=budget["visual_dimensions"],
                    source_shape=spec["source_shape"],
                    threshold=spec["mask_threshold"],
                    clip=spec["clip"],
                )
                values = out.astype(np.float16)
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite or overflowed subspace output")
                arrays[budget["name"]][i] = values
        for name, array in arrays.items():
            array.flush()
            dest = root / "representations" / name
            (dest / "annual.partial.npy").replace(dest / "annual.npy")
            status = {
                "state": "complete",
                "shape": list(shape),
                "sha256": sha(dest / "annual.npy"),
                "index_sha256": spec["index"]["sha256"],
                "identity": ident,
                "model_sha256": sha(root / "projection.npz"),
                "primary_or_weak_labels_loaded": False,
                "neural_encoder_updated": False,
                "unit_normalized": False,
            }
            dump(dest / "status.json", status)
            records.append(status)
        return records


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment public-visual-subspace")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["fit", "export"], required=True)
    args = parser.parse_args(argv)
    fit(args.spec) if args.phase == "fit" else export(args.spec)
    return 0
