"""Matched time-conditioned refinement with audited real-view/calendar provenance."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from pyproj import CRS

from xuannv_embedding.data_process import observation_time
from xuannv_embedding.data_process.observation_time import ObservationInterval, month_support
from xuannv_embedding.export.context import sha
from xuannv_embedding.models import time_modulated
from xuannv_embedding.models.time_modulated import TimeModulatedFusion, diversity_loss
from xuannv_embedding.training import semantic_refinement as base
from xuannv_embedding.training.frozen_fusion import source_array, verified


def specification(path):
    return validate_specification(base._json(Path(path)))


def validate_specification(value):
    if (
        set(value) != {"schema", "base", "temporal", "diversity"}
        or value["schema"] != "time-modulated-refinement-v1"
        or not isinstance(value["temporal"], dict)
        or set(value["temporal"]) != {"views", "public_manifest", "time_contract"}
        or not isinstance(value["diversity"], dict)
        or set(value["diversity"]) != {"variance", "covariance"}
        or any(
            type(v) not in {int, float} or not math.isfinite(v) or v < 0
            for v in value["diversity"].values()
        )
    ):
        raise ValueError("Invalid time-modulated specification")
    spec, index = base.validate_specification(value["base"])
    if spec["schema"] != "semantic-refinement-v1":
        raise ValueError("Time-modulated base must use the registered residual architecture")
    return {**spec, "temporal": value["temporal"], "diversity": value["diversity"]}, index


def identity(path):
    return {
        "spec_sha256": sha(path),
        "implementation": {
            **base.implementation(),
            "engine": sha(base.__file__),
            "training": sha(__file__),
            "time_model": sha(time_modulated.__file__),
            "physical_time": sha(observation_time.__file__),
        },
    }


def _same_geography(a, b):
    if len(a) != len(b):
        raise ValueError("Conditional observation geography differs")
    for x, y in zip(a, b):
        if (
            x["patch_id"] != y["patch_id"]
            or x["bounds"] != y["bounds"]
            or CRS.from_user_input(x["crs"]) != CRS.from_user_input(y["crs"])
        ):
            raise ValueError("Conditional observation geographic order differs")


class TrainingData(base.TrainingData):
    def __init__(self, spec, index):
        super().__init__(spec, index)
        temporal = spec["temporal"]
        public = base._json(verified(temporal["public_manifest"]))
        records = [{**r, "crs": r.get("crs", index.get("crs"))} for r in index["records"]]
        if (
            public.get("state") != "complete"
            or public.get("schema") != "temporal-public-queries-v1"
            or public.get("index") != spec["index"]
            or public.get("map_targets_loaded_for_inference") is not False
            or public.get("local_images_used") is not False
        ):
            raise ValueError("Expected frozen genuine-public monthly guidance")
        _same_geography(records, public["records"])
        self.queries = np.load(verified(public["queries"]), mmap_mode="r")
        self.query_mask = np.load(verified(public["coverage"]), mmap_mode="r")
        t = len(public["months"])
        native = tuple(math.ceil(v / 16) for v in self.context_shape)
        if (
            list(self.queries.shape) != public["shape"]
            or self.queries.shape != (self.n, t, 64, *native)
            or self.query_mask.shape != (self.n, t, 1, *native)
        ):
            raise ValueError("Public query geometry differs from real local features")
        contract = base._json(verified(temporal["time_contract"]))
        if (
            contract.get("state") != "complete"
            or contract.get("months") != public["months"]
            or contract.get("physical_time_used_for_queries") is not True
            or contract.get("one_row_per_actual_view_not_month_copy") is not True
        ):
            raise ValueError("Expected physical observation time contract")
        views = temporal["views"]
        if (
            not isinstance(views, list)
            or not views
            or len(views) != len(contract["views"])
            or any(set(r) != {"slot", "features", "coverage", "geometry"} for r in views)
            or [v["slot"] for v in views] != list(range(len(views)))
            or [v["slot"] for v in contract["views"]] != list(range(len(views)))
        ):
            raise ValueError("Actual observation view inventory differs")
        expected = month_support(
            [ObservationInterval.from_mapping(v["source_time"]) for v in contract["views"]],
            public["months"],
        )
        self.support = np.load(verified(contract["weights"]))
        if self.support.dtype != np.float32 or not np.array_equal(expected, self.support):
            raise ValueError("Stored support differs from physical observation intervals")
        if (
            spec["inputs"]["context"] != views[0]["features"]
            or spec["inputs"]["context_mask"] != views[0]["coverage"]
        ):
            raise ValueError("Base context must bind the first actual observation")
        self.views = []
        self.view_masks = []
        for view in views:
            geometry = base._json(verified(view["geometry"]))
            _same_geography(records, geometry["records"])
            if geometry["months"] != public["months"]:
                raise ValueError("Local feature calendar registration differs")
            arrays = []
            for key, c in [("features", 1024), ("coverage", 1)]:
                a = source_array(view[key], spec["index"]["sha256"], self.n)
                if a.shape != (self.n, c, *native):
                    raise ValueError("Actual view array dimensions differ")
                arrays.append(a)
            self.views.append(arrays[0])
            self.view_masks.append(arrays[1])
        for a in [self.query_mask, *self.view_masks]:
            if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
                raise ValueError("Conditional source coverage must be in [0,1]")

    def batch(self, positions):
        p, s, d, _, _, target, visible, labels = super().batch(positions)
        mask = np.stack([np.array(a[positions], np.float32, copy=True) for a in self.view_masks], 1)
        local = np.stack([np.array(a[positions], np.float32, copy=True) for a in self.views], 1)
        query_mask = np.array(self.query_mask[positions], np.float32, copy=True)
        query = np.array(self.queries[positions], np.float32, copy=True)
        local = np.where(mask > 0, local, 0)
        query = np.where(query_mask > 0, query, 0)
        if not np.isfinite(local).all() or not np.isfinite(query).all():
            raise ValueError("Nonfinite observed conditional features")
        support = np.broadcast_to(self.support, (len(positions), *self.support.shape)).copy()
        conditional = tuple(
            torch.from_numpy(np.ascontiguousarray(a))
            for a in [local, mask, query, query_mask, support]
        )
        return p, s, d, conditional, None, target, visible, labels

    def moments(self, *, seed):
        result = super().moments(seed=seed)
        result["context"] = sampled_observed_moments(self.views, self.view_masks, seed=seed)
        result["query"] = sampled_observed_moments(
            [self.queries[:, t] for t in range(self.queries.shape[1])],
            [self.query_mask[:, t] for t in range(self.query_mask.shape[1])],
            seed=seed,
        )
        return result


def sampled_observed_moments(arrays, masks, *, seed):
    """Normalize each encoder in its own coordinate basis without map labels."""
    rng = np.random.default_rng(seed)
    rows, valid = [], []
    channels = arrays[0].shape[1]
    for array, mask in zip(arrays, masks):
        for i in range(len(array)):
            ids = rng.choice(
                array.shape[-2] * array.shape[-1],
                min(64, array.shape[-2] * array.shape[-1]),
                replace=False,
            )
            rows.append(np.asarray(array[i], np.float32).reshape(channels, -1)[:, ids].T)
            valid.append(np.asarray(mask[i]).reshape(-1)[ids] > 0)
    x, m = np.concatenate(rows), np.concatenate(valid)
    if len(x) > 65536:
        ids = rng.choice(len(x), 65536, replace=False)
        x, m = x[ids], m[ids]
    x = x[m].astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite observed normalization sample")
    mean = x.mean(0) if len(x) else np.zeros(channels)
    std = x.std(0) if len(x) > 1 else np.ones(channels)
    std[std < 1e-6] = 1
    return mean.astype(np.float32), std.astype(np.float32)


def forward(model, batch, shape, *, burnin=False):
    p, s, d, conditional, _, target, visible, _ = batch
    local, mask, queries, query_mask, support = conditional
    if not torch.equal(support, support[:1].expand_as(support)):
        raise ValueError("Observation calendar support differs inside a regional batch")
    output = model(
        p,
        s,
        d,
        local,
        mask,
        queries,
        query_mask,
        support[0],
        spectral_mask=visible,
        context_source_shape=shape,
        burnin=burnin,
    )
    # A fixed auxiliary teacher can have less coverage than the new actual views.
    # Its absent zero vectors are unknown, rather than reconstruction targets.
    if target is not None:
        output.context_mask = output.context_mask & (
            target.detach().abs().sum(1, keepdim=True) > 1e-8
        )
    return output


def extra_loss(output, visible, step, spec):
    losses = diversity_loss(output.white_features, visible.any(1, keepdim=True), step=step)
    return {**losses, "loss": sum(spec["diversity"][k] * v for k, v in losses.items())}


RUNTIME = base.RefinementRuntime(
    specification, identity, TrainingData, TimeModulatedFusion, forward, extra_loss
)


def train(path, *, max_updates=None):
    return base.train(path, max_updates=max_updates, runtime=RUNTIME)


def export(path):
    return base.export(path, runtime=RUNTIME)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment time-modulated")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["train", "export"], required=True)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args(argv)
    if args.phase == "export" and args.max_updates is not None:
        parser.error("--max-updates applies only to training")
    train(args.spec, max_updates=args.max_updates) if args.phase == "train" else export(args.spec)
    return 0
