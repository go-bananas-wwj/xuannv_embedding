"""Coarse public semantics and separate local feature-geometry learning."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from xuannv_embedding.export.context import sha
from xuannv_embedding.models import decoupled_geometry
from xuannv_embedding.models import gated_geometry as geometry_loss_source
from xuannv_embedding.models.decoupled_geometry import PublicSemanticGeometryFusion
from xuannv_embedding.models.gated_geometry import relational_geometry_loss
from xuannv_embedding.training import semantic_refinement as base
from xuannv_embedding.training import time_modulated as temporal


def specification(path):
    value = base._json(Path(path))
    if (
        set(value) != {"schema", "base", "temporal", "relation"}
        or value["schema"] != "decoupled-geometry-refinement-v1"
        or not isinstance(value["relation"], dict)
        or set(value["relation"]) != {"weight", "samples", "same_class_weight", "seed"}
    ):
        raise ValueError("Invalid gated geometry/relation specification")
    r = value["relation"]
    if (
        type(r["weight"]) not in {int, float}
        or not math.isfinite(r["weight"])
        or not 0 <= r["weight"] <= 4
        or type(r["samples"]) is not int
        or not 2 <= r["samples"] <= 4096
        or type(r["same_class_weight"]) not in {int, float}
        or not math.isfinite(r["same_class_weight"])
        or not 1 <= r["same_class_weight"] <= 4
        or type(r["seed"]) is not int
        or not 0 <= r["seed"] < 2**32
    ):
        raise ValueError("Invalid relational geometry controls")
    spec, index = temporal.validate_specification(
        {
            "schema": "time-modulated-refinement-v1",
            "base": value["base"],
            "temporal": value["temporal"],
            "diversity": {"variance": 0.0, "covariance": 0.0},
        }
    )
    return {**spec, "relation": r}, index


def identity(path):
    result = temporal.identity(path)
    result["implementation"].update(
        temporal_training=sha(temporal.__file__),
        training=sha(__file__),
        decoupled_geometry_model=sha(decoupled_geometry.__file__),
        geometry_loss_source=sha(geometry_loss_source.__file__),
    )
    return result


def batch_loss(output, batch, step, spec):
    if output.local_teacher is None or output.local_teacher_mask is None:
        raise ValueError("Local feature geometry requires actual teacher/mask tensors")
    visible = batch[6]
    labels = batch[7]["worldcover"]
    r = spec["relation"]
    with torch.autocast(device_type=visible.device.type, enabled=False):
        public = (
            F.interpolate(
                visible.any(1, keepdim=True).float(),
                size=output.local_teacher.shape[-2:],
                mode="area",
            )
            > 0.05
        )
        domain = public & (output.local_teacher_mask > 0.05)
    with torch.set_grad_enabled(torch.is_grad_enabled() and r["weight"] > 0):
        value = relational_geometry_loss(
            output.white_features,
            output.local_teacher,
            domain,
            labels,
            step=step,
            samples=r["samples"],
            same_class_weight=r["same_class_weight"],
            seed=r["seed"],
        )
    return {"local_geometry": value, "loss": r["weight"] * value}


RUNTIME = base.RefinementRuntime(
    specification=specification,
    identity=identity,
    data=temporal.TrainingData,
    model=PublicSemanticGeometryFusion,
    forward=temporal.forward,
    batch_loss=batch_loss,
)


def train(path, *, max_updates=None):
    return base.train(path, max_updates=max_updates, runtime=RUNTIME)


def export(path):
    return base.export(path, runtime=RUNTIME)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment decoupled-geometry")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["train", "export"], required=True)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args(argv)
    if args.phase == "export" and args.max_updates is not None:
        parser.error("--max-updates applies only to training")
    train(args.spec, max_updates=args.max_updates) if args.phase == "train" else export(args.spec)
    return 0
