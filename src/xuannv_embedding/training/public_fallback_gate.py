"""Staged weak-head adaptation and a shared three-expert regional embedding gate."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.export.context import sha
from xuannv_embedding.models import public_fallback_gate as model_source
from xuannv_embedding.models.public_fallback_gate import PublicFallbackGate
from xuannv_embedding.training import content_adaptive_residual as parent
from xuannv_embedding.training.frozen_fusion import source_array, verified


def _adapt(spec):
    adapted = copy.deepcopy(spec)
    adapted.pop("fallback_teacher")
    adapted.pop("head_warmup_steps")
    adapted["inputs"].pop("foundation")
    adapted["schema"] = "content-adaptive-residual-v1"
    weights = spec["model"]["initial_weights"]
    adapted["model"] = {
        "width": spec["model"]["width"],
        "initial_alpha": weights[2] / (weights[1] + weights[2]),
    }
    return adapted


def specification(path):
    spec = parent._read(path)
    expected = {
        "schema",
        "index",
        "inputs",
        "worldcover",
        "osm",
        "teacher",
        "model",
        "loss",
        "steps",
        "warmup_steps",
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip",
        "seed",
        "amp",
        "device",
        "reserve_gib",
        "output",
        "head_warmup_steps",
        "fallback_teacher",
    }
    if (
        not isinstance(spec, dict)
        or set(spec) != expected
        or spec["schema"] != "public-fallback-gate-v1"
        or not isinstance(spec["inputs"], dict)
        or set(spec["inputs"]) != {"public", "semantic", "geometry", "foundation", "spectral"}
        or not isinstance(spec["model"], dict)
        or set(spec["model"]) != {"width", "initial_weights"}
        or not isinstance(spec["fallback_teacher"], dict)
        or set(spec["fallback_teacher"]) != {"checkpoint", "spec", "export", "public_lineage"}
        or type(spec["head_warmup_steps"]) is not int
        or type(spec["steps"]) is not int
        or not 0 <= spec["head_warmup_steps"] < spec["steps"]
    ):
        raise ValueError("Invalid self-contained public fallback specification")
    PublicFallbackGate(**spec["model"])
    parent._reference(spec["inputs"]["foundation"], array=True)
    for value in spec["fallback_teacher"].values():
        parent._reference(value)
    _, index = parent.validate_specification(_adapt(spec))
    return spec, index


def identity(path):
    result = parent.identity(path)
    result["implementation"].update(
        training=sha(__file__),
        adaptive_engine=sha(parent.__file__),
        public_fallback_model=sha(model_source.__file__),
    )
    return result


class TrainingData(parent.TrainingData):
    def __init__(self, spec, index, *, load_weak=True):
        super().__init__(_adapt(spec), index, load_weak=load_weak)
        fallback = spec["fallback_teacher"]
        original = parent._read(verified(fallback["spec"]))
        original = original.get("base", original)
        saved = torch.load(
            str(verified(fallback["checkpoint"])), weights_only=True, map_location="cpu"
        )
        exported = parent._read(verified(fallback["export"]))
        lineage = parent._read(verified(fallback["public_lineage"]))
        if (
            lineage.get("state") != "passed"
            or lineage.get("only_public_observations") is not True
            or lineage.get("primary_reference_maps_loaded") is not False
            or lineage.get("checkpoint") != fallback["checkpoint"]
            or lineage.get("specification") != fallback["spec"]
            or lineage.get("embedding")
            != {k: spec["inputs"]["foundation"][k] for k in ["path", "sha256"]}
            or lineage.get("index_sha256") != self.index_sha256
        ):
            raise ValueError("Genuine-public fallback lineage binding differs")
        if (
            saved["identity"]["spec_sha256"] != fallback["spec"]["sha256"]
            or saved["step"] != original["steps"]
            or original["index"]["sha256"] != self.index_sha256
            or original["inputs"]["public"]["sha256"] != spec["inputs"]["public"]["sha256"]
            or original["inputs"]["spectral"] != spec["inputs"]["spectral"]
            or any(original[k]["sha256"] != spec[k]["sha256"] for k in ["worldcover", "osm"])
            or exported.get("state") != "complete"
            or exported.get("evaluation_maps_used") is not False
            or exported["index_sha256"] != self.index_sha256
            or exported["sha256"] != spec["inputs"]["foundation"]["sha256"]
            or exported["model_sha256"] != fallback["checkpoint"]["sha256"]
        ):
            raise ValueError("Public fallback checkpoint/export binding changed")
        self.foundation = source_array(spec["inputs"]["foundation"], self.index_sha256, self.n)
        if (
            self.foundation.shape != self.public.shape
            or list(self.foundation.shape) != exported["shape"]
            or self.foundation.dtype not in [np.float16, np.float32]
        ):
            raise ValueError("Public fallback source geometry differs")
        prefix = "base." if "base.public_mean" in saved["model"] else ""
        for key in ["public_mean", "public_std", "spectral_mean", "spectral_std"]:
            if not torch.equal(self.teacher_state[key], saved["model"][prefix + key]):
                raise ValueError("Public fallback coordinate statistics differ")
        if load_weak and (
            saved["summary"]["weak_domain_sha256"] != self.domain_sha256
            or any(
                saved["summary"]["class_weights"][k] != v.tolist()
                for k, v in self.class_weights.items()
            )
        ):
            raise ValueError("Public fallback weak-label exposure differs")

    def batch(self, positions):
        result = super().batch(positions)
        foundation = torch.from_numpy(np.array(self.foundation[positions], np.float32, copy=True))
        return (*result, foundation)


def forward(model, batch, spec, current):
    return model(*batch[:5], batch[7], heads_only=current < spec["head_warmup_steps"])


RUNTIME = parent.AdaptiveRuntime(
    specification=specification,
    identity=identity,
    data=TrainingData,
    model=PublicFallbackGate,
    forward=forward,
)


def update(model, batch, weights, optimizer, spec, current):
    return parent.update(model, batch, weights, optimizer, spec, current, forward_function=forward)


def train(path, *, max_updates=None):
    return parent.train(path, max_updates=max_updates, runtime=RUNTIME)


def export(path):
    return parent.export(path, runtime=RUNTIME)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment public-fallback-gate")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=["train", "export"], required=True)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args(argv)
    if args.phase == "export" and args.max_updates is not None:
        parser.error("--max-updates applies only to training")
    train(args.spec, max_updates=args.max_updates) if args.phase == "train" else export(args.spec)
    return 0
