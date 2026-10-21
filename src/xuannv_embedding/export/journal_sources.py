"""Whole-source absence in the selected frozen regional fusion checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training import decoupled_geometry as geometry
from xuannv_embedding.training import semantic_refinement as semantic
from xuannv_embedding.training.cli import _setup_device
from xuannv_embedding.training.frozen_fusion import verified


def remove_local_inputs(batch):
    """Remove all local values, coverage and teachers before either fusion encoder."""
    public, spectral, detail, context, mask, _, visible, labels = batch
    detail = torch.zeros_like(detail) if detail is not None else None
    if isinstance(context, tuple):
        local, coverage, query, query_mask, support = context
        context = (torch.zeros_like(local), torch.zeros_like(coverage), query, query_mask, support)
        mask = None
    else:
        context, mask = None, None
    return public, spectral, detail, context, mask, None, visible, labels


def archived_semantic_runtime(references):
    """Bind the original model and export engine recorded in the trained checkpoint."""
    import importlib.util
    import sys

    loaded = {}
    for name in ["model", "training"]:
        path = verified(references[name])
        module_name = "_jstars_original_semantic_" + name
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        loaded[name] = module
    engine, model = loaded["training"], loaded["model"]
    engine.semantic_residual = model
    engine.SemanticResidualFusion = model.SemanticResidualFusion
    return semantic.RefinementRuntime(
        specification=engine.specification,
        identity=engine.identity,
        data=engine.TrainingData,
        model=model.SemanticResidualFusion,
        forward=engine._forward,
    )


def export_absent(path, device):
    spec = semantic._json(path)
    if set(spec) != {"schema", "semantic", "geometry", "semantic_source", "alpha", "output"} or (
        spec["schema"] != "jstars-whole-local-absence-v1" or spec["alpha"] != 0.15
    ):
        raise ValueError("Invalid selected whole-source absence contract")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {"spec": sha(path), "code": sha(__file__)}
    if (root / "status.json").exists():
        result = json.loads((root / "status.json").read_text())
        if result["identity"] != identity:
            raise ValueError("Whole-source absence identity changed")
        verified(result["features"])
        return result
    dev, _, _ = _setup_device(device)
    started = time.monotonic()
    outputs, weights = [], []
    for name, runtime in [
        ("semantic", archived_semantic_runtime(spec["semantic_source"])),
        ("geometry", geometry.RUNTIME),
    ]:
        source = verified(spec[name])
        settings, index = runtime.specification(source)
        checkpoint = Path(settings["output"]) / "model.weights.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["identity"] != runtime.identity(source) or saved["step"] != settings["steps"]:
            raise ValueError("Whole-source export requires the unchanged terminal checkpoint")
        data = runtime.data(settings, index)
        model = runtime.model(**settings["model"]).to(dev).eval()
        model.load_state_dict(saved["model"], strict=True)
        target = root / (name + ".npy")
        shape = (data.n, 64, *data.public.shape[-2:])
        out = np.lib.format.open_memmap(
            target.with_suffix(".partial.npy"), mode="w+", dtype="float16", shape=shape
        )
        with torch.inference_mode():
            for start in range(0, data.n, 2):
                ids = list(range(start, min(start + 2, data.n)))
                batch = semantic._move(remove_local_inputs(data.batch(ids)), dev)
                with semantic._autocast(settings, dev):
                    result = runtime.forward(model, batch, data.context_shape)
                if not torch.equal(result.embedding, result.public_embedding):
                    raise ValueError("Missing local inputs must recover the learned public branch")
                values = result.embedding.float().cpu().numpy().astype("float16")
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite absent-source feature")
                out[ids] = values
                if start % 20 == 0:
                    dump(
                        root / "progress.json",
                        {"component": name, "completed": start + len(ids), "total": data.n},
                    )
        out.flush()
        del out
        target.with_suffix(".partial.npy").replace(target)
        outputs.append(target)
        weights.append({"path": str(checkpoint), "sha256": sha(checkpoint)})
        del model, data
        if dev.type == "npu":
            torch.npu.empty_cache()
    a, b = [np.load(p, mmap_mode="r") for p in outputs]
    target = root / "annual.npy"
    out = np.lib.format.open_memmap(
        root / "annual.partial.npy", mode="w+", dtype="float16", shape=a.shape
    )
    for i in range(len(a)):
        out[i] = (
            np.float32(0.85) * a[i].astype("float32") + np.float32(0.15) * b[i].astype("float32")
        ).astype("float16")
    out.flush()
    del out
    (root / "annual.partial.npy").replace(target)
    result = {
        "state": "complete",
        "identity": identity,
        "features": {"path": str(target), "sha256": sha(target)},
        "weights": weights,
        "wall_seconds": time.monotonic() - started,
        "scope": "all_local_sources_absent_before_regional_fusion_and_terminal_guidance",
        "all_local_detail_context_coverage_and_teacher_removed": True,
        "learned_public_branch_exact": True,
        "heads_refitted": False,
        "parameters_refitted": False,
    }
    dump(root / "status.json", result)
    return result


def run_absent_case(path, device):
    from xuannv_embedding.export.jstars_revision import score_frozen

    spec = semantic._json(path)
    if set(spec) != {"schema", "source", "region", "readers", "output"} or (
        spec["schema"] != "jstars-whole-source-case-v1"
    ):
        raise ValueError("Invalid whole-source case")
    result = export_absent(verified(spec["source"]), device)
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    score_path = root / "score.json"
    value = {
        "schema": "jstars-frozen-intervention-v1",
        "region": spec["region"],
        "feature": result["features"],
        "readers": spec["readers"],
        "output": str(root / "evaluation"),
    }
    if score_path.exists() and semantic._json(score_path) != value:
        raise ValueError("Whole-source scoring identity changed")
    dump(score_path, value)
    return score_frozen(score_path, device)
