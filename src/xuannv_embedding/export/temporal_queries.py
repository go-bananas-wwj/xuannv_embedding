"""Inference-only monthly public guidance with observed-center geometry and recovery."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import math
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from xuannv_embedding.data_process.osm_weak_targets import _json
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.models.temporal_feature_fusion import TemporalFeatureFusion
from xuannv_embedding.training import temporal_fusion
from xuannv_embedding.training.cli import _setup_device


def verified(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Invalid public query reference")
    path = Path(ref["path"])
    if not path.is_absolute() or sha(path) != ref["sha256"]:
        raise ValueError("Public query source changed")
    return path


def masked_resize(monthly, observed, shape, *, minimum_coverage):
    if (
        monthly.ndim != 5
        or observed.shape != (*monthly.shape[:2], 1, *monthly.shape[-2:])
        or observed.dtype not in {torch.float32, torch.bool}
    ):
        raise ValueError("Public monthly guidance and observation geometry differ")
    if not torch.isfinite(observed).all() or (observed < 0).any() or (observed > 1).any():
        raise ValueError("Invalid public observed fraction")
    valid = observed > 0
    clean = torch.where(valid, monthly.float(), 0)
    if not torch.isfinite(clean).all():
        raise ValueError("Nonfinite observed monthly guidance")
    weight = observed.float()
    size = tuple(shape)
    coverage = F.interpolate(weight.flatten(0, 1), size=size, mode="bilinear", align_corners=False)
    numerator = F.interpolate(
        (clean * weight).flatten(0, 1), size=size, mode="bilinear", align_corners=False
    )
    result = torch.where(coverage > minimum_coverage, numerator / coverage.clamp_min(1e-8), 0)
    return (
        result.reshape(*monthly.shape[:2], monthly.shape[2], *size),
        coverage.reshape(*monthly.shape[:2], 1, *size),
    )


def specification(path):
    spec = _json(Path(path))
    if (
        set(spec)
        != {
            "schema",
            "training_spec",
            "checkpoint",
            "annual_reference",
            "output_shape",
            "minimum_coverage",
            "maximum_absolute_replay_error",
            "maximum_relative_replay_rmse",
            "device",
            "reserve_gib",
            "output",
        }
        or spec["schema"] != "temporal-public-queries-v1"
        or not isinstance(spec["output_shape"], list)
        or len(spec["output_shape"]) != 2
        or any(type(v) is not int or v < 1 for v in spec["output_shape"])
        or type(spec["reserve_gib"]) is not int
        or spec["reserve_gib"] < 0
        or not isinstance(spec["device"], str)
        or not re.fullmatch(r"cpu|npu:[0-9]+", spec["device"])
        or not Path(spec["output"]).is_absolute()
        or any(
            type(spec[k]) not in {int, float} or not math.isfinite(spec[k]) or spec[k] < 0
            for k in [
                "minimum_coverage",
                "maximum_absolute_replay_error",
                "maximum_relative_replay_rmse",
            ]
        )
        or not 0 <= spec["minimum_coverage"] <= 1
    ):
        raise ValueError("Invalid public temporal query specification")
    training_path = verified(spec["training_spec"])
    training = _json(training_path)
    if training.get("static", {}).get("local", "missing") is not None:
        raise ValueError("Monthly guidance requires a genuinely public teacher without local input")
    return spec, training_path


def _chain(previous, values, masks):
    h = hashlib.sha256(bytes.fromhex(previous))
    h.update(np.ascontiguousarray(values).tobytes())
    h.update(np.ascontiguousarray(masks).tobytes())
    return h.hexdigest()


def export_queries(path, *, max_tiles=None):
    if max_tiles is not None and (type(max_tiles) is not int or max_tiles < 1):
        raise ValueError("Partial query export requires a positive tile limit")
    path = Path(path)
    spec, training_path = specification(path)
    # This inference-only loader verifies observation/static inputs but never map targets.
    data = temporal_fusion.Data(training_path, targets=False)
    assert data.labels is None
    calendar = data.months.tolist()
    serial = [v // 100 * 12 + v % 100 - 1 for v in calendar]
    if len(set(calendar)) != 12 or any(b - a != 1 for a, b in zip(serial, serial[1:])):
        raise ValueError("Public guidance requires twelve unique consecutive calendar months")
    modelpath = verified(spec["checkpoint"])
    saved = torch.load(str(modelpath), weights_only=True, map_location="cpu")
    source_identity = temporal_fusion.identity(training_path)
    if (
        saved["identity"] != source_identity
        or saved["step"] != data.spec["steps"]
        or saved.get("training_summary", {}).get("high_resolution_used") is not False
    ):
        raise ValueError("Frozen public teacher identity or terminal checkpoint differs")
    annual_path = verified(spec["annual_reference"])
    annual = np.load(annual_path, mmap_mode="r")
    if annual.shape != (data.n, 64, *data.frames.shape[-2:]):
        raise ValueError("Public annual replay geographic dimensions differ")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    ident = {
        "spec_sha256": sha(path),
        "exporter_sha256": sha(__file__),
        "teacher": source_identity,
        "checkpoint_sha256": sha(modelpath),
    }
    with (root / "export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        header = root / "identity.json"
        if header.exists():
            if _json(header) != ident:
                raise ValueError("Public query implementation or specification changed")
        elif any(p.name != "export.lock" for p in root.iterdir()):
            raise ValueError("Unowned monthly query output")
        dump(header, ident)
        manifest = root / "manifest.json"
        if manifest.exists():
            result = _json(manifest)
            if result.get("state") != "complete":
                raise ValueError("Invalid completed query state")
            for key in ["queries", "coverage"]:
                verified(result[key])
            return result
        progress = root / "progress.json"
        payload = root / "payload"
        pending = root / "payload.partial"
        state = (
            _json(progress)
            if progress.exists()
            else {
                "completed": 0,
                "chain": "0" * 64,
                "max_error": 0.0,
                "squared_error": 0.0,
                "squared_reference": 0.0,
            }
        )
        start = state["completed"]
        if type(start) is not int or not 0 <= start <= data.n:
            raise ValueError("Invalid public query recovery prefix")
        if payload.exists():
            if pending.exists() or start != data.n:
                raise ValueError("Invalid finalized public query prefix")
            directory = payload
        else:
            pending.mkdir(exist_ok=True)
            directory = pending
        shape = (data.n, 12, 64, *spec["output_shape"])
        maskshape = (data.n, 12, 1, *spec["output_shape"])
        needed = (math.prod(shape) + math.prod(maskshape)) * 4 if start == 0 else 0
        if shutil.disk_usage(root).free < spec["reserve_gib"] * 2**30 + needed:
            raise RuntimeError("Public query disk reserve reached")
        mode = "r+" if start else "w+"
        values = np.lib.format.open_memmap(
            directory / "monthly.npy", mode=mode, dtype=np.float32, shape=shape
        )
        coverage = np.lib.format.open_memmap(
            directory / "coverage.npy", mode=mode, dtype=np.float32, shape=maskshape
        )
        if (
            values.shape != shape
            or coverage.shape != maskshape
            or values.dtype != np.float32
            or coverage.dtype != np.float32
        ):
            raise ValueError("Public query recovery array shape or dtype changed")
        chain = "0" * 64
        for i in range(start):
            chain = _chain(chain, values[i], coverage[i])
        if chain != state["chain"]:
            raise ValueError("Committed public query prefix changed")
        limit = min(data.n, start + max_tiles) if max_tiles is not None else data.n
        if start < limit:
            device, _, _ = _setup_device(spec["device"])
            if device.type == "npu":
                torch.npu.config.allow_internal_format = False
            model = TemporalFeatureFusion(**data.spec["model"]).to(device)
            model.load_state_dict(saved["model"], strict=True)
            model.eval()
            with torch.inference_mode():
                for i in range(start, limit):
                    batch = tuple(v.to(device) if v is not None else None for v in data.batch([i]))
                    with temporal_fusion.cast(device, data.spec["amp"]):
                        result = model(*batch[:6])
                    if not torch.equal(result.monthly_embedding.mean(1), result.embedding):
                        raise ValueError(
                            "Monthly mean no longer reproduces the teacher annual output"
                        )
                    current = result.embedding[0].float().cpu().numpy()
                    expected = np.asarray(annual[i], np.float32)
                    if not np.isfinite(current).all() or not np.isfinite(expected).all():
                        raise FloatingPointError("Nonfinite frozen teacher annual replay")
                    difference = current.astype(np.float64) - expected
                    state["max_error"] = max(state["max_error"], float(np.max(abs(difference))))
                    state["squared_error"] += float(np.sum(difference**2))
                    state["squared_reference"] += float(np.sum(expected.astype(np.float64) ** 2))
                    observed = batch[1].any(2, keepdim=True).float()
                    query, mask = masked_resize(
                        result.monthly_embedding,
                        observed,
                        spec["output_shape"],
                        minimum_coverage=spec["minimum_coverage"],
                    )
                    values[i] = query[0].float().cpu().numpy()
                    coverage[i] = mask[0].float().cpu().numpy()
                    if not np.isfinite(values[i]).all():
                        raise FloatingPointError("Nonfinite public query output")
                    chain = _chain(chain, values[i], coverage[i])
                    state.update(completed=i + 1, chain=chain)
                    if (i + 1) % 8 == 0 or i + 1 == limit:
                        values.flush()
                        coverage.flush()
                        dump(progress, state)
                        dump(
                            root / "status.json",
                            {"state": "exporting", "completed": i + 1, "total": data.n},
                        )
        if limit < data.n:
            result = {"state": "checkpointed", "completed": limit, "total": data.n}
            dump(root / "status.json", result)
            return result
        relative = math.sqrt(state["squared_error"] / max(state["squared_reference"], 1e-30))
        if (
            state["max_error"] > spec["maximum_absolute_replay_error"]
            or relative > spec["maximum_relative_replay_rmse"]
        ):
            raise ValueError("Public annual replay exceeds registered numerical tolerance")
        if directory == pending:
            pending.rename(payload)
        source_index = _json(Path(data.spec["index"]["path"]))
        final = {
            "state": "complete",
            "schema": spec["schema"],
            "shape": list(shape),
            "months": data.months.tolist(),
            "index": data.spec["index"],
            "records": [
                {
                    "patch_id": r["patch_id"],
                    "bounds": r["bounds"],
                    "crs": r.get("crs") or source_index.get("crs"),
                }
                for r in data.records
            ],
            "queries": {
                "path": str(payload / "monthly.npy"),
                "sha256": sha(payload / "monthly.npy"),
            },
            "coverage": {
                "path": str(payload / "coverage.npy"),
                "sha256": sha(payload / "coverage.npy"),
            },
            "annual_replay_max_absolute_error": state["max_error"],
            "annual_replay_relative_rmse": relative,
            "map_targets_loaded_for_inference": False,
            "local_images_used": False,
            "checkpoint_sha256": sha(modelpath),
            "native_pixel_center_interpolation": True,
            "output_field_is_not_a_new_observation": True,
        }
        dump(manifest, final)
        dump(root / "status.json", {"state": "complete", "completed": data.n})
        return final


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment temporal-public-queries")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--max-tiles", type=int)
    args = parser.parse_args(argv)
    export_queries(args.spec, max_tiles=args.max_tiles)
    return 0
