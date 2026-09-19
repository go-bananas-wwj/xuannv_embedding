"""Geographically checked comparison features for the paired development protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.training.cli import _git_sha
from xuannv_embedding.training.experiment import _json, _sha


def validate_grid(reference, bounds):
    if reference["shape"] != [128, 128] or not np.allclose(
        reference["bounds"], bounds, rtol=0, atol=1e-4
    ):
        raise ValueError("external feature grid differs from the reference grid")


def raw_features(sample, sources):
    channels = []
    for source in sources:
        x = sample["source_frames"][source][-1]
        mask = sample["source_masks"][source][-1].bool()
        clean = torch.where(mask, x, 0)
        channels.extend((clean, mask.expand(1, *x.shape[-2:]).float()))
    return torch.cat(channels).float().numpy()


def run(args):
    if args.output.exists():
        raise FileExistsError("comparison export already exists")
    cache_path = args.cache / "cache.json"
    cache = json.loads(cache_path.read_text())
    source_records = {}
    if args.kind == "alphaearth":
        if args.source is None:
            raise ValueError("AlphaEarth requires the audited source directory")
        manifest = json.loads((args.source / "aef_source_manifest.json").read_text())
        source_records = {r["patch_id"]: r for r in manifest["records"]}
    args.output.mkdir(parents=True)
    records = []
    for record in cache["records"]:
        if args.kind == "raw":
            if _sha(Path(record["path"])) != record["sha256"]:
                raise ValueError("raw observation cache changed")
            sample = torch.load(record["path"], map_location="cpu", weights_only=True, mmap=True)
            feature = raw_features(sample, list(cache["model_inputs"]))
            provenance = {"source_sha256": record["sha256"]}
        else:
            source = source_records[record["patch_id"]]
            validate_grid(source["reference_grid"], record["bounds"])
            if source["valid_pixel_count"] != 128 * 128:
                raise ValueError("external source has missing pixels; matched coverage is required")
            mask_path = args.source / source["valid_mask_path"]
            if _sha(mask_path) != source["valid_mask_sha256"] or not np.load(mask_path).all():
                raise ValueError("external valid-mask audit failed")
            path = args.source / source["output_path"]
            feature = torch.load(str(path), map_location="cpu", weights_only=True).float().numpy()
            if feature.shape != (64, 128, 128):
                raise ValueError("unexpected AlphaEarth tensor shape")
            provenance = {"source": str(path), "source_sha256": _sha(path)}
        if not np.isfinite(feature).all():
            raise ValueError("comparison features contain nonfinite values")
        path = args.output / (record["patch_id"] + ".npz")
        np.savez_compressed(path, embedding=feature[None])
        records.append({**record, "path": str(path), "sha256": _sha(path), **provenance})
        _json(args.output / "status.json", {"state": "running", "patches": len(records)})
    months = ["annual_2025"] if args.kind == "alphaearth" else [cache["data"]["months"][-1]]
    metadata = {
        "kind": args.kind,
        "git_sha": _git_sha(),
        "cache_sha256": _sha(cache_path),
        "months": months,
        "split": cache["split"],
        "feature_description": (
            "official annual 2025 features; geographic grid and full validity checked"
            if args.kind == "alphaearth"
            else "last-month standardized public observations plus per-source availability"
        ),
        "source_manifest_sha256": (
            _sha(args.source / "aef_source_manifest.json") if args.kind == "alphaearth" else None
        ),
    }
    _json(args.output / "manifest.json", {**metadata, "records": records})
    _json(args.output / "status.json", {"state": "complete", "patches": len(records)})
