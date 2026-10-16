"""Prepare paired scratch runs and frozen spatial downstream folds for one region."""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import RegionRasterDataset
from xuannv_embedding.export.context import dump, sha


def public_control_config(full: dict) -> dict:
    """Project the full configuration without changing shared architecture settings."""
    public = copy.deepcopy(full)
    model = public["model"]
    excluded = {k for k, v in model["input_sources"].items() if v["role"] == "highres"}
    if not excluded:
        raise ValueError("The full configuration must contain a highres input")
    model["input_sources"] = {k: v for k, v in model["input_sources"].items() if k not in excluded}
    model["target_heads"] = {
        k: v for k, v in model["target_heads"].items() if v["source"] not in excluded
    }
    model["highres_transformer"] = None
    for dataset in public["data"]["datasets"]:
        dataset["source_map"] = {
            k: v for k, v in dataset["source_map"].items() if k not in excluded
        }
    return public


def spatial_five_folds(bounds: list, *, seed: int = 41) -> list[dict]:
    """Balance 2x2-tile blocks over five folds, without looking at model scores."""
    b = np.asarray(bounds, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 4 or not np.isfinite(b).all():
        raise ValueError("Finite projected bounds are required")
    sizes = b[:, 2:] - b[:, :2]
    if np.any(sizes <= 0) or not np.allclose(sizes, sizes[0], atol=0.01, rtol=0):
        raise ValueError("Uniform tile extents are required")
    xy = (b[:, :2] - b[:, :2].min(0)) / sizes[0]
    if not np.allclose(xy, np.round(xy), atol=1e-6, rtol=0):
        raise ValueError("Tile origins must be on the common grid")
    positions = [tuple(row) for row in np.round(xy).astype(int).tolist()]
    if len(set(positions)) != len(positions):
        raise ValueError("Duplicate tile positions")
    blocks: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(positions):
        blocks.setdefault((x // 2, y // 2), []).append(i)
    if len(blocks) < 5:
        raise ValueError("At least five independent spatial blocks are required")
    keys = sorted(blocks)
    order = np.random.default_rng(seed).permutation(len(keys))
    groups: list[list[int]] = [[] for _ in range(5)]
    for j, i in enumerate(order):
        groups[j % 5].extend(blocks[keys[i]])
    return [
        {
            "fold": i,
            "train": sorted(
                v for j, group in enumerate(groups) if j not in {i, (i + 1) % 5} for v in group
            ),
            "validation": sorted(groups[(i + 1) % 5]),
            "test": sorted(groups[i]),
            "buffer": [],
        }
        for i in range(5)
    ]


def prepare_static_targets(sample: dict) -> dict:
    """Register the same WorldCover weak target for monthly and annual objectives."""
    target = sample["targets"]["worldcover"]
    mask = sample["target_masks"]["worldcover"]
    if target.ndim != 3 or not torch.equal(target, target[:1].expand_as(target)):
        raise ValueError("WorldCover must be one static reference, not changing monthly labels")
    if not torch.equal(mask, mask[:1].expand_as(mask)):
        raise ValueError("WorldCover validity must be static")
    if target.min() < 0 or target.max() > 11:
        raise ValueError("WorldCover must use 1..11 with unknown=0")
    result = dict(sample)
    result["targets"] = {k: v for k, v in sample["targets"].items() if k != "worldcover"}
    result["target_masks"] = {k: v for k, v in sample["target_masks"].items() if k != "worldcover"}
    result["targets"].update(osm_landcover=target, esa_worldcover=target[0])
    result["target_masks"].update(osm_landcover=mask, esa_worldcover=mask[0])
    return result


_DATASET: RegionRasterDataset | None = None
_CACHE_ROOT: Path | None = None


def cache_data(config: Config) -> dict:
    """Keep nested configuration paths portable in runtime JSON documents."""
    return json.loads(json.dumps(asdict(config.data), default=str))


def _worker_init(config: str, output: str) -> None:
    global _DATASET, _CACHE_ROOT
    torch.set_num_threads(1)
    cfg = Config.from_yaml(config)
    _DATASET = RegionRasterDataset(cfg, cfg.data.datasets[0])
    _CACHE_ROOT = Path(output)


def _cache_sample(index: int) -> dict:
    assert _DATASET is not None and _CACHE_ROOT is not None
    if shutil.disk_usage(_CACHE_ROOT).free < 100 * 2**30:
        raise RuntimeError("Disk reserve reached")
    sample = prepare_static_targets(_DATASET[index])
    record = _DATASET.records[index]
    source = next(iter(_DATASET.config.model.input_sources))
    paths = _DATASET._record_value(record, source)
    path = paths[0] if isinstance(paths, list) else paths
    with rasterio.open(_DATASET._absolute(path)) as raster:
        bounds, crs = list(raster.bounds), str(raster.crs)
    row = {
        "index": index,
        "patch_id": sample["patch_id"],
        "region": sample["region"],
        "bounds": bounds,
        "crs": crs,
        "models": {},
    }
    for name in ("highres", "public"):
        value = (
            sample if name == "highres" else {**sample, "highres_frames": {}, "highres_masks": {}}
        )
        destination = _CACHE_ROOT / name / "samples" / f"{index:06d}.pt"
        temporary = destination.with_suffix(".partial")
        torch.save(value, temporary)
        temporary.replace(destination)
        row["models"][name] = {"path": str(destination), "sha256": sha(destination)}
    dump(_CACHE_ROOT / "receipts" / f"{index:06d}.json", row)
    return row


def prepare(args: argparse.Namespace) -> dict:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Config.from_yaml(args.highres_config)
    if len(source.data.datasets) != 1:
        raise ValueError("The trial must contain one independent training region")
    identity = {
        "source_sha256": sha(args.highres_config),
        "seed": args.seed,
        "epochs": args.epochs,
        "protocol": "paired-region-v1",
    }
    identity_path = root / "preparation-identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Preparation identity changed; use a separate trial directory")
    dump(identity_path, identity)
    (root / "configs").mkdir(exist_ok=True)
    raw = yaml.safe_load(args.highres_config.read_text())
    configurations = {"highres": copy.deepcopy(raw), "public": public_control_config(raw)}
    for name, config in configurations.items():
        config["experiment"].update(name=name, seed=args.seed, output_dir=str(root / "runs" / name))
        config["training"]["epochs"] = args.epochs
        config["data"]["num_workers"] = 0
        path = root / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        Config.from_yaml(path)
        (root / "cache" / name / "samples").mkdir(parents=True, exist_ok=True)
    (root / "cache/receipts").mkdir(exist_ok=True)
    dataset = RegionRasterDataset(source, source.data.datasets[0])
    rows: dict[int, dict] = {}
    for path in (root / "cache/receipts").glob("*.json"):
        row = json.loads(path.read_text())
        if all(
            Path(ref["path"]).exists() and sha(ref["path"]) == ref["sha256"]
            for ref in row["models"].values()
        ):
            rows[row["index"]] = row
    pending = [i for i in range(len(dataset)) if i not in rows]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=_worker_init,
        initargs=(str(root / "configs/highres.yaml"), str(root / "cache")),
    ) as pool:
        for row in pool.map(_cache_sample, pending):
            rows[row["index"]] = row
            dump(
                root / "preparation-status.json",
                {"state": "caching", "completed": len(rows), "total": len(dataset)},
            )
    ordered = [rows[i] for i in range(len(dataset))]
    folds = spatial_five_folds([r["bounds"] for r in ordered], seed=args.seed)
    dump(
        root / "folds.json",
        {
            "schema": "spatial-five-fold-v1",
            "seed": args.seed,
            "block_tiles": 2,
            "pixel_buffer_m": 100,
            "folds": folds,
        },
    )
    steps = math.ceil(args.epochs * len(dataset) / 48)
    for name in configurations:
        config_path = root / "configs" / f"{name}.yaml"
        cfg = Config.from_yaml(config_path)
        records = [
            {k: v for k, v in r.items() if k != "models"} | r["models"][name] for r in ordered
        ]
        cache_path = root / "cache" / name / "cache.json"
        dump(
            cache_path,
            {
                "state": "complete",
                "months": cfg.data.months,
                "records": records,
                "split": folds[0],
                "data": cache_data(cfg),
                "domains": [
                    {
                        "region": records[0]["region"],
                        "months": cfg.data.months,
                        "indices": list(range(len(records))),
                    }
                ],
            },
        )
        spec = {
            "schema": "regional-base-v1",
            "mode": "scratch",
            "checkpoint": None,
            "config": {"path": str(config_path), "sha256": sha(config_path)},
            "cache": {"path": str(cache_path), "sha256": sha(cache_path)},
            "output": str(root / "runs" / name),
            "training": {
                "steps": steps,
                "head_steps": 0,
                "warmup_steps": min(200, steps),
                "seed": args.seed,
                "effective_batch": 48,
                "rates": {"public": 1e-4, "highres": 1e-4, "heads": 1e-4},
                "weight_decay": 0.05,
                "save_every": 100,
                "static_weight": 0.25,
                "min_free_gib": 100,
            },
        }
        if name == "public":
            initial = root / "configs/highres.yaml"
            spec["initialization_config"] = {"path": str(initial), "sha256": sha(initial)}
        spec_path = root / f"{name}-spec.json"
        dump(spec_path, spec)
    report = {
        "state": "prepared",
        "samples": len(ordered),
        "encoder_training": "all tiles",
        "months": source.data.months,
        "steps": steps,
        "effective_batch": 48,
        "sample_exposure_epochs": steps * 48 / len(ordered),
        "downstream_folds": 5,
        "training_started": False,
        "evaluation_references_in_training": False,
    }
    dump(root / "preparation-status.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv experiment paired-region")
    parser.add_argument("--highres-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        raise ValueError("Positive exposure budget required")
    print(json.dumps(prepare(args), ensure_ascii=False))
    return 0
