"""Export frozen weights with real neighbouring input context on a common grid."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def context_window(tiles, position, margin):
    """Stitch neighbouring tiles; absent exterior is zero, never fabricated reflection."""
    center = tiles[position]
    h, w = center.shape[-2:]
    if h != w or not 0 <= margin <= h:
        raise ValueError("context requires square tiles and margin in [0, size]")
    if margin == 0:
        return center
    out = center.new_zeros(*center.shape[:-2], h + 2 * margin, w + 2 * margin)
    x, y = position
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            tile = tiles.get((x + dx, y + dy))
            if tile is None:
                continue
            oy, ox = margin + dy * h, margin + dx * w
            y0, y1 = max(0, oy), min(h + 2 * margin, oy + h)
            x0, x1 = max(0, ox), min(w + 2 * margin, ox + w)
            if y1 > y0 and x1 > x0:
                out[..., y0:y1, x0:x1] = tile[..., y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def run(args):
    from xuannv_embedding.config import Config
    from xuannv_embedding.data.raster_dataset import RegionRasterDataset
    from xuannv_embedding.training.checkpoint import load_training_checkpoint
    from xuannv_embedding.training.cli import _git_sha, _setup_device, build_training_system
    from xuannv_embedding.training.compatibility import load_compatible_checkpoint

    faulthandler.register(signal.SIGUSR1)
    torch.set_num_threads(2)
    spec = json.loads(args.spec.read_text())
    cfg = Config.from_yaml(spec["models"][args.model]["config"])
    definition = spec["models"][args.model]
    checkpoint = Path(definition["checkpoint"])
    if sha(checkpoint) != definition["sha256"]:
        raise ValueError("checkpoint identity changed")
    cache = json.loads(Path(spec["cache"]).read_text())
    out = Path(spec["output"]) / "exports" / args.model
    out.mkdir(parents=True, exist_ok=True)
    identity = {
        "model": args.model,
        "checkpoint_sha256": definition["sha256"],
        "cache_sha256": sha(spec["cache"]),
        "config_sha256": sha(definition["config"]),
        "spec_sha256": sha(args.spec),
        "code_sha": _git_sha(),
        "context_margin": 16,
        "center_size": 128,
        "grid_m": 10,
        "highres_preprocessing": (
            "original static aggregation; bilinear to 10m before "
            "neighbour assembly; no new observations"
        ),
        "outside_coverage": (
            "zero input, missing highres mask; " "central temporal availability retained"
        ),
        "months": list(cfg.data.months),
        "records": cache["records"],
        "split": cache["split"],
    }
    if (out / "identity.json").exists():
        previous = json.loads((out / "identity.json").read_text())
        for k in ("checkpoint_sha256", "cache_sha256", "config_sha256", "spec_sha256"):
            if previous[k] != identity[k]:
                raise ValueError("resume identity differs")
    dump(out / "identity.json", identity)
    started = time.monotonic()
    samples, positions = [], []
    b = np.array([r["bounds"] for r in cache["records"]])
    origin = (b[:, 0].min(), b[:, 3].max())
    for r in cache["records"]:
        if sha(r["path"]) != r["sha256"]:
            raise ValueError("cache sample changed")
        s = torch.load(r["path"], weights_only=True, mmap=False)
        samples.append({k: s[k] for k in ("source_frames", "source_masks", "timestamps")})
        positions.append(
            (round((r["bounds"][0] - origin[0]) / 1280), round((origin[1] - r["bounds"][3]) / 1280))
        )
    spatial = {
        name: {pos: s["source_frames"][name] for pos, s in zip(positions, samples)}
        for name in samples[0]["source_frames"]
    }
    hr, hm = {}, {}
    if definition.get("compatibility_profile"):
        ds = RegionRasterDataset(cfg, cfg.data.datasets[0])
        lookup = {r.patch_id: r for r in ds.records}
        hrdir = out / "highres_inputs"
        hrdir.mkdir(exist_ok=True)
        for name, sc in cfg.model.input_sources.items():
            if sc.role != "highres":
                continue
            hr[name], hm[name] = {}, {}
            for i, (r, pos) in enumerate(zip(cache["records"], positions)):
                path = hrdir / f"{r['patch_id']}_{name}.npz"
                if path.exists():
                    with np.load(path) as z:
                        x, m = torch.from_numpy(z["x"]), torch.from_numpy(z["m"])
                else:
                    observations = ds._observations(lookup[r["patch_id"]], name, sc.channels)
                    x, m = ds._highres_input(name, observations, sc.channels)
                    x = F.interpolate(
                        x[None], size=(128, 128), mode="bilinear", align_corners=False
                    )[0]
                    m = F.interpolate(m[None], size=(128, 128), mode="nearest")[0]
                    np.savez(path, x=x.numpy(), m=m.numpy())
                hr[name][pos], hm[name][pos] = x, m
                if i % 40 == 0:
                    print("highres", name, i, flush=True)
    model = build_training_system(cfg).model
    if definition.get("compatibility_profile"):
        load_compatible_checkpoint(checkpoint, model, profile=definition["compatibility_profile"])
    else:
        load_training_checkpoint(
            checkpoint,
            model=model,
            expected_config_sha256=sha(definition["config"]),
            expected_source_schema={k: asdict(v) for k, v in cfg.model.input_sources.items()},
            expected_regions=[d.region for d in cfg.data.datasets],
        )
    device, _, _ = _setup_device(args.device)
    model.to(device).eval()
    records = []
    for i, (r, pos, s) in enumerate(zip(cache["records"], positions, samples)):
        path = out / f"{r['patch_id']}.npz"
        if not path.exists():
            tic = time.monotonic()
            results = {}
            with torch.inference_mode():
                for margin in (16, 0):
                    frame = {
                        k: context_window(v, pos, margin)[None].clone().to(device)
                        for k, v in spatial.items()
                    }
                    masks = {k: v[None].to(device) for k, v in s["source_masks"].items()}
                    high = {
                        k: context_window(v, pos, margin)[None].clone().to(device)
                        for k, v in hr.items()
                    }
                    high_masks = {
                        k: context_window(v, pos, margin)[None].clone().to(device)
                        for k, v in hm.items()
                    }
                    output = model(frame, masks, s["timestamps"][None].to(device), high, high_masks)
                    z = output.embedding_map[0].float().cpu().numpy()
                    if not np.isfinite(z).all():
                        raise ValueError("nonfinite embedding")
                    if margin:
                        z = z[..., margin:-margin, margin:-margin]
                        results["embedding"] = z
                    else:
                        results["without_context"] = z[-1]
            np.savez(path, **results, seconds=time.monotonic() - tic)
        records.append(
            {
                "patch_id": r["patch_id"],
                "bounds": r["bounds"],
                "path": str(path),
                "sha256": sha(path),
            }
        )
        if i % 10 == 0:
            print("export", args.model, i, flush=True)
            dump(
                out / "status.json",
                {
                    "state": "running",
                    "patches": i + 1,
                    "total": len(samples),
                    "elapsed": time.monotonic() - started,
                },
            )
    dump(
        out / "manifest.json",
        {
            **identity,
            "records": records,
            "wall_seconds": time.monotonic() - started,
            "bytes": sum(Path(r["path"]).stat().st_size for r in records),
            "forward_export_seconds": sum(float(np.load(r["path"])["seconds"]) for r in records),
        },
    )
    dump(
        out / "status.json",
        {"state": "complete", "patches": len(records), "elapsed": time.monotonic() - started},
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cpu")
    run(p.parse_args(argv))
    return 0
