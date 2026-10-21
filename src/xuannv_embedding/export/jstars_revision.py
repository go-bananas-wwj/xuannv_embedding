"""Registered observation interventions for the fixed journal representation."""

from __future__ import annotations

import hashlib

import numpy as np


def retain_observation(image, coverage, fraction, tile_id, seed):
    """Retain a nested edge-anchored prefix of actually observed raw pixels."""
    image, coverage = np.asarray(image), np.asarray(coverage)
    if (
        type(fraction) not in {int, float}
        or fraction not in {0, 0.25, 0.5, 1}
        or image.ndim != 3
        or coverage.shape != (1, *image.shape[-2:])
        or not np.isfinite(coverage).all()
        or (coverage < 0).any()
        or (coverage > 1).any()
        or type(seed) is not int
        or not isinstance(tile_id, str)
    ):
        raise ValueError("Invalid registered observation retention")
    edge = hashlib.sha256(f"{seed}:{tile_id}".encode()).digest()[0] % 4
    # The same edge is used for all dates at a location; existing gaps stay gaps.
    ordered = np.rot90(coverage[0], edge)
    valid = np.flatnonzero(ordered.ravel() > 0)
    keep = np.zeros(ordered.size, bool)
    keep[valid[: int(len(valid) * fraction)]] = True
    keep = np.rot90(keep.reshape(ordered.shape), -edge)[None]
    mask = np.where(keep, coverage, 0).copy()
    values = np.where(mask > 0, image, 0).copy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite retained observation")
    return values, mask


def select_observations(frames, slots, fraction, tile_id, seed):
    """Hide raw values and coverage together without renumbering calendar slots."""
    if slots != sorted(set(slots)) or any(type(i) is not int or i < 0 for i in slots):
        raise ValueError("Observation slots must be unique ordered nonnegative integers")
    count = 0
    for count, (image, mask) in enumerate(frames, 1):
        yield retain_observation(image, mask, fraction if count - 1 in slots else 0, tile_id, seed)
    if slots and slots[-1] >= count:
        raise ValueError("Selected observation slot exceeds the actual inventory")


def export_guidance(path, device):
    """Intervene before terminal image guidance, retaining its upstream representation."""
    import json
    import shutil
    import time
    from pathlib import Path

    import torch

    from xuannv_embedding.data_process import subpixel_rgb as rgb
    from xuannv_embedding.export import rgb_guided_refinement as legacy
    from xuannv_embedding.export import viewwise_guided_refinement as viewwise
    from xuannv_embedding.export.context import dump, sha
    from xuannv_embedding.training.cli import _setup_device
    from xuannv_embedding.training.frozen_fusion import verified

    spec = viewwise.shared.read_json(path)
    if (
        set(spec) != {"schema", "guide", "mode", "fraction", "slots", "seed", "output"}
        or spec["schema"] != "jstars-guidance-intervention-v1"
        or spec["mode"] not in {"viewwise", "mean_first", "no_guide"}
        or spec["fraction"] not in {0, 0.25, 0.5, 1}
        or type(spec["seed"]) is not int
        or not Path(spec["output"]).is_absolute()
    ):
        raise ValueError("Invalid standalone journal guidance intervention")
    guide = viewwise.specification(verified(spec["guide"]))
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {"spec": sha(path), "code": sha(__file__), "upstream_features": guide["features"]}
    status = root / "status.json"
    if status.exists():
        previous = json.loads(status.read_text())
        if previous["identity"] != identity:
            raise ValueError("Guidance intervention identity changed")
        verified(previous["features"])
        return previous
    started = time.monotonic()
    data = viewwise.SourceData(guide)
    shape = data.base.shape
    if shutil.disk_usage(root).free < 100 * 2**30 + np.prod(shape) * 2:
        raise RuntimeError("Journal export violates 100 GiB reserve")
    slots = spec["slots"]
    if slots != sorted(set(slots)) or any(i < 0 or i >= data.view_count for i in slots):
        raise ValueError("Invalid actual calendar slots")
    dev, _, _ = _setup_device(device)
    model = None
    if spec["mode"] != "no_guide":
        model = (
            legacy.load_anyup(guide["upsampler"]["source_bundle"], guide["upsampler"]["checkpoint"])
            .to(dev)
            .eval()
        )
    dest = root / "annual.npy"
    values = np.lib.format.open_memmap(
        root / "annual.partial.npy", mode="w+", dtype="float16", shape=shape
    )
    with torch.inference_mode():
        for i in range(0, data.n, 2):
            stop = min(i + 2, data.n)
            base = torch.tensor(np.array(data.base[i:stop], np.float32), device=dev)
            batches = []
            for position in range(i, stop):
                frames = list(
                    select_observations(
                        data.raw.frames(position),
                        slots,
                        spec["fraction"],
                        str(position),
                        spec["seed"],
                    )
                )
                if spec["mode"] == "mean_first":
                    frames = [rgb.pool_rgb(frames, data.raw_spec["source_shape"])]
                batches.append(frames)
            packed = []
            for observations in zip(*batches, strict=True):
                images, masks = [], []
                for image, mask in observations:
                    image, mask = rgb.pool_rgb([(image, mask)], data.raw_spec["source_shape"])
                    image, mask = rgb.pack_subpixels(image, mask, factor=5)
                    images.append(image.astype(np.float16).astype(np.float32))
                    masks.append(mask.astype(np.float16).astype(np.float32))
                packed.append(
                    (torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(masks)))
                )
            if model is None:
                out = base
            else:
                out = viewwise.refine_views(
                    base,
                    iter(packed),
                    model,
                    guide["normalization"],
                    guide["q_chunk_size"],
                    0.75,
                    len(packed),
                )
            values[i:stop] = out.float().cpu().numpy().astype(np.float16)
            if i % 20 == 0:
                dump(root / "progress.json", {"completed": i + 1, "total": data.n})
                print("guidance", root.name, i + 1, data.n, flush=True)
    values.flush()
    del values
    (root / "annual.partial.npy").replace(dest)
    result = {
        "state": "complete",
        "identity": identity,
        "features": {"path": str(dest), "sha256": sha(dest)},
        "shape": list(shape),
        "export_batch_size": 2,
        "wall_seconds": time.monotonic() - started,
        "intervention_scope": "terminal_image_guidance",
        "upstream_local_information_retained": True,
        "all_registered_calendar_slots_preserved": True,
        "heads_refitted": False,
    }
    dump(status, result)
    return result


def score_frozen(path, device):
    """Read changed features with the original heads, support and thresholds."""
    import json
    import time
    from pathlib import Path

    from xuannv_embedding.downstream import full_region_readouts as full
    from xuannv_embedding.export.context import dump, sha
    from xuannv_embedding.training.frozen_fusion import verified
    from xuannv_embedding.training.manuscript_followup import _outside_metrics
    from xuannv_embedding.training.shared_fusion_search import reader_jobs

    spec = full.primary._load(path)
    if (
        set(spec) != {"schema", "region", "feature", "readers", "output"}
        or spec["schema"] != "jstars-frozen-intervention-v1"
    ):
        raise ValueError("Invalid frozen journal readout specification")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {"spec_sha256": sha(path), "code": sha(__file__), "readout": full.code()}
    dest = root / "result.json"
    if dest.exists():
        previous = json.loads(dest.read_text())
        if previous["identity"] != identity:
            raise ValueError("Completed journal scores changed")
        return previous
    started = time.monotonic()
    feature = verified(spec["feature"])
    jobs = reader_jobs(
        {"output": str(root / "projection")},
        spec["region"],
        {"name": "frozen_intervention", "kind": "journal_intervention"},
        feature,
    )
    rows = []
    for job in jobs:
        candidate, data, conditions = full.specification(job["spec"])
        candidate["device"] = device
        batch, labels, valid = full._arrays(candidate, data)
        original, reference, _ = full.specification(verified(spec["readers"][job["head"]]))
        for key in [
            "indices",
            "records",
            "valid",
            "labels",
            "train_positions",
            "calibration_positions",
        ]:
            if data[key] != reference[key]:
                raise ValueError("Frozen intervention geographic or label contract changed")
        for condition in conditions:
            cal = Path(original["output"]) / "calibration/records" / (condition["key"] + ".json")
            record = json.loads(cal.read_text())
            model = full.load_head(
                record["readout"], record["readout_identity_sha256"], job["head"], device
            )
            row = full._predict(
                root / job["head"],
                candidate,
                condition,
                batch,
                labels[condition["task"]],
                valid,
                list(range(len(batch.indices))),
                model,
                record["support"],
            )
            if row["metrics"]["threshold"] != record["row"]["metrics"]["threshold"]:
                raise ValueError("Journal intervention changed a frozen threshold")
            prediction = root / job["head"] / "predictions/candidate" / (condition["key"] + ".npz")
            excluded = set(record["support"]["support_tiles"]) | {
                data["indices"][i] for i in data["calibration_positions"]
            }
            with np.load(prediction) as z:
                keep = ~np.isin(z["tiles"], list(excluded))
                outside = _outside_metrics(
                    z["truth"][keep], z["scores"][keep], row["metrics"]["threshold"]
                )
            rows.append({**row, "outside_support_and_calibration": outside})
            dump(root / "progress.json", {"completed": len(rows), "total": 8})
            del model
    result = {
        "state": "complete",
        "identity": identity,
        "rows": rows,
        "heads_refitted": False,
        "thresholds_refitted": False,
        "wall_seconds": time.monotonic() - started,
    }
    dump(dest, result)
    return result


def run_case(path, device):
    """Export and score a registered case, with immutable input/output identities."""
    from pathlib import Path

    from xuannv_embedding.downstream.paired_multitask import _load
    from xuannv_embedding.export.context import dump
    from xuannv_embedding.training.frozen_fusion import verified

    spec = _load(path)
    if (
        set(spec) != {"schema", "guide", "region", "readers", "output"}
        or spec["schema"] != "jstars-guidance-case-v1"
    ):
        raise ValueError("Invalid journal intervention case")
    result = export_guidance(verified(spec["guide"]), device)
    score_path = Path(spec["output"]) / "score.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score = {
        "schema": "jstars-frozen-intervention-v1",
        "region": spec["region"],
        "feature": result["features"],
        "readers": spec["readers"],
        "output": str(Path(spec["output"]) / "evaluation"),
    }
    if score_path.exists() and _load(score_path) != score:
        raise ValueError("Registered case feature identity changed")
    dump(score_path, score)
    return score_frozen(score_path, device)


def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["guide", "score", "case", "source-case", "raw-case", "figures"]
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.phase == "raw-case":
        from xuannv_embedding.export.journal_observations import run_raw_case

        run_raw_case(args.spec, args.device)
    elif args.phase == "source-case":
        from xuannv_embedding.export.journal_sources import run_absent_case

        run_absent_case(args.spec, args.device)
    elif args.phase == "figures":
        from xuannv_embedding.downstream.journal_figures import geographic_cases

        geographic_cases(args.spec)
    else:
        {"guide": export_guidance, "score": score_frozen, "case": run_case}[args.phase](
            args.spec, args.device
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
