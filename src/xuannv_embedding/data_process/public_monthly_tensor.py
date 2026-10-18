"""Public monthly observations with calendar order, named bands and fresh normalization."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.export import public_statistics
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS, calibrate_public


def ref(path: str | Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha(path)}


def verified(value: dict[str, str]) -> Path:
    if set(value) != {"path", "sha256"} or sha(value["path"]) != value["sha256"]:
        raise ValueError("Monthly public source identity changed")
    return Path(value["path"])


def read_inputs(path: str | Path) -> tuple[dict, dict, list[dict], dict, list[int]]:
    spec = json.loads(Path(path).read_text())
    if (
        set(spec) != {"schema", "observations", "period", "index", "output", "reserve_gib"}
        or spec["schema"] != "public-monthly-tensor-v1"
        or type(spec["reserve_gib"]) is not int
        or spec["reserve_gib"] < 0
    ):
        raise ValueError("Invalid public monthly tensor specification")
    observations = json.loads(verified(spec["observations"]).read_text())
    period = json.loads(verified(spec["period"]).read_text())
    index = json.loads(verified(spec["index"]).read_text())
    if period["observations"] != spec["observations"] or period["index"] != spec["index"]:
        raise ValueError("Monthly period and geographic identity differ")
    dates = [dt.datetime.strptime(s, "%Y-%m") for s in period["input_months"]]
    ordinal = [d.year * 12 + d.month for d in dates]
    if (
        len(dates) != 12
        or any(s != d.strftime("%Y-%m") for s, d in zip(period["input_months"], dates))
        or np.any(np.diff(ordinal) != 1)
    ):
        raise ValueError("Twelve consecutive calendar months are required")
    months = [d.year * 100 + d.month for d in dates]
    if (
        observations.get("schema") != "named-public-statistics-v2"
        or observations.get("kind") not in ["physical_npz", "normalized_tensor"]
        or set(observations["bands"]) != set(PUBLIC_BANDS)
        or any(
            len(observations["bands"][s]) != len(set(observations["bands"][s]))
            or set(observations["bands"][s]) != set(names)
            for s, names in PUBLIC_BANDS.items()
        )
    ):
        raise ValueError("Canonical named public observation bands are required")
    records = json.loads(verified(observations["inputs"]).read_text())["records"]
    if (
        not records
        or len(records) != len(index["records"])
        or len({r["patch_id"] for r in records}) != len(records)
    ):
        raise ValueError("Monthly public geographic count differs")
    for a, b in zip(records, index["records"]):
        if any(a.get(k) != b.get(k, index.get(k)) for k in ["patch_id", "bounds", "crs"]):
            raise ValueError("Monthly public geographic order differs")
    stats = json.loads(verified(observations["statistics"]).read_text())
    return spec, observations, records, stats, months


def read_record(
    record: dict, observations: dict, stats: dict, months: list[int]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    path = verified(record["input"])
    frames, masks = {}, {}
    normalized = observations["kind"] == "normalized_tensor"
    sample = (
        torch.load(str(path), weights_only=True, mmap=True, map_location="cpu")
        if normalized
        else np.load(path)
    )
    try:
        if normalized and not np.array_equal(np.asarray(sample["timestamps"]), months):
            raise ValueError("Intrinsic monthly tensor timestamps differ")
        for s, declared in observations["bands"].items():
            a = np.asarray(sample["source_frames"][s] if normalized else sample[s])
            mask = np.asarray(
                sample["target_masks"][s + "_recon"] if normalized else sample[s + "_valid"]
            )
            allowed = (
                {len(declared), len(declared) + 1}
                if normalized and s in ["s2", "landsat"]
                else {len(declared)}
            )
            if (
                a.ndim != 4
                or not np.issubdtype(a.dtype, np.number)
                or np.iscomplexobj(a)
                or len(a) != len(months)
                or a.shape[1] not in allowed
                or mask.shape != (len(months), *a.shape[-2:])
                or not np.isfinite(mask).all()
                or ((mask < 0) | (mask > 1)).any()
            ):
                raise ValueError("Monthly source frame or mask shape differs")
            mask = mask > 0
            a = a[:, : len(declared)].astype(np.float32)
            if not np.isfinite(a.transpose(0, 2, 3, 1)[mask]).all():
                raise ValueError("Nonfinite valid monthly observation")
            if normalized:
                mean = np.asarray(stats[s]["mean"][: len(declared)], np.float32)
                std = np.asarray(stats[s]["std"][: len(declared)], np.float32)
                if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                    raise ValueError("Invalid archived monthly normalization")
                a = a * std[None, :, None, None] + mean[None, :, None, None]
            frames[s], masks[s] = a, mask
    finally:
        if not normalized:
            sample.close()
    calibrated = calibrate_public(frames, observations["bands"], observations["radiometry"])
    for s, names in PUBLIC_BANDS.items():
        calibrated[s] = calibrated[s][:, [observations["bands"][s].index(b) for b in names]]
        if not np.isfinite(calibrated[s].transpose(0, 2, 3, 1)[masks[s]]).all():
            raise ValueError("Nonfinite calibrated public observation")
    if len({a.shape[-2:] for a in calibrated.values()}) != 1:
        raise ValueError("Monthly public source footprints differ")
    return calibrated, masks


def prepare(path: str | Path) -> Path:
    spec, observations, records, stats, months = read_inputs(path)
    root = Path(spec["output"])
    identity = {
        "spec_sha256": sha(path),
        "implementation_sha256": sha(__file__),
        "public_observation_implementation_sha256": sha(public_statistics.__file__),
    }
    header = root / "identity.json"
    if root.exists() and any(root.iterdir()) and not header.exists():
        raise ValueError("Unowned public monthly output directory")
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError("Monthly public output identity changed")
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.json"
    if (root / "status.json").exists():
        state = json.loads((root / "status.json").read_text())
        if state.get("stage") == "complete":
            metadata = json.loads(metadata_path.read_text())
            if sha(metadata_path) != state["metadata_sha256"] or any(
                sha(metadata[k]["path"]) != metadata[k]["sha256"]
                for k in ["frames", "masks", "normalization"]
            ):
                raise ValueError("Completed monthly public output changed")
            return metadata_path
    first, _ = read_record(records[0], observations, stats, months)
    h, w = first["s2"].shape[-2:]
    shape = (len(records), len(months), 19, h, w)
    reserve = spec["reserve_gib"] * 2**30
    required = int(np.prod(shape)) * 4 + len(records) * len(months) * 3 * h * w
    if shutil.disk_usage(root).free < required + reserve:
        raise RuntimeError("Monthly tensor preparation would exceed the disk reserve")
    dump(header, identity)
    frames = np.lib.format.open_memmap(
        root / "frames.partial.npy", mode="w+", dtype=np.float32, shape=shape
    )
    masks = np.lib.format.open_memmap(
        root / "masks.partial.npy",
        mode="w+",
        dtype=bool,
        shape=(len(records), len(months), 3, h, w),
    )
    totals = {s: [0, np.zeros(len(bs)), np.zeros(len(bs))] for s, bs in PUBLIC_BANDS.items()}
    for i, record in enumerate(records):
        values, valid = read_record(record, observations, stats, months)
        offset = 0
        for j, (s, bands) in enumerate(PUBLIC_BANDS.items()):
            a, v = values[s], valid[s]
            if a.shape[-2:] != (h, w):
                raise ValueError("Monthly source shape changes between geographic records")
            frames[i, :, offset : offset + len(bands)] = np.where(v[:, None], a, 0)
            masks[i, :, j] = v
            selected = a.transpose(0, 2, 3, 1)[v].astype(np.float64)
            totals[s][0] += len(selected)
            totals[s][1] += selected.sum(0)
            totals[s][2] += np.square(selected).sum(0)
            offset += len(bands)
        if i % 20 == 0:
            if shutil.disk_usage(root).free < reserve:
                raise RuntimeError("Monthly tensor disk reserve reached")
            dump(root / "status.json", {"stage": "physical_observations", "completed": i + 1})
    normalization, slices, offset = {}, {}, 0
    for s, bands in PUBLIC_BANDS.items():
        count, total, square = totals[s]
        mean = total / max(count, 1)
        scale = (
            np.maximum(np.sqrt(np.maximum(square / max(count, 1) - mean**2, 0)), 1e-6)
            if count
            else np.ones(len(bands))
        )
        normalization[s] = {"mean": mean.tolist(), "std": scale.tolist(), "observations": count}
        slices[s] = [offset, offset + len(bands)]
        offset += len(bands)
    dump(root / "normalization.json", normalization)
    for i in range(len(records)):
        for j, (s, (start, stop)) in enumerate(slices.items()):
            mean = np.asarray(normalization[s]["mean"], np.float32)[None, :, None, None]
            scale = np.asarray(normalization[s]["std"], np.float32)[None, :, None, None]
            a = np.clip((frames[i, :, start:stop] - mean) / scale, -6, 6)
            frames[i, :, start:stop] = np.where(masks[i, :, j : j + 1], a, 0)
    frames.flush()
    masks.flush()
    (root / "frames.partial.npy").replace(root / "frames.npy")
    (root / "masks.partial.npy").replace(root / "masks.npy")
    dump(
        metadata_path,
        {
            "schema": spec["schema"],
            "months": months,
            "index": spec["index"],
            "source_observations": spec["observations"],
            "period_provenance": spec["period"],
            "band_slices": slices,
            "bands": PUBLIC_BANDS,
            "frames": ref(root / "frames.npy"),
            "masks": ref(root / "masks.npy"),
            "normalization": ref(root / "normalization.json"),
            "shape": list(shape),
            "normalization_clip": [-6, 6],
            "map_or_local_image_values_used": False,
        },
    )
    dump(root / "status.json", {"stage": "complete", "metadata_sha256": sha(metadata_path)})
    return metadata_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv experiment public-monthly-tensor")
    parser.add_argument("--spec", type=Path, required=True)
    prepare(parser.parse_args(argv).spec)
    return 0
