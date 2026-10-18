"""Named-band public observation summaries, without high-resolution or map inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.export.context import dump, sha

PUBLIC_BANDS = {
    "s2": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
    "s1": ["vv", "vh"],
    "landsat": ["blue", "green", "red", "nir08", "swir16", "swir22"],
}


def _summaries(values, valid):
    observed = valid[:, None] if valid.ndim == 3 else valid
    count = observed.sum(0)
    clean = np.where(observed, values, 0)
    mean = clean.sum(0) / np.maximum(count, 1)
    variance = (clean**2).sum(0) / np.maximum(count, 1) - mean**2
    minimum = np.where(count > 0, np.where(observed, values, np.inf).min(0), 0)
    maximum = np.where(count > 0, np.where(observed, values, -np.inf).max(0), 0)
    return [mean, np.sqrt(np.maximum(variance, 0)), minimum, maximum]


def summarize_public(frames: dict, masks: dict, bands: dict) -> tuple[np.ndarray, list[str]]:
    if any(set(item) != set(PUBLIC_BANDS) for item in [frames, masks, bands]):
        raise ValueError("Three canonical public sources are required")
    arrays, names, shape = [], [], None
    for source, canonical in PUBLIC_BANDS.items():
        declared = bands[source]
        values, valid = np.asarray(frames[source]), np.asarray(masks[source])
        if (
            len(declared) != len(set(declared))
            or set(declared) != set(canonical)
            or values.ndim != 4
            or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values)
            or values.shape[1] != len(declared)
            or valid.dtype != bool
            or valid.shape != (len(values), *values.shape[-2:])
            or not np.isfinite(values.transpose(0, 2, 3, 1)[valid]).all()
        ):
            raise ValueError("Invalid band contract, observations or public validity")
        if shape is not None and shape != values.shape[-2:]:
            raise ValueError("Public source footprints differ")
        shape = values.shape[-2:]
        values = values[:, [declared.index(b) for b in canonical]].astype(np.float32)
        stats = _summaries(values, valid)
        for statistic, result in zip(["mean", "std", "min", "max"], stats):
            arrays.extend(result)
            names.extend(f"{source}:{b}:{statistic}" for b in canonical)
        arrays.append(valid.mean(0))
        names.append(f"{source}:availability")
        if source in {"s2", "landsat"}:
            red, green, nir = ("B04", "B03", "B08") if source == "s2" else ("red", "green", "nir08")
            indices, index_valid = [], []
            for first, second in [(nir, red), (green, nir)]:
                a, b = values[:, canonical.index(first)], values[:, canonical.index(second)]
                usable = valid & (a >= 0) & (b >= 0) & ((a + b) > 1e-6)
                index = np.divide(a - b, a + b, out=np.zeros_like(a), where=usable)
                indices.append(index)
                index_valid.append(usable)
            for statistic, result in zip(
                ["mean", "std", "min", "max"],
                _summaries(np.stack(indices, 1), np.stack(index_valid, 1)),
            ):
                arrays.extend(result)
                names.extend(f"{source}:{b}:{statistic}" for b in ["ndvi", "ndwi"])
    output = np.stack(arrays).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite public summary")
    return output, names


def calibrate_public(frames: dict, bands: dict, radiometry: dict) -> dict:
    if set(frames) != set(radiometry) or set(frames) != set(bands):
        raise ValueError("Radiometry must explicitly cover every public source")
    output = {}
    for source, values in frames.items():
        if set(radiometry[source]) != set(bands[source]):
            raise ValueError("Radiometry must be declared by band name")
        scales, offsets = [], []
        for band in bands[source]:
            item = radiometry[source][band]
            if (
                set(item) != {"scale", "offset"}
                or any(type(v) not in {int, float} or not np.isfinite(v) for v in item.values())
                or item["scale"] <= 0
            ):
                raise ValueError("Invalid scale/offset radiometry contract")
            scales.append(item["scale"])
            offsets.append(item["offset"])
        scale = np.asarray(scales, np.float32)[None, :, None, None]
        offset = np.asarray(offsets, np.float32)[None, :, None, None]
        output[source] = np.asarray(values, np.float32) * scale + offset
    return output


def _verified(ref):
    if set(ref) != {"path", "sha256"} or sha(ref["path"]) != ref["sha256"]:
        raise ValueError("Public statistics input identity changed")
    return Path(ref["path"])


def export(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    if (
        set(spec) != {"schema", "inputs", "bands", "statistics", "kind", "output", "radiometry"}
        or spec["schema"] != "named-public-statistics-v2"
        or spec["kind"] not in {"normalized_tensor", "physical_npz"}
    ):
        raise ValueError("Invalid public summary specification")
    inputs = json.loads(_verified(spec["inputs"]).read_text())
    if set(inputs) != {"records"} or not inputs["records"]:
        raise ValueError("Invalid public summary observation index")
    stats = json.loads(_verified(spec["statistics"]).read_text()) if spec["statistics"] else None
    if stats is None:
        raise ValueError("Archived statistics are required for the common model-domain clipping")
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {"spec_sha256": sha(spec_path), "implementation_sha256": sha(__file__)}
    if (root / "identity.json").exists() and json.loads(
        (root / "identity.json").read_text()
    ) != identity:
        raise ValueError("Public summary recipe changed")
    dump(root / "identity.json", identity)
    if (root / "status.json").exists() and (root / "annual.npy").exists():
        status = json.loads((root / "status.json").read_text())
        if status.get("stage") == "complete" and sha(root / "annual.npy") == status["sha256"]:
            return
    count = len(inputs["records"])
    output, names = None, None
    for i, record in enumerate(inputs["records"]):
        path = _verified(record["input"])
        frames, valid = {}, {}
        if spec["kind"] == "normalized_tensor":
            sample = torch.load(str(path), weights_only=True, mmap=True)
            for source, declared in spec["bands"].items():
                value = sample["source_frames"][source].numpy()[:, : len(declared)]
                mean = np.asarray(stats[source]["mean"][: len(declared)], np.float32)[
                    None, :, None, None
                ]
                std = np.asarray(stats[source]["std"][: len(declared)], np.float32)[
                    None, :, None, None
                ]
                frames[source] = np.clip(value, -6, 6) * std + mean
                valid[source] = sample["target_masks"][source + "_recon"].numpy() > 0
        else:
            with np.load(path) as stored:
                for source, declared in spec["bands"].items():
                    value = stored[source][:, : len(declared)]
                    mean = np.asarray(stats[source]["mean"][: len(declared)], np.float32)[
                        None, :, None, None
                    ]
                    std = np.asarray(stats[source]["std"][: len(declared)], np.float32)[
                        None, :, None, None
                    ]
                    frames[source] = np.clip((value - mean) / std, -6, 6) * std + mean
                    valid[source] = stored[source + "_valid"].astype(bool)
        frames = calibrate_public(frames, spec["bands"], spec["radiometry"])
        values, columns = summarize_public(frames, valid, spec["bands"])
        if output is None:
            output = np.lib.format.open_memmap(
                root / "annual.partial.npy",
                mode="w+",
                dtype=np.float32,
                shape=(count, *values.shape),
            )
            names = columns
        if values.shape != output.shape[1:] or columns != names:
            raise ValueError("Public statistic schema varies across tiles")
        output[i] = values
        if (i + 1) % 20 == 0 or i + 1 == count:
            output.flush()
            dump(root / "status.json", {"stage": "exporting", "completed": i + 1, "total": count})
    output.flush()
    (root / "annual.partial.npy").replace(root / "annual.npy")
    dump(
        root / "columns.json",
        {
            "columns": names,
            "bands": spec["bands"],
            "map_labels_used": False,
            "radiometry": spec["radiometry"],
            "model_domain_clip": [-6, 6],
            "NDVI": "(B08-B04)/(B08+B04)",
            "NDWI": "(B03-B08)/(B03+B08)",
        },
    )
    dump(
        root / "status.json",
        {
            "stage": "complete",
            "shape": list(output.shape),
            "sha256": sha(root / "annual.npy"),
            "map_labels_used": False,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment public-statistics")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.spec)
    return 0
