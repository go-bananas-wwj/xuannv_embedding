"""Native-grid annual observations with explicit provenance and quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from xuannv_embedding.data_process.annual_quality import LOWRES_CHANNELS, relative_file
from xuannv_embedding.data_process.observation_raster import parent_geometry
from xuannv_embedding.utils.manifest import load_manifest


class AnnualObservationDataset(Dataset):
    """Read annual manifests without resizing, averaging or inventing observations."""

    def __init__(self, manifest: Path, *, allow_candidates: bool = False) -> None:
        self.directory = manifest.parent
        summary = json.loads((self.directory / "summary.json").read_text())
        if not allow_candidates and not summary.get("training_ready"):
            raise ValueError("Annual dataset has not passed scientific quality gates")
        self.root = Path(summary["data_root"])
        document = load_manifest(manifest)
        self.records = document.records
        years = {record.provenance["year"] for record in self.records}
        if len(years) != 1 or document.meta.months != [
            f"{next(iter(years))}-{month:02}" for month in range(1, 13)
        ]:
            raise ValueError("Annual manifest must contain exactly one complete calendar year")
        self.schemas = json.loads((self.directory / "sources.json").read_text())
        self.statistics = {}
        for source, schema in self.schemas.items():
            path = self.directory / "statistics" / f"{source}_stats.json"
            if not path.exists():
                continue
            values = json.loads(path.read_text())
            if values.get("fit_split") != "train":
                raise ValueError(f"Annual statistics must be fit on train only: {source}")
            mean = torch.tensor(values["mean"], dtype=torch.float32)[:, None, None]
            std = torch.tensor(values["std"], dtype=torch.float32)[:, None, None]
            if (
                mean.shape != std.shape
                or len(mean) != schema["channels"]
                or not bool(
                    torch.isfinite(mean).all() & torch.isfinite(std).all() & (std > 0).all()
                )
            ):
                raise ValueError(f"Invalid statistics: {source}")
            self.statistics[source] = mean, std

    def __len__(self) -> int:
        return len(self.records)

    def _read(self, observation: dict) -> tuple[torch.Tensor, torch.Tensor]:
        path = relative_file(self.root, observation["path"])
        with rasterio.open(path) as raster:
            epsg, bounds = parent_geometry(observation["parent_key"])
            if (
                raster.crs is None
                or raster.crs.to_epsg() != epsg
                or not np.allclose(raster.bounds, bounds, rtol=0, atol=0.01)
            ):
                raise ValueError("Annual observation parent geometry mismatch")
            values = raster.read(out_dtype="float32")
            geometry = (raster.crs, raster.transform, raster.shape)
            valid = (raster.read_masks() > 0).all(axis=0) & np.isfinite(values).all(axis=0)
        with rasterio.open(relative_file(self.root, observation["mask"])) as raster:
            if (raster.crs, raster.transform, raster.shape) != geometry:
                raise ValueError("Annual mask geometry mismatch")
            valid &= raster.read(1) > 0
        if values.shape != (observation["channels"], observation["height"], observation["width"]):
            raise ValueError("Annual observation shape mismatch")
        if observation["source"] in LOWRES_CHANNELS and (values[:, valid] <= -32768).any():
            raise ValueError("Annual observation contains undeclared negative fill")
        mean, std = self.statistics[observation["source"]]
        frame = (torch.from_numpy(np.nan_to_num(values)) - mean) / std
        mask = torch.from_numpy(valid)
        frame = torch.where(mask[None], frame, torch.zeros_like(frame))
        if not bool(torch.isfinite(frame).all()):
            raise ValueError("Nonfinite annual normalized input")
        return frame, mask

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        year = record.provenance["year"]
        metadata = record.provenance["observations"]
        if set(metadata) - set(self.schemas) or set(record.sources) - set(self.schemas):
            raise ValueError("Unknown source in annual manifest")
        frames, masks, highres = {}, {}, []
        for source, schema in self.schemas.items():
            if source not in self.statistics:
                continue
            observations = metadata.get(source, [])
            paths = record.sources.get(source) or []
            if paths != [item["path"] for item in observations]:
                raise ValueError("Annual manifest paths disagree with metadata")
            if schema["role"] == "temporal":
                size = 43 if source == "landsat" else 128
                shape = (schema["channels"], size, size)
                if observations:
                    shape = tuple(observations[0][key] for key in ("channels", "height", "width"))
                frames[source] = torch.zeros(12, *shape)
                masks[source] = torch.zeros(12, *shape[-2:], dtype=torch.bool)
            seen_months = set()
            for observation in observations:
                if observation["source"] != source:
                    raise ValueError("Annual observation source mismatch")
                if (
                    int(observation["date"][:4]) != year
                    or observation["parent_key"] != record.grid["parent_key"]
                ):
                    raise ValueError("Annual observation year/parent mismatch")
                frame, mask = self._read(observation)
                if schema["role"] == "temporal":
                    month = int(observation["date"][5:7]) - 1
                    if month not in range(12) or month in seen_months:
                        raise ValueError("Duplicate or invalid annual month")
                    seen_months.add(month)
                    frames[source][month] = frame
                    masks[source][month] = mask
                else:
                    highres.append({"values": frame, "mask": mask, "metadata": observation})
        epsg, bounds = parent_geometry(record.grid["parent_key"])
        return {
            "patch_id": record.patch_id,
            "year": year,
            "timestamps": torch.tensor([year * 100 + month for month in range(1, 13)]),
            "source_frames": frames,
            "source_pixel_masks": masks,
            "source_masks": {source: mask.flatten(1).any(dim=1) for source, mask in masks.items()},
            "highres_observations": highres,
            "quality": record.quality,
            "output_grid": {
                "parent_key": record.grid["parent_key"],
                "crs": f"EPSG:{epsg}",
                "transform": [5, 0, bounds[0], 0, -5, bounds[3]],
                "spacing_m": 5,
                "shape": [256, 256],
            },
        }


def collate_annual_observations(samples: list[dict]) -> dict:
    return {
        "source_frames": {
            source: torch.stack([item["source_frames"][source] for item in samples])
            for source in samples[0]["source_frames"]
        },
        "source_pixel_masks": {
            source: torch.stack([item["source_pixel_masks"][source] for item in samples])
            for source in samples[0]["source_pixel_masks"]
        },
        "source_masks": {
            source: torch.stack([item["source_masks"][source] for item in samples])
            for source in samples[0]["source_masks"]
        },
        "timestamps": torch.stack([item["timestamps"] for item in samples]),
        "highres_observations": [item["highres_observations"] for item in samples],
        "metadata": [
            {key: item[key] for key in ("patch_id", "year", "quality", "output_grid")}
            for item in samples
        ],
    }


def check_manifest(task: tuple[Path, int]) -> dict:
    """Exercise real native-grid reads and collation for a deterministic sample."""
    path, count = task
    torch.set_num_threads(2)
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
    dataset = AnnualObservationDataset(path, allow_candidates=True)
    ranked = sorted(
        (hashlib.sha256(record.patch_id.encode()).digest(), index)
        for index, record in enumerate(dataset.records)
    )
    indices = [index for _, index in ranked[:count]]
    highres_count, lowres_months, selected_ids = 0, 0, []
    highres_sources, highres_shapes = Counter(), defaultdict(set)
    for index in indices:
        sample = dataset[index]
        batch = collate_annual_observations([sample])
        if not any(bool(value.any()) for value in batch["source_masks"].values()):
            raise ValueError("Annual sample has no valid lowres observations")
        selected_ids.append(sample["patch_id"])
        highres_count += len(sample["highres_observations"])
        for observation in sample["highres_observations"]:
            source = observation["metadata"]["source"]
            highres_sources[source] += 1
            highres_shapes[source].add(tuple(observation["values"].shape))
        lowres_months += sum(int(mask.sum()) for mask in sample["source_masks"].values())
    return {
        "samples_read": len(indices),
        "valid_lowres_source_months": lowres_months,
        "highres_observations_read": highres_count,
        "highres_observations_by_source": dict(highres_sources),
        "native_highres_shapes": {key: sorted(value) for key, value in highres_shapes.items()},
        "sample_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        "native_grids_preserved": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv data check-annual")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--samples-per-manifest", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 6:
        parser.error("workers must be between 1 and 6")
    if args.samples_per_manifest <= 0:
        parser.error("samples-per-manifest must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    report = {}
    paths = sorted(args.dataset.glob("*.manifest.jsonl"))
    jobs = [(path, args.samples_per_manifest) for path in paths]
    if args.workers == 1:
        results = map(check_manifest, jobs)
        report = {path.name: result for path, result in zip(paths, results)}
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            report = dict(zip((p.name for p in paths), pool.map(check_manifest, jobs)))
    for name, result in report.items():
        print(json.dumps({name: result}), flush=True)
    if not report:
        raise ValueError("No annual manifests found")
    result = {"loader_passed": True, "scientific_quality_validated": False, "manifests": report}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0
