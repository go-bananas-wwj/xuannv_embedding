"""Local-ZIP V2 datasets with independent per-product observation axes."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from rasterio.io import MemoryFile
from torch.utils.data import Dataset

from xuannv_embedding.config import V2Config

_DENSE_SHAPES = {
    "s2_local": (10, 128, 128),
    "s1_local": (2, 128, 128),
    "landsat_local": (6, 43, 43),
}


def _epoch_days(value) -> float:
    return float(value.timestamp() / 86400.0)


def _read_member(path: str, member: str) -> tuple[torch.Tensor, torch.Tensor]:
    with ZipFile(path) as archive:
        payload = archive.read(member)
    with MemoryFile(payload) as memory:
        with memory.open() as dataset:
            values = dataset.read()
    finite = np.isfinite(values)
    tensor = torch.from_numpy(np.where(finite, values, 0).astype(np.float32, copy=False))
    mask = torch.from_numpy(finite.all(axis=0, keepdims=True).astype(np.float32))
    return tensor, mask


class V2LocalZipDataset(Dataset):
    """Read frozen observations directly from ZIP; missing observations stay explicit."""

    def __init__(
        self,
        config: V2Config,
        registry_path: Path,
        *,
        spatial_size: int,
        max_records: int | None = None,
        january_pair_only: bool = False,
    ) -> None:
        self.config = config
        self.spatial_size = spatial_size
        registry = pq.read_table(registry_path)
        if max_records is not None:
            registry = registry.slice(0, max_records)
        self.records = registry.to_pylist()
        patch_ids = [str(row["patch_id"]) for row in self.records]
        filters: list[tuple[str, str, Any]] = [("patch_id", "in", patch_ids)]
        if january_pair_only:
            filters.extend([("month", "=", 1), ("year", "in", [2020, 2021])])
        observations = pq.read_table(
            config.paths.data_root / "observations" / "index" / "availability.parquet",
            filters=filters,
        )
        lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in observations.to_pylist():
            lookup[(str(row["patch_id"]), str(row["product_id"]))].append(row)
        self.observations = {
            key: sorted(value, key=lambda row: (row["year"], row["month"]))
            for key, value in lookup.items()
        }
        self.dense_products = tuple(
            product_id for product_id, product in config.products.items() if product.role == "dense"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        patch_id = str(record["patch_id"])
        source_frames: dict[str, torch.Tensor] = {}
        pixel_masks: dict[str, torch.Tensor] = {}
        observation_masks: dict[str, torch.Tensor] = {}
        time_bounds: dict[str, torch.Tensor] = {}
        available_at: dict[str, torch.Tensor] = {}
        targets: dict[str, torch.Tensor] = {}
        target_masks: dict[str, torch.Tensor] = {}
        reference_intervals: torch.Tensor | None = None
        for product_id in self.dense_products:
            rows = self.observations.get((patch_id, product_id), [])
            if not rows:
                raise ValueError(f"availability index 缺少 {patch_id}/{product_id}")
            channels, height, width = _DENSE_SHAPES[product_id]
            frames = []
            masks = []
            present = []
            bounds = []
            availability = []
            for row in rows:
                if row["present"]:
                    frame, mask = _read_member(row["archive_path"], row["member_name"])
                else:
                    frame = torch.zeros(channels, height, width)
                    mask = torch.zeros(1, height, width)
                frames.append(frame)
                masks.append(mask)
                present.append(bool(row["present"]))
                bounds.append(
                    [_epoch_days(row["interval_start"]), _epoch_days(row["interval_end"])]
                )
                availability.append(_epoch_days(row["available_at"]))
            source_frames[product_id] = torch.stack(frames)
            pixel_masks[product_id] = torch.stack(masks)
            observation_masks[product_id] = torch.tensor(present, dtype=torch.bool)
            time_bounds[product_id] = torch.tensor(bounds, dtype=torch.float32)
            available_at[product_id] = torch.tensor(availability, dtype=torch.float32)
            target = F.interpolate(
                source_frames[product_id],
                size=(self.spatial_size, self.spatial_size),
                mode="area",
            )
            target_mask = F.interpolate(
                pixel_masks[product_id],
                size=(self.spatial_size, self.spatial_size),
                mode="area",
            )
            targets[product_id] = target
            target_masks[product_id] = target_mask
            if reference_intervals is None:
                reference_intervals = time_bounds[product_id]
        assert reference_intervals is not None
        bounds = [float(value) for value in record["utm_bounds"]]
        pixel_size = (bounds[2] - bounds[0]) / self.spatial_size
        output_transform = torch.tensor(
            [pixel_size, 0.0, bounds[0], 0.0, -pixel_size, bounds[3]],
            dtype=torch.float32,
        )
        return {
            "patch_id": patch_id,
            "model_inputs": {
                "source_frames": source_frames,
                "source_pixel_masks": pixel_masks,
                "source_observation_masks": observation_masks,
                "source_time_bounds": time_bounds,
                "source_available_at": available_at,
                "output_intervals": reference_intervals,
                "output_geotransforms": output_transform,
                "output_size": (self.spatial_size, self.spatial_size),
            },
            "targets": targets,
            "target_masks": target_masks,
        }


def collate_v2(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("collate_v2 samples 为空")
    first_inputs = samples[0]["model_inputs"]
    dense_fields = (
        "source_frames",
        "source_pixel_masks",
        "source_observation_masks",
        "source_time_bounds",
        "source_available_at",
    )
    model_inputs: dict[str, Any] = {}
    for field in dense_fields:
        model_inputs[field] = {
            product_id: torch.stack(
                [sample["model_inputs"][field][product_id] for sample in samples]
            )
            for product_id in first_inputs[field]
        }
    model_inputs["output_intervals"] = torch.stack(
        [sample["model_inputs"]["output_intervals"] for sample in samples]
    )
    model_inputs["output_geotransforms"] = torch.stack(
        [sample["model_inputs"]["output_geotransforms"] for sample in samples]
    )
    model_inputs["output_size"] = first_inputs["output_size"]
    return {
        "patch_ids": [sample["patch_id"] for sample in samples],
        "model_inputs": model_inputs,
        "targets": {
            product_id: torch.stack([sample["targets"][product_id] for sample in samples])
            for product_id in samples[0]["targets"]
        },
        "target_masks": {
            product_id: torch.stack([sample["target_masks"][product_id] for sample in samples])
            for product_id in samples[0]["target_masks"]
        },
    }
