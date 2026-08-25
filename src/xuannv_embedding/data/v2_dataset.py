"""Local-ZIP V2 datasets with independent per-product observation axes."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal
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


def _decode_member(archive: ZipFile, member: str) -> tuple[torch.Tensor, torch.Tensor]:
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
        output_selection: Literal["all", "random_single", "fixed"] = "all",
        fixed_output_months: tuple[tuple[int, int], ...] = (),
        random_seed: int = 42,
        context_days: int | None = None,
        include_targets: bool = True,
    ) -> None:
        if output_selection == "fixed" and not fixed_output_months:
            raise ValueError("fixed output_selection 必须提供 fixed_output_months")
        if output_selection != "fixed" and fixed_output_months:
            raise ValueError("fixed_output_months 只能与 fixed output_selection 一起使用")
        self.config = config
        self.spatial_size = spatial_size
        self.output_selection = output_selection
        self.fixed_output_months = fixed_output_months
        self.random_seed = random_seed
        self.context_days = context_days
        self.include_targets = include_targets
        self._archives: dict[str, ZipFile] = {}
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

    def _output_rows(self, patch_id: str) -> list[dict[str, Any]]:
        candidates = self.observations[(patch_id, self.dense_products[0])]
        if self.output_selection == "all":
            return candidates
        if self.output_selection == "fixed":
            by_month = {(int(row["year"]), int(row["month"])): row for row in candidates}
            missing = [month for month in self.fixed_output_months if month not in by_month]
            if missing:
                raise ValueError(f"{patch_id} 缺少固定输出月份: {missing}")
            return [by_month[month] for month in self.fixed_output_months]
        eligible = []
        for row in candidates:
            key = (int(row["year"]), int(row["month"]))
            product_rows = [
                next(
                    candidate
                    for candidate in self.observations[(patch_id, product_id)]
                    if (int(candidate["year"]), int(candidate["month"])) == key
                )
                for product_id in self.dense_products
            ]
            if any(bool(candidate["present"]) for candidate in product_rows):
                eligible.append(row)
        if not eligible:
            raise ValueError(f"{patch_id} 没有任何可监督输出月份")
        digest = hashlib.sha256(f"{self.random_seed}:{patch_id}".encode("utf-8")).digest()
        return [eligible[int.from_bytes(digest[:8], "big") % len(eligible)]]

    def _context_rows(
        self,
        rows: list[dict[str, Any]],
        output_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any] | None]:
        if self.context_days is None:
            return rows
        selected: list[dict[str, Any]] = []
        for row in rows:
            available = _epoch_days(row["available_at"])
            end = _epoch_days(row["interval_end"])
            if any(
                available <= _epoch_days(output["interval_end"])
                and end > _epoch_days(output["interval_end"]) - self.context_days
                for output in output_rows
            ):
                selected.append(row)
        # Monthly local products need at most 12 slots per 365-day output window.
        slots = 12 * len(output_rows)
        selected = selected[-slots:]
        return [None] * (slots - len(selected)) + selected

    def __len__(self) -> int:
        return len(self.records)

    def close(self) -> None:
        for archive in getattr(self, "_archives", {}).values():
            archive.close()
        if hasattr(self, "_archives"):
            self._archives.clear()

    def __del__(self) -> None:
        self.close()

    def _read_member(self, path: str, member: str) -> tuple[torch.Tensor, torch.Tensor]:
        archive = self._archives.get(path)
        if archive is None:
            archive = ZipFile(path)
            self._archives[path] = archive
        return _decode_member(archive, member)

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
        output_rows = self._output_rows(patch_id)
        reference_intervals = torch.tensor(
            [
                [_epoch_days(row["interval_start"]), _epoch_days(row["interval_end"])]
                for row in output_rows
            ],
            dtype=torch.float32,
        )
        for product_id in self.dense_products:
            rows = self.observations.get((patch_id, product_id), [])
            if not rows:
                raise ValueError(f"availability index 缺少 {patch_id}/{product_id}")
            channels, height, width = _DENSE_SHAPES[product_id]
            output_keys = {(int(row["year"]), int(row["month"])) for row in output_rows}
            target_rows = [
                row for row in rows if (int(row["year"]), int(row["month"])) in output_keys
            ]
            target_by_key = {(int(row["year"]), int(row["month"])): row for row in target_rows}
            target_rows = [
                target_by_key[(int(row["year"]), int(row["month"]))] for row in output_rows
            ]
            rows_with_padding = self._context_rows(rows, output_rows)
            frames = []
            masks = []
            present = []
            bounds = []
            availability = []
            for row in rows_with_padding:
                if row is not None and row["present"]:
                    frame, mask = self._read_member(row["archive_path"], row["member_name"])
                else:
                    frame = torch.zeros(channels, height, width)
                    mask = torch.zeros(1, height, width)
                frames.append(frame)
                masks.append(mask)
                present.append(bool(row is not None and row["present"]))
                if row is None:
                    bounds.append([0.0, 0.0])
                    availability.append(0.0)
                else:
                    bounds.append(
                        [_epoch_days(row["interval_start"]), _epoch_days(row["interval_end"])]
                    )
                    availability.append(_epoch_days(row["available_at"]))
            source_frames[product_id] = torch.stack(frames)
            pixel_masks[product_id] = torch.stack(masks)
            observation_masks[product_id] = torch.tensor(present, dtype=torch.bool)
            time_bounds[product_id] = torch.tensor(bounds, dtype=torch.float32)
            available_at[product_id] = torch.tensor(availability, dtype=torch.float32)
            if self.include_targets:
                target_frames = []
                target_pixel_masks = []
                for row in target_rows:
                    if row["present"]:
                        frame, mask = self._read_member(row["archive_path"], row["member_name"])
                    else:
                        frame = torch.zeros(channels, height, width)
                        mask = torch.zeros(1, height, width)
                    target_frames.append(frame)
                    target_pixel_masks.append(mask)
                target = F.interpolate(
                    torch.stack(target_frames),
                    size=(self.spatial_size, self.spatial_size),
                    mode="area",
                )
                target_mask = F.interpolate(
                    torch.stack(target_pixel_masks),
                    size=(self.spatial_size, self.spatial_size),
                    mode="area",
                )
                targets[product_id] = target
                target_masks[product_id] = target_mask
        bounds = [float(value) for value in record["utm_bounds"]]
        pixel_size = (bounds[2] - bounds[0]) / self.spatial_size
        output_transform = torch.tensor(
            [pixel_size, 0.0, bounds[0], 0.0, -pixel_size, bounds[3]],
            dtype=torch.float32,
        )
        return {
            "patch_id": patch_id,
            "macro_id": str(record["macro_id"]),
            "split": str(record["split"]),
            "grid_epsg": int(record["grid_epsg"]),
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
        "macro_ids": [sample["macro_id"] for sample in samples],
        "splits": [sample["split"] for sample in samples],
        "grid_epsgs": torch.tensor([sample["grid_epsg"] for sample in samples]),
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
