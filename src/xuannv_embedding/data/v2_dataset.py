"""Local-ZIP V2 datasets with independent per-product observation axes."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq
import rasterio
import torch
import torch.nn.functional as F
import zarr
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds
from torch.utils.data import Dataset

from xuannv_embedding.config import V2Config

_DENSE_SHAPES = {
    "s2_local": (10, 128, 128),
    "s1_local": (2, 128, 128),
    "landsat_local": (6, 43, 43),
}


def _load_statistics(
    config: V2Config, product_id: str, *, allow_incomplete: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    path = config.paths.data_root / "statistics" / f"{product_id}.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 V2 波段统计量 {path}: {exc}") from exc
    product = config.products[product_id]
    expected = {
        "schema_version": "xuannv_v2_band_statistics_v1",
        "product_id": product_id,
        "bands": list(product.bands),
        "split": "train",
        "representation": "stored_dn",
        "scaling_applied": False,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"V2 波段统计量合同不匹配: {path}:{key}")
    if not allow_incomplete and document.get("complete_training_split") is not True:
        raise ValueError(f"V2 生产训练要求完整训练划分统计量: {path}")
    mean = torch.tensor(document.get("mean", []), dtype=torch.float32)
    std = torch.tensor(document.get("std", []), dtype=torch.float32)
    if mean.numel() != len(product.bands) or std.numel() != len(product.bands):
        raise ValueError(f"V2 波段统计量通道数不匹配: {path}")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise ValueError(f"V2 波段统计量包含 NaN/Inf: {path}")
    if bool((std <= 0).any()):
        raise ValueError(f"V2 波段统计量 std 必须为正: {path}")
    return mean[:, None, None], std[:, None, None]


def _epoch_days(value) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return float(value.timestamp() / 86400.0)


def _stored_pixel_validity(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values).all(axis=0)
    all_zero = (values == 0).all(axis=0)
    explicit_fill = (values == -32768).any(axis=0)
    return finite & ~all_zero & ~explicit_fill


def _decode_member(archive: ZipFile, member: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = archive.read(member)
    with MemoryFile(payload) as memory:
        with memory.open() as dataset:
            values = dataset.read()
    valid = _stored_pixel_validity(values)
    tensor = torch.from_numpy(np.where(valid[None], values, 0).astype(np.float32, copy=False))
    mask = torch.from_numpy(valid[None].astype(np.float32))
    return tensor, mask


def _parse_time_days(value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() / 86400.0


def _select_highres_candidates(
    rows: list[dict[str, Any]],
    output_intervals: torch.Tensor,
    *,
    mode: Literal["within_period", "causal_window", "centered_window"],
    structure_days: int,
    appearance_days: int,
    structure_max: int,
    appearance_max: int,
) -> list[dict[str, Any]]:
    """Return the union of per-interval memory candidates before pixel decoding."""
    selected_ids: set[str] = set()
    parsed = [
        (
            row,
            _parse_time_days(str(row["acquired_at"])),
            _parse_time_days(str(row["available_at"])),
        )
        for row in rows
    ]
    for interval in output_intervals:
        start, end = (float(interval[0]), float(interval[1]))
        center = (start + end) * 0.5
        for window_days, maximum in (
            (structure_days, structure_max),
            (appearance_days, appearance_max),
        ):
            eligible: list[tuple[dict[str, Any], float]] = []
            for row, acquired, available in parsed:
                if mode == "causal_window":
                    keep = available <= end and acquired < end and acquired > end - window_days
                elif mode == "within_period":
                    keep = acquired < end and acquired > start
                elif mode == "centered_window":
                    keep = abs(acquired - center) <= window_days
                else:
                    raise ValueError(f"未知 temporal mode: {mode}")
                if keep:
                    eligible.append((row, acquired))
            ranked = sorted(
                eligible,
                key=lambda item: (
                    -float(item[0].get("intersection_fraction") or 0.0)
                    * float(item[0].get("clear_percent") or 0.0),
                    abs(item[1] - center),
                    item[1],
                    str(item[0]["scene_id"]),
                ),
            )
            selected_ids.update(str(row["scene_id"]) for row, _ in ranked[:maximum])
    return sorted(
        [row for row in rows if str(row["scene_id"]) in selected_ids],
        key=lambda row: (row["acquired_at"], row["scene_id"]),
    )


def _read_highres_patch(
    row: dict[str, Any],
    *,
    bands: int,
    stored_gsd_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read a native-grid scene window; pad only, never resize image pixels."""
    bounds = tuple(float(value) for value in row["patch_bounds"])
    target_size = int(math.ceil((bounds[2] - bounds[0]) / stored_gsd_m)) + 2
    with rasterio.open(row["image_path"]) as dataset:
        window = from_bounds(*bounds, transform=dataset.transform).round_offsets().round_lengths()
        values = dataset.read(window=window, boundless=True, fill_value=0)
        transform = dataset.window_transform(window)
    if values.shape[0] != bands:
        raise ValueError(f"高分场景通道错误: {row['image_path']}")
    valid = _stored_pixel_validity(values)
    if row.get("qa_present") and row.get("qa_path"):
        with rasterio.open(row["qa_path"]) as qa:
            qa_window = from_bounds(*bounds, transform=qa.transform)
            quality = qa.read(
                window=qa_window,
                boundless=True,
                fill_value=0,
                out_shape=(qa.count, values.shape[1], values.shape[2]),
                resampling=rasterio.enums.Resampling.nearest,
            )
            descriptions = tuple(item or "" for item in qa.descriptions)
        try:
            clear = quality[descriptions.index("clear")] > 0
        except ValueError:
            clear = np.ones_like(valid)
        try:
            unusable = quality[descriptions.index("udm1")] > 0
        except ValueError:
            unusable = np.zeros_like(valid)
        valid &= clear & ~unusable
    height = min(values.shape[1], target_size)
    width = min(values.shape[2], target_size)
    padded = np.zeros((bands, target_size, target_size), dtype=np.float32)
    padded_mask = np.zeros((1, target_size, target_size), dtype=np.float32)
    padded[:, :height, :width] = values[:, :height, :width]
    padded_mask[:, :height, :width] = valid[:height, :width]
    padded *= padded_mask
    return (
        torch.from_numpy(padded),
        torch.from_numpy(padded_mask),
        torch.tensor(list(transform)[:6], dtype=torch.float32),
    )


def _detail_statistics(
    frame: torch.Tensor, mask: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    gray = frame.mean(dim=0, keepdim=True)
    local_mean = F.avg_pool2d(gray[None], 3, stride=1, padding=1)[0]
    local_square = F.avg_pool2d((gray * gray)[None], 3, stride=1, padding=1)[0]
    variance = (local_square - local_mean.square()).clamp_min(0)
    gradient = torch.zeros_like(gray)
    gradient[:, :, 1:] += (gray[:, :, 1:] - gray[:, :, :-1]).abs()
    gradient[:, 1:, :] += (gray[:, 1:, :] - gray[:, :-1, :]).abs()
    stats = torch.cat((gray, variance, gradient), dim=0)
    pooled = F.adaptive_avg_pool2d((stats * mask)[None], (size, size))[0]
    pooled_mask = F.adaptive_avg_pool2d(mask[None], (size, size))[0]
    return pooled, pooled_mask


def _read_supervised_label(
    rows: list[dict[str, Any]],
    *,
    epsg: int,
    output_transform: torch.Tensor,
    output_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    merged = np.zeros((output_size, output_size), dtype=np.float32)
    valid = np.zeros((output_size, output_size), dtype=np.uint8)
    destination_transform = rasterio.Affine(*output_transform.tolist())
    for row in rows:
        with rasterio.open(row["label_path"]) as source:
            values = source.read(1)
            projected = np.zeros_like(merged)
            coverage = np.zeros_like(valid)
            reproject(
                values,
                projected,
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=destination_transform,
                dst_crs=f"EPSG:{epsg}",
                resampling=Resampling.nearest,
                dst_nodata=0,
            )
            reproject(
                np.ones(values.shape, dtype=np.uint8),
                coverage,
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=destination_transform,
                dst_crs=f"EPSG:{epsg}",
                resampling=Resampling.nearest,
                dst_nodata=0,
            )
        merged = np.maximum(merged, projected)
        valid = np.maximum(valid, coverage)
    return torch.from_numpy(merged), torch.from_numpy(valid.astype(np.float32))


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
        normalize: bool = True,
        zarr_cache_path: Path | None = None,
        allow_incomplete_statistics: bool = False,
        output_intervals_override: tuple[tuple[float, float], ...] = (),
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
        self.output_intervals_override = output_intervals_override
        self._archives: dict[str, ZipFile] = {}
        registry = pq.read_table(registry_path)
        if max_records is not None:
            registry = registry.slice(0, max_records)
        self.records = registry.to_pylist()
        patch_ids = [str(row["patch_id"]) for row in self.records]
        lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._zarr = None
        self._zarr_patch_index: dict[str, int] = {}
        if zarr_cache_path is None:
            filters: list[tuple[str, str, Any]] = [("patch_id", "in", patch_ids)]
            if january_pair_only:
                filters.extend([("month", "=", 1), ("year", "in", [2020, 2021])])
            observations = pq.read_table(
                config.paths.data_root / "observations" / "index" / "availability.parquet",
                filters=filters,
            )
            for row in observations.to_pylist():
                lookup[(str(row["patch_id"]), str(row["product_id"]))].append(row)
        else:
            self._zarr = zarr.open_group(str(zarr_cache_path), mode="r")
            if self._zarr.attrs.get("schema_version") != "xuannv_v2_smoke_dense_cache_v2":
                raise ValueError(f"Zarr cache schema 非法: {zarr_cache_path}")
            if self._zarr.attrs.get("network_remote_pixels") is not False:
                raise ValueError(f"Zarr cache 未证明 remote pixels 禁用: {zarr_cache_path}")
            lock_path = config.paths.data_root / "locks" / "local_archive_sha256.jsonl"
            lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
            if self._zarr.attrs.get("source_archive_lock_sha256") != lock_sha:
                raise ValueError(f"Zarr cache 与当前 archive lock 不一致: {zarr_cache_path}")
            current_registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
            if self._zarr.attrs.get("registry_sha256") != current_registry_sha:
                raise ValueError(f"Zarr cache 与当前 registry 不一致: {zarr_cache_path}")
            availability_path = (
                config.paths.data_root / "observations" / "index" / "availability.parquet"
            )
            current_availability_sha = hashlib.sha256(availability_path.read_bytes()).hexdigest()
            if self._zarr.attrs.get("availability_sha256") != current_availability_sha:
                raise ValueError(f"Zarr cache 与当前 availability index 不一致: {zarr_cache_path}")
            manifest_path = zarr_cache_path / "cache_manifest.json"
            if (
                not manifest_path.is_file()
                or self._zarr.attrs.get("cache_manifest_sha256")
                != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            ):
                raise ValueError(f"Zarr cache manifest 缺失或摘要不匹配: {zarr_cache_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("zip_zarr_audit_passed") is not True:
                raise ValueError(f"Zarr cache 未通过 ZIP↔Zarr audit: {zarr_cache_path}")
            from xuannv_embedding.data_process.v2_zarr_cache import verify_smoke_zarr_cache

            verify_smoke_zarr_cache(zarr_cache_path, full=False)
            cached_patch_ids = [str(value) for value in self._zarr.attrs["patch_ids"]]
            self._zarr_patch_index = {
                patch_id: index for index, patch_id in enumerate(cached_patch_ids)
            }
            missing_patches = sorted(set(patch_ids) - set(cached_patch_ids))
            if missing_patches:
                raise ValueError(f"Zarr cache 缺少 registry patch: {missing_patches[:5]}")
            years = self._zarr.attrs["years"]
            months = self._zarr.attrs["months"]
            month_indices = {
                (int(year), int(month)): index
                for index, (year, month) in enumerate(zip(years, months, strict=True))
            }
            filters = [("patch_id", "in", patch_ids)]
            if january_pair_only:
                filters.extend([("month", "=", 1), ("year", "in", [2020, 2021])])
            indexed = pq.read_table(availability_path, filters=filters)
            for row in indexed.to_pylist():
                patch_id = str(row["patch_id"])
                product_id = str(row["product_id"])
                month_index = month_indices[(int(row["year"]), int(row["month"]))]
                cached_present = bool(
                    self._zarr[product_id]["present"][self._zarr_patch_index[patch_id], month_index]
                )
                if cached_present != bool(row["present"]):
                    raise ValueError(
                        f"Zarr cache present 与 availability 不一致: {patch_id}/{product_id}"
                    )
                row["cache_month_index"] = month_index
                lookup[(patch_id, product_id)].append(row)
        self.observations = {
            key: sorted(value, key=lambda row: (row["year"], row["month"]))
            for key, value in lookup.items()
        }
        self.dense_products = tuple(
            product_id for product_id, product in config.products.items() if product.role == "dense"
        )
        self.highres_products = tuple(
            product_id
            for product_id, product in config.products.items()
            if product.role == "highres"
        )
        highres_lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for product_id in self.highres_products:
            path = (
                config.paths.data_root
                / "observations"
                / "highres"
                / product_id
                / "patch_observations.parquet"
            )
            if not path.is_file():
                continue
            for row in pq.read_table(path, filters=[("patch_id", "in", patch_ids)]).to_pylist():
                highres_lookup[(str(row["patch_id"]), product_id)].append(row)
        self.highres_observations = {
            key: sorted(value, key=lambda row: (row["acquired_at"], row["scene_id"]))
            for key, value in highres_lookup.items()
        }
        self.highres_statistics = (
            {
                product_id: _load_statistics(
                    config, product_id, allow_incomplete=allow_incomplete_statistics
                )
                for product_id in self.highres_products
                if any(key[1] == product_id for key in self.highres_observations)
            }
            if normalize
            else {}
        )
        label_lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for task in config.training.semantic_probe_tasks:
            path = config.paths.data_root / "labels" / task / "patch_observations.parquet"
            if not path.is_file():
                if config.training.semantic_probe_weight > 0:
                    raise ValueError(f"semantic probe 缺少真实 label index: {path}")
                continue
            for row in pq.read_table(path, filters=[("patch_id", "in", patch_ids)]).to_pylist():
                label_lookup[(str(row["patch_id"]), task)].append(row)
        self.supervised_label_observations = dict(label_lookup)
        self.statistics = (
            {
                product_id: _load_statistics(
                    config, product_id, allow_incomplete=allow_incomplete_statistics
                )
                for product_id in self.dense_products
            }
            if normalize
            else {}
        )

    def _normalize(self, product_id: str, frame: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if product_id not in self.statistics:
            return frame
        mean, std = self.statistics[product_id]
        return ((frame - mean) / std) * mask

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
        # All-missing sentinel patches still belong to validation. Selecting a
        # deterministic interval yields zero source/target masks and a legal
        # zero-gradient backward pass instead of silently dropping the record.
        if not eligible:
            eligible = candidates
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

    def _read_row(
        self, patch_id: str, product_id: str, row: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._zarr is None:
            return self._read_member(row["archive_path"], row["member_name"])
        patch_index = self._zarr_patch_index[patch_id]
        month_index = int(row["cache_month_index"])
        group = self._zarr[product_id]
        frame = torch.from_numpy(group["frames"][patch_index, month_index].astype(np.float32))
        mask = torch.from_numpy(group["masks"][patch_index, month_index].astype(np.float32))
        return frame, mask

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
        observation_candidates: dict[str, list[dict[str, object] | None]] = {}
        output_rows = self._output_rows(patch_id)
        reference_intervals = torch.tensor(
            [
                [_epoch_days(row["interval_start"]), _epoch_days(row["interval_end"])]
                for row in output_rows
            ],
            dtype=torch.float32,
        )
        if self.output_intervals_override:
            reference_intervals = torch.tensor(self.output_intervals_override, dtype=torch.float32)
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
            observation_candidates[product_id] = [
                (
                    None
                    if row is None
                    else {
                        "archive_path": str(row.get("archive_path", "")),
                        "member_name": str(row.get("member_name", "")),
                        "year": int(row["year"]),
                        "month": int(row["month"]),
                        "present": bool(row["present"]),
                    }
                )
                for row in rows_with_padding
            ]
            frames = []
            masks = []
            present = []
            bounds = []
            availability = []
            for row in rows_with_padding:
                if row is not None and row["present"]:
                    frame, mask = self._read_row(patch_id, product_id, row)
                else:
                    frame = torch.zeros(channels, height, width)
                    mask = torch.zeros(1, height, width)
                frame = self._normalize(product_id, frame, mask)
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
                        frame, mask = self._read_row(patch_id, product_id, row)
                    else:
                        frame = torch.zeros(channels, height, width)
                        mask = torch.zeros(1, height, width)
                    frame = self._normalize(product_id, frame, mask)
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
        highres_frames: dict[str, torch.Tensor] = {}
        highres_masks: dict[str, torch.Tensor] = {}
        highres_acquired_at: dict[str, torch.Tensor] = {}
        highres_available_at: dict[str, torch.Tensor] = {}
        highres_geotransforms: dict[str, torch.Tensor] = {}
        detail_targets: dict[str, torch.Tensor] = {}
        detail_masks: dict[str, torch.Tensor] = {}
        max_scenes = self.config.temporal.highres_structure_max_observations
        for product_id in self.highres_products:
            product = self.config.products[product_id]
            rows = list(self.highres_observations.get((patch_id, product_id), []))
            rows = _select_highres_candidates(
                rows,
                reference_intervals,
                mode=self.config.temporal.mode,
                structure_days=self.config.temporal.highres_structure_days,
                appearance_days=self.config.temporal.highres_appearance_days,
                structure_max=max_scenes,
                appearance_max=self.config.temporal.highres_appearance_max_observations,
            )
            native_size = int(math.ceil((bounds[2] - bounds[0]) / product.stored_gsd_m)) + 2
            frames: list[torch.Tensor] = []
            masks: list[torch.Tensor] = []
            transforms: list[torch.Tensor] = []
            acquired: list[float] = []
            available: list[float] = []
            for row in rows:
                frame, mask, transform = _read_highres_patch(
                    row, bands=len(product.bands), stored_gsd_m=product.stored_gsd_m
                )
                if product_id in self.highres_statistics:
                    mean, std = self.highres_statistics[product_id]
                    frame = ((frame - mean) / std) * mask
                frames.append(frame)
                masks.append(mask)
                transforms.append(transform)
                acquired.append(_parse_time_days(row["acquired_at"]))
                available.append(_parse_time_days(row["available_at"]))
            while not frames:
                frames.append(torch.zeros(len(product.bands), native_size, native_size))
                masks.append(torch.zeros(1, native_size, native_size))
                transforms.append(
                    torch.tensor(
                        [
                            product.stored_gsd_m,
                            0.0,
                            bounds[0],
                            0.0,
                            -product.stored_gsd_m,
                            bounds[3],
                        ],
                        dtype=torch.float32,
                    )
                )
                acquired.append(0.0)
                available.append(0.0)
            highres_frames[product_id] = torch.stack(frames)
            highres_masks[product_id] = torch.stack(masks)
            highres_geotransforms[product_id] = torch.stack(transforms)
            highres_acquired_at[product_id] = torch.tensor(acquired, dtype=torch.float32)
            highres_available_at[product_id] = torch.tensor(available, dtype=torch.float32)
            product_targets = []
            product_masks = []
            for interval in reference_intervals:
                interval_start = float(interval[0])
                interval_end = float(interval[1])
                interval_center = (interval_start + interval_end) * 0.5
                eligible = [
                    scene_index
                    for scene_index in range(len(rows))
                    if (
                        self.config.temporal.mode == "causal_window"
                        and available[scene_index] <= interval_end
                        and acquired[scene_index] < interval_end
                        and acquired[scene_index]
                        > interval_end - self.config.temporal.highres_structure_days
                    )
                    or (
                        self.config.temporal.mode == "within_period"
                        and interval_start < acquired[scene_index] < interval_end
                    )
                    or (
                        self.config.temporal.mode == "centered_window"
                        and abs(acquired[scene_index] - interval_center)
                        <= self.config.temporal.highres_structure_days
                    )
                ]
                if eligible:
                    selected = max(
                        eligible,
                        key=lambda item: (float(masks[item].sum()), acquired[item], -item),
                    )
                    detail, detail_mask = _detail_statistics(
                        frames[selected], masks[selected], self.spatial_size
                    )
                else:
                    detail = torch.zeros(3, self.spatial_size, self.spatial_size)
                    detail_mask = torch.zeros(1, self.spatial_size, self.spatial_size)
                product_targets.append(detail)
                product_masks.append(detail_mask)
            detail_targets[product_id] = torch.stack(product_targets)
            detail_masks[product_id] = torch.stack(product_masks)
            observation_candidates[product_id] = [
                {
                    "scene_id": str(row["scene_id"]),
                    "acquired_at": str(row["acquired_at"]),
                    "available_at": str(row["available_at"]),
                    "image_path": str(row["image_path"]),
                    "qa_path": str(row.get("qa_path") or ""),
                }
                for row in rows
            ]
            while len(observation_candidates[product_id]) < len(frames):
                observation_candidates[product_id].append(None)
        supervised_labels: dict[str, torch.Tensor] = {}
        supervised_label_masks: dict[str, torch.Tensor] = {}
        for task in self.config.training.semantic_probe_tasks:
            label_rows = self.supervised_label_observations.get((patch_id, task), [])
            if label_rows:
                label, label_mask = _read_supervised_label(
                    label_rows,
                    epsg=int(record["grid_epsg"]),
                    output_transform=output_transform,
                    output_size=self.spatial_size,
                )
            else:
                label = torch.zeros(self.spatial_size, self.spatial_size)
                label_mask = torch.zeros(self.spatial_size, self.spatial_size)
            supervised_labels[task] = label[None].repeat(len(reference_intervals), 1, 1)
            supervised_label_masks[task] = label_mask[None].repeat(len(reference_intervals), 1, 1)
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
                "highres_frames": highres_frames,
                "highres_masks": highres_masks,
                "highres_acquired_at": highres_acquired_at,
                "highres_available_at": highres_available_at,
                "highres_geotransforms": highres_geotransforms,
            },
            "targets": targets,
            "target_masks": target_masks,
            "observation_candidates": observation_candidates,
            "detail_targets": detail_targets,
            "detail_masks": detail_masks,
            "supervised_labels": supervised_labels,
            "supervised_label_masks": supervised_label_masks,
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
    highres_fields = (
        "highres_frames",
        "highres_masks",
        "highres_acquired_at",
        "highres_available_at",
        "highres_geotransforms",
    )
    for field in highres_fields:
        model_inputs[field] = {}
        for product_id in first_inputs[field]:
            tensors = [sample["model_inputs"][field][product_id] for sample in samples]
            maximum = max(tensor.shape[0] for tensor in tensors)
            padded = []
            for tensor in tensors:
                if tensor.shape[0] < maximum:
                    shape = (maximum - tensor.shape[0], *tensor.shape[1:])
                    padding = torch.zeros(shape, dtype=tensor.dtype)
                    if field == "highres_geotransforms":
                        padding[:] = tensor[-1]
                    tensor = torch.cat((tensor, padding), dim=0)
                padded.append(tensor)
            model_inputs[field][product_id] = torch.stack(padded)
    return {
        "patch_ids": [sample["patch_id"] for sample in samples],
        "macro_ids": [sample["macro_id"] for sample in samples],
        "splits": [sample["split"] for sample in samples],
        "grid_epsgs": torch.tensor([sample["grid_epsg"] for sample in samples]),
        "observation_candidates": [
            {
                product_id: candidates
                + [None]
                * (
                    model_inputs[
                        (
                            "highres_frames"
                            if product_id in first_inputs["highres_frames"]
                            else "source_frames"
                        )
                    ][product_id].shape[1]
                    - len(candidates)
                )
                for product_id, candidates in sample["observation_candidates"].items()
            }
            for sample in samples
        ],
        "model_inputs": model_inputs,
        "targets": {
            product_id: torch.stack([sample["targets"][product_id] for sample in samples])
            for product_id in samples[0]["targets"]
        },
        "target_masks": {
            product_id: torch.stack([sample["target_masks"][product_id] for sample in samples])
            for product_id in samples[0]["target_masks"]
        },
        "detail_targets": {
            product_id: torch.stack([sample["detail_targets"][product_id] for sample in samples])
            for product_id in samples[0]["detail_targets"]
        },
        "detail_masks": {
            product_id: torch.stack([sample["detail_masks"][product_id] for sample in samples])
            for product_id in samples[0]["detail_masks"]
        },
        "supervised_labels": {
            task: torch.stack([sample["supervised_labels"][task] for sample in samples])
            for task in samples[0]["supervised_labels"]
        },
        "supervised_label_masks": {
            task: torch.stack([sample["supervised_label_masks"][task] for sample in samples])
            for task in samples[0]["supervised_label_masks"]
        },
    }
