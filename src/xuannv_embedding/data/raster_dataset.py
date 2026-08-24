"""按区域合同读取真实多源 GeoTIFF，并生成统一 P10C batch。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from xuannv_embedding.config import Config, RegionDatasetConfig
from xuannv_embedding.data.contracts import load_region_records
from xuannv_embedding.utils.manifest import ManifestRecord, SourceValue, manifest_meta_path

_DATE_PATTERN = re.compile(r"(?<!\d)(\d{8})(?!\d)")


@dataclass(frozen=True)
class _Observation:
    month: int
    values: torch.Tensor
    mask: torch.Tensor


def _paths(value: SourceValue) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _month(path: str) -> int:
    match = _DATE_PATTERN.search(Path(path).name)
    if match is None:
        return 0
    return int(match.group(1)[:6])


def _resize(values: torch.Tensor, size: tuple[int, int], *, nearest: bool) -> torch.Tensor:
    if values.shape[-2:] == size:
        return values
    mode = "nearest" if nearest else "bilinear"
    result = F.interpolate(
        values.unsqueeze(0), size=size, mode=mode, align_corners=False if not nearest else None
    )
    return result[0]


class RegionRasterDataset(Dataset[dict[str, Any]]):
    """一个区域一个 Dataset；多区域训练由相同运行时轮转这些 Dataset。"""

    def __init__(
        self,
        config: Config,
        dataset_config: RegionDatasetConfig,
        *,
        max_records: int | None = None,
    ) -> None:
        self.config = config
        self.dataset_config = dataset_config
        self.records = load_region_records(dataset_config, expected_months=config.data.months)
        if max_records is not None:
            if max_records <= 0:
                raise ValueError("max_records 必须大于 0")
            self.records = self.records[:max_records]
        if not self.records:
            raise ValueError(f"区域 {dataset_config.region!r} 没有 manifest 记录")
        self.months = [int(value.replace("-", "")) for value in config.data.months]
        self.output_size = (config.data.patch_size, config.data.patch_size)
        self.source_root = (
            config.paths.data_root
            if manifest_meta_path(dataset_config.manifest_path).is_file()
            else dataset_config.manifest_path.parent.parent
        )
        self.physical_by_canonical = {
            canonical: physical for physical, canonical in dataset_config.source_map.items()
        }
        self.statistics = self._load_statistics()
        self.highres_sizes = self._infer_highres_sizes()

    def _load_statistics(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for source, source_config in self.config.model.input_sources.items():
            path = self.dataset_config.statistics_dir / f"{source}_stats.json"
            if not path.is_file():
                available = any(
                    _paths(self._record_value(record, source)) for record in self.records
                )
                if available:
                    raise ValueError(
                        f"区域 {self.dataset_config.region!r} 的可用 source {source!r} "
                        f"缺少归一化统计量: {path}"
                    )
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"无法读取统计量: {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"统计量必须是 JSON object: {path}")
            mean = torch.tensor(value.get("mean", []), dtype=torch.float32)
            std = torch.tensor(value.get("std", []), dtype=torch.float32)
            if len(mean) != source_config.channels or len(std) != source_config.channels:
                raise ValueError(f"统计量通道冲突: {path}")
            if not bool(torch.isfinite(mean).all().item()) or not bool(
                torch.isfinite(std).all().item()
            ):
                raise ValueError(f"统计量 mean/std 必须全部有限: {path}")
            if bool((std <= 0).any().item()):
                raise ValueError(f"统计量 std 必须为正: {path}")
            result[source] = (mean[:, None, None], std[:, None, None])
        return result

    def _record_value(self, record: ManifestRecord, canonical_source: str) -> SourceValue:
        physical = self.physical_by_canonical.get(canonical_source)
        return record.sources.get(physical) if physical is not None else None

    def _absolute(self, relative: str) -> Path:
        path = self.source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest source 不存在: {path}")
        return path

    def _read(
        self,
        relative: str,
        *,
        source: str,
        channels: int,
        categorical: bool,
    ) -> _Observation:
        path = self._absolute(relative)
        with rasterio.open(path) as raster:
            values = torch.from_numpy(raster.read(out_dtype="float32"))
            nodata = raster.nodata
        expected_bands = 1 if categorical else channels
        if values.shape[0] != expected_bands:
            raise ValueError(
                f"source 通道冲突: {path}={values.shape[0]}, expected={expected_bands}"
            )
        finite = torch.isfinite(values).all(dim=0)
        if nodata is not None and np.isfinite(nodata):
            finite &= (values != float(nodata)).any(dim=0)
        companion = path.with_name(f"{path.stem}_mask.tif")
        if companion.is_file():
            with rasterio.open(companion) as raster:
                explicit_mask = torch.from_numpy(raster.read(1, out_dtype="float32")) > 0
            if explicit_mask.shape != finite.shape:
                raise ValueError(f"像素 mask 尺寸冲突: {companion}")
            finite &= explicit_mask
        values = torch.nan_to_num(values)
        if not categorical and source in self.statistics:
            mean, std = self.statistics[source]
            values = (values - mean) / std
        values = values * finite[None]
        return _Observation(_month(relative), values, finite.float())

    def _observations(
        self, record: ManifestRecord, source: str, channels: int, *, categorical: bool = False
    ) -> list[_Observation]:
        return [
            self._read(path, source=source, channels=channels, categorical=categorical)
            for path in _paths(self._record_value(record, source))
        ]

    def _infer_highres_sizes(self) -> dict[str, tuple[int, int]]:
        sizes: dict[str, tuple[int, int]] = {}
        for source, source_config in self.config.model.input_sources.items():
            if source_config.role != "highres":
                continue
            for record in self.records:
                paths = _paths(self._record_value(record, source))
                if not paths:
                    continue
                with rasterio.open(self._absolute(paths[0])) as raster:
                    sizes[source] = (raster.height, raster.width)
                break
            sizes.setdefault(source, self.output_size)
        return sizes

    def _monthly_continuous(
        self, observations: list[_Observation], channels: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frames = torch.zeros(len(self.months), channels, *self.output_size)
        masks = torch.zeros(len(self.months))
        pixel_masks = torch.zeros(len(self.months), *self.output_size)
        for index, month in enumerate(self.months):
            selected = [item for item in observations if item.month == month]
            if not selected:
                continue
            values = torch.stack(
                [_resize(item.values, self.output_size, nearest=False) for item in selected]
            )
            valid = torch.stack(
                [_resize(item.mask[None], self.output_size, nearest=True)[0] for item in selected]
            )
            count = valid.sum(dim=0)
            frames[index] = (values * valid[:, None]).sum(dim=0) / count.clamp(min=1)[None]
            pixel_masks[index] = (count > 0).float()
            masks[index] = float(bool(pixel_masks[index].any().item()))
        return frames, masks, pixel_masks

    def _highres_input(
        self, source: str, observations: list[_Observation], channels: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.highres_sizes[source]
        if not observations:
            return torch.zeros(channels, *size), torch.zeros(1, *size)
        values = torch.stack([_resize(item.values, size, nearest=False) for item in observations])
        valid = torch.stack(
            [_resize(item.mask[None], size, nearest=True)[0] for item in observations]
        )
        count = valid.sum(dim=0)
        frame = (values * valid[:, None]).sum(dim=0) / count.clamp(min=1)[None]
        return frame, (count > 0).float()[None]

    def _categorical_target(
        self, observations: list[_Observation]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not observations:
            return (
                torch.zeros(len(self.months), *self.output_size, dtype=torch.long),
                torch.zeros(len(self.months), *self.output_size),
            )
        selected = observations[-1]
        labels = _resize(selected.values, self.output_size, nearest=True)[0].long()
        valid = _resize(selected.mask[None], self.output_size, nearest=True)[0]
        return labels[None].repeat(len(self.months), 1, 1), valid[None].repeat(
            len(self.months), 1, 1
        )

    def _supervised_labels(
        self, record: ManifestRecord
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        labels: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        patch_ids = [record.patch_id]
        if record.source_patch_id:
            patch_ids.append(record.source_patch_id)
        for task, root in self.dataset_config.supervised_label_roots.items():
            mask_root = root / "masks"
            exact = [mask_root / f"{patch_id}.tif" for patch_id in patch_ids]
            path = next((item for item in exact if item.is_file()), None)
            if path is None:
                dated = [
                    item for patch_id in patch_ids for item in mask_root.glob(f"{patch_id}_*.tif")
                ]
                path = max(dated, default=None)
            if path is None:
                labels[task] = torch.zeros(*self.output_size)
                masks[task] = torch.tensor(0.0)
                continue
            with rasterio.open(path) as raster:
                value = torch.from_numpy(raster.read(1, out_dtype="float32"))[None]
            labels[task] = (_resize(value, self.output_size, nearest=True)[0] > 0).float()
            masks[task] = torch.tensor(1.0)
        return labels, masks

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source_frames: dict[str, torch.Tensor] = {}
        source_masks: dict[str, torch.Tensor] = {}
        highres_frames: dict[str, torch.Tensor] = {}
        highres_masks: dict[str, torch.Tensor] = {}
        observations_by_source: dict[str, list[_Observation]] = {}
        monthly_by_source: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for source, source_config in self.config.model.input_sources.items():
            observations = self._observations(record, source, source_config.channels)
            observations_by_source[source] = observations
            if source_config.role == "temporal":
                monthly = self._monthly_continuous(observations, source_config.channels)
                monthly_by_source[source] = monthly
                source_frames[source], source_masks[source], _ = monthly
            else:
                highres_frames[source], highres_masks[source] = self._highres_input(
                    source, observations, source_config.channels
                )

        targets: dict[str, torch.Tensor] = {}
        target_masks: dict[str, torch.Tensor] = {}
        for name, head in self.config.model.target_heads.items():
            if head.loss_type == "categorical":
                observations = self._observations(record, head.source, 1, categorical=True)
                targets[name], target_masks[name] = self._categorical_target(observations)
            elif head.source in monthly_by_source:
                targets[name] = monthly_by_source[head.source][0].clone()
                target_masks[name] = monthly_by_source[head.source][2].clone()
            else:
                observations = observations_by_source.get(head.source, [])
                targets[name], _, target_masks[name] = self._monthly_continuous(
                    observations, head.channels
                )

        supervised_labels, supervised_label_masks = self._supervised_labels(record)
        return {
            "patch_id": record.patch_id,
            "region": record.region,
            "source_frames": source_frames,
            "source_masks": source_masks,
            "timestamps": torch.tensor(self.months),
            "highres_frames": highres_frames,
            "highres_masks": highres_masks,
            "targets": targets,
            "target_masks": target_masks,
            "supervised_labels": supervised_labels,
            "supervised_label_masks": supervised_label_masks,
        }


def collate_region_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples 不能为空")
    regions = {sample["region"] for sample in samples}
    if len(regions) != 1:
        raise ValueError("一个 raster batch 只能包含同一区域；多区域由 loader 轮转")

    def stack_mapping(name: str) -> dict[str, torch.Tensor]:
        keys = samples[0][name]
        return {key: torch.stack([sample[name][key] for sample in samples]) for key in keys}

    return {
        "patch_ids": [sample["patch_id"] for sample in samples],
        "regions": [sample["region"] for sample in samples],
        "source_frames": stack_mapping("source_frames"),
        "source_masks": stack_mapping("source_masks"),
        "timestamps": torch.stack([sample["timestamps"] for sample in samples]),
        "highres_frames": stack_mapping("highres_frames"),
        "highres_masks": stack_mapping("highres_masks"),
        "targets": stack_mapping("targets"),
        "target_masks": stack_mapping("target_masks"),
        "supervised_labels": stack_mapping("supervised_labels"),
        "supervised_label_masks": stack_mapping("supervised_label_masks"),
    }
