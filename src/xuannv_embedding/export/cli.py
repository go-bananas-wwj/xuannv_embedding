"""从严格 checkpoint 导出真实区域 manifest 的月度 embedding。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from xuannv_embedding.config import Config
from xuannv_embedding.export.embedding import export_embedding_batches
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.compatibility import load_compatible_checkpoint


def _device(value: str | None) -> torch.device:
    if value is not None:
        device = torch.device(value)
    else:
        try:
            import torch_npu  # noqa: F401

            device = torch.device("npu:0") if torch.npu.is_available() else torch.device("cpu")
        except ImportError:
            device = torch.device("cpu")
    if device.type == "npu":
        torch.npu.set_device(device)
    return device


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv export")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--region", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--compatibility-profile", choices=["haidian_p10c_v1"])
    args = parser.parse_args(argv)

    from torch.utils.data import DataLoader

    from xuannv_embedding.data.raster_dataset import RegionRasterDataset, collate_region_batch

    config = Config.from_yaml(args.config)
    system = build_training_system(config)
    if args.compatibility_profile:
        load_compatible_checkpoint(
            args.checkpoint,
            system.model,
            profile=args.compatibility_profile,
        )
    else:
        load_training_checkpoint(
            args.checkpoint,
            model=system.model,
            expected_config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
            expected_source_schema={
                name: asdict(value) for name, value in config.model.input_sources.items()
            },
            expected_regions=[dataset.region for dataset in config.data.datasets],
        )
    device = _device(args.device)
    selected = set(args.region or [item.region for item in config.data.datasets])
    known = {item.region for item in config.data.datasets}
    if not selected <= known:
        parser.error(f"配置中不存在 region: {sorted(selected - known)}")
    written: dict[str, list[str]] = {}
    for dataset_config in config.data.datasets:
        if dataset_config.region not in selected:
            continue
        dataset = RegionRasterDataset(config, dataset_config, max_records=args.limit)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size or config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_region_batch,
            pin_memory=True,
        )
        paths = export_embedding_batches(
            system.model,
            loader,
            args.output_root / dataset_config.region,
            device=device,
        )
        written[dataset_config.region] = [str(path) for path in paths]
    print(json.dumps({"written": written}, ensure_ascii=False, indent=2))
    return 0
