from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from xuannv_embedding.config import (
    Config,
    DataConfig,
    ExperimentConfig,
    InputSourceConfig,
    ModelConfig,
    PathsConfig,
    RegionDatasetConfig,
    STPConfig,
    TargetHeadConfig,
    TrainingConfig,
)
from xuannv_embedding.data.raster_dataset import RegionRasterDataset, collate_region_batch
from xuannv_embedding.export.embedding import export_embedding_batches
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest


def _write(path: Path, array: np.ndarray, *, nodata: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=array.shape[0],
        height=array.shape[1],
        width=array.shape[2],
        dtype="float32",
        transform=from_origin(0, 160, 10, 10),
        crs="EPSG:32650",
        nodata=nodata,
    ) as target:
        target.write(array.astype(np.float32))


def _config(tmp_path: Path, manifest: Path) -> Config:
    statistics = tmp_path / "statistics" / "region-a"
    statistics.mkdir(parents=True)
    (statistics / "s2_stats.json").write_text(
        '{"mean": [10.0, 20.0], "std": [2.0, 4.0]}', encoding="utf-8"
    )
    dataset = RegionDatasetConfig(
        region="region-a",
        manifest_path=manifest,
        statistics_dir=statistics,
        patch_grid_path=tmp_path / "grid.json",
        source_map={"physical_s2": "s2", "physical_aerial": "highres_optical", "wc": "worldcover"},
        supervised_label_roots={},
    )
    return Config(
        schema_version="1",
        paths=PathsConfig(tmp_path, tmp_path / "outputs", tmp_path / "artifacts"),
        experiment=ExperimentConfig("test"),
        model=ModelConfig(
            embed_dim=8,
            input_sources={
                "s2": InputSourceConfig(2, "temporal"),
                "highres_optical": InputSourceConfig(1, "highres"),
            },
            target_heads={
                "s2_recon": TargetHeadConfig("s2", "continuous", 2, 1.0),
                "highres_optical_recon": TargetHeadConfig("highres_optical", "continuous", 1, 1.0),
                "worldcover": TargetHeadConfig("worldcover", "categorical", 3, 1.0),
            },
            stem_dim=8,
            num_months=2,
            ref_year=2025,
            ref_month=12,
            stp=STPConfig(
                space_dim=16,
                time_dim=16,
                precision_dim=16,
                precision_scale=1,
                num_blocks=1,
                num_heads=2,
                temporal_fusion="gated_sum",
                time_attention_mode="none",
            ),
        ),
        training=TrainingConfig(1, 1e-3, 0.0, 0, 1, 1, 1, amp=False),
        data=DataConfig(["2025-12", "2026-01"], [dataset], 1, 0, 16),
    )


def test_region_raster_dataset_maps_sources_months_statistics_and_missingness(
    tmp_path: Path,
) -> None:
    s2_december = tmp_path / "region-a" / "s2_20251201_patch_1.tif"
    s2_january = tmp_path / "region-a" / "s2_20260103_patch_1.tif"
    worldcover = tmp_path / "region-a" / "worldcover_20230101_patch_1.tif"
    _write(s2_december, np.stack([np.full((16, 16), 12), np.full((16, 16), 24)]))
    _write(s2_january, np.stack([np.full((16, 16), 14), np.full((16, 16), 28)]))
    _write(worldcover, np.ones((1, 16, 16)))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            ManifestRecord(
                "patch-1",
                "region-a",
                {
                    "physical_s2": [
                        "region-a/s2_20251201_patch_1.tif",
                        "region-a/s2_20260103_patch_1.tif",
                    ],
                    "physical_aerial": None,
                    "wc": "region-a/worldcover_20230101_patch_1.tif",
                },
            )
        ],
        months=["2025-12", "2026-01"],
    )
    config = _config(tmp_path, manifest)
    dataset = RegionRasterDataset(config, config.data.datasets[0])

    sample = dataset[0]
    assert sample["region"] == "region-a"
    assert sample["source_frames"]["s2"][:, :, 0, 0].tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert torch.count_nonzero(sample["highres_masks"]["highres_optical"]) == 0
    assert torch.count_nonzero(sample["target_masks"]["highres_optical_recon"]) == 0
    assert sample["targets"]["worldcover"].dtype == torch.int64
    assert sample["targets"]["worldcover"].shape == (2, 16, 16)

    batch = collate_region_batch([sample])
    assert batch["source_frames"]["s2"].shape == (1, 2, 2, 16, 16)
    assert batch["highres_frames"]["highres_optical"].shape == (1, 1, 16, 16)

    system = build_training_system(config).eval()
    paths = export_embedding_batches(system.model, [batch], tmp_path / "export", device="cpu")
    with np.load(paths[0]) as payload:
        assert payload["embedding"].shape == (2, 8, 16, 16)
