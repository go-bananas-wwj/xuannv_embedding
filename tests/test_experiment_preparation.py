import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from xuannv_embedding.config import Config
from xuannv_embedding.training.cli import synthetic_batch
from xuannv_embedding.training.experiment import public_base_config, run, spatial_partition


def test_public_base_excludes_highres_inputs_targets_and_mappings(tmp_path: Path) -> None:
    original = yaml.safe_load(Path("configs/production/haidian_p10c_v1.yaml").read_text())
    raw = public_base_config(original, seed=41, lr=1e-4)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = Config.from_yaml(path)
    assert set(config.model.input_sources) == {"s1", "s2", "landsat"}
    assert all("highres" not in h.source for h in config.model.target_heads.values())
    assert config.data.datasets[0].source_map["worldcover"] == "worldcover"
    assert "highres_optical" in original["model"]["input_sources"]


def test_spatial_partition_keeps_adjacent_tiles_out_of_training() -> None:
    centers = np.array([(x * 1280, y * 1280) for x in range(20) for y in range(20)])
    split = spatial_partition(centers, tile_size=1280)
    assert split == spatial_partition(centers, tile_size=1280)
    groups = [set(split[k]) for k in ["train", "validation", "test", "buffer"]]
    assert set.union(*groups) == set(range(len(centers)))
    assert sum(map(len, groups)) == len(centers)
    train = centers[split["train"]]
    held = centers[split["validation"] + split["test"]]
    assert np.max(np.abs(train[:, None] - held[None]), axis=-1).min() > 1280


def test_spatial_partition_rejects_nonfinite_coordinates() -> None:
    with pytest.raises(ValueError):
        spatial_partition(np.array([[0, float("nan")]]), tile_size=1280)


def test_cached_training_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    raw = public_base_config(
        yaml.safe_load(Path("configs/production/haidian_p10c_v1.yaml").read_text()),
        seed=41,
        lr=1e-4,
    )
    raw["model"]["embed_dim"] = 8
    raw["model"]["stem_dim"] = 8
    raw["model"]["num_months"] = 2
    raw["model"]["stp"].update(
        space_dim=16,
        time_dim=16,
        precision_dim=16,
        num_blocks=1,
        num_heads=2,
        time_attention_mode="none",
    )
    raw["data"]["months"] = ["2025-12", "2026-01"]
    raw["data"]["patch_size"] = 16
    raw["data"]["batch_size"] = 1
    raw["data"]["num_workers"] = 0
    raw["training"]["amp"] = False
    raw["training"]["input_masking"]["max_months_per_sample"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = Config.from_yaml(path)
    cache = tmp_path / "cache"
    cache.mkdir()
    records = []
    for index in range(4):
        batch = synthetic_batch(config, batch_size=1, spatial_size=16)
        sample = {
            k: ({s: v[0] for s, v in value.items()} if isinstance(value, dict) else value[0])
            for k, value in batch.items()
            if k not in ("patch_ids", "regions")
        }
        sample.update(patch_id=f"p{index}", region="haidian")
        output = cache / f"{index}.pt"
        torch.save(sample, output)
        records.append(
            {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
        )
    document = {
        "model_inputs": {k: asdict(v) for k, v in config.model.input_sources.items()},
        "model_targets": {k: asdict(v) for k, v in config.model.target_heads.items()},
        "data": asdict(config.data),
        "records": records,
        "split": {"train": [0, 1, 2], "validation": [3]},
    }
    (cache / "cache.json").write_text(json.dumps(document, default=str))

    def args(output, epochs, resume=None):
        return argparse.Namespace(
            config=path,
            cache=cache,
            output=output,
            device="cpu",
            epochs=epochs,
            pilot=False,
            resume=resume,
        )

    full = tmp_path / "full"
    run(args(full, 4))
    resumed = tmp_path / "resumed"
    run(args(resumed, 2))
    run(args(resumed, 4, resumed / "latest.pt"))
    left = torch.load(full / "latest.pt", weights_only=True)
    right = torch.load(resumed / "latest.pt", weights_only=True)
    for key in left["model"]:
        torch.testing.assert_close(left["model"][key], right["model"][key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        run(args(resumed, 1))
