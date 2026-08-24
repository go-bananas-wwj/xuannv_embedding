from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.checkpoint import (
    CheckpointError,
    load_training_checkpoint,
    save_training_checkpoint,
)
from xuannv_embedding.training.compatibility import (
    CompatibilityError,
    load_compatible_checkpoint,
    remap_haidian_p10c_state_dict,
)

REAL_P10C = Path(
    "/data/xuannv_embedding/outputs/"
    "v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704/epoch_800.pt"
)
REAL_SHA256 = "69dfd81c898544413a747f5c7304cc9210ad1cf420ce724864b8bd7deb6ed790"


def _model(*, legacy: bool) -> AEFModel:
    optical = "highres_optical_haidian" if legacy else "highres_optical"
    sar = "highres_sar_haidian" if legacy else "highres_sar"
    return AEFModel(
        sensor_channels={"s2": 2, optical: 3, sar: 1},
        embed_dim=8,
        target_heads={
            "s2_recon": ("continuous", 2),
            f"{optical}_recon": ("continuous", 3),
            f"{sar}_recon": ("continuous", 1),
        },
        stem_dim=8,
        stp={
            "space_dim": 16,
            "time_dim": 16,
            "precision_dim": 16,
            "precision_scale": 1,
            "num_blocks": 1,
            "num_heads": 2,
            "temporal_fusion": "gated_sum",
            "time_attention_mode": "none",
        },
        num_months=2,
        ref_year=2025,
        ref_month=12,
        source_roles={"s2": "temporal", optical: "highres", sar: "highres"},
    ).eval()


def _registry(path: Path, checkpoint: Path, key_count: int) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "artifacts": {
                    "haidian_p10c_v1": {
                        "filename": checkpoint.name,
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "model_key_count": key_count,
                        "compatibility_profile": "haidian_p10c_v1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_profile_maps_all_keys_and_preserves_embedding_exactly(tmp_path: Path) -> None:
    torch.manual_seed(7)
    legacy = _model(legacy=True)
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"epoch": 799, "model": legacy.state_dict()}, checkpoint)
    registry = _registry(tmp_path / "artifacts.json", checkpoint, len(legacy.state_dict()))
    canonical = _model(legacy=False)

    report = load_compatible_checkpoint(
        checkpoint,
        canonical,
        profile="haidian_p10c_v1",
        artifact_manifest_path=registry,
    )

    assert report.consumed_keys == len(legacy.state_dict())
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()

    source_frames = {"s2": torch.randn(1, 2, 2, 16, 16)}
    source_masks = {"s2": torch.ones(1, 2)}
    timestamps = torch.tensor([[202512, 202601]])
    optical = torch.randn(1, 3, 16, 16)
    sar = torch.randn(1, 1, 16, 16)
    masks = torch.ones(1, 1, 16, 16)
    with torch.no_grad():
        old_output = legacy(
            source_frames,
            source_masks,
            timestamps,
            {
                "highres_optical_haidian": optical,
                "highres_sar_haidian": sar,
            },
            {
                "highres_optical_haidian": masks,
                "highres_sar_haidian": masks,
            },
        )
        new_output = canonical(
            source_frames,
            source_masks,
            timestamps,
            {"highres_optical": optical, "highres_sar": sar},
            {"highres_optical": masks, "highres_sar": masks},
        )

    assert torch.equal(old_output.embedding_map, new_output.embedding_map)
    assert torch.equal(old_output.embedding, new_output.embedding)


def test_profile_rejects_unregistered_hash_before_loading(tmp_path: Path) -> None:
    legacy = _model(legacy=True)
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": legacy.state_dict()}, checkpoint)
    registry = _registry(tmp_path / "artifacts.json", checkpoint, len(legacy.state_dict()))
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    with pytest.raises(CompatibilityError, match="SHA-256"):
        load_compatible_checkpoint(
            checkpoint,
            _model(legacy=False),
            profile="haidian_p10c_v1",
            artifact_manifest_path=registry,
        )


def test_profile_rejects_mapping_collisions() -> None:
    state = {
        "highres_encoders.highres_optical_haidian.conv.weight": torch.ones(1),
        "highres_encoders.highres_optical.conv.weight": torch.zeros(1),
    }

    with pytest.raises(CompatibilityError, match="映射冲突"):
        remap_haidian_p10c_state_dict(state)


def test_new_checkpoint_format_round_trip(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "checkpoint.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=4,
        config_sha256="a" * 64,
        git_sha="1234567890abcdef",
        source_schema={"s2": {"channels": 12, "role": "temporal"}},
        regions=["haidian", "harbin"],
        metrics={"loss": 1.0},
    )
    restored = nn.Linear(3, 2)
    state = load_training_checkpoint(
        path,
        model=restored,
        expected_config_sha256="a" * 64,
        expected_source_schema={"s2": {"channels": 12, "role": "temporal"}},
        expected_regions=["haidian", "harbin"],
    )

    assert state["format_version"] == "1"
    assert state["config_sha256"] == "a" * 64
    assert state["git_sha"] == "1234567890abcdef"
    assert state["regions"] == ["haidian", "harbin"]
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))

    with pytest.raises(CheckpointError, match="config_sha256"):
        load_training_checkpoint(
            path,
            model=nn.Linear(3, 2),
            expected_config_sha256="b" * 64,
            expected_source_schema={"s2": {"channels": 12, "role": "temporal"}},
            expected_regions=["haidian", "harbin"],
        )
    with pytest.raises(CheckpointError, match="source_schema"):
        load_training_checkpoint(
            path,
            model=nn.Linear(3, 2),
            expected_config_sha256="a" * 64,
            expected_source_schema={"s2": {"channels": 2, "role": "temporal"}},
            expected_regions=["haidian", "harbin"],
        )
    with pytest.raises(CheckpointError, match="regions"):
        load_training_checkpoint(
            path,
            model=nn.Linear(3, 2),
            expected_config_sha256="a" * 64,
            expected_source_schema={"s2": {"channels": 12, "role": "temporal"}},
            expected_regions=["haidian"],
        )


def test_new_checkpoint_rejects_missing_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"format_version": "1", "model": nn.Linear(1, 1).state_dict()}, path)

    with pytest.raises(CheckpointError, match="缺少字段"):
        load_training_checkpoint(path, model=nn.Linear(1, 1))


@pytest.mark.skipif(not REAL_P10C.is_file(), reason="local production artifact is unavailable")
def test_real_p10c_artifact_sha_and_key_count() -> None:
    assert hashlib.sha256(REAL_P10C.read_bytes()).hexdigest() == REAL_SHA256
    state = torch.load(REAL_P10C, map_location="cpu", weights_only=True)

    mapped = remap_haidian_p10c_state_dict(state["model"])

    assert len(state["model"]) == 431
    assert len(mapped) == 431
