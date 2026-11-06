from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from xuannv_embedding.data.contracts import ProductSpec
from xuannv_embedding.models.v2_model import XuannvV2Model
from xuannv_embedding.training.checkpoint import (
    CheckpointError,
    capture_rng_state,
    load_v2_training_checkpoint,
    save_v2_training_checkpoint,
)
from xuannv_embedding.training.losses import V2TotalLoss
from xuannv_embedding.training.runtime import (
    V2TrainingSystem,
    train_v2_accumulation_steps,
    train_v2_steps,
)
from xuannv_embedding.training.validation_profiles import (
    assert_macro_disjoint,
    data_manifest_sha256,
)


def _system() -> V2TrainingSystem:
    product = ProductSpec(
        product_id="dense",
        role="dense",
        bands=("a", "b"),
        native_gsd_m=(10.0, 10.0),
        stored_gsd_m=10.0,
        dtype="float32",
        time_precision="month",
        already_resampled=True,
        qa_available=False,
    )
    model = XuannvV2Model(
        {"dense": product},
        embedding_dim=8,
        stem_dim=8,
        spatial_dim=16,
        temporal_dim=16,
        precision_dim=8,
        num_blocks=1,
        num_heads=2,
        temporal_mode="causal_window",
        gradient_checkpointing=True,
    )
    criterion = V2TotalLoss(
        embed_dim=8,
        reconstruction_weights={"dense": 1.0},
        uniformity_weight=0.01,
    )
    return V2TrainingSystem(model, criterion)


def _batch() -> dict[str, object]:
    frames = torch.randn(2, 2, 2, 8, 8)
    return {
        "patch_ids": ["p0", "p1"],
        "model_inputs": {
            "source_frames": {"dense": frames},
            "source_pixel_masks": {"dense": torch.ones(2, 2, 1, 8, 8)},
            "source_observation_masks": {"dense": torch.ones(2, 2, dtype=torch.bool)},
            "source_time_bounds": {
                "dense": torch.tensor([[[0.0, 30.0], [30.0, 60.0]]]).repeat(2, 1, 1)
            },
            "source_available_at": {"dense": torch.tensor([[30.0, 60.0]]).repeat(2, 1)},
            "output_intervals": torch.tensor([[[0.0, 30.0], [30.0, 60.0]]]).repeat(2, 1, 1),
            "output_size": (8, 8),
        },
        "targets": {"dense": frames.clone()},
        "target_masks": {"dense": torch.ones(2, 2, 1, 8, 8)},
    }


def _checkpoint_contract() -> dict[str, object]:
    return {
        "config_sha256": "a" * 64,
        "git_sha": "1234567890abcdef",
        "data_manifest_sha256": "b" * 64,
        "product_schema": {"dense": {"bands": ["a", "b"]}},
        "temporal_contract": {"mode": "causal_window"},
    }


def test_data_manifest_rejects_changed_indexed_highres_content(tmp_path: Path) -> None:
    required = (
        "locks/local_archive_sha256.jsonl",
        "registry/local_archive_inventory.parquet",
        "registry/split_80_10_10.parquet",
        "observations/index/availability.parquet",
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"locked")
    statistics = tmp_path / "statistics" / "dense.json"
    statistics.parent.mkdir(parents=True)
    statistics.write_text("{}", encoding="utf-8")
    image = tmp_path / "scene.tif"
    image.write_bytes(b"original pixels")
    import hashlib

    scenes = tmp_path / "observations" / "highres" / "hr" / "scenes.parquet"
    scenes.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "product_id": "hr",
                    "image_path": str(image),
                    "image_size_bytes": image.stat().st_size,
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "qa_path": None,
                    "qa_present": False,
                    "qa_size_bytes": None,
                    "qa_sha256": None,
                }
            ]
        ),
        scenes,
    )
    patch_row = pq.read_table(scenes).to_pylist()[0]
    patch_row.update({"patch_id": "p1", "scene_id": "s1"})
    pq.write_table(
        pa.Table.from_pylist([patch_row]), scenes.with_name("patch_observations.parquet")
    )

    first = data_manifest_sha256(tmp_path)
    image.write_bytes(b"modified pixels")

    assert len(first) == 64
    import pytest

    with pytest.raises(ValueError, match="内容.*变化"):
        data_manifest_sha256(tmp_path)


def test_v2_training_updates_parameters_and_optimizer() -> None:
    torch.manual_seed(10)
    system = _system()
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-3)
    before = next(system.model.parameters()).detach().clone()

    summary = train_v2_steps(
        system,
        [_batch()],
        optimizer,
        device=torch.device("cpu"),
        max_steps=2,
        amp=False,
    )

    assert summary["steps"] == 2
    assert torch.isfinite(torch.tensor(summary["loss"]))
    assert not torch.equal(before, next(system.model.parameters()).detach())
    assert optimizer.state


def test_v2_gradient_accumulation_counts_optimizer_and_micro_steps() -> None:
    system = _system()
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-3)
    batches = [_batch() for _ in range(6)]

    summary = train_v2_accumulation_steps(
        system,
        batches,
        optimizer,
        device=torch.device("cpu"),
        optimizer_steps=2,
        gradient_accumulation_steps=3,
        amp=False,
    )

    assert summary["optimizer_steps"] == 2
    assert summary["micro_batches"] == 6
    assert summary["samples"] == 12
    assert {int(state["step"].item()) for state in optimizer.state.values()} == {2}


def test_v2_checkpoint_round_trip_carries_data_provenance(tmp_path: Path) -> None:
    system = _system()
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-3)
    train_v2_steps(
        system,
        [_batch()],
        optimizer,
        device=torch.device("cpu"),
        max_steps=1,
        amp=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "v2.pt"
    contract = _checkpoint_contract()
    save_v2_training_checkpoint(
        path,
        model=system.model,
        criterion=system.criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        step=2,
        metrics={"loss": 1.0},
        rank_rng_states=[capture_rng_state()],
        **contract,
    )
    restored = _system()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)

    state = load_v2_training_checkpoint(
        path,
        model=restored.model,
        criterion=restored.criterion,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_config_sha256=contract["config_sha256"],
        expected_data_manifest_sha256=contract["data_manifest_sha256"],
        expected_product_schema=contract["product_schema"],
        expected_temporal_contract=contract["temporal_contract"],
    )

    assert state["format_version"] == "2"
    assert state["data_manifest_sha256"] == "b" * 64
    assert state["step"] == 2
    assert restored_optimizer.state
    assert len(state["rank_rng_states"]) == 1


def test_v2_checkpoint_restores_python_numpy_and_torch_rng(tmp_path: Path) -> None:
    system = _system()
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-3)
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    path = tmp_path / "rng.pt"
    save_v2_training_checkpoint(
        path,
        model=system.model,
        criterion=system.criterion,
        optimizer=optimizer,
        scheduler=None,
        step=0,
        metrics={},
        sampler_state={"epoch": 3, "offset": 17},
        **_checkpoint_contract(),
    )
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    state = load_v2_training_checkpoint(
        path,
        model=system.model,
        criterion=system.criterion,
        optimizer=optimizer,
        scheduler=None,
        expected_config_sha256="a" * 64,
        expected_data_manifest_sha256="b" * 64,
        expected_product_schema={"dense": {"bands": ["a", "b"]}},
        expected_temporal_contract={"mode": "causal_window"},
    )

    assert state["sampler_state"] == {"epoch": 3, "offset": 17}
    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected


def test_v2_checkpoint_loader_explicitly_rejects_v1(tmp_path: Path) -> None:
    path = tmp_path / "v1.pt"
    state = {
        "format_version": "1",
        "config_sha256": "a" * 64,
        "git_sha": "1234567",
        "data_manifest_sha256": "b" * 64,
        "product_schema": {"dense": {}},
        "temporal_contract": {"mode": "causal_window"},
        "step": 0,
        "model": {},
        "criterion": {},
        "optimizer": {},
        "scheduler": None,
        "metrics": {},
        "rng_state": torch.get_rng_state(),
    }
    torch.save(state, path)
    system = _system()
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-3)
    contract = _checkpoint_contract()

    try:
        load_v2_training_checkpoint(
            path,
            model=system.model,
            criterion=system.criterion,
            optimizer=optimizer,
            scheduler=None,
            expected_config_sha256=contract["config_sha256"],
            expected_data_manifest_sha256=contract["data_manifest_sha256"],
            expected_product_schema=contract["product_schema"],
            expected_temporal_contract=contract["temporal_contract"],
        )
    except CheckpointError as exc:
        assert "拒绝 V1" in str(exc)
    else:
        raise AssertionError("V2 loader 不得接受 V1 checkpoint")


def test_smoke_registry_rejects_macro_cross_split_leakage(tmp_path: Path) -> None:
    path = tmp_path / "registry.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"macro_id": "shared", "split": "train"},
                {"macro_id": "shared", "split": "test"},
            ]
        ),
        path,
    )

    try:
        assert_macro_disjoint(path)
    except ValueError as exc:
        assert "跨 split" in str(exc)
    else:
        raise AssertionError("应拒绝跨 split macro_id")
