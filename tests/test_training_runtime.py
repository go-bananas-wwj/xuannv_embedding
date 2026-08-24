from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from xuannv_embedding.config import Config
from xuannv_embedding.export.embedding import export_embedding_batches
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from xuannv_embedding.training.cli import (
    RegionBatchStream,
    _epoch_count,
    _git_sha,
    synthetic_batch,
)
from xuannv_embedding.training.losses import TotalLoss
from xuannv_embedding.training.runtime import TrainingSystem, train_steps


def _system() -> TrainingSystem:
    model = AEFModel(
        sensor_channels={"s2": 2, "aerial": 1},
        embed_dim=8,
        target_heads={
            "s2_recon": ("continuous", 2),
            "aerial_recon": ("continuous", 1),
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
        source_roles={"s2": "temporal", "aerial": "highres"},
    )
    criterion = TotalLoss(
        {
            "s2_recon": {"loss_type": "l1", "channels": 2, "weight": 1.0},
            "aerial_recon": {"loss_type": "l1", "channels": 1, "weight": 0.5},
        },
        uniformity_weight=0.01,
        semantic_probe_embed_dim=8,
        semantic_probe_weight=0.1,
        semantic_probe_tasks=["osm_building"],
        semantic_probe_hidden_dim=0,
    )
    return TrainingSystem(model, criterion)


def _batch() -> dict[str, object]:
    return {
        "patch_ids": ["p1"],
        "source_frames": {"s2": torch.randn(1, 2, 2, 16, 16)},
        "source_masks": {"s2": torch.ones(1, 2)},
        "timestamps": torch.tensor([[202512, 202601]]),
        "highres_frames": {"aerial": torch.randn(1, 1, 16, 16)},
        "highres_masks": {"aerial": torch.ones(1, 1, 16, 16)},
        "targets": {
            "s2_recon": torch.randn(1, 2, 2, 16, 16),
            "aerial_recon": torch.randn(1, 2, 1, 16, 16),
        },
        "target_masks": {
            "s2_recon": torch.ones(1, 2, 16, 16),
            "aerial_recon": torch.ones(1, 2, 16, 16),
        },
        "supervised_labels": {"osm_building": torch.zeros(1, 16, 16)},
        "supervised_label_masks": {"osm_building": torch.ones(1)},
    }


def test_runtime_updates_model_and_saves_complete_training_state(tmp_path: Path) -> None:
    torch.manual_seed(5)
    system = _system()
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    before = system.model.temporal_stem_bank.encoders["s2"].conv.weight.detach().clone()

    summary = train_steps(
        system,
        [_batch()],
        optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        epochs=1,
        gradient_accumulation_steps=1,
        amp=False,
    )

    assert summary["optimizer_steps"] == 1
    assert np.isfinite(summary["loss"])
    assert not torch.equal(before, system.model.temporal_stem_bank.encoders["s2"].conv.weight)

    checkpoint = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        checkpoint,
        model=system.model,
        criterion=system.criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        config_sha256="a" * 64,
        git_sha="1234567890abcdef",
        source_schema={"s2": {"channels": 2, "role": "temporal"}},
        regions=["test-region"],
        metrics=summary,
    )
    restored = _system()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = load_training_checkpoint(
        checkpoint,
        model=restored.model,
        criterion=restored.criterion,
        optimizer=restored_optimizer,
    )
    assert state["criterion"] is not None
    assert all(
        torch.equal(left, right)
        for left, right in zip(system.criterion.parameters(), restored.criterion.parameters())
    )


def test_runtime_preserves_absolute_epoch_for_resume_warmups() -> None:
    system = _system()
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-3)

    summary = train_steps(
        system,
        [_batch()],
        optimizer,
        device=torch.device("cpu"),
        epochs=2,
        start_epoch=4,
        gradient_accumulation_steps=1,
        amp=False,
    )

    assert summary["start_epoch"] == 4
    assert summary["end_epoch"] == 5
    assert system.criterion.current_epoch == 5


def test_region_batch_stream_resumes_sampler_at_absolute_epoch() -> None:
    class RecordingSampler:
        def __init__(self) -> None:
            self.epochs: list[int] = []

        def set_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)

    class FakeLoader:
        def __init__(self) -> None:
            self.sampler = RecordingSampler()

        def __len__(self) -> int:
            return 1

        def __iter__(self):
            yield {"epoch_marker": len(self.sampler.epochs)}

    loader = FakeLoader()
    stream = RegionBatchStream(
        [loader],
        [1.0],
        seed=7,
        max_steps=1,
        masking_config={"enabled": False},
        start_epoch=4,
    )

    list(stream)
    list(stream)

    assert loader.sampler.epochs == [4, 5]


def test_configured_epochs_are_a_total_but_cli_override_is_incremental() -> None:
    assert _epoch_count(800, None, 400) == 400
    assert _epoch_count(800, 1, 400) == 1
    with pytest.raises(ValueError, match="没有待训练"):
        _epoch_count(800, None, 800)


def test_git_sha_is_independent_of_training_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _git_sha()
    monkeypatch.chdir(tmp_path)

    assert _git_sha() == expected


def test_explicit_git_sha_must_be_a_real_hex_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XUANNV_GIT_SHA", "not-a-commit")
    with pytest.raises(RuntimeError, match="XUANNV_GIT_SHA"):
        _git_sha()


def test_export_writes_one_atomic_finite_embedding_per_patch(tmp_path: Path) -> None:
    system = _system().eval()
    batch = _batch()
    paths = export_embedding_batches(system.model, [batch], tmp_path, device="cpu")
    assert paths == [tmp_path / "p1.npz"]
    assert not list(tmp_path.glob("*.partial"))
    with np.load(paths[0]) as payload:
        assert payload["embedding"].shape == (2, 8, 16, 16)
        assert np.isfinite(payload["embedding"]).all()
        assert payload["timestamps"].tolist() == [202512, 202601]


def test_synthetic_batch_keeps_missing_highres_out_of_input_and_supervision() -> None:
    config = Config.from_yaml("configs/production/harbin_p10c.yaml")
    batch = synthetic_batch(
        config,
        batch_size=1,
        spatial_size=16,
        missing_sources={"highres_sar"},
    )
    assert torch.count_nonzero(batch["highres_masks"]["highres_sar"]) == 0
    assert torch.count_nonzero(batch["target_masks"]["highres_sar_recon"]) == 0
    assert torch.count_nonzero(batch["highres_masks"]["highres_optical"]) > 0
