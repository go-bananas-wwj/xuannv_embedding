from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from xuannv_embedding.training.losses import (
    SemanticProbeLoss,
    TotalLoss,
    batch_uniformity_loss,
    reconstruction_loss,
)
from xuannv_embedding.training.masking import InputMaskingConfig, apply_input_masking


def test_reconstruction_loss_l1_respects_mask() -> None:
    pred = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    assert torch.allclose(reconstruction_loss(pred, target, mask), torch.tensor(3.0))


def test_reconstruction_loss_ce_ignores_class_zero() -> None:
    pred = torch.randn(2, 3, 4, 4)
    target = torch.randint(0, 3, (2, 4, 4))
    target[:, 0] = 0
    mask = torch.ones(2, 4, 4)

    actual = reconstruction_loss(pred, target, mask, loss_type="ce")
    per_pixel = F.cross_entropy(pred, target, ignore_index=0, reduction="none")
    valid = mask * (target != 0)
    expected = (per_pixel * valid).sum() / valid.sum().clamp(min=1.0)

    assert torch.allclose(actual, expected)


def test_uniformity_accepts_monthly_embeddings() -> None:
    embedding = torch.randn(2, 3, 8)
    loss = batch_uniformity_loss(embedding)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_semantic_probe_supports_hard_negative_warmup() -> None:
    probe = SemanticProbeLoss(
        embed_dim=4,
        tasks=["osm_building"],
        hidden_dim=0,
        hard_negative_ratio=0.5,
        hard_negative_weight=0.4,
        hard_negative_warmup_epochs=4,
    )
    probe.set_epoch(1)
    embedding_map = torch.randn(2, 1, 4, 4, 4)
    labels = {"osm_building": torch.zeros(2, 4, 4)}
    labels["osm_building"][:, 0, 0] = 1.0
    label_masks = {"osm_building": torch.ones(2)}

    loss, stats = probe(embedding_map, labels, label_masks)

    assert torch.isfinite(loss)
    assert torch.allclose(
        stats["semantic_probe_osm_building_hard_negative_weight"],
        torch.tensor(0.2),
    )
    assert stats["semantic_probe_positive_pixels"].item() == 2.0


def test_total_loss_contains_only_p10c_objectives() -> None:
    criterion = TotalLoss(
        target_cfg={"s2_recon": {"loss_type": "l1", "channels": 2, "weight": 0.8}},
        uniformity_weight=0.06,
        semantic_probe_embed_dim=4,
        semantic_probe_weight=0.14,
        semantic_probe_tasks=["osm_building"],
        semantic_probe_hidden_dim=0,
    )
    output = SimpleNamespace(
        embedding_map=F.normalize(torch.randn(2, 1, 4, 4, 4), p=2, dim=2),
        reconstructions={"s2_recon": torch.randn(2, 1, 2, 4, 4)},
    )
    targets = {"s2_recon": torch.randn(2, 1, 2, 4, 4)}
    masks = {"s2_recon": torch.ones(2, 1, 4, 4)}
    labels = {"osm_building": torch.zeros(2, 4, 4)}
    label_masks = {"osm_building": torch.ones(2)}

    losses = criterion(output, targets, masks, labels, label_masks)

    assert {"total", "recon", "uniformity", "semantic_probe"} <= losses.keys()
    assert (
        not {
            "covariance",
            "patch_discrimination",
            "distillation",
            "prototype",
            "boundary",
            "latent_reconstruction",
        }
        & losses.keys()
    )
    assert torch.isfinite(losses["total"])


def test_input_masking_drops_modality_without_changing_targets() -> None:
    original_target = torch.ones(2, 2, 1, 4, 4)
    prepared = {
        "source_frames": {"s2": torch.ones_like(original_target)},
        "source_masks": {"s2": torch.ones(2, 2)},
        "highres_frames": {},
        "highres_masks": {},
        "targets": {"s2_recon": original_target.clone()},
    }
    config = InputMaskingConfig(
        enabled=True,
        modality_dropout_probs={"s2": 1.0},
        month_dropout_prob=0.0,
        spatial_block_prob=0.0,
    )

    masked = apply_input_masking(prepared, config)

    assert torch.count_nonzero(masked["source_frames"]["s2"]) == 0
    assert torch.count_nonzero(masked["source_masks"]["s2"]) == 0
    assert torch.equal(masked["targets"]["s2_recon"], original_target)
