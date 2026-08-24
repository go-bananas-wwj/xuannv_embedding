from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xuannv_embedding.cli import main
from xuannv_embedding.downstream.heads import (
    Conv3x3Head,
    DeepLabLiteHead,
    DeepWideMLPHead,
    LinearHead,
    MLPHead,
    UNetHead,
    WideMLPHead,
    build_head,
)
from xuannv_embedding.downstream.metrics import evaluate_binary
from xuannv_embedding.downstream.protocol import (
    EvaluationProtocol,
    ProtocolError,
    SpatialFold,
    choose_validation_threshold,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("linear", LinearHead),
        ("mlp", MLPHead),
        ("wide_mlp", WideMLPHead),
        ("deep_wide_mlp", DeepWideMLPHead),
        ("conv3x3", Conv3x3Head),
        ("unet", UNetHead),
        ("deeplab_lite", DeepLabLiteHead),
    ],
)
def test_standard_heads_preserve_dense_shape(name: str, expected: type) -> None:
    head = build_head(name, embed_dim=64, num_classes=1).eval()
    output = head(torch.randn(2, 64, 16, 16))
    output.mean().backward()

    assert isinstance(head, expected)
    assert output.shape == (2, 1, 16, 16)
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_nonstandard_head_is_not_registered() -> None:
    with pytest.raises(ValueError, match="未知标准 head"):
        build_head("upernet", embed_dim=64, num_classes=1)


def test_metrics_report_f1_ap_auc_with_fixed_threshold() -> None:
    logits = np.array([[-4.0, 4.0], [3.0, -3.0]], dtype=np.float32)
    target = np.array([[0, 1], [1, 0]], dtype=np.int64)

    metrics = evaluate_binary(logits, target, threshold=0.7)

    assert metrics["f1"] == 1.0
    assert metrics["ap"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["threshold"] == 0.7


def test_metrics_mark_undefined_ap_auc_as_null() -> None:
    metrics = evaluate_binary(
        np.array([-2.0, -1.0], dtype=np.float32),
        np.array([0, 0], dtype=np.int64),
        threshold=0.5,
    )

    assert metrics["ap"] is None
    assert metrics["auc"] is None


def test_threshold_is_selected_only_from_validation_predictions() -> None:
    validation_logits = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    validation_target = np.array([0, 0, 1, 1], dtype=np.int64)
    threshold = choose_validation_threshold(validation_logits, validation_target)
    test_metrics = evaluate_binary(
        np.array([-2.0, 2.0], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        threshold=threshold,
    )

    assert test_metrics["threshold"] == threshold


def test_spatial_fold_and_shot_protocol_are_fixed(tmp_path: Path) -> None:
    fold = SpatialFold(
        fold=0,
        train=("p1", "p2", "p3", "p4", "p5", "p6"),
        validation=("v1",),
        test=("t1", "t2"),
    )
    protocol = EvaluationProtocol(fold=fold, shot=5, seed=42)

    assert len(protocol.training_patch_ids()) == 5
    assert (
        protocol.training_patch_ids()
        == EvaluationProtocol(fold=fold, shot=5, seed=42).training_patch_ids()
    )
    assert protocol.report_scope == "5-shot"

    path = tmp_path / "folds.json"
    path.write_text(
        json.dumps({"folds": [{"fold": 0, "train": ["p1"], "val": ["v1"], "test": ["t1"]}]}),
        encoding="utf-8",
    )
    loaded = SpatialFold.from_file(path, fold=0)
    assert loaded.validation == ("v1",)


def test_spatial_fold_rejects_leakage() -> None:
    with pytest.raises(ProtocolError, match="空间 fold 泄漏"):
        SpatialFold(fold=0, train=("p1",), validation=("p1",), test=("t1",))


def test_downstream_cli_has_train_and_evaluate_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as train_exit:
        main(["downstream", "train", "--help"])
    assert train_exit.value.code == 0
    assert "--dataset" in capsys.readouterr().out

    with pytest.raises(SystemExit) as eval_exit:
        main(["downstream", "evaluate", "--help"])
    assert eval_exit.value.code == 0
    assert "--checkpoint" in capsys.readouterr().out


def test_downstream_train_evaluate_mini_e2e(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    dataset = tmp_path / "dataset.npz"
    patch_ids = np.array([f"p{index}" for index in range(8)])
    embeddings = rng.normal(size=(8, 4, 4, 4)).astype(np.float32)
    labels = (embeddings[:, 0] > 0).astype(np.float32)
    np.savez(dataset, embeddings=embeddings, labels=labels, patch_ids=patch_ids)
    folds = tmp_path / "folds.json"
    folds.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "fold": 0,
                        "train": ["p0", "p1", "p2", "p3"],
                        "val": ["p4", "p5"],
                        "test": ["p6", "p7"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "head.pt"
    report = tmp_path / "metrics.json"

    assert (
        main(
            [
                "downstream",
                "train",
                "--dataset",
                str(dataset),
                "--folds",
                str(folds),
                "--head",
                "linear",
                "--output",
                str(checkpoint),
                "--epochs",
                "2",
                "--batch-size",
                "2",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "downstream",
                "evaluate",
                "--dataset",
                str(dataset),
                "--folds",
                str(folds),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(report),
                "--batch-size",
                "2",
            ]
        )
        == 0
    )

    metrics = json.loads(report.read_text(encoding="utf-8"))
    assert {"f1", "ap", "auc"} <= metrics.keys()
    assert metrics["report_scope"] == "full-label"
    assert metrics["threshold_source"] == "validation"
