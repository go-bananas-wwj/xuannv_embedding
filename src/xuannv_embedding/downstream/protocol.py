from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


class ProtocolError(ValueError):
    """下游空间 fold、shot 或阈值协议不合法。"""


@dataclass(frozen=True)
class SpatialFold:
    fold: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = {
            "train": set(self.train),
            "validation": set(self.validation),
            "test": set(self.test),
        }
        if any(len(values) != len(getattr(self, name)) for name, values in groups.items()):
            raise ProtocolError("空间 fold 内包含重复 patch_id")
        if (
            groups["train"] & groups["validation"]
            or groups["train"] & groups["test"]
            or groups["validation"] & groups["test"]
        ):
            raise ProtocolError("空间 fold 泄漏: train/validation/test 必须互斥")

    @classmethod
    def from_file(cls, path: str | Path, *, fold: int) -> "SpatialFold":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"无法读取 fold 文件: {exc}") from exc
        folds = raw.get("folds") if isinstance(raw, dict) else None
        if not isinstance(folds, list):
            raise ProtocolError("fold 文件必须包含 folds 列表")
        matches = [item for item in folds if isinstance(item, dict) and item.get("fold") == fold]
        if len(matches) != 1:
            raise ProtocolError(f"fold={fold} 必须且只能出现一次")
        item = matches[0]
        try:
            return cls(
                fold=fold,
                train=tuple(str(value) for value in item["train"]),
                validation=tuple(str(value) for value in item["val"]),
                test=tuple(str(value) for value in item["test"]),
            )
        except (KeyError, TypeError) as exc:
            raise ProtocolError(f"fold={fold} 缺少 train/val/test 列表") from exc


@dataclass(frozen=True)
class EvaluationProtocol:
    fold: SpatialFold
    shot: int | None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.shot not in {None, 5, 10, 50}:
            raise ProtocolError("shot 只允许 full-label(None)、5、10 或 50")

    @property
    def report_scope(self) -> str:
        return "full-label" if self.shot is None else f"{self.shot}-shot"

    def training_patch_ids(
        self,
        *,
        positive_patch_ids: set[str] | None = None,
    ) -> tuple[str, ...]:
        if self.shot is None:
            return self.fold.train
        if positive_patch_ids is None:
            raise ProtocolError("few-shot 必须提供由标签计算的正样本 patch 集合")
        train = set(self.fold.train)
        positives = train & positive_patch_ids
        negatives = train - positive_patch_ids
        if len(positives) < self.shot:
            raise ProtocolError(
                f"fold={self.fold.fold} 只有 {len(positives)} 个正样本 patch，"
                f"不足 {self.shot}-shot"
            )
        if len(negatives) < self.shot:
            raise ProtocolError(
                f"fold={self.fold.fold} 只有 {len(negatives)} 个负样本 patch，"
                f"不足与 {self.shot}-shot 正样本配对"
            )

        def rank(pool: set[str], label: str) -> list[str]:
            return sorted(
                pool,
                key=lambda patch_id: hashlib.sha256(
                    f"{self.seed}:{self.fold.fold}:{label}:{patch_id}".encode()
                ).digest(),
            )

        return tuple(
            rank(positives, "positive")[: self.shot] + rank(negatives, "negative")[: self.shot]
        )


def choose_validation_threshold(
    logits: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    ignore_index: int = -1,
) -> float:
    """只根据 validation split 选择 F1 最大阈值。"""
    from sklearn.metrics import precision_recall_curve

    logits_array = (
        logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    ).reshape(-1)
    target_array = (
        target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)
    ).reshape(-1)
    valid = target_array != ignore_index
    target_valid = (target_array[valid] > 0).astype(np.uint8)
    if not valid.any() or np.unique(target_valid).size < 2:
        raise ProtocolError("validation 必须同时包含正负有效像素才能选择阈值")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits_array[valid], -80.0, 80.0)))
    precision, recall, thresholds = precision_recall_curve(target_valid, probabilities)
    if thresholds.size == 0:
        raise ProtocolError("validation 无法生成阈值候选")
    f1 = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1))])
