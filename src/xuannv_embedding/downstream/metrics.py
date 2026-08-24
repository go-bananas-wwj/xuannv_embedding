from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def evaluate_binary(
    logits: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    threshold: float,
    ignore_index: int = -1,
) -> dict[str, Any]:
    """使用外部固定阈值报告二分类 F1/AP/AUC。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    logits_array = _numpy(logits).reshape(-1)
    target_array = _numpy(target).reshape(-1)
    valid = target_array != ignore_index
    if not valid.any():
        raise ValueError("评测没有有效像素")
    target_valid = (target_array[valid] > 0).astype(np.uint8)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits_array[valid], -80.0, 80.0)))
    prediction = probabilities >= float(threshold)
    positive = target_valid == 1
    true_positive = int((prediction & positive).sum())
    false_positive = int((prediction & ~positive).sum())
    false_negative = int((~prediction & positive).sum())
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    if np.unique(target_valid).size < 2:
        ap = None
        auc = None
    else:
        ap = float(average_precision_score(target_valid, probabilities))
        auc = float(roc_auc_score(target_valid, probabilities))
    return {
        "f1": float(f1),
        "ap": ap,
        "auc": auc,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }
