from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from xuannv_embedding.downstream.heads import build_head
from xuannv_embedding.downstream.metrics import evaluate_binary
from xuannv_embedding.downstream.protocol import EvaluationProtocol, ProtocolError, SpatialFold


class DownstreamWorkflowError(ValueError):
    """下游数据、checkpoint 或评测身份合同失败。"""


@dataclass(frozen=True)
class DenseDataset:
    embeddings: torch.Tensor
    labels: torch.Tensor
    patch_ids: tuple[str, ...]
    dataset_sha256: str
    label_sha256: str

    @classmethod
    def from_npz(cls, path: str | Path) -> "DenseDataset":
        dataset_path = Path(path)
        try:
            with np.load(dataset_path, allow_pickle=False) as archive:
                embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
                labels = np.asarray(archive["labels"], dtype=np.float32)
                patch_ids = tuple(str(value) for value in archive["patch_ids"].tolist())
        except (OSError, KeyError, ValueError) as exc:
            raise DownstreamWorkflowError(f"无法读取下游 NPZ 数据集: {exc}") from exc
        if embeddings.ndim != 4:
            raise DownstreamWorkflowError("embeddings 必须是 [N,D,H,W]")
        if labels.ndim != 3 or labels.shape != (
            embeddings.shape[0],
            embeddings.shape[2],
            embeddings.shape[3],
        ):
            raise DownstreamWorkflowError("labels 必须是与 embeddings 对齐的 [N,H,W]")
        if len(patch_ids) != embeddings.shape[0] or len(set(patch_ids)) != len(patch_ids):
            raise DownstreamWorkflowError("patch_ids 必须与样本一一对应且唯一")
        if not np.isfinite(embeddings).all() or not np.isfinite(labels).all():
            raise DownstreamWorkflowError("embedding 或 label 包含 NaN/Inf")
        label_digest = hashlib.sha256()
        for patch_id, label in zip(patch_ids, labels):
            label_digest.update(patch_id.encode("utf-8"))
            label_digest.update(b"\0")
            label_digest.update(np.ascontiguousarray(label).tobytes())
        return cls(
            embeddings=torch.from_numpy(embeddings.copy()),
            labels=torch.from_numpy(labels.copy()),
            patch_ids=patch_ids,
            dataset_sha256=_sha256_file(dataset_path),
            label_sha256=label_digest.hexdigest(),
        )

    def subset(self, patch_ids: tuple[str, ...]) -> TensorDataset:
        index = {patch_id: position for position, patch_id in enumerate(self.patch_ids)}
        missing = sorted(set(patch_ids) - set(index))
        if missing:
            raise DownstreamWorkflowError(f"数据集缺少 fold patch_id: {missing}")
        positions = torch.tensor([index[patch_id] for patch_id in patch_ids], dtype=torch.long)
        return TensorDataset(
            self.embeddings.index_select(0, positions),
            self.labels.index_select(0, positions),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_sha256(fold: SpatialFold) -> str:
    payload = json.dumps(
        {
            "fold": fold.fold,
            "train": fold.train,
            "validation": fold.validation,
            "test": fold.test,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _shot(value: str) -> int | None:
    return None if value == "full" else int(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _collect_logits(
    model: torch.nn.Module,
    dataset: TensorDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for embedding, label in loader:
            prediction = model(embedding.to(device))[:, 0]
            logits.append(prediction.cpu())
            labels.append(label.cpu())
    return torch.cat(logits), torch.cat(labels)


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def train_downstream(args: Namespace) -> dict[str, Any]:
    """训练标准二分类头，阈值只从 validation split 得到。"""
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise DownstreamWorkflowError("epochs、batch_size 与 lr 必须大于 0")
    _seed_everything(args.seed)
    dataset = DenseDataset.from_npz(args.dataset)
    fold = SpatialFold.from_file(args.folds, fold=args.fold)
    protocol = EvaluationProtocol(fold=fold, shot=_shot(args.shot), seed=args.seed)
    train_ids = protocol.training_patch_ids()
    train_data = dataset.subset(train_ids)
    validation_data = dataset.subset(fold.validation)
    device = torch.device(args.device)
    model = build_head(
        args.head,
        embed_dim=int(dataset.embeddings.shape[1]),
        num_classes=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model.train()
    for _ in range(args.epochs):
        for embedding, label in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(embedding.to(device))[:, 0]
            loss = F.binary_cross_entropy_with_logits(logits, label.to(device))
            if not torch.isfinite(loss):
                raise DownstreamWorkflowError("下游训练 loss 出现 NaN/Inf")
            loss.backward()
            optimizer.step()

    validation_logits, validation_labels = _collect_logits(
        model,
        validation_data,
        batch_size=args.batch_size,
        device=device,
    )
    threshold = choose_validation_threshold_checked(validation_logits, validation_labels)
    state = {
        "format_version": "downstream-1",
        "head": args.head,
        "embed_dim": int(dataset.embeddings.shape[1]),
        "num_classes": 1,
        "model": model.state_dict(),
        "fold": fold.fold,
        "fold_sha256": _fold_sha256(fold),
        "shot": protocol.shot,
        "report_scope": protocol.report_scope,
        "seed": args.seed,
        "dataset_sha256": dataset.dataset_sha256,
        "label_sha256": dataset.label_sha256,
        "validation_threshold": threshold,
        "threshold_source": "validation",
    }
    _atomic_torch_save(state, args.output)
    summary = {key: value for key, value in state.items() if key != "model"}
    _atomic_json(summary, args.output.with_suffix(args.output.suffix + ".train.json"))
    return summary


def choose_validation_threshold_checked(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    from xuannv_embedding.downstream.protocol import choose_validation_threshold

    try:
        return choose_validation_threshold(logits, labels)
    except ProtocolError as exc:
        raise DownstreamWorkflowError(str(exc)) from exc


def evaluate_downstream(args: Namespace) -> dict[str, Any]:
    """用训练 checkpoint 中固定的 validation 阈值评测 test split。"""
    dataset = DenseDataset.from_npz(args.dataset)
    fold = SpatialFold.from_file(args.folds, fold=args.fold)
    try:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as exc:
        raise DownstreamWorkflowError(f"无法读取下游 checkpoint: {exc}") from exc
    required = {
        "format_version",
        "head",
        "embed_dim",
        "num_classes",
        "model",
        "fold",
        "fold_sha256",
        "shot",
        "report_scope",
        "seed",
        "dataset_sha256",
        "label_sha256",
        "validation_threshold",
        "threshold_source",
    }
    if not isinstance(state, dict) or required - set(state):
        raise DownstreamWorkflowError("下游 checkpoint 缺少严格评测元数据")
    if state["format_version"] != "downstream-1":
        raise DownstreamWorkflowError("下游 checkpoint format_version 不兼容")
    identity_checks = {
        "fold": (state["fold"], fold.fold),
        "fold_sha256": (state["fold_sha256"], _fold_sha256(fold)),
        "dataset_sha256": (state["dataset_sha256"], dataset.dataset_sha256),
        "label_sha256": (state["label_sha256"], dataset.label_sha256),
        "threshold_source": (state["threshold_source"], "validation"),
    }
    mismatches = [
        name for name, (actual, expected) in identity_checks.items() if actual != expected
    ]
    if mismatches:
        raise DownstreamWorkflowError(f"评测协议身份不一致: {', '.join(mismatches)}")
    device = torch.device(args.device)
    model = build_head(
        state["head"],
        embed_dim=int(state["embed_dim"]),
        num_classes=int(state["num_classes"]),
    ).to(device)
    try:
        model.load_state_dict(state["model"], strict=True)
    except RuntimeError as exc:
        raise DownstreamWorkflowError(f"下游 head 严格加载失败: {exc}") from exc
    logits, labels = _collect_logits(
        model,
        dataset.subset(fold.test),
        batch_size=args.batch_size,
        device=device,
    )
    metrics = evaluate_binary(logits, labels, threshold=float(state["validation_threshold"]))
    report = {
        **metrics,
        "fold": fold.fold,
        "report_scope": state["report_scope"],
        "shot": state["shot"],
        "threshold_source": "validation",
        "label_sha256": dataset.label_sha256,
        "fold_sha256": state["fold_sha256"],
    }
    _atomic_json(report, args.output)
    return report
