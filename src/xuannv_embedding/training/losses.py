from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "l1",
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算带掩码的重建损失。

    支持非月度输入 ``[B, C, H, W]`` 与月度输入 ``[B, T, C, H, W]``（或 CE 的
    ``[B, T, H, W]`` target）。对于月度输入，时间维度会与 batch 维度合并后计算。

    Args:
        pred: 预测值。L1 时为 ``[B, C, H, W]`` 或 ``[B, T, C, H, W]``；
            CE 时为 logits ``[B, C, H, W]`` 或 ``[B, T, C, H, W]``。
        target: 目标值。L1 时为 ``[B, C, H, W]`` 或 ``[B, T, C, H, W]``；
            CE 时为类别索引 ``[B, H, W]`` 或 ``[B, T, H, W]`` (int64)。
        mask: 空间有效掩码，形状可为 ``[H, W]``、``[B, H, W]``、``[B, 1, H, W]``、
            ``[B, T, H, W]`` 或 ``[B, T, 1, H, W]``。
        loss_type: ``"l1"`` 或 ``"ce"``。
        eps: 防止除零的小常数。

    Returns:
        标量张量，表示掩码平均后的损失。
    """
    is_temporal = pred.dim() == 5

    if is_temporal:
        B, T, C, H, W = pred.shape
        pred = pred.reshape(B * T, C, H, W)
        if target.dim() == 5:
            target = target.reshape(B * T, C, H, W)
        elif target.dim() == 4:
            # CE target: (B, T, H, W)
            target = target.reshape(B * T, H, W)
        if mask.dim() == 5:
            mask = mask.reshape(B * T, *mask.shape[2:])
        elif mask.dim() == 4 and mask.shape[1] == T:
            mask = mask.reshape(B * T, *mask.shape[2:])
        elif mask.dim() == 3:
            if mask.shape == (B, T, 1):
                mask = mask.reshape(B * T, 1, 1).expand(B * T, H, W)
            else:
                # (B, T, H, W) 已在 reshape 分支处理，其它形状保留供 expand_as 处理。
                pass
        elif mask.dim() == 2:
            # (B, T) 时间掩码，应用到所有空间位置。
            mask = mask.reshape(B * T, 1, 1).expand(B * T, H, W)

    if loss_type == "l1":
        # 逐元素 L1，然后在通道维度取平均，得到 [B, H, W]。
        loss = F.l1_loss(pred, target, reduction="none").mean(dim=1)
    elif loss_type == "ce":
        # cross_entropy 输出 [B, H, W]；以 0 作为 nodata/背景类别，不参与损失。
        loss = F.cross_entropy(pred, target, ignore_index=0, reduction="none")
        # 即使外部 mask 未显式屏蔽 class-0 像素，也确保其不进入平均 denominator。
        mask = mask * (target != 0).float()
    else:
        raise ValueError(f"不支持的 loss_type: {loss_type!r}，仅支持 'l1' 或 'ce'")

    # 将 mask 广播到 [B, H, W] 后应用。
    mask = mask.expand_as(loss)
    masked_sum = (loss * mask).sum()
    masked_count = mask.sum()
    return masked_sum / (masked_count + eps)


def batch_uniformity_loss(emb: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """计算 batch 内场景级嵌入的均匀性损失。

    先将每个嵌入 L2 归一化到单位球面，再计算 Wang-Isola 风格的
    ``log(mean(exp(-temperature * pairwise_squared_distance)))``。该值在嵌入
    更分散时更小，因此可以用正权重直接加到总损失里进行最小化。

    Args:
        emb: 场景级嵌入，形状 ``[B, D]`` 或月度 ``[B, T, D]``。
        temperature: 距离温度，值越大越强调近邻排斥。

    Returns:
        标量张量，表示均匀性损失。
    """
    # 月度输出合并为 (B*T, D)。
    if emb.dim() == 3:
        emb = emb.reshape(-1, emb.shape[-1])

    # L2 归一化，避免除零。
    emb = F.normalize(emb, p=2, dim=1)
    batch_size = emb.shape[0]

    # pairwise squared distance = ||u_i - u_j||^2 = 2 - 2 * u_i @ u_j。
    similarity = emb @ emb.t()  # [B, B]
    squared_dist = 2.0 - 2.0 * similarity

    # 排除对角线。
    off_diag_count = batch_size * (batch_size - 1)
    if off_diag_count == 0:
        return torch.tensor(0.0, device=emb.device, dtype=emb.dtype)

    diag_mask = ~torch.eye(batch_size, device=emb.device, dtype=torch.bool)
    off_diag_dist = squared_dist[diag_mask]
    return torch.log(torch.exp(-temperature * off_diag_dist).mean())


class SemanticProbeLoss(nn.Module):
    """Training-only semantic probes that make embedding maps directly decodable.

    Each task is a tiny probe applied to one monthly embedding map. By default
    this is a 1x1 MLP. When ``hidden_dim <= 0`` it becomes a pure 1x1 linear
    probe, which is useful when we want to force linearly readable embeddings.
    The probes are optimized during embedding training and discarded after
    training.
    """

    def __init__(
        self,
        embed_dim: int,
        tasks: list[str] | tuple[str, ...],
        hidden_dim: int = 64,
        task_weights: dict[str, float] | None = None,
        pos_weight: float = 1.0,
        pos_weights: dict[str, float] | None = None,
        month_index: int = -1,
        hard_negative_ratio: float = 0.0,
        hard_negative_weight: float = 0.0,
        hard_negative_warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.tasks = tuple(tasks)
        self.task_weights = dict(task_weights or {})
        self.pos_weight = float(pos_weight)
        self.pos_weights = dict(pos_weights or {})
        self.month_index = int(month_index)
        self.hard_negative_ratio = max(0.0, float(hard_negative_ratio))
        self.hard_negative_weight = max(0.0, float(hard_negative_weight))
        self.hard_negative_warmup_epochs = max(0, int(hard_negative_warmup_epochs))
        self.current_epoch = 0
        hidden_dim = int(hidden_dim)
        modules: dict[str, nn.Module] = {}
        for task in self.tasks:
            if hidden_dim <= 0:
                modules[task] = nn.Conv2d(embed_dim, 1, kernel_size=1)
            else:
                modules[task] = nn.Sequential(
                    nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, 1, kernel_size=1),
                )
        self.probes = nn.ModuleDict(modules)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _current_hard_negative_weight(self) -> float:
        if self.hard_negative_weight == 0.0:
            return 0.0
        if self.hard_negative_warmup_epochs <= 0:
            return self.hard_negative_weight
        progress = min(
            1.0,
            float(self.current_epoch + 1) / self.hard_negative_warmup_epochs,
        )
        return self.hard_negative_weight * progress

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def _dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs * mask
        target = target * mask
        intersection = (probs * target).sum()
        union = probs.sum() + target.sum()
        return 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

    def _hard_negative_loss(
        self,
        bce_map: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.hard_negative_ratio <= 0.0 or self.hard_negative_weight <= 0.0:
            return bce_map.sum() * 0.0
        negative_mask = (target < 0.5) & (mask > 0.0)
        values = bce_map[negative_mask]
        if values.numel() == 0:
            return bce_map.sum() * 0.0
        k = max(1, int(values.numel() * self.hard_negative_ratio))
        k = min(k, values.numel())
        return values.topk(k).values.mean()

    def forward(
        self,
        embedding_map: torch.Tensor,
        labels: dict[str, torch.Tensor] | None,
        label_masks: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = embedding_map.sum() * 0.0
        if not self.tasks or labels is None:
            return zero, {
                "semantic_probe_positive_pixels": zero.detach(),
                "semantic_probe_valid_pixels": zero.detach(),
            }

        emb = embedding_map[:, self.month_index]
        total = zero
        task_weight_sum = zero
        total_positive = zero
        total_valid = zero
        stats: dict[str, torch.Tensor] = {}
        hard_negative_weight = self._current_hard_negative_weight()

        for task in self.tasks:
            if task not in labels or task not in self.probes:
                continue
            label = labels[task].to(device=emb.device, dtype=emb.dtype)
            if label.dim() != 3:
                raise ValueError(
                    f"semantic label {task!r} 形状应为 [B,H,W]，实际为 {tuple(label.shape)}"
                )
            label = label[:, None]
            if label.shape[-2:] != emb.shape[-2:]:
                label = F.interpolate(label, size=emb.shape[-2:], mode="nearest")
            label = (label > 0.5).to(dtype=emb.dtype)

            sample_mask = None
            if label_masks is not None and task in label_masks:
                sample_mask = label_masks[task].to(device=emb.device, dtype=emb.dtype)
            if sample_mask is None:
                sample_mask = torch.ones((emb.shape[0],), device=emb.device, dtype=emb.dtype)
            valid = sample_mask[:, None, None, None].expand_as(label)
            if bool((valid.sum() <= 0).item()):
                stats[f"semantic_probe_{task}_loss"] = zero.detach()
                stats[f"semantic_probe_{task}_positive_pixels"] = zero.detach()
                stats[f"semantic_probe_{task}_valid_pixels"] = zero.detach()
                continue

            logits = self.probes[task](emb)
            pw = float(self.pos_weights.get(task, self.pos_weight))
            bce_map = F.binary_cross_entropy_with_logits(
                logits,
                label,
                pos_weight=torch.tensor(pw, device=emb.device, dtype=emb.dtype),
                reduction="none",
            )
            bce = self._masked_mean(bce_map, valid)
            dice = self._dice_loss(logits, label, valid)
            hard_negative = self._hard_negative_loss(bce_map, label, valid)
            task_loss = bce + dice + hard_negative_weight * hard_negative
            weight = torch.tensor(
                float(self.task_weights.get(task, 1.0)),
                device=emb.device,
                dtype=emb.dtype,
            )
            total = total + weight * task_loss
            task_weight_sum = task_weight_sum + weight
            positive_pixels = (label * valid).sum()
            valid_pixels = valid.sum()
            total_positive = total_positive + positive_pixels
            total_valid = total_valid + valid_pixels
            stats[f"semantic_probe_{task}_loss"] = task_loss.detach()
            stats[f"semantic_probe_{task}_hard_negative"] = hard_negative.detach()
            stats[f"semantic_probe_{task}_hard_negative_weight"] = torch.tensor(
                hard_negative_weight,
                device=emb.device,
                dtype=emb.dtype,
            ).detach()
            stats[f"semantic_probe_{task}_positive_pixels"] = positive_pixels.detach()
            stats[f"semantic_probe_{task}_valid_pixels"] = valid_pixels.detach()

        loss = total / task_weight_sum.clamp(min=1.0)
        stats["semantic_probe_positive_pixels"] = total_positive.detach()
        stats["semantic_probe_valid_pixels"] = total_valid.detach()
        return loss, stats


class TotalLoss(nn.Module):
    """P10C 总损失：重建、uniformity 与区域无关 OSM 弱语义 probe。"""

    def __init__(
        self,
        target_cfg: dict[str, dict],
        uniformity_weight: float = 1.0,
        uniformity_warmup_epochs: int = 0,
        uniformity_temperature: float = 2.0,
        semantic_probe_embed_dim: int | None = None,
        semantic_probe_weight: float = 0.0,
        semantic_probe_warmup_epochs: int = 0,
        semantic_probe_tasks: list[str] | tuple[str, ...] = (),
        semantic_probe_task_weights: dict[str, float] | None = None,
        semantic_probe_pos_weight: float = 1.0,
        semantic_probe_pos_weights: dict[str, float] | None = None,
        semantic_probe_hidden_dim: int = 64,
        semantic_probe_hard_negative_ratio: float = 0.0,
        semantic_probe_hard_negative_weight: float = 0.0,
        semantic_probe_hard_negative_warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.target_cfg = target_cfg
        self.uniformity_weight = float(uniformity_weight)
        self.uniformity_warmup_epochs = int(uniformity_warmup_epochs)
        self.uniformity_temperature = float(uniformity_temperature)
        self.semantic_probe_weight = float(semantic_probe_weight)
        self.semantic_probe_warmup_epochs = int(semantic_probe_warmup_epochs)
        self.semantic_probe_tasks = tuple(semantic_probe_tasks)
        if self.semantic_probe_tasks and semantic_probe_embed_dim is None:
            raise ValueError(
                "semantic_probe_embed_dim is required when semantic_probe_tasks is not empty"
            )
        self.semantic_probe = (
            SemanticProbeLoss(
                embed_dim=int(semantic_probe_embed_dim or 1),
                tasks=self.semantic_probe_tasks,
                hidden_dim=int(semantic_probe_hidden_dim),
                task_weights=semantic_probe_task_weights,
                pos_weight=semantic_probe_pos_weight,
                pos_weights=semantic_probe_pos_weights,
                hard_negative_ratio=semantic_probe_hard_negative_ratio,
                hard_negative_weight=semantic_probe_hard_negative_weight,
                hard_negative_warmup_epochs=semantic_probe_hard_negative_warmup_epochs,
            )
            if self.semantic_probe_tasks
            else None
        )
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，用于两个 warmup。"""
        self.current_epoch = int(epoch)
        if self.semantic_probe is not None:
            self.semantic_probe.set_epoch(epoch)

    @staticmethod
    def _warmup_weight(weight: float, warmup_epochs: int, epoch: int) -> float:
        if weight == 0.0:
            return 0.0
        if warmup_epochs <= 0:
            return weight
        progress = min(1.0, float(epoch + 1) / warmup_epochs)
        return weight * progress

    def forward(
        self,
        output,
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        supervised_labels: dict[str, torch.Tensor] | None = None,
        supervised_label_masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        embedding_map = output.embedding_map
        embedding = F.normalize(embedding_map.mean(dim=[3, 4]), p=2, dim=-1)
        total_recon = embedding_map.sum() * 0.0
        result: dict[str, torch.Tensor] = {}

        for name, cfg in self.target_cfg.items():
            try:
                pred = output.reconstructions[name]
                target = targets[name]
                mask = masks[name]
            except KeyError as exc:
                raise KeyError(f"重建目标 {name!r} 缺少 prediction、target 或 mask") from exc
            loss = reconstruction_loss(
                pred,
                target,
                mask,
                loss_type=cfg["loss_type"],
            )
            total_recon = total_recon + float(cfg["weight"]) * loss
            result[f"recon_{name}"] = loss

        uniformity = batch_uniformity_loss(
            embedding,
            temperature=self.uniformity_temperature,
        )
        uniformity_weight = self._warmup_weight(
            self.uniformity_weight,
            self.uniformity_warmup_epochs,
            self.current_epoch,
        )
        weighted_uniformity = uniformity * uniformity_weight

        if self.semantic_probe is None:
            semantic_probe = embedding_map.sum() * 0.0
            semantic_stats = {
                "semantic_probe_positive_pixels": semantic_probe.detach(),
                "semantic_probe_valid_pixels": semantic_probe.detach(),
            }
        else:
            semantic_probe, semantic_stats = self.semantic_probe(
                embedding_map,
                supervised_labels,
                supervised_label_masks,
            )
        semantic_weight = self._warmup_weight(
            self.semantic_probe_weight,
            self.semantic_probe_warmup_epochs,
            self.current_epoch,
        )
        weighted_semantic = semantic_probe * semantic_weight
        total = total_recon + weighted_uniformity + weighted_semantic

        result.update(
            {
                "total": total,
                "recon": total_recon,
                "uniformity": uniformity,
                "uniformity_weighted": weighted_uniformity,
                "uniformity_weight": embedding_map.new_tensor(uniformity_weight),
                "semantic_probe": semantic_probe,
                "semantic_probe_weighted": weighted_semantic,
                "semantic_probe_weight": embedding_map.new_tensor(semantic_weight),
            }
        )
        for name, value in semantic_stats.items():
            if name.startswith("semantic_probe_"):
                result[name] = value
        return result
