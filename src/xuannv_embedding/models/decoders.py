from __future__ import annotations

# 连续/分类 Decoder 模块。
import torch
import torch.nn as nn


class ContinuousDecoder(nn.Module):
    """连续值解码器，将 VMF 嵌入映射到连续输出（如归一化后的遥感反射率）。

    结构为逐像素 1x1 卷积 MLP，不引入额外空间上下文。
    在月度模型中，调用方会将 ``(B, T, D, H, W)`` 折叠为 ``(B*T, D, H, W)``
    后传入，本模块本身不处理时间维度。
    """

    def __init__(self, embed_dim: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 逐像素特征变换。
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            # 输出目标通道数。
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """输入嵌入 (B, embed_dim, H, W)，输出 (B, out_channels, H, W)。"""
        return self.net(emb)


class CategoricalDecoder(nn.Module):
    """分类解码器，将 VMF 嵌入映射到类别 logits。

    通过 1x1 卷积在每个空间位置独立预测类别分数。
    在月度模型中，调用方会将 ``(B, T, D, H, W)`` 折叠为 ``(B*T, D, H, W)``
    后传入，本模块本身不处理时间维度。
    """

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """输入嵌入 (B, embed_dim, H, W)，输出 (B, num_classes, H, W)。"""
        return self.conv(emb)
