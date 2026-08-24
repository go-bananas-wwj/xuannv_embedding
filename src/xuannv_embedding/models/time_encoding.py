from __future__ import annotations

import math

import torch
from torch import nn

# 时间/窗口编码模块：将时间戳或窗口标识映射为可学习的 embedding。


class TimeEncoding(nn.Module):
    """基于正弦/余弦位置编码 + 可学习投影的时间编码器。

    支持输入为毫秒时间戳或月份索引等标量时间信息，输出与模型维度一致的
    时间 embedding，便于注入到时序特征中。
    """

    def __init__(self, embed_dim: int, max_periods: int = 64) -> None:
        """初始化 TimeEncoding。

        Args:
            embed_dim: 输出 embedding 维度，与模型主维度一致。
            max_periods: 正弦/余弦编码的最大周期基数，决定频率范围。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_periods = max_periods
        # 使用偶数维度的正弦/余弦基，避免奇数维度时 sin/cos 数量不一致。
        self.sinusoidal_dim = (embed_dim // 2) * 2

        # 可学习投影：将正弦/余弦基映射到目标维度。
        self.proj = nn.Linear(self.sinusoidal_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """将时间戳编码为 embedding。

        Args:
            timestamps: 时间戳张量，形状为 (B, T)，可以是毫秒或月份索引。

        Returns:
            时间 embedding，形状为 (B, T, embed_dim)。
        """
        batch_size, time_steps = timestamps.shape
        # 将时间戳展开到 (B, T, 1) 以与频率项广播。
        position = timestamps.float().unsqueeze(-1)

        # 计算正弦/余弦频率项：1 / max_periods^(2i / sinusoidal_dim)。
        div_term = torch.exp(
            torch.arange(0, self.sinusoidal_dim, 2, device=timestamps.device, dtype=torch.float32)
            * (-math.log(self.max_periods) / self.sinusoidal_dim)
        )

        # angle: (B, T, sinusoidal_dim // 2)
        angle = position * div_term

        # 构建正弦/余弦位置编码。
        pe = torch.zeros(
            batch_size,
            time_steps,
            self.sinusoidal_dim,
            device=timestamps.device,
            dtype=torch.float32,
        )
        pe[:, :, 0::2] = torch.sin(angle)
        pe[:, :, 1::2] = torch.cos(angle)

        # 可学习投影 + LayerNorm。
        pe = self.proj(pe)
        pe = self.norm(pe)
        return pe


class WindowCode(nn.Module):
    """时间窗口编码器：将离散的窗口 ID 映射为可学习 embedding。"""

    def __init__(self, num_windows: int, embed_dim: int) -> None:
        """初始化 WindowCode。

        Args:
            num_windows: 窗口类别数。
            embed_dim: 每个窗口的 embedding 维度。
        """
        super().__init__()
        self.num_windows = num_windows
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_windows, embed_dim)

    def forward(self, window_ids: torch.Tensor) -> torch.Tensor:
        """查找窗口 embedding。

        Args:
            window_ids: 窗口 ID，形状为任意整数张量。

        Returns:
            窗口 embedding，形状为 (*window_ids.shape, embed_dim)。
        """
        return self.embedding(window_ids)
