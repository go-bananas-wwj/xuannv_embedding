from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# 空间/时间 Transformer 算子：在 patch 序列或时间序列上做自注意力。


class SinusoidalTimeEncoding(nn.Module):
    """正弦/余弦时间编码。

    将标量时间戳（毫秒、天或任意连续单位）编码为 ``dim`` 维正弦嵌入，
    用于 ``STPTimeOperator`` 提供时间先验。
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        """初始化 SinusoidalTimeEncoding。

        Args:
            dim: 输出编码维度。
            max_period: 正弦周期上限，默认 10000。
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """对时间戳进行正弦编码。

        Args:
            timestamps: 形状为 ``(B, T)`` 或 ``(B,)`` 的标量时间戳。

        Returns:
            编码结果，形状为 ``(B, T, dim)``；当输入为 ``(B,)`` 且内部 T=1 时，
            返回 ``(B, dim)``。
        """
        half_dim = self.dim // 2
        # 频率序列：(half_dim,)
        freq = torch.exp(
            torch.arange(half_dim, device=timestamps.device, dtype=torch.float32)
            * (math.log(self.max_period) / max(half_dim - 1, 1))
        )

        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(1)  # (B, 1)

        t = timestamps.unsqueeze(-1).float()  # (B, T, 1)
        f = freq.view(1, 1, -1)  # (1, 1, half_dim)
        sin_emb = torch.sin(t * f)
        cos_emb = torch.cos(t * f)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)  # (B, T, dim)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        if timestamps.shape[1] == 1:
            emb = emb.squeeze(1)  # (B, dim)

        return emb


class STPSpaceOperator(nn.Module):
    """STP 空间算子（1/16L 分辨率）。

    将每帧的 ``H*W`` 视为序列长度，做预归一化多头自注意力 + MLP，
    用于捕获全局空间依赖。
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """初始化 STPSpaceOperator。

        Args:
            dim: 输入与输出通道数。
            num_heads: 注意力头数，必须能整除 ``dim``。
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 ``(B, T, H, W, C)``。

        Returns:
            输出张量，形状 ``(B, T, H, W, C)``。
        """
        B, T, H, W, C = x.shape
        x_flat = x.reshape(B * T, H * W, C)
        residual = x_flat

        x_norm = self.norm1(x_flat)
        qkv = self.qkv(x_norm).view(B * T, H * W, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (BT, heads, HW, d)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)  # (BT, heads, HW, d)
        out = out.permute(0, 2, 1, 3).reshape(B * T, H * W, C)
        x_flat = residual + self.proj(out)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        return x_flat.view(B, T, H, W, C)


class STPTimeOperator(nn.Module):
    """STP 时间算子（1/8L 分辨率）。

    为每个空间位置独立做时间自注意力，并加入正弦时间编码。
    支持通过 ``mask`` 屏蔽无效时间步。
    """

    def __init__(self, dim: int, num_heads: int = 8, attention_mode: str = "full") -> None:
        """初始化 STPTimeOperator。

        Args:
            dim: 输入与输出通道数。
            num_heads: 注意力头数，必须能整除 ``dim``。
            attention_mode: ``"full"`` 表示跨月自注意力，``"none"`` 表示只保留
                逐月时间编码与逐 token MLP，不在月份之间交换信息。
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除")
        if attention_mode not in {"full", "none"}:
            raise ValueError(
                "STPTimeOperator attention_mode 仅支持 'full' 或 'none'，"
                f"实际为 {attention_mode!r}"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention_mode = attention_mode

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3) if attention_mode == "full" else None
        self.proj = nn.Linear(dim, dim) if attention_mode == "full" else None
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.time_encoding = SinusoidalTimeEncoding(dim)

    def _pad_or_trim(self, tensor: torch.Tensor, target_len: int, dim: int = 1) -> torch.Tensor:
        """将张量沿指定维度补齐或截断到目标长度（复制最后一个值）。"""
        cur_len = tensor.shape[dim]
        if cur_len == target_len:
            return tensor
        if cur_len > target_len:
            slices = [slice(None)] * tensor.dim()
            slices[dim] = slice(None, target_len)
            return tensor[tuple(slices)]
        repeat_times = target_len - cur_len
        last = tensor.index_select(dim, torch.tensor([cur_len - 1], device=tensor.device))
        repeats = [1] * tensor.dim()
        repeats[dim] = repeat_times
        padding = last.repeat(*repeats)
        return torch.cat([tensor, padding], dim=dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 ``(B, T, H, W, C)``。
            timestamps: 时间戳，形状 ``(B, T)`` 或 ``(B,)``。
            mask: 可选时间有效掩码，形状 ``(B, T)``。

        Returns:
            输出张量，形状 ``(B, T, H, W, C)``。
        """
        B, T, H, W, C = x.shape
        if timestamps.dim() == 1:
            timestamps = timestamps.view(B, T)
        timestamps = self._pad_or_trim(timestamps, T, dim=1)

        time_enc = self.time_encoding(timestamps)  # (B, T, C) 或 (B, C)
        if time_enc.dim() == 2:
            time_enc = time_enc.unsqueeze(1).expand(-1, T, -1)
        else:
            time_enc = self._pad_or_trim(time_enc, T, dim=1)
        time_enc = time_enc.unsqueeze(2).unsqueeze(3)
        x = x + time_enc

        x_flat = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        residual = x_flat

        if self.attention_mode == "none":
            x_flat = residual + self.mlp(self.norm2(x_flat))
            return x_flat.view(B, H, W, T, C).permute(0, 3, 1, 2, 4)

        x_norm = self.norm1(x_flat)
        if self.qkv is None or self.proj is None:
            raise RuntimeError("full attention 模式缺少 qkv/proj 参数")
        qkv = self.qkv(x_norm).view(B * H * W, T, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (BHW, heads, T, d)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        if mask is not None:
            # logits 形状为 (BHW, heads, T_query, T_key)，掩码作用在 key 维度上。
            mask_flat = mask.view(B, 1, T).expand(-1, H * W, -1)
            mask_flat = mask_flat.reshape(B * H * W, 1, T).unsqueeze(2)  # (BHW, 1, 1, T)
            logits = logits.masked_fill(mask_flat == 0, float("-inf"))
            # 防止全被掩码的样本产生 NaN：将这些行全部置 0 后再 softmax。
            row_sum = mask.sum(dim=-1)  # (B,)
            empty_row = (row_sum == 0).view(B, 1, 1, 1).expand(B, H * W, self.num_heads, T)
            empty_row = empty_row.reshape(B * H * W, self.num_heads, 1, T)
            logits = torch.where(empty_row, torch.zeros_like(logits), logits)
        attn = F.softmax(logits, dim=-1)

        if mask is not None:
            row_valid = mask.sum(dim=-1) > 0  # (B,)
            row_valid = row_valid.view(B, 1).expand(B, H * W).reshape(B * H * W, 1, 1)
            attn = attn * row_valid.unsqueeze(-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B * H * W, T, C)
        x_flat = residual + self.proj(out)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        return x_flat.view(B, H, W, T, C).permute(0, 3, 1, 2, 4)


class STPPrecisionOperator(nn.Module):
    """STP 精度算子（1/2L 分辨率）。

    使用两组 3x3 卷积 + GroupNorm + GELU 在局部空间上做精化，
    保留高分辨率细节。
    """

    def __init__(self, dim: int) -> None:
        """初始化 STPPrecisionOperator。

        Args:
            dim: 输入与输出通道数。
        """
        super().__init__()
        self.dim = dim
        num_groups1 = 8 if dim % 8 == 0 else dim
        num_groups2 = 8 if (dim * 4) % 8 == 0 else dim * 4
        self.norm1 = nn.GroupNorm(num_groups1, dim)
        self.norm2 = nn.GroupNorm(num_groups2, dim * 4)
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 ``(B, T, H, W, C)``。

        Returns:
            输出张量，形状 ``(B, T, H, W, C)``。
        """
        B, T, H, W, C = x.shape
        x_conv = x.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        residual = x_conv

        x_conv = self.conv1(self.norm1(x_conv))
        x_conv = F.gelu(x_conv)
        x_conv = self.conv2(self.norm2(x_conv))
        x_conv = residual + x_conv

        return x_conv.view(B, T, C, H, W).permute(0, 1, 3, 4, 2)


class LearnedSpatialResampling(nn.Module):
    """可学习的空间重采样层。

    先用无参数重采样对齐空间尺寸，再用小卷积做通道投影。早期实现用
    ``ConvTranspose2d`` 和大 stride/downsample kernel，在 Ascend NPU 反传时容易
    触发 ``Conv2DBackpropInput`` L1 tiling 限制，因此改为避免转置卷积和大卷积核。
    该结构现已固化进已登记权重，无论目标加速器为何都不得改动：改变它会破坏 431 键
    旧 checkpoint 兼容映射与 embedding 逐元素一致性门禁。
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: float) -> None:
        """初始化 LearnedSpatialResampling。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            scale_factor: 空间缩放因子；大于 1 为上采样，小于 1 为下采样。
        """
        super().__init__()
        self.scale_factor = scale_factor
        groups = 8 if out_channels % 8 == 0 else 1
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int] | None = None) -> torch.Tensor:
        """可学习重采样。

        Args:
            x: 输入张量，形状 ``(N, C, H, W)``。
            target_size: 可选的目标空间尺寸 ``(H_target, W_target)``；
                当卷积输出尺寸不一致时，使用双线性插值兜底。

        Returns:
            输出张量，形状 ``(N, out_channels, H_target, W_target)``。
        """
        if target_size is None:
            target_size = (
                max(1, int(round(x.shape[2] * self.scale_factor))),
                max(1, int(round(x.shape[3] * self.scale_factor))),
            )
        if x.shape[2:] != target_size:
            if target_size[0] < x.shape[2] or target_size[1] < x.shape[3]:
                out = F.adaptive_avg_pool2d(x, target_size)
            else:
                out = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        else:
            out = x
        return self.proj(out)


class MultiResolutionSTPBlock(nn.Module):
    """多分辨率 STP 块。

    在 1/16L 空间路径、1/8L 时间路径、可配置精度路径上分别执行对应算子，
    并通过六个跨尺度交换分支实现信息融合。
    """

    def __init__(
        self,
        space_dim: int,
        time_dim: int,
        precision_dim: int,
        num_heads: int = 8,
        time_attention_mode: str = "full",
        precision_scale: int = 2,
    ) -> None:
        """初始化 MultiResolutionSTPBlock。

        Args:
            space_dim: 空间路径通道数。
            time_dim: 时间路径通道数。
            precision_dim: 精度路径通道数。
            num_heads: 注意力算子头数。
            precision_scale: 精度路径相对输入的下采样倍数，1 表示保持原始空间分辨率。
        """
        super().__init__()
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.precision_dim = precision_dim
        if precision_scale not in {1, 2}:
            raise ValueError(f"precision_scale 仅支持 1 或 2，实际为 {precision_scale}")
        self.precision_scale = int(precision_scale)

        self.space_op = STPSpaceOperator(space_dim, num_heads)
        self.time_op = STPTimeOperator(time_dim, num_heads, attention_mode=time_attention_mode)
        self.precision_op = STPPrecisionOperator(precision_dim)

        self.space_to_time = LearnedSpatialResampling(space_dim, time_dim, 2.0)
        self.space_to_precision = LearnedSpatialResampling(
            space_dim, precision_dim, 16.0 / self.precision_scale
        )
        self.time_to_space = LearnedSpatialResampling(time_dim, space_dim, 0.5)
        self.time_to_precision = LearnedSpatialResampling(
            time_dim, precision_dim, 8.0 / self.precision_scale
        )
        self.precision_to_space = LearnedSpatialResampling(
            precision_dim, space_dim, self.precision_scale / 16.0
        )
        self.precision_to_time = LearnedSpatialResampling(
            precision_dim, time_dim, self.precision_scale / 8.0
        )

    def forward(
        self,
        space_x: torch.Tensor,
        time_x: torch.Tensor,
        precision_x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            space_x: 空间路径输入，形状 ``(B, T, Hs, Ws, space_dim)``。
            time_x: 时间路径输入，形状 ``(B, T, Ht, Wt, time_dim)``。
            precision_x: 精度路径输入，形状 ``(B, T, Hp, Wp, precision_dim)``。
            timestamps: 时间戳，形状 ``(B, T)``。
            mask: 可选时间有效掩码，形状 ``(B, T)``。

        Returns:
            三个路径的输出，形状分别与输入一致。
        """
        space_out = self.space_op(space_x)
        time_out = self.time_op(time_x, timestamps, mask=mask)
        precision_out = self.precision_op(precision_x)

        B, T = space_out.shape[:2]
        space_H, space_W = space_out.shape[2:4]
        time_H, time_W = time_out.shape[2:4]
        precision_H, precision_W = precision_out.shape[2:4]

        space_2d = space_out.permute(0, 1, 4, 2, 3).reshape(B * T, self.space_dim, space_H, space_W)
        time_2d = time_out.permute(0, 1, 4, 2, 3).reshape(B * T, self.time_dim, time_H, time_W)
        precision_2d = precision_out.permute(0, 1, 4, 2, 3).reshape(
            B * T, self.precision_dim, precision_H, precision_W
        )

        time_to_space = self.time_to_space(time_2d, target_size=(space_H, space_W))
        precision_to_space = self.precision_to_space(precision_2d, target_size=(space_H, space_W))
        space_exchange = space_2d + time_to_space + precision_to_space

        space_to_time = self.space_to_time(space_2d, target_size=(time_H, time_W))
        precision_to_time = self.precision_to_time(precision_2d, target_size=(time_H, time_W))
        time_exchange = time_2d + space_to_time + precision_to_time

        space_to_precision = self.space_to_precision(
            space_2d, target_size=(precision_H, precision_W)
        )
        time_to_precision = self.time_to_precision(time_2d, target_size=(precision_H, precision_W))
        precision_exchange = precision_2d + space_to_precision + time_to_precision

        space_out = space_exchange.view(B, T, self.space_dim, space_H, space_W).permute(
            0, 1, 3, 4, 2
        )
        time_out = time_exchange.view(B, T, self.time_dim, time_H, time_W).permute(0, 1, 3, 4, 2)
        precision_out = precision_exchange.view(
            B, T, self.precision_dim, precision_H, precision_W
        ).permute(0, 1, 3, 4, 2)

        return space_out, time_out, precision_out


class STPEncoder(nn.Module):
    """Space-Time-Precision 多分辨率编码器。

    将时序多源特征投影到三个分辨率路径，依次通过若干 ``MultiResolutionSTPBlock``，
    最终将所有路径对齐到精度路径分辨率并相加，返回特征与原始输入空间尺寸。
    """

    # 各路径相对于输入的空间缩放倍数。
    SPACE_SCALE = 16
    TIME_SCALE = 8

    def __init__(
        self,
        input_channels: int,
        space_dim: int = 512,
        time_dim: int = 256,
        precision_dim: int = 128,
        num_blocks: int = 6,
        num_heads: int = 8,
        gradient_checkpointing: bool = False,
        time_attention_mode: str = "full",
        precision_scale: int = 2,
    ) -> None:
        """初始化 STPEncoder。

        Args:
            input_channels: 拼接后的输入通道数。
            space_dim: 空间路径通道数。
            time_dim: 时间路径通道数。
            precision_dim: 精度路径通道数。
            num_blocks: STP 块数量。
            num_heads: 注意力头数。
            gradient_checkpointing: 是否启用梯度检查点以节省显存。
            precision_scale: 精度路径相对输入的下采样倍数；1 表示保持 128x128
                原生网格，2 表示旧版 64x64 精度路径。
        """
        super().__init__()
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.precision_dim = precision_dim
        self.gradient_checkpointing = gradient_checkpointing
        if precision_scale not in {1, 2}:
            raise ValueError(f"precision_scale 仅支持 1 或 2，实际为 {precision_scale}")
        self.precision_scale = int(precision_scale)
        if time_attention_mode not in {"full", "none"}:
            raise ValueError(
                "STPEncoder time_attention_mode 仅支持 'full' 或 'none'，"
                f"实际为 {time_attention_mode!r}"
            )
        self.time_attention_mode = time_attention_mode

        self.input_projection = nn.Linear(input_channels, precision_dim)
        self.space_projection = nn.Linear(precision_dim, space_dim)
        self.time_projection = nn.Linear(precision_dim, time_dim)

        self.blocks = nn.ModuleList(
            [
                MultiResolutionSTPBlock(
                    space_dim,
                    time_dim,
                    precision_dim,
                    num_heads,
                    time_attention_mode=time_attention_mode,
                    precision_scale=self.precision_scale,
                )
                for _ in range(num_blocks)
            ]
        )

        # 最终重采样到精度路径分辨率；仅 2× 上采样是可学习的，其余靠插值兜底。
        self.final_space_resample = LearnedSpatialResampling(
            space_dim, precision_dim, float(self.SPACE_SCALE / self.precision_scale)
        )
        self.final_time_resample = LearnedSpatialResampling(
            time_dim, precision_dim, float(self.TIME_SCALE / self.precision_scale)
        )
        self.norm = nn.LayerNorm(precision_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """前向传播。

        要求输入空间尺寸至少为 ``SPACE_SCALE``（默认 16），否则无法产生
        1/16L 空间路径特征。

        Args:
            x: 输入张量，形状 ``(B, T, H, W, input_channels)``。
            timestamps: 时间戳，形状 ``(B, T)``。
            mask: 可选时间有效掩码，形状 ``(B, T)``。

        Returns:
            (features, input_size) 元组：
            - features: ``(B, T, H//precision_scale, W//precision_scale, precision_dim)``
            - input_size: ``(H, W)``，用于后续上采样对齐。
        """
        B, T, H, W, C = x.shape
        if H < self.SPACE_SCALE or W < self.SPACE_SCALE:
            raise ValueError(
                f"STPEncoder requires input height/width >= {self.SPACE_SCALE}, got {H}x{W}"
            )
        input_size = (H, W)

        x_proj = self.input_projection(x)

        # 空间路径：投影到 space_dim 并下采样到 1/16L
        space_features = self.space_projection(x_proj)
        space_features = space_features.permute(0, 1, 4, 2, 3).reshape(B * T, self.space_dim, H, W)
        space_features = F.adaptive_avg_pool2d(
            space_features, (H // self.SPACE_SCALE, W // self.SPACE_SCALE)
        )
        space_features = space_features.view(
            B, T, self.space_dim, H // self.SPACE_SCALE, W // self.SPACE_SCALE
        ).permute(0, 1, 3, 4, 2)

        # 时间路径：投影到 time_dim 并下采样到 1/8L
        time_features = self.time_projection(x_proj)
        time_features = time_features.permute(0, 1, 4, 2, 3).reshape(B * T, self.time_dim, H, W)
        time_features = F.adaptive_avg_pool2d(
            time_features, (H // self.TIME_SCALE, W // self.TIME_SCALE)
        )
        time_features = time_features.view(
            B, T, self.time_dim, H // self.TIME_SCALE, W // self.TIME_SCALE
        ).permute(0, 1, 3, 4, 2)

        precision_h = H // self.precision_scale
        precision_w = W // self.precision_scale

        # 精度路径：保持 precision_dim，可选保持原始分辨率或下采样到 1/2L。
        precision_features = x_proj.permute(0, 1, 4, 2, 3).reshape(B * T, self.precision_dim, H, W)
        precision_features = F.adaptive_avg_pool2d(precision_features, (precision_h, precision_w))
        precision_features = precision_features.view(
            B, T, self.precision_dim, precision_h, precision_w
        ).permute(0, 1, 3, 4, 2)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                try:
                    space_features, time_features, precision_features = checkpoint.checkpoint(
                        block,
                        space_features,
                        time_features,
                        precision_features,
                        timestamps,
                        mask,
                        use_reentrant=False,
                    )
                except TypeError:
                    # 旧版 PyTorch 不支持 use_reentrant 参数，回退默认行为。
                    space_features, time_features, precision_features = checkpoint.checkpoint(
                        block,
                        space_features,
                        time_features,
                        precision_features,
                        timestamps,
                        mask,
                    )
            else:
                space_features, time_features, precision_features = block(
                    space_features, time_features, precision_features, timestamps, mask=mask
                )

        # 将各路径 reshape 为 (BT, C, H, W) 以便重采样
        space_2d = space_features.permute(0, 1, 4, 2, 3).reshape(
            B * T, self.space_dim, H // self.SPACE_SCALE, W // self.SPACE_SCALE
        )
        time_2d = time_features.permute(0, 1, 4, 2, 3).reshape(
            B * T, self.time_dim, H // self.TIME_SCALE, W // self.TIME_SCALE
        )
        precision_2d = precision_features.permute(0, 1, 4, 2, 3).reshape(
            B * T, self.precision_dim, precision_h, precision_w
        )

        target_size = (precision_h, precision_w)
        space_resampled = self.final_space_resample(space_2d, target_size=target_size)
        time_resampled = self.final_time_resample(time_2d, target_size=target_size)

        final_features = space_resampled + time_resampled + precision_2d
        final_features = final_features.view(
            B, T, self.precision_dim, precision_h, precision_w
        ).permute(0, 1, 3, 4, 2)

        return self.norm(final_features), input_size


class EmbeddingUpsampleHead(nn.Module):
    """嵌入上采样头。

    将低分辨率嵌入特征通过双线性插值上采样回原始输入分辨率，并使用小卷积
    精化。避免 ``ConvTranspose2d``，使 NPU 反传更稳定。
    """

    def __init__(self, in_dim: int, out_dim: int | None = None) -> None:
        """初始化 EmbeddingUpsampleHead。

        Args:
            in_dim: 输入通道数。
            out_dim: 输出通道数，默认等于 ``in_dim``。
        """
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        self.out_dim = out_dim
        num_groups = 8 if out_dim % 8 == 0 else out_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_size: tuple[int, int] | None = None) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 ``(B, H, W, C)``。
            target_size: 可选的目标空间尺寸 ``(H_target, W_target)``。当输入的
                若未提供，默认上采样到 ``(2H, 2W)``。

        Returns:
            输出张量，形状 ``(B, H_target, W_target, out_dim)``（未提供 target_size
            时默认为 ``(2H, 2W)``）。
        """
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        if target_size is None:
            target_size = (H * 2, W * 2)
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.net(x)
        return x.permute(0, 2, 3, 1)


class MonthlyEmbeddingModule(nn.Module):
    """月度嵌入模块。

    将时序特征按 ``YYYYMM`` 时间戳分配到固定月度 bin，对每个 bin 内的有效
    观测做空间位置级别的加权平均；无观测的月份/位置使用可学习 ``missing_token``
    填充，保证输出形状固定且不会出现 NaN。
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_months: int,
        ref_year: int = 2025,
        ref_month: int = 1,
        missing_token_init: float = 0.02,
    ) -> None:
        """初始化 MonthlyEmbeddingModule。

        Args:
            in_channels: 输入特征通道数（STP 精度路径维度）。
            embed_dim: 输出月度嵌入维度。
            num_months: 固定月度 bin 数量。
            ref_year: 月度 bin 起始年份。
            ref_month: 月度 bin 起始月份。
            missing_token_init: ``missing_token`` 的初始值。
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_months = num_months
        self.ref_year = ref_year
        self.ref_month = ref_month

        self.proj = nn.Linear(in_channels, embed_dim)
        self.missing_token = nn.Parameter(torch.full((1, 1, 1, 1, embed_dim), missing_token_init))

    def _yyyymm_to_index(self, timestamps: torch.Tensor) -> torch.Tensor:
        """将 ``YYYYMM`` 整数时间戳映射到以 ``ref_year/ref_month`` 为 0 的月度索引。

        Args:
            timestamps: 形状 ``(B, T)`` 的整数时间戳。

        Returns:
            形状 ``(B, T)`` 的月度索引，越界值为负数或大于等于 ``num_months``。
        """
        years = timestamps // 100
        months = timestamps % 100
        return (years - self.ref_year) * 12 + (months - self.ref_month)

    def forward(
        self,
        feats: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            feats: 输入特征，形状 ``(B, T_obs, H, W, C)``。
            timestamps: 时间戳，形状 ``(B, T_obs)``，应为 ``YYYYMM`` 整数格式。
            mask: 可选时间有效掩码，形状 ``(B, T_obs)``；为 None 时视为全 1。

        Returns:
            ``(monthly_feats, monthly_mask)`` 元组：
            - monthly_feats: ``(B, num_months, H, W, embed_dim)``。
            - monthly_mask: ``(B, num_months)``，1 表示该月份至少有一个有效观测。
        """
        B, T, H, W, C = feats.shape
        M = self.num_months
        device = feats.device

        z = self.proj(feats)  # (B, T, H, W, embed_dim)

        month_index = self._yyyymm_to_index(timestamps)  # (B, T)
        in_range = (month_index >= 0) & (month_index < M)

        if mask is None:
            valid = in_range
        else:
            valid = mask.bool() & in_range

        # 构建 scatter_add 所需的线性索引。
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, T, H, W).reshape(-1)
        m_idx = month_index.view(B, T, 1, 1).expand(B, T, H, W).reshape(-1)
        h_idx = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, T, H, W).reshape(-1)
        w_idx = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, T, H, W).reshape(-1)

        valid_flat = valid.view(B, T, 1, 1).expand(B, T, H, W).reshape(-1)
        flat_index = ((b_idx * M + m_idx) * H + h_idx) * W + w_idx

        z_flat = z.reshape(B * T * H * W, self.embed_dim)

        acc = torch.zeros(B * M * H * W, self.embed_dim, device=device, dtype=z.dtype)
        acc.scatter_add_(
            0,
            flat_index[valid_flat].unsqueeze(-1).expand(-1, self.embed_dim),
            z_flat[valid_flat],
        )

        count = torch.zeros(B * M * H * W, device=device, dtype=z.dtype)
        count.scatter_add_(
            0,
            flat_index[valid_flat],
            torch.ones(valid_flat.sum(), device=device, dtype=z.dtype),
        )

        acc = acc.view(B, M, H, W, self.embed_dim)
        count = count.view(B, M, H, W)

        count_safe = count.clamp(min=1.0).unsqueeze(-1)
        avg = acc / count_safe
        has_obs = (count > 0).float().unsqueeze(-1)
        monthly_feats = has_obs * avg + (1 - has_obs) * self.missing_token

        monthly_mask = (count.view(B, M, H * W).sum(-1) > 0).float()
        return monthly_feats, monthly_mask
