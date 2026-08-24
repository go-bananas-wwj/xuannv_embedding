from __future__ import annotations

# Availability-aware 高分辨率数据融合模块。
import torch
from torch import nn


class AvailabilityAwareFusion(nn.Module):
    """基于可用性掩码的高分辨率特征融合模块。

    要求 ``base_feat`` 与 ``highres_feat`` 已经对齐到相同空间尺寸 ``(H, W)``；
    将高分辨率数据是否可用显式编码为嵌入，并与基础特征相加，
    随后通过 1x1 卷积融合基础特征与高分辨率特征。
    """

    def __init__(self, dim: int) -> None:
        """初始化融合模块。

        Args:
            dim: 输入与输出通道数，基础特征与高分辨率特征通道数需一致。
        """
        super().__init__()
        self.dim = dim

        # 将单通道可用性掩码映射到 dim 维嵌入。
        self.avail_embed = nn.Conv2d(1, dim, kernel_size=1)

        # GroupNorm 要求 num_channels 能被 num_groups 整除；
        # 当 dim 不能被 8 整除时，回退到 dim 个 group（即逐通道）。
        num_groups = 8 if dim % 8 == 0 else dim
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        base_feat: torch.Tensor,
        highres_feat: torch.Tensor,
        avail_mask: torch.Tensor,
    ) -> torch.Tensor:
        """融合基础特征与高分辨率特征。

        Args:
            base_feat: 基础特征，形状 (B, C, H, W)。
            highres_feat: 高分辨率特征，形状 (B, C, H, W)；
                调用方需负责将其池化/重采样到与 ``base_feat`` 相同尺寸。
                当高分辨率数据缺失时，可传入全 0 张量。
            avail_mask: 高分辨率数据可用性掩码，形状 (B, 1, H, W)；
                1 表示可用，0 表示缺失。

        Returns:
            融合后的特征，形状 (B, C, H, W)。

        Raises:
            AssertionError: 当 ``base_feat`` 与 ``highres_feat`` 空间尺寸不一致时。
        """
        assert base_feat.shape[2:] == highres_feat.shape[2:], (
            f"base_feat spatial size {base_feat.shape[2:]} does not match "
            f"highres_feat spatial size {highres_feat.shape[2:]}"
        )
        if avail_mask.shape[2:] != base_feat.shape[2:]:
            raise ValueError(
                f"avail_mask spatial size {avail_mask.shape[2:]} does not match "
                f"base_feat spatial size {base_feat.shape[2:]}"
            )
        highres_feat = highres_feat * avail_mask.float()
        avail_embed = self.avail_embed(avail_mask.float())  # (B, C, H, W)
        combined = torch.cat([base_feat + avail_embed, highres_feat], dim=1)  # (B, 2C, H, W)
        fused = self.fusion(combined)  # (B, C, H, W)
        return fused
