from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DenseHead(nn.Module):
    """密集下游头公共接口。"""

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class LinearHead(DenseHead):
    """严格线性 probe：一个 1×1 卷积。"""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.projection(embedding_map)


class MLPHead(DenseHead):
    """逐像素 MLP，不引入空间上下文。"""

    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.network(embedding_map)


class WideMLPHead(DenseHead):
    """宽逐像素 MLP：D→128→64→C。"""

    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.network(embedding_map)


class DeepWideMLPHead(DenseHead):
    """深宽逐像素 MLP：D→256→128→C。"""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.network(embedding_map)


class Conv3x3Head(DenseHead):
    """带局部上下文的两层 3×3 卷积头。"""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.network(embedding_map)


class UNetHead(DenseHead):
    """两级轻量 UNet decoder，skip 均来自冻结 embedding map。"""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        half = max(8, embed_dim // 2)
        quarter = max(8, embed_dim // 4)
        self.up1 = nn.ConvTranspose2d(embed_dim, half, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim + half, half, kernel_size=3, padding=1),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(half, quarter, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim + quarter, quarter, kernel_size=3, padding=1),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(quarter, num_classes, kernel_size=1)

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first = self.up1(embedding_map)
        skip2 = F.interpolate(
            embedding_map, size=first.shape[-2:], mode="bilinear", align_corners=False
        )
        first = self.conv1(torch.cat([first, skip2], dim=1))
        second = self.up2(first)
        skip4 = F.interpolate(
            embedding_map, size=second.shape[-2:], mode="bilinear", align_corners=False
        )
        second = self.conv2(torch.cat([second, skip4], dim=1))
        logits = self.classifier(second)
        return F.interpolate(
            logits,
            size=embedding_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class DeepLabLiteHead(DenseHead):
    """轻量 ASPP head，面向已经是 128×128 的 dense embedding。"""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        rates: tuple[int, int, int] = (3, 6, 12),
    ) -> None:
        super().__init__()
        branch_dim = max(16, hidden_dim // 4)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dim, branch_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(branch_dim),
                    nn.ReLU(inplace=True),
                ),
                *[
                    nn.Sequential(
                        nn.Conv2d(
                            embed_dim,
                            branch_dim,
                            kernel_size=3,
                            padding=rate,
                            dilation=rate,
                            bias=False,
                        ),
                        nn.BatchNorm2d(branch_dim),
                        nn.ReLU(inplace=True),
                    )
                    for rate in rates
                ],
            ]
        )
        fused_channels = branch_dim * len(self.branches)
        self.project = nn.Sequential(
            nn.Conv2d(fused_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.project(torch.cat([branch(embedding_map) for branch in self.branches], dim=1))


_STANDARD_HEADS: dict[str, type[DenseHead]] = {
    "linear": LinearHead,
    "mlp": MLPHead,
    "wide_mlp": WideMLPHead,
    "deep_wide_mlp": DeepWideMLPHead,
    "conv3x3": Conv3x3Head,
    "unet": UNetHead,
    "deeplab_lite": DeepLabLiteHead,
}


def build_head(name: str, *, embed_dim: int, num_classes: int) -> DenseHead:
    """只构造发布协议登记的标准头。"""
    try:
        head_type = _STANDARD_HEADS[name]
    except KeyError as exc:
        raise ValueError(f"未知标准 head: {name!r}") from exc
    return head_type(embed_dim, num_classes)


STANDARD_HEAD_NAMES = tuple(_STANDARD_HEADS)

__all__ = [
    "Conv3x3Head",
    "DeepLabLiteHead",
    "DeepWideMLPHead",
    "DenseHead",
    "LinearHead",
    "MLPHead",
    "STANDARD_HEAD_NAMES",
    "UNetHead",
    "WideMLPHead",
    "build_head",
]
