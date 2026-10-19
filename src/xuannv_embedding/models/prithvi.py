# Copyright (c) IBM Corp. 2024. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Modified encoder-only implementation of the publisher's Prithvi EO 2.0:
# https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
# Publisher revision: 9eb1b1102806593963daa333bcc491b1c6f8562f
# Retains parameter names and positional mathematics; uses native PyTorch
# attention/reshapes, rejects silent border cropping and excludes the decoder.
"""Portable non-TL Prithvi encoder with strict published-state consumption."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PrithviEncoderConfig:
    image_size: int = 224
    num_frames: int = 4
    in_channels: int = 6
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    patch_size: tuple[int, int, int] = (1, 16, 16)
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        values = [
            self.image_size,
            self.num_frames,
            self.in_channels,
            self.embed_dim,
            self.depth,
            self.num_heads,
        ]
        if any(type(n) is not int or n < 1 for n in values):
            raise ValueError("Prithvi dimensions must be positive integers")
        if (
            type(self.patch_size) is not tuple
            or len(self.patch_size) != 3
            or any(type(n) is not int or n < 1 for n in self.patch_size)
            or self.embed_dim % 16
            or self.embed_dim % self.num_heads
            or any(
                n % p
                for n, p in zip(
                    (self.num_frames, self.image_size, self.image_size), self.patch_size
                )
            )
        ):
            raise ValueError("Unsupported Prithvi patch or embedding dimensions")
        if (
            any(
                type(n) not in (int, float) or not math.isfinite(n) or n <= 0
                for n in (self.mlp_ratio, self.layer_norm_eps)
            )
            or int(self.embed_dim * self.mlp_ratio) < 1
        ):
            raise ValueError("Invalid Prithvi expansion or normalization")


def sincos_positions(embed_dim: int, grid: tuple[int, int, int]) -> np.ndarray:
    """Publisher width/height/time sine-cosine positions, including zero CLS."""
    if (
        type(embed_dim) is not int
        or embed_dim < 16
        or embed_dim % 16
        or len(grid) != 3
        or any(type(n) is not int or n < 1 for n in grid)
    ):
        raise ValueError("Invalid Prithvi positional grid")

    def axis(width: int, length: int) -> np.ndarray:
        omega = np.arange(width // 2, dtype=float) / (width / 2.0)
        omega = 1.0 / 10000**omega
        phase = np.einsum("m,d->md", np.arange(length), omega)
        return np.concatenate((np.sin(phase), np.cos(phase)), axis=1)

    t, h, w = grid
    spatial_width = embed_dim // 16 * 6
    temporal_width = embed_dim // 16 * 4
    width = np.tile(axis(spatial_width, w), (t * h, 1))
    height = np.tile(np.repeat(axis(spatial_width, h), w, axis=0), (t, 1))
    time = np.repeat(axis(temporal_width, t), h * w, axis=0)
    values = np.concatenate((width, height, time), axis=1)
    return np.concatenate((np.zeros((1, embed_dim)), values), axis=0)


class _PatchEmbed(nn.Module):
    def __init__(self, config: PrithviEncoderConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.grid_size = [
            n // p
            for n, p in zip(
                (config.num_frames, config.image_size, config.image_size), config.patch_size
            )
        ]
        self.proj = nn.Conv3d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, config: PrithviEncoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, d // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        values = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(values.transpose(1, 2).reshape(b, n, d))


class _Mlp(nn.Module):
    def __init__(self, config: PrithviEncoderConfig) -> None:
        super().__init__()
        width = int(config.embed_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.embed_dim, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, config: PrithviEncoderConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.attn = _Attention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.mlp = _Mlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PrithviEncoder(nn.Module):
    """Inference encoder for the published non-coordinate-encoded MAE weights."""

    def __init__(self, config: PrithviEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.num_features = config.embed_dim
        self.patch_embed = _PatchEmbed(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        positions = sincos_positions(config.embed_dim, tuple(self.patch_embed.grid_size))
        self.register_buffer("pos_embed", torch.from_numpy(positions).float()[None])
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def position_encoding(self, shape: tuple[int, int, int]) -> torch.Tensor:
        grid = [n // p for n, p in zip(shape, self.config.patch_size)]
        original = self.patch_embed.grid_size
        if grid == original:
            return self.pos_embed
        if grid[0] != original[0]:
            base = (grid[0], *original[1:])
            values = sincos_positions(self.config.embed_dim, base)
            # The publisher recomputes temporal positions on CPU. Transfer them
            # explicitly so dynamic sequences also work on accelerator devices.
            positions = torch.from_numpy(values).float()[None].to(self.pos_embed.device)
        else:
            base, positions = tuple(original), self.pos_embed
        patch = positions[:, 1:].reshape(*base, self.config.embed_dim).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, size=tuple(grid[1:]), mode="bicubic", align_corners=True)
        patch = patch.permute(0, 2, 3, 1).reshape(1, -1, self.config.embed_dim)
        return torch.cat((positions[:, :1], patch), dim=1)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.ndim != 5
            or x.shape[0] < 1
            or x.shape[1] != self.config.in_channels
            or any(n < 1 for n in x.shape[-3:])
            or not x.is_floating_point()
        ):
            raise ValueError("Prithvi input shape must be floating B,C,T,H,W")
        if any(n % p for n, p in zip(x.shape[-3:], self.config.patch_size)):
            raise ValueError("Prithvi input must be divisible by patches; pad explicitly")
        if not torch.isfinite(x).all():
            raise ValueError("Prithvi input must be finite after explicit missing-data handling")
        positions = self.position_encoding(tuple(x.shape[-3:])).to(x.device)
        values = self.patch_embed(x) + positions[:, 1:]
        cls = (self.cls_token + positions[:, :1]).expand(len(x), -1, -1)
        values = torch.cat((cls, values), dim=1)
        for block in self.blocks:
            values = block(values)
        return self.norm(values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)[:, 1:]
        grid = tuple(n // p for n, p in zip(x.shape[-3:], self.config.patch_size))
        return tokens.transpose(1, 2).reshape(len(x), self.config.embed_dim, *grid)


def load_encoder_state(
    model: PrithviEncoder, checkpoint: Mapping[str, torch.Tensor]
) -> dict[str, int]:
    """Consume every encoder tensor strictly and explicitly count excluded decoder tensors."""
    if any(not k.startswith(("encoder.", "decoder.")) for k in checkpoint):
        raise ValueError("Unexpected checkpoint prefix")
    state = {
        k.removeprefix("encoder."): v for k, v in checkpoint.items() if k.startswith("encoder.")
    }
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Pretrained encoder keys differ")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
            raise ValueError("Pretrained encoder tensor shape differs: " + name)
        if not torch.isfinite(value).all():
            raise ValueError("Pretrained encoder weights must be finite: " + name)
    model.load_state_dict(state, strict=True)
    return {
        "encoder_keys": len(state),
        "ignored_decoder_keys": sum(k.startswith("decoder.") for k in checkpoint),
    }
