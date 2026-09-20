"""Optional source-specific residual adaptation of a public-observation base."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from xuannv_embedding.models.decoders import ContinuousDecoder
from xuannv_embedding.models.model import AEFOutput


def masked_resize(x: torch.Tensor, mask: torch.Tensor, size: tuple[int, int]):
    clean = torch.where(mask.bool(), x, torch.zeros_like(x))
    numerator = F.interpolate(clean, size=size, mode="bilinear", align_corners=False)
    weight = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    valid = F.interpolate(mask, size=size, mode="nearest") > 0
    return torch.where(valid, numerator / weight.clamp_min(1e-6), 0), valid


class HighResResidual(nn.Module):
    def __init__(self, channels: int, embed_dim: int, *, native: bool):
        super().__init__()
        self.native = native
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(1, b), nn.GELU())
                for a, b in [(channels, 32), (32, embed_dim), (embed_dim, embed_dim)]
            ]
        )
        self.correction = nn.Conv2d(2 * embed_dim, embed_dim, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, z, image, mask):
        b, t, d, h, w = z.shape
        monthly = image.ndim == 5
        if monthly:
            if image.shape[:2] != (b, t) or mask.shape != (b, t, 1, *image.shape[-2:]):
                raise ValueError("monthly highres image/mask must match embedding batch and time")
            image = image.flatten(0, 1)
            mask = mask.flatten(0, 1)
        if mask.shape != (image.shape[0], 1, *image.shape[-2:]):
            raise ValueError("highres quality mask must be [B,1,H,W]")
        mask = mask.to(dtype=image.dtype)
        feature = torch.where(mask.bool(), image, torch.zeros_like(image))
        if not self.native:
            feature, mask = masked_resize(feature, mask, z.shape[-2:])
        for layer in self.layers:
            feature = layer(feature) * mask
        feature, valid = masked_resize(feature, mask.to(feature.dtype), z.shape[-2:])
        if monthly:
            repeated = feature.reshape(b, t, d, h, w)
            valid = valid.reshape(b, t, 1, h, w)
        else:
            repeated = feature[:, None].expand(-1, t, -1, -1, -1)
            valid = valid[:, None]
        delta = self.correction(torch.cat([z, repeated], dim=2).reshape(b * t, 2 * d, h, w))
        delta = delta.reshape(b, t, d, h, w)
        # Preserve the exact pretrained value at zero initialization, including FP rounding.
        corrected = z + (F.normalize(z + delta, dim=2) - F.normalize(z, dim=2))
        return torch.where(valid, corrected, z)


class IncrementalHighResModel(nn.Module):
    def __init__(self, base, sources, targets, *, native: bool, freeze_base: bool):
        super().__init__()
        self.base = base
        self.freeze_base = freeze_base
        self.frozen_sources: set[str] = set()
        self.embed_dim = base.embed_dim
        self.branches = nn.ModuleDict(
            {s: HighResResidual(c, base.embed_dim, native=native) for s, c in sources.items()}
        )
        self.new_decoders = nn.ModuleDict(
            {h: ContinuousDecoder(base.embed_dim, c) for h, c in targets.items()}
        )
        if freeze_base:
            self.base.requires_grad_(False)
            self.base.eval()

    def extend(self, sources, targets, *, native: bool, freeze_existing: bool):
        if set(sources) & set(self.branches) or set(targets) & set(self.new_decoders):
            raise ValueError("new sources and decoders must not overwrite existing modules")
        if not sources or not targets:
            raise ValueError("extension requires new sources and targets")
        self.requires_grad_(not freeze_existing)
        self.freeze_base = freeze_existing
        self.frozen_sources = set(self.branches) if freeze_existing else set()
        for source, channels in sources.items():
            self.branches[source] = HighResResidual(channels, self.embed_dim, native=native)
        for name, channels in targets.items():
            self.new_decoders[name] = ContinuousDecoder(self.embed_dim, channels)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
        for source in self.frozen_sources:
            self.branches[source].eval()
        return self

    def forward(
        self, source_frames, source_masks, timestamps, highres_frames=None, highres_masks=None
    ):
        highres_frames, highres_masks = highres_frames or {}, highres_masks or {}
        if set(highres_frames) != set(highres_masks):
            raise ValueError("each supplied highres source requires a quality mask")
        if set(highres_frames) - set(self.branches):
            raise ValueError("unregistered highres source")
        if self.freeze_base:
            with torch.no_grad():
                output = self.base(source_frames, source_masks, timestamps)
        else:
            output = self.base(source_frames, source_masks, timestamps)
        z = output.embedding_map
        for name, branch in self.branches.items():
            if name in highres_frames:
                z = branch(z, highres_frames[name], highres_masks[name])
        b, t, d, h, w = z.shape
        flat = z.reshape(b * t, d, h, w)
        recon = {
            name: decoder(flat).reshape(b, t, -1, h, w)
            for bank in (self.base.decoders, self.new_decoders)
            for name, decoder in bank.items()
        }
        return AEFOutput(z, F.normalize(z.mean((-2, -1)), dim=-1), recon)
