"""Monthly high-resolution window transformers injected inside a frozen STP base.

Inputs must be north-up grids with the same audited geographic footprint as the
public grid. GSD follows that footprint, not an assumed integer resize factor.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from xuannv_embedding.models.decoders import ContinuousDecoder
from xuannv_embedding.models.model import AEFOutput


@lru_cache(maxsize=32)
def _layout(h, w, qh, qw, patch, window):
    if qh % window or qw % window:
        raise ValueError("public grid must be divisible by window_cells")
    th, tw = math.ceil(h / patch), math.ceil(w / patch)
    # Pixel centers include partial boundary patches; no crop or implicit stretch.
    ys = (torch.arange(th) * patch + (torch.arange(th) * patch + patch).clamp(max=h)) / 2
    xs = (torch.arange(tw) * patch + (torch.arange(tw) * patch + patch).clamp(max=w)) / 2
    yy, xx = torch.meshgrid(ys * qh / h, xs * qw / w, indexing="ij")
    wy, wx = (yy / window).long(), (xx / window).long()
    ids = (wy * (qw // window) + wx).flatten()
    count = (qh // window) * (qw // window)
    groups = [torch.where(ids == i)[0] for i in range(count)]
    width = max(len(g) for g in groups)
    indices = torch.zeros(count, width, dtype=torch.long)
    present = torch.zeros(count, width, dtype=torch.bool)
    for i, group in enumerate(groups):
        indices[i, : len(group)] = group
        present[i, : len(group)] = True
    xy = torch.stack((xx - wx * window, yy - wy * window), dim=-1).reshape(-1, 2)
    return indices, present, xy[indices] / window


class MetricAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query, keys, visible, qxy, kxy):
        n, nq, d = query.shape
        nk = keys.shape[1]
        q = self.q(query).reshape(n, nq, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(keys).reshape(n, nk, 2, self.heads, d // self.heads).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // self.heads)
        distance = (qxy[:, :, None] - kxy[:, None]).square().sum(-1)
        scores = scores - distance[:, None].to(scores.dtype)
        # Finite softmax followed by explicit masking gives zero for all-missing windows.
        scores = scores.masked_fill(~visible[:, None, None], -1e4)
        weights = scores.softmax(-1) * visible[:, None, None].to(scores.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        value = (weights @ v).transpose(1, 2).reshape(n, nq, d)
        return self.proj(value) * visible.any(-1)[:, None, None]


class LocalBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = MetricAttention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x, visible, xy):
        normal = self.norm1(x)
        x = x + self.attention(normal, normal, visible, xy, xy)
        return (x + self.mlp(self.norm2(x))) * visible[..., None]


class HighResWindowEncoder(nn.Module):
    def __init__(self, channels, settings):
        super().__init__()
        self.settings = settings
        self.projection = nn.Linear(channels * settings.patch_pixels**2, settings.dim)
        self.position = nn.Linear(5, settings.dim)
        self.time = nn.Linear(3, settings.dim)
        self.blocks = nn.ModuleList(
            [LocalBlock(settings.dim, settings.heads) for _ in range(settings.layers)]
        )

    def forward(self, image, mask, public_shape, timestamps):
        if image.ndim != 5:
            raise ValueError("transformer requires monthly highres [B,T,C,H,W]")
        b, t, c, h, w = image.shape
        if mask.shape != (b, t, 1, h, w) or timestamps.shape != (b, t):
            raise ValueError("monthly highres mask/time dimensions differ")
        s, (qh, qw) = self.settings, public_shape
        indices, occupied, xy = _layout(h, w, qh, qw, s.patch_pixels, s.window_cells)
        indices, occupied, xy = [v.to(image.device) for v in (indices, occupied, xy)]
        nw, nk = indices.shape
        pad = (0, (-w) % s.patch_pixels, 0, (-h) % s.patch_pixels)
        clean = torch.where(mask.bool(), image, 0).flatten(0, 1)
        pixels = F.unfold(F.pad(clean, pad), s.patch_pixels, stride=s.patch_pixels).transpose(1, 2)
        fraction = F.avg_pool2d(F.pad(mask.flatten(0, 1).float(), pad), s.patch_pixels)
        fraction = fraction.flatten(1)[:, indices]
        visible = (fraction > 0) & occupied[None]
        value = self.projection(pixels[:, indices])
        coordinates = xy[None].expand(b * t, -1, -1, -1)
        gsd = image.new_tensor(
            [math.log(qw / w * s.reference_gsd_m), math.log(qh / h * s.reference_gsd_m)]
        )
        metadata = torch.cat(
            (coordinates, gsd.expand(b * t, nw, nk, 2), fraction[..., None]), dim=-1
        )
        month = timestamps.flatten().float() % 100
        year = timestamps.flatten().float() // 100
        date = torch.stack(
            (torch.sin(month * math.pi / 6), torch.cos(month * math.pi / 6), (year - 2000) / 100),
            dim=-1,
        )
        value = value + self.position(metadata.to(value.dtype)) + self.time(date)[:, None, None]
        value = value * visible[..., None]
        flat = value.reshape(-1, nk, s.dim)
        valid = visible.reshape(-1, nk)
        pos = coordinates.reshape(-1, nk, 2)
        for block in self.blocks:
            pieces = []
            for start in range(0, len(flat), s.window_chunk):
                args = (
                    flat[start : start + s.window_chunk],
                    valid[start : start + s.window_chunk],
                    pos[start : start + s.window_chunk],
                )
                pieces.append(
                    checkpoint(block, *args, use_reentrant=False) if self.training else block(*args)
                )
            flat = torch.cat(pieces)
        return flat.reshape(b, t, nw, nk, s.dim), visible.reshape(b, t, nw, nk), xy


class CrossResolutionInjector(nn.Module):
    def __init__(self, precision_dim, sources, settings):
        super().__init__()
        self.settings = settings
        self.query = nn.Linear(precision_dim, settings.dim)
        self.norm = nn.LayerNorm(settings.dim)
        self.attention = nn.ModuleDict(
            {name: MetricAttention(settings.dim, settings.heads) for name in sources}
        )
        self.gates = nn.ModuleDict({name: nn.Linear(2 * settings.dim, 1) for name in sources})
        self.output = nn.Linear(settings.dim, precision_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, precision, encoded):
        b, t, h, w, d = precision.shape
        s = self.settings
        side = s.window_cells
        q = precision.reshape(b, t, h // side, side, w // side, side, d)
        q = q.permute(0, 1, 2, 4, 3, 5, 6).reshape(-1, side * side, d)
        query = self.norm(self.query(q))
        axis = (torch.arange(side, device=q.device).float() + 0.5) / side
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        qxy = torch.stack((xx, yy), -1).reshape(1, side * side, 2)
        parts = []
        for start in range(0, len(q), s.window_chunk):
            end = min(start + s.window_chunk, len(q))
            qc = query[start:end]
            # Source-specific attention avoids a source winning solely through token count.
            values, gates, active = [], [], []
            for name, (tokens, valid, xy) in encoded.items():
                nk = tokens.shape[-2]
                keys = tokens.reshape(-1, nk, s.dim)[start:end]
                visible = valid.reshape(-1, nk)[start:end]
                positions = xy[torch.arange(start, end, device=q.device) % xy.shape[0]]
                value = self.attention[name](
                    qc, self.norm(keys), visible, qxy.expand(end - start, -1, -1), positions
                )
                present = visible.any(-1)[:, None, None]
                gate = self.gates[name](torch.cat((qc, value), -1))
                values.append(value)
                gates.append(gate.masked_fill(~present, -1e4))
                active.append(present)
            weights = torch.stack(gates, -1).softmax(-1) * torch.stack(active, -1)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            fused = (torch.stack(values, -1) * weights).sum(-1)
            any_active = torch.stack(active, -1).any(-1)
            parts.append(self.output(fused) * any_active)
        delta = torch.cat(parts).reshape(b, t, h // side, w // side, side, side, d)
        delta = delta.permute(0, 1, 2, 4, 3, 5, 6).reshape_as(precision)
        return precision + delta


class HighResTransformerModel(nn.Module):
    def __init__(self, base, sources, targets, *, settings, freeze_base):
        super().__init__()
        if not freeze_base:
            raise ValueError("transformer adapter currently requires freeze_base")
        if base.stp_encoder.precision_scale != 1:
            raise ValueError("transformer adapter requires precision_scale=1")
        if not settings.injection_blocks or max(settings.injection_blocks) >= len(
            base.stp_encoder.blocks
        ):
            raise ValueError("injection must precede a later STP block")
        self.base, self.settings, self.freeze_base = base, settings, freeze_base
        self.embed_dim = base.embed_dim
        self.encoders = nn.ModuleDict(
            {k: HighResWindowEncoder(c, settings) for k, c in sources.items()}
        )
        self.injectors = nn.ModuleDict(
            {
                str(i): CrossResolutionInjector(base.stp_encoder.precision_dim, sources, settings)
                for i in settings.injection_blocks
            }
        )
        self.new_decoders = nn.ModuleDict(
            {k: ContinuousDecoder(base.embed_dim, c) for k, c in targets.items()}
        )
        base.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Preserve deterministic base vMF output; only STP needs training-mode checkpointing.
        self.base.eval()
        self.base.stp_encoder.train(mode)
        return self

    def forward(
        self, source_frames, source_masks, timestamps, highres_frames=None, highres_masks=None
    ):
        highres_frames, highres_masks = highres_frames or {}, highres_masks or {}
        if set(highres_frames) != set(highres_masks) or set(highres_frames) - set(self.encoders):
            raise ValueError("unregistered or mismatched highres sources")
        shape = next(iter(source_frames.values())).shape[-2:]
        encoded = {
            name: self.encoders[name](image, highres_masks[name], shape, timestamps)
            for name, image in highres_frames.items()
        }

        def inject(index, feature):
            key = str(index)
            return (
                self.injectors[key](feature, encoded)
                if key in self.injectors and encoded
                else feature
            )

        # Frozen parameters still propagate gradients to earlier injection blocks.
        out = self.base(source_frames, source_masks, timestamps, stp_injector=inject)
        b, t, d, h, w = out.embedding_map.shape
        recon = dict(out.reconstructions)
        for name, decoder in self.new_decoders.items():
            recon[name] = decoder(out.embedding_map.reshape(b * t, d, h, w)).reshape(b, t, -1, h, w)
        return AEFOutput(out.embedding_map, out.embedding, recon)
