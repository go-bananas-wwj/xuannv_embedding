"""Scale-aware residual learning from sensor features and known weak semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def align_patch_features(values, source_shape, output_shape, *, patch):
    """Resample padded token centers onto the unpadded observation footprint."""
    h, w = source_shape
    if values.ndim != 4 or values.shape[-2:] != (math.ceil(h / patch), math.ceil(w / patch)):
        raise ValueError("Native patch grid does not match the padded source footprint")
    oh, ow = output_shape
    yy = (torch.arange(oh, device=values.device).float() + 0.5) * h / oh
    xx = (torch.arange(ow, device=values.device).float() + 0.5) * w / ow
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack(
        (2 * xx / (values.shape[-1] * patch) - 1, 2 * yy / (values.shape[-2] * patch) - 1), -1
    )
    return F.grid_sample(
        values.float(),
        grid[None].expand(len(values), -1, -1, -1),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


@dataclass
class SemanticOutput:
    embedding: torch.Tensor
    public_embedding: torch.Tensor
    white_delta: torch.Tensor
    gates: torch.Tensor
    worldcover_logits: torch.Tensor
    area_logits: torch.Tensor
    geometry_logits: torch.Tensor
    spectral_reconstruction: torch.Tensor
    detail_reconstruction: torch.Tensor
    context_reconstruction: torch.Tensor
    detail_mask: torch.Tensor
    context_mask: torch.Tensor


class SemanticResidualFusion(nn.Module):
    def __init__(
        self,
        *,
        width: int = 128,
        public_residual_scale: float = 0.25,
        local_residual_scale: float = 0.5,
        interaction_layers: int = 0,
    ) -> None:
        super().__init__()
        if type(interaction_layers) is not int or interaction_layers not in {0, 1, 2}:
            raise ValueError("Semantic interaction layer count must be zero, one or two")
        if (
            type(width) is not int
            or width < 8
            or any(
                type(v) not in {int, float} or not math.isfinite(v) or not 0 < v <= 1
                for v in [public_residual_scale, local_residual_scale]
            )
        ):
            raise ValueError("Invalid semantic residual dimensions/scales")
        self.public_residual_scale = public_residual_scale
        self.local_residual_scale = local_residual_scale
        for name, c in [
            ("public", 64),
            ("spectral", 95),
            ("detail", 64),
            ("context", 1024),
            ("context_target", 64),
        ]:
            self.register_buffer(name + "_mean", torch.zeros(1, c, 1, 1))
            self.register_buffer(name + "_std", torch.ones(1, c, 1, 1))

        def block(a, kernel):
            return nn.Sequential(
                nn.Conv2d(a, width, kernel, padding=kernel // 2), nn.GroupNorm(1, width), nn.GELU()
            )

        self.public_encoder = block(159, 3)
        self.detail_encoder = block(65, 3)
        self.context_encoder = block(1025, 1)
        self.public_delta = nn.Conv2d(width, 64, 1)
        self.detail_delta = nn.Conv2d(width, 64, 1)
        self.context_delta = nn.Conv2d(width, 64, 1)
        self.selector = nn.Conv2d(3 * width + 2, 3, 1)
        nn.init.zeros_(self.selector.weight)
        with torch.no_grad():
            self.selector.bias.copy_(torch.tensor([2.0, 0.0, 0.0]))
        for layer in [self.public_delta, self.detail_delta, self.context_delta]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.worldcover_head = nn.Conv2d(64, 12, 1)
        self.area_head = nn.Conv2d(64, 7, 1)
        self.geometry_head = nn.Conv2d(64, 4, 1)
        self.spectral_head = nn.Conv2d(64, 95, 1)
        self.detail_head = nn.Conv2d(64, 64, 1)
        self.context_head = nn.Conv2d(64, 64, 1)
        # Construct after the original heads to retain common seeded initialization.
        mixer = []
        for layer in range(interaction_layers):
            dilation = layer + 1
            mixer.extend(
                [
                    nn.Conv2d(
                        3 * width + 2 if layer == 0 else width,
                        width,
                        3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.GroupNorm(1, width),
                    nn.GELU(),
                ]
            )
        self.local_mixer = nn.Sequential(*mixer) if mixer else None

    def set_statistics(self, name: str, mean: torch.Tensor, std: torch.Tensor) -> None:
        if name not in {"public", "spectral", "detail", "context", "context_target"}:
            raise ValueError("Unknown semantic feature source")
        target = getattr(self, name + "_mean")
        mean, std = [
            torch.as_tensor(v, dtype=torch.float32).reshape(1, -1, 1, 1) for v in [mean, std]
        ]
        if (
            mean.shape != target.shape
            or std.shape != target.shape
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError("Invalid observed feature statistics")
        with torch.no_grad():
            target.copy_(mean)
            getattr(self, name + "_std").copy_(std)

    def standardize(self, x: torch.Tensor, name: str, mask: torch.Tensor) -> torch.Tensor:
        clean = torch.where(mask, x.detach().float(), 0)
        return torch.where(
            mask,
            ((clean - getattr(self, name + "_mean")) / getattr(self, name + "_std")).clamp(-6, 6),
            0,
        )

    def forward(
        self,
        public: torch.Tensor,
        spectral: torch.Tensor,
        detail: torch.Tensor | None,
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        *,
        spectral_mask: torch.Tensor,
        context_source_shape: tuple[int, int] | None = None,
        burnin: bool = False,
    ) -> SemanticOutput:
        if (
            public.ndim != 4
            or public.shape[1] != 64
            or spectral.shape != (len(public), 95, *public.shape[-2:])
            or spectral_mask.shape != spectral.shape
            or spectral_mask.dtype != torch.bool
        ):
            raise ValueError("Semantic fusion requires aligned 64/95 public maps and masks")
        if not torch.isfinite(public).all() or not torch.isfinite(spectral).all():
            raise ValueError("Public observations must be finite")
        public, spectral = public.detach().float(), spectral.detach().float()
        b, _, h, w = public.shape
        p_mask = public.abs().sum(1, keepdim=True) > 1e-8
        base = self.standardize(public, "public", p_mask)
        if detail is None:
            detail = torch.zeros_like(public)
        if detail.shape != public.shape or not torch.isfinite(detail).all():
            raise ValueError("Detail feature shape/values differ")
        detail = detail.detach().float()
        d_mask = detail.abs().sum(1, keepdim=True) > 1e-8
        if (context is None) != (context_mask is None):
            raise ValueError("Native context and mask must be supplied together")
        if context is not None:
            if (
                context.ndim != 4
                or context.shape[:2] != (b, 1024)
                or context_mask.shape != (b, 1, *context.shape[-2:])
                or context_source_shape is None
                or len(context_source_shape) != 2
                or any(type(v) is not int or v < 1 for v in context_source_shape)
                or context.shape[-2:] != tuple(math.ceil(v / 16) for v in context_source_shape)
            ):
                raise ValueError("Native context source/patch geometry differs")
            if (
                not torch.isfinite(context_mask).all()
                or (context_mask < 0).any()
                or (context_mask > 1).any()
            ):
                raise ValueError("Native coverage must be finite in [0,1]")
            context_mask = context_mask.detach().float()
            c_valid = context_mask > 0
            context = context.detach().float()
            if not torch.isfinite(torch.where(c_valid, context, 0)).all():
                raise ValueError("Observed native context must be finite")
            coverage = align_patch_features(context_mask, context_source_shape, (h, w), patch=16)
        else:
            coverage = public.new_zeros(b, 1, h, w)
        c_mask = coverage > 0.05
        zeros = torch.zeros_like(public)
        gates = public.new_zeros(b, 3, h, w)
        gates[:, 0] = 1
        pub_delta = zeros
        local_delta = zeros
        if not burnin:
            p_hidden = self.public_encoder(
                torch.cat([base, self.standardize(spectral, "spectral", spectral_mask)], 1)
            )
            pub_delta = self.public_residual_scale * self.public_delta(p_hidden).tanh()
            d_hidden = (
                self.detail_encoder(
                    torch.cat([self.standardize(detail, "detail", d_mask), d_mask.float()], 1)
                )
                * d_mask
            )
            if context is not None:
                c_hidden = (
                    self.context_encoder(
                        torch.cat([self.standardize(context, "context", c_valid), context_mask], 1)
                    )
                    * c_valid
                )
                numerator = align_patch_features(
                    c_hidden * context_mask, context_source_shape, (h, w), patch=16
                )
                c_hidden = torch.where(c_mask, numerator / coverage.clamp_min(1e-6), 0)
            else:
                c_hidden = torch.zeros_like(d_hidden)
            if self.local_mixer is not None:
                joint = self.local_mixer(
                    torch.cat([p_hidden, d_hidden, c_hidden, d_mask.float(), coverage], 1)
                )
                d_hidden = (d_hidden + joint) * d_mask
                c_hidden = (c_hidden + joint) * c_mask
            scores = self.selector(
                torch.cat([p_hidden, d_hidden, c_hidden, d_mask.float(), coverage], 1)
            )
            allowed = torch.cat([torch.ones_like(d_mask), d_mask, c_mask], 1)
            gates = torch.softmax(torch.where(allowed, scores.float(), -10000), 1)
            local_delta = self.local_residual_scale * (
                gates[:, 1:2] * self.detail_delta(d_hidden).tanh()
                + gates[:, 2:3] * self.context_delta(c_hidden).tanh()
            )
        delta = pub_delta + local_delta
        embedding = public + self.public_std * delta
        public_embedding = public + self.public_std * pub_delta
        white = base + delta
        return SemanticOutput(
            embedding,
            public_embedding,
            delta,
            gates,
            self.worldcover_head(white),
            self.area_head(white),
            self.geometry_head(white),
            self.spectral_head(white),
            self.detail_head(white),
            self.context_head(white),
            d_mask,
            c_mask,
        )


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = torch.broadcast_to(mask, x.shape)
    return torch.where(mask, x, 0).sum() / mask.sum().clamp_min(1)


def semantic_loss(
    output: SemanticOutput,
    labels: dict,
    targets: dict,
    spectral_mask: torch.Tensor,
    class_weights: dict,
    *,
    weights: dict | None = None,
) -> dict:
    coefficients = {
        "worldcover": 1.0,
        "area": 1.0,
        "geometry": 0.5,
        "spectral": 0.2,
        "detail": 0.05,
        "context": 0.05,
        "preservation": 0.05,
    }
    if weights is not None:
        if set(weights) != set(coefficients):
            raise ValueError("Unknown semantic loss coefficient")
        coefficients = weights
    if any(
        type(v) not in {int, float} or not math.isfinite(v) or v < 0 for v in coefficients.values()
    ):
        raise ValueError("Invalid semantic loss coefficient")
    if (
        set(labels) != {"worldcover", "area", "geometry"}
        or set(class_weights) != set(labels)
        or set(targets) != {"spectral", "detail", "context"}
        or spectral_mask.dtype != torch.bool
        or spectral_mask.shape != output.spectral_reconstruction.shape
    ):
        raise ValueError("Semantic loss schema or spectral domain differs")
    domain = spectral_mask.any(1)
    losses = {}
    for name, logits in [
        ("worldcover", output.worldcover_logits),
        ("area", output.area_logits),
        ("geometry", output.geometry_logits),
    ]:
        y = labels[name]
        cw = class_weights[name]
        if (
            y.shape != domain.shape
            or y.dtype != torch.long
            or (y < 0).any()
            or (y >= logits.shape[1]).any()
            or cw.shape != (logits.shape[1],)
            or not torch.isfinite(cw).all()
            or (cw < 0).any()
        ):
            raise ValueError("Invalid known weak semantic labels/weights")
        valid = (y > 0) & domain
        y = torch.where(valid, y, 0)
        cw = cw.to(y.device).float()
        ce = F.cross_entropy(logits.float(), y, weight=cw, ignore_index=0, reduction="sum")
        losses[name] = ce / torch.where(valid, cw[y], 0).sum().clamp_min(1)
    if targets["spectral"].shape != output.spectral_reconstruction.shape:
        raise ValueError("Physical teacher differs")
    losses["spectral"] = _masked_mean(
        (output.spectral_reconstruction.float() - targets["spectral"].detach().float()).square(),
        spectral_mask,
    )
    for name, pred, valid in [
        ("detail", output.detail_reconstruction, output.detail_mask),
        ("context", output.context_reconstruction, output.context_mask),
    ]:
        target = targets[name]
        if target is None:
            losses[name] = output.embedding.sum() * 0
        else:
            if target.shape != pred.shape:
                raise ValueError("Visual reconstruction teacher differs")
            losses[name] = _masked_mean((pred.float() - target.detach().float()).square(), valid)
    losses["preservation"] = _masked_mean(output.white_delta.float().square(), domain[:, None])
    losses["loss"] = sum(coefficients[k] * v for k, v in losses.items())
    return losses
