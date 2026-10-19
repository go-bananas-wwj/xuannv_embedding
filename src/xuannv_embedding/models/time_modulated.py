"""Condition actual local views on observed coeval public satellite features."""

from __future__ import annotations

import math

import torch
from torch import nn

from xuannv_embedding.models.semantic_residual import SemanticResidualFusion


class TimeModulatedFusion(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.base = SemanticResidualFusion(**kwargs)
        self.affine = nn.Conv2d(64, 2048, 1)
        self.channel_gate = nn.Conv2d(64, 1024, 1)
        self.attention = nn.Conv2d(1090, 1, 1)
        for layer in [self.affine, self.channel_gate, self.attention]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.channel_gate.bias, -2)

    def set_statistics(self, name, mean, std):
        self.base.set_statistics(name, mean, std)

    def standardize(self, *args, **kwargs):
        return self.base.standardize(*args, **kwargs)

    def forward(
        self,
        public,
        spectral,
        detail,
        views,
        view_mask,
        queries,
        query_mask,
        support,
        *,
        spectral_mask,
        context_source_shape,
        burnin=False,
    ):
        b = len(public)
        if (
            views.ndim != 5
            or views.shape[0] != b
            or views.shape[2] != 1024
            or view_mask.shape != (b, views.shape[1], 1, *views.shape[-2:])
            or queries.ndim != 5
            or queries.shape[0] != b
            or queries.shape[2] != 64
            or queries.shape[-2:] != views.shape[-2:]
            or query_mask.shape != (b, queries.shape[1], 1, *queries.shape[-2:])
            or support.shape != (views.shape[1], queries.shape[1])
            or views.shape[1] < 1
            or queries.shape[1] < 1
        ):
            raise ValueError("Local views/public queries/support geometry differs")
        if (
            not torch.isfinite(support).all()
            or (support < 0).any()
            or not torch.allclose(support.sum(1), torch.ones_like(support[:, 0]), atol=1e-6)
        ):
            raise ValueError("Invalid physical month support")
        for mask in [view_mask, query_mask]:
            if not torch.isfinite(mask).all() or ((mask < 0) | (mask > 1)).any():
                raise ValueError("Observation coverage must be in [0,1]")
        for values, mask in [(views, view_mask), (queries, query_mask)]:
            if not torch.isfinite(torch.where(mask > 0, values, 0)).all():
                raise ValueError("Observed conditional inputs must be finite")
        v, _, h, w = views.shape[1:]
        m = view_mask.detach().float()
        qm = query_mask.detach().float()
        local = torch.where(m > 0, views.detach().float(), 0)
        local = torch.where(
            m > 0,
            ((local - self.base.context_mean[:, None]) / self.base.context_std[:, None]).clamp(
                -6, 6
            ),
            0,
        )
        query = torch.where(qm > 0, queries.detach().float(), 0)
        query = torch.where(
            qm > 0,
            ((query - self.base.public_mean[:, None]) / self.base.public_std[:, None]).clamp(-6, 6),
            0,
        )
        support = support.detach().float()
        available = torch.einsum("vt,btchw->bvchw", support, qm)
        paired = torch.einsum("vt,btchw->bvchw", support, query * qm)
        paired = torch.where(available > 0, paired / available.clamp_min(1e-6), 0)
        q = paired.reshape(b * v, 64, h, w)
        gamma, beta = self.affine(q).float().chunk(2, 1)
        gate = self.channel_gate(q).float().sigmoid()
        modulated = local.reshape(b * v, 1024, h, w) + 0.5 * gate * (
            gamma.tanh() * local.reshape(b * v, 1024, h, w) + beta.tanh()
        ) * (available.reshape(b * v, 1, h, w) > 0)
        entropy = -(support * support.clamp_min(1e-8).log()).sum(1)
        entropy = entropy / math.log(max(2, support.shape[1]))
        entropy = entropy[None, :, None, None, None].expand(b, v, 1, h, w)
        scores = self.attention(
            torch.cat([local, paired, m, entropy], 2).reshape(b * v, 1090, h, w)
        )
        scores = scores.float().reshape(b, v, 1, h, w)
        scores = torch.where(m > 0, scores, -10000)
        weights = torch.softmax(scores, 1) * m
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        pooled = (modulated.reshape(b, v, 1024, h, w) * weights).sum(1)
        coverage = m.amax(1)
        raw = torch.where(coverage > 0, pooled * self.base.context_std + self.base.context_mean, 0)
        return self.base(
            public,
            spectral,
            detail,
            raw,
            coverage,
            spectral_mask=spectral_mask,
            context_source_shape=context_source_shape,
            burnin=burnin,
            trainable_context=True,
        )


def diversity_loss(white_features, domain, *, step):
    """VICReg-inspired masked moments, without a twin-view invariance objective."""
    if (
        white_features.ndim != 4
        or white_features.shape[1] != 64
        or domain.shape != (len(white_features), 1, *white_features.shape[-2:])
        or domain.dtype != torch.bool
        or type(step) is not int
        or step < 0
    ):
        raise ValueError("Invalid diversity feature/domain/step")
    dy, dx = divmod(step % 16, 4)
    x = white_features.float()[:, :, dy::4, dx::4].permute(0, 2, 3, 1).reshape(-1, 64)
    m = domain[:, :, dy::4, dx::4].reshape(-1, 1)
    n = m.float().sum()
    x = torch.where(m, x, 0)
    mean = x.sum(0) / n.clamp_min(1)
    centered = torch.where(m, x - mean, 0)
    covariance = centered.T @ centered / (n - 1).clamp_min(1)
    diagonal = covariance.diagonal()
    variance = (1 - torch.sqrt(diagonal.clamp_min(0) + 1e-4)).relu().mean()
    offdiag = covariance - torch.diag_embed(diagonal)
    return {
        "variance": torch.where(n > 1, variance, 0),
        "covariance": torch.where(n > 1, offdiag.square().sum() / 64, 0),
    }
