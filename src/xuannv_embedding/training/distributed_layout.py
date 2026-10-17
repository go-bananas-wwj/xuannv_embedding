"""Verify semantic parameter identity before DDP performs positional broadcasts."""

from __future__ import annotations

import hashlib
import json

import torch.distributed as dist
from torch import nn


def validate_distributed_layout(module: nn.Module) -> str:
    """Reject differing named layouts on every rank, before mutating any tensors.

    DDP checks shapes, but two sensors can have equal shapes and different names.
    Include frozen parameters and buffers so future architecture changes cannot
    silently reorder a registered branch either.
    """
    layout = [
        [kind, name, list(value.shape), str(value.dtype), value.requires_grad]
        for kind, values in (
            ("parameter", module.named_parameters()),
            ("buffer", module.named_buffers()),
        )
        for name, value in values
    ]
    fingerprint = hashlib.sha256(
        json.dumps(layout, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    if dist.is_available() and dist.is_initialized():
        layouts = [None] * dist.get_world_size()
        dist.all_gather_object(layouts, layout)
        reference = layouts[0]
        mismatches = [rank for rank, other in enumerate(layouts) if other != reference]
        if mismatches:
            raise ValueError(
                "DDP named parameter/buffer registration differs from rank 0 on ranks "
                f"{mismatches}; broadcast rejected before initialization"
            )
    return fingerprint
