"""Observe actual parameter changes without modifying gradients or Adam's state."""

from __future__ import annotations

from collections import defaultdict

import torch


def parameter_group(name):
    if name.startswith("criterion.target_supervision."):
        return "target_heads"
    if not name.startswith("model."):
        return None
    name = name.removeprefix("model.")
    if name.startswith("temporal_encoder."):
        return "stp"
    if name.startswith(("temporal_encoders.", "lowres_pool.", "source_weights")):
        return "lowres_stems_pool"
    if name.startswith(("highres_encoders.", "highres_projections.", "highres_pools.")):
        return "highres"
    if name.startswith(("joint.", "gate.", "branch_norms.")):
        return "fusion"
    if name.startswith("bottleneck.proj."):
        return "vmf_projection"
    if name.startswith(("decoders.", "highres_decoders.")):
        return "image_decoders"
    return "other_model"


def snapshot_parameters(system):
    groups = defaultdict(list)
    with torch.no_grad():
        for name, parameter in system.named_parameters():
            group = parameter_group(name)
            if group is not None:
                groups[group].append((parameter, parameter.detach().clone()))
    return dict(groups)


def measure_updates(snapshot):
    result = {}
    with torch.no_grad():
        for name, rows in snapshot.items():
            values = []
            count = 0
            for parameter, before in rows:
                delta = parameter.detach().float() - before.float()
                grad = parameter.grad
                gradient = torch.zeros_like(delta) if grad is None else grad.detach().float()
                values.append(
                    torch.stack(
                        (
                            delta.square().sum(),
                            before.float().square().sum(),
                            gradient.square().sum(),
                            -(delta * gradient).sum(),
                        )
                    )
                )
                count += parameter.numel()
            total = torch.stack(values).sum(0)
            if not bool(torch.isfinite(total).all()):
                raise FloatingPointError("Nonfinite actual parameter update diagnostics")
            delta_norm, weight_norm, gradient_norm = total[:3].sqrt().tolist()
            result[name] = {
                "parameters": count,
                "update_l2": delta_norm,
                "weight_l2_before": weight_norm,
                "raw_gradient_l2": gradient_norm,
                "relative_update_l2": delta_norm / weight_norm if weight_norm else None,
                "update_vs_negative_raw_gradient_cosine": (
                    max(-1.0, min(1.0, float(total[3]) / (delta_norm * gradient_norm)))
                    if delta_norm and gradient_norm
                    else None
                ),
            }
    return result
