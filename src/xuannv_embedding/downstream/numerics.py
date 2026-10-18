"""Scoped deterministic execution for frozen-embedding readouts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def city_readout_numerics(
    device: torch.device, *, deterministic: bool = False
) -> Iterator[dict | None]:
    """Keep legacy execution unchanged, or require deterministic CPU/CUDA readouts.

    Torch backend flags are process-global and restored on exit. A default CUDA
    workspace setting, installed before CUDA initialization, remains in the
    process environment because an initialized library cannot be reconfigured.
    Do not overlap this context with other numeric policies in the same process.
    """
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be a boolean")
    if not deterministic:
        yield None
        return
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Deterministic city readouts support CPU and CUDA only")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device.type == "cuda":
        if workspace is None:
            if torch.cuda.is_initialized():
                raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
            workspace = ":4096:8"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
        elif workspace not in {":4096:8", ":16:8"}:
            raise ValueError("Deterministic city readouts require a deterministic cuBLAS workspace")
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        policy = {
            "schema": "xuannv.city-readout-numerics.v1",
            "deterministic_algorithms": True,
            "warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_config": workspace if device.type == "cuda" else None,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        }
        yield policy
    finally:
        enabled, warn_only, cudnn_deterministic, cudnn_benchmark = previous
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
