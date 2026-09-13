"""Experimental Ascend OM runtime for the two official OmniCloudMask V4 models."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np


def _host_array_pointer(acl: Any, array: np.ndarray) -> int:
    """Retain the established NumPy buffer API without flooding data-processing logs."""
    with warnings.catch_warnings(record=True) as captured:
        pointer = acl.util.numpy_to_ptr(array)
    for warning in captured:
        if not str(warning.message).startswith("acl.util.numpy_to_ptr will be deprecated."):
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    return pointer


class OmLogitRunner(Protocol):
    def infer(self, array: np.ndarray) -> list[np.ndarray]: ...

    def close(self) -> None: ...


def _check(ret: int, operation: str, acl: Any) -> None:
    if ret != 0:
        raise RuntimeError(f"Ascend ACL {operation} 失败: ret={ret}: {acl.get_recent_err_msg()}")


class AscendAclOmEnsemble:
    """Load fixed-shape OM models once and reuse their device buffers."""

    _ACL_MEM_MALLOC_HUGE_FIRST = 0
    _ACL_MEMCPY_HOST_TO_DEVICE = 1
    _ACL_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self, model_paths: list[Path], device_id: int) -> None:
        if len(model_paths) != 2:
            raise ValueError("OmniCloudMask V4 NPU ensemble 必须提供两个 OM 子模型")
        missing = [str(path) for path in model_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"OmniCloudMask OM 模型不存在: {missing}")
        try:
            import acl
        except ImportError as exc:
            ascend_home = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/latest"))
            acl_site = ascend_home / "python" / "site-packages"
            if acl_site.is_dir() and str(acl_site) not in sys.path:
                sys.path.insert(0, str(acl_site))
            try:
                import acl
            except ImportError:
                raise RuntimeError(f"当前环境没有 CANN Python ACL runtime: {acl_site}") from exc
        self.acl = acl
        self.device_id = device_id
        self.closed = False
        _check(acl.init(), "init", acl)
        _check(acl.rt.set_device(device_id), "set_device", acl)
        self.context, ret = acl.rt.create_context(device_id)
        _check(ret, "create_context", acl)
        self.models: list[dict[str, Any]] = []
        try:
            for path in model_paths:
                self.models.append(self._load_model(path))
        except BaseException:
            self.close()
            raise

    def _load_model(self, path: Path) -> dict[str, Any]:
        acl = self.acl
        model_id, ret = acl.mdl.load_from_file(str(path))
        _check(ret, f"load_from_file({path})", acl)
        desc = acl.mdl.create_desc()
        _check(acl.mdl.get_desc(desc, model_id), "get_desc", acl)
        input_dims, ret = acl.mdl.get_input_dims(desc, 0)
        _check(ret, "get_input_dims", acl)
        output_dims, ret = acl.mdl.get_output_dims(desc, 0)
        _check(ret, "get_output_dims", acl)
        if input_dims["dims"] != [1, 3, 96, 96] or output_dims["dims"] != [1, 4, 96, 96]:
            raise ValueError(f"OM 固定形状合同错误: {input_dims['dims']} -> {output_dims['dims']}")
        input_size = acl.mdl.get_input_size_by_index(desc, 0)
        output_size = acl.mdl.get_output_size_by_index(desc, 0)
        input_ptr, ret = acl.rt.malloc(input_size, self._ACL_MEM_MALLOC_HUGE_FIRST)
        _check(ret, "malloc(input)", acl)
        output_ptr, ret = acl.rt.malloc(output_size, self._ACL_MEM_MALLOC_HUGE_FIRST)
        _check(ret, "malloc(output)", acl)
        input_buffer = acl.create_data_buffer(input_ptr, input_size)
        output_buffer = acl.create_data_buffer(output_ptr, output_size)
        input_dataset = acl.mdl.create_dataset()
        output_dataset = acl.mdl.create_dataset()
        _, ret = acl.mdl.add_dataset_buffer(input_dataset, input_buffer)
        _check(ret, "add_dataset_buffer(input)", acl)
        _, ret = acl.mdl.add_dataset_buffer(output_dataset, output_buffer)
        _check(ret, "add_dataset_buffer(output)", acl)
        return {
            "model_id": model_id,
            "desc": desc,
            "input_size": input_size,
            "output_size": output_size,
            "input_ptr": input_ptr,
            "output_ptr": output_ptr,
            "input_buffer": input_buffer,
            "output_buffer": output_buffer,
            "input_dataset": input_dataset,
            "output_dataset": output_dataset,
        }

    def infer(self, array: np.ndarray) -> list[np.ndarray]:
        values = np.ascontiguousarray(array[None], dtype=np.float32)
        if values.shape != (1, 3, 96, 96):
            raise ValueError(f"Ascend OM 输入必须为 [3,96,96]: {array.shape}")
        results = []
        acl = self.acl
        for model in self.models:
            _check(
                acl.rt.memcpy(
                    model["input_ptr"],
                    model["input_size"],
                    _host_array_pointer(acl, values),
                    values.nbytes,
                    self._ACL_MEMCPY_HOST_TO_DEVICE,
                ),
                "memcpy(H2D)",
                acl,
            )
            _check(
                acl.mdl.execute(model["model_id"], model["input_dataset"], model["output_dataset"]),
                "execute",
                acl,
            )
            output = np.empty(model["output_size"] // 4, dtype=np.float32)
            _check(
                acl.rt.memcpy(
                    _host_array_pointer(acl, output),
                    output.nbytes,
                    model["output_ptr"],
                    model["output_size"],
                    self._ACL_MEMCPY_DEVICE_TO_HOST,
                ),
                "memcpy(D2H)",
                acl,
            )
            results.append(output.reshape(4, 96, 96))
        return results

    def close(self) -> None:
        if getattr(self, "closed", True):
            return
        acl = self.acl
        for model in reversed(getattr(self, "models", [])):
            acl.mdl.destroy_dataset(model["input_dataset"])
            acl.mdl.destroy_dataset(model["output_dataset"])
            acl.destroy_data_buffer(model["input_buffer"])
            acl.destroy_data_buffer(model["output_buffer"])
            acl.rt.free(model["input_ptr"])
            acl.rt.free(model["output_ptr"])
            acl.mdl.destroy_desc(model["desc"])
            acl.mdl.unload(model["model_id"])
        if getattr(self, "context", None) is not None:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        self.closed = True

    def __del__(self) -> None:
        self.close()


def _patch_indexes(height: int, width: int) -> list[tuple[int, int, int, int]]:
    patch_size, stride = 96, 64
    max_top, max_left = height - patch_size, width - patch_size
    indexes = []
    for top in range(0, height, stride):
        top = min(top, max_top)
        for left in range(0, width, stride):
            left = min(left, max_left)
            index = (top, top + patch_size, left, left + patch_size)
            if index not in indexes:
                indexes.append(index)
    return indexes


def _channel_norm(patch: np.ndarray) -> np.ndarray:
    result = np.zeros(patch.shape, dtype=np.float32)
    for index, band in enumerate(patch):
        valid = band != 0
        if valid.any():
            values = band[valid]
            std = float(values.std()) or 1.0
            result[index, valid] = (values - float(values.mean())) / std
    return result


def _gradient() -> np.ndarray:
    overlap = 32
    gradient = np.ones((96, 96), dtype=np.float32) * overlap
    gradient[:, :overlap] = np.arange(1, overlap + 1)
    gradient[:, -overlap:] = np.arange(overlap, 0, -1)
    gradient /= overlap
    return np.rot90(gradient) * gradient


class AscendOmniCloudMaskV4Predictor:
    """Run official V4 weights as two ATC-compiled models and ensemble their logits."""

    version = "4.0-ascend-om"

    def __init__(
        self,
        *,
        model_paths: list[Path],
        device_id: int,
        runner_factory: Callable[[list[Path], int], OmLogitRunner] = AscendAclOmEnsemble,
    ) -> None:
        self.device = f"npu:{device_id}"
        self.runner = runner_factory(model_paths, device_id)

    def predict_batch(self, arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        labels = []
        confidences = []
        gradient = _gradient()
        for array in arrays:
            if array.ndim != 3 or array.shape[0] != 3 or min(array.shape[1:]) < 96:
                raise ValueError(f"OmniCloudMask NPU 输入形状非法: {array.shape}")
            logits = np.zeros((4, *array.shape[1:]), dtype=np.float32)
            weights = np.zeros(array.shape[1:], dtype=np.float32)
            for top, bottom, left, right in _patch_indexes(*array.shape[1:]):
                patch = _channel_norm(array[:, top:bottom, left:right])
                if not np.any(patch):
                    continue
                ensemble = np.mean(self.runner.infer(patch), axis=0)
                logits[:, top:bottom, left:right] += ensemble * gradient[None]
                weights[top:bottom, left:right] += gradient
            normalized = np.divide(
                logits,
                weights[None],
                out=np.zeros_like(logits),
                where=weights[None] > 0,
            )
            shifted = normalized - normalized.max(axis=0, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=0, keepdims=True).clip(min=1e-12)
            probabilities = np.clip(probabilities + 0.001, 0.001, 0.999)
            data_valid = ~np.all(array == 0, axis=0)
            probabilities *= data_valid[None]
            labels.append(np.argmax(probabilities, axis=0).astype(np.uint8))
            confidences.append(np.max(probabilities, axis=0).astype(np.float32))
        return np.stack(labels), np.stack(confidences)

    def close(self) -> None:
        self.runner.close()
