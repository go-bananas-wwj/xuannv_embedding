"""可重复执行的生产模型发布门禁。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import RegionRasterDataset, collate_region_batch
from xuannv_embedding.models.model import AEFModel, AEFOutput
from xuannv_embedding.training.compatibility import load_compatible_checkpoint


class ReleaseValidationError(RuntimeError):
    """模型、数据或制品没有通过发布合同。"""


_LEGACY_SOURCE_NAMES = {
    "highres_optical": "highres_optical_haidian",
    "highres_sar": "highres_sar_haidian",
}
_LEGACY_TARGET_NAMES = {
    "highres_optical_recon": "highres_optical_haidian_recon",
    "highres_sar_recon": "highres_sar_haidian_recon",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_leaf(path: str, tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().cpu().contiguous().numpy()
    return {
        "path": path,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def model_input_evidence(batch: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return a serialization-independent digest for exactly the model inputs."""
    required = {
        "patch_ids",
        "source_frames",
        "source_masks",
        "timestamps",
        "highres_frames",
        "highres_masks",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise ReleaseValidationError(f"模型输入缺少字段: {missing}")
    patch_ids = batch["patch_ids"]
    if not isinstance(patch_ids, list) or not all(isinstance(item, str) for item in patch_ids):
        raise ReleaseValidationError("模型输入 patch_ids 必须是字符串列表")
    leaves: list[dict[str, Any]] = []
    for group in ("source_frames", "source_masks", "highres_frames", "highres_masks"):
        values = batch[group]
        if not isinstance(values, dict):
            raise ReleaseValidationError(f"模型输入 {group} 必须是 mapping")
        for name in sorted(values):
            tensor = values[name]
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ReleaseValidationError(f"模型输入 {group} 包含非法条目")
            leaves.append(_tensor_leaf(f"{group}/{name}", tensor))
    timestamps = batch["timestamps"]
    if not isinstance(timestamps, torch.Tensor):
        raise ReleaseValidationError("模型输入 timestamps 必须是 tensor")
    leaves.append(_tensor_leaf("timestamps", timestamps))
    document = {
        "schema_version": "xuannv_model_input_digest_v1",
        "patch_ids": list(patch_ids),
        "tensors": leaves,
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), document


def tensor_evidence(value: torch.Tensor | np.ndarray) -> tuple[str, dict[str, Any]]:
    """Return an exact dtype/shape/content digest for one output tensor."""
    array = (
        value.detach().float().cpu().contiguous().numpy()
        if isinstance(value, torch.Tensor)
        else np.ascontiguousarray(value)
    )
    document = {
        "schema_version": "xuannv_tensor_digest_v1",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), document


def _default_legacy_reference_manifest() -> Path:
    return Path(str(resources.files("xuannv_embedding.export").joinpath("legacy_reference.json")))


def _reference_member(root: Path, relative: Any, description: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ReleaseValidationError(f"{description} path 必须是相对路径")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReleaseValidationError(f"{description} path 逃逸 reference root") from exc
    if not path.is_file():
        raise ReleaseValidationError(f"{description} 不存在: {path}")
    return path


def validate_independent_legacy_reference(
    checkpoint: str | Path,
    batch: dict[str, Any],
    current_embedding: torch.Tensor,
    *,
    reference_root: str | Path,
    reference_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare current output with a frozen output produced by archived legacy source."""
    manifest_path = (
        Path(reference_manifest_path)
        if reference_manifest_path is not None
        else _default_legacy_reference_manifest()
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"无法读取独立旧运行时 reference manifest: {exc}") from exc
    required = {
        "schema_version",
        "profile",
        "legacy_runtime",
        "checkpoint_sha256",
        "input",
        "output",
        "environment",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ReleaseValidationError("独立旧运行时 reference manifest 字段不符合严格合同")
    if manifest.get("schema_version") != "xuannv_independent_legacy_reference_v1":
        raise ReleaseValidationError("独立旧运行时 reference schema_version 不受支持")
    if manifest.get("profile") != "haidian_p10c_v1":
        raise ReleaseValidationError("独立旧运行时 reference profile 不受支持")
    if _sha256(Path(checkpoint)) != manifest.get("checkpoint_sha256"):
        raise ReleaseValidationError("独立旧运行时 reference checkpoint SHA-256 不一致")

    legacy_runtime = manifest["legacy_runtime"]
    legacy_fields = {
        "archive_tag",
        "original_git_sha",
        "sanitized_git_sha",
        "models_tree_git_sha1",
    }
    if not isinstance(legacy_runtime, dict) or set(legacy_runtime) != legacy_fields:
        raise ReleaseValidationError("独立旧运行时 source provenance 不完整")
    for name in ("original_git_sha", "sanitized_git_sha", "models_tree_git_sha1"):
        value = legacy_runtime[name]
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ReleaseValidationError(f"独立旧运行时 {name} 非法")

    root = Path(reference_root).resolve()
    input_meta = manifest["input"]
    output_meta = manifest["output"]
    input_fields = {
        "path",
        "bytes",
        "file_sha256",
        "tensor_sha256",
        "patch_ids",
        "tensor_evidence",
    }
    output_fields = {"path", "bytes", "file_sha256", "tensor_sha256", "tensor_evidence"}
    if not isinstance(input_meta, dict) or set(input_meta) != input_fields:
        raise ReleaseValidationError("独立旧运行时 input metadata 不符合严格合同")
    if not isinstance(output_meta, dict) or set(output_meta) != output_fields:
        raise ReleaseValidationError("独立旧运行时 output metadata 不符合严格合同")
    input_path = _reference_member(root, input_meta["path"], "legacy input")
    output_path = _reference_member(root, output_meta["path"], "legacy output")
    for path, metadata, description in (
        (input_path, input_meta, "legacy input"),
        (output_path, output_meta, "legacy output"),
    ):
        if path.stat().st_size != metadata.get("bytes") or _sha256(path) != metadata.get(
            "file_sha256"
        ):
            raise ReleaseValidationError(f"{description} 文件大小或 SHA-256 不一致")

    try:
        frozen_input = torch.load(input_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as exc:
        raise ReleaseValidationError(f"无法安全读取 legacy input: {exc}") from exc
    if not isinstance(frozen_input, dict):
        raise ReleaseValidationError("legacy input 顶层必须是 mapping")
    frozen_input_digest, frozen_input_document = model_input_evidence(frozen_input)
    if frozen_input_digest != input_meta.get(
        "tensor_sha256"
    ) or frozen_input_document != input_meta.get("tensor_evidence"):
        raise ReleaseValidationError("独立旧运行时 input tensor evidence 不一致")
    live_input_digest, _ = model_input_evidence(batch)
    if live_input_digest != frozen_input_digest:
        raise ReleaseValidationError("当前真实 batch 与独立旧运行时 reference input 不一致")

    try:
        frozen_output = np.load(output_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ReleaseValidationError(f"无法安全读取 legacy output: {exc}") from exc
    frozen_output_digest, frozen_output_document = tensor_evidence(frozen_output)
    if frozen_output_digest != output_meta.get(
        "tensor_sha256"
    ) or frozen_output_document != output_meta.get("tensor_evidence"):
        raise ReleaseValidationError("独立旧运行时 output tensor evidence 不一致")
    current = current_embedding.detach().float().cpu().contiguous().numpy()
    if not np.array_equal(current, frozen_output):
        max_difference = (
            float(np.max(np.abs(current - frozen_output)))
            if current.shape == frozen_output.shape
            else float("inf")
        )
        raise ReleaseValidationError(
            f"当前 embedding 与独立旧运行时 reference 不一致: "
            f"max_abs_difference={max_difference:.8g}"
        )
    return {
        "embedding_exact": True,
        "input_tensor_sha256": live_input_digest,
        "output_tensor_sha256": frozen_output_digest,
        "legacy_runtime": dict(legacy_runtime),
        "reference_environment": dict(manifest["environment"]),
    }


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(child, device) for key, child in value.items()}
    return value


def _model_inputs(batch: dict[str, Any]) -> tuple[Any, ...]:
    return (
        batch["source_frames"],
        batch["source_masks"],
        batch["timestamps"],
        batch["highres_frames"],
        batch["highres_masks"],
    )


def validate_embedding_output(
    output: AEFOutput,
    *,
    expected_batch: int,
    expected_months: int,
    expected_dim: int,
    expected_size: tuple[int, int],
    norm_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """核验 embedding 形状、有限值与 vMF 单位范数。"""
    expected_shape = (
        expected_batch,
        expected_months,
        expected_dim,
        *expected_size,
    )
    actual_shape = tuple(output.embedding_map.shape)
    if actual_shape != expected_shape:
        raise ReleaseValidationError(
            f"embedding 形状错误: expected={expected_shape}, actual={actual_shape}"
        )
    if not bool(torch.isfinite(output.embedding_map).all().item()):
        raise ReleaseValidationError("embedding 包含非有限值")
    norms = torch.linalg.vector_norm(output.embedding_map.float(), dim=2)
    max_abs_error = float((norms - 1.0).abs().max().cpu())
    if max_abs_error > norm_tolerance:
        raise ReleaseValidationError(
            f"vMF embedding 未保持单位范数: max_abs_error={max_abs_error:.8g}"
        )
    return {
        "shape": list(actual_shape),
        "finite": True,
        "vmf_mean_norm": float(norms.mean().cpu()),
        "vmf_std_norm": float(norms.std().cpu()),
        "vmf_max_abs_error": max_abs_error,
    }


def validate_missing_source(batch: dict[str, Any], source: str) -> dict[str, Any]:
    """缺失模态不得进入有效输入或重建监督。"""
    try:
        availability = batch["highres_masks"][source]
        supervision = batch["target_masks"][f"{source}_recon"]
    except KeyError as exc:
        raise ReleaseValidationError(f"缺失模态门禁找不到字段: {exc}") from exc
    availability_nonzero = int(torch.count_nonzero(availability).item())
    supervision_nonzero = int(torch.count_nonzero(supervision).item())
    if availability_nonzero:
        raise ReleaseValidationError(f"缺失模态 {source!r} 仍有有效 availability")
    if supervision_nonzero:
        raise ReleaseValidationError(f"缺失模态 {source!r} 仍有重建监督")
    return {
        "source": source,
        "availability_nonzero": availability_nonzero,
        "supervision_nonzero": supervision_nonzero,
    }


def build_legacy_haidian_model(config: Config) -> AEFModel:
    """构造原 P10C 物理 source 命名的只读对照模型。"""
    sensor_channels = {
        _LEGACY_SOURCE_NAMES.get(name, name): source.channels
        for name, source in config.model.input_sources.items()
    }
    source_roles = {
        _LEGACY_SOURCE_NAMES.get(name, name): source.role
        for name, source in config.model.input_sources.items()
    }
    target_heads = {
        _LEGACY_TARGET_NAMES.get(name, name): (head.loss_type, head.channels)
        for name, head in config.model.target_heads.items()
    }
    return AEFModel(
        sensor_channels=sensor_channels,
        embed_dim=config.model.embed_dim,
        target_heads=target_heads,
        stem_dim=config.model.stem_dim,
        stp=asdict(config.model.stp),
        num_months=config.model.num_months,
        ref_year=config.model.ref_year,
        ref_month=config.model.ref_month,
        gradient_checkpointing=config.training.gradient_checkpointing,
        source_roles=source_roles,
    ).eval()


def _legacy_batch(batch: dict[str, Any]) -> dict[str, Any]:
    result = dict(batch)
    result["highres_frames"] = {
        _LEGACY_SOURCE_NAMES.get(name, name): value
        for name, value in batch["highres_frames"].items()
    }
    result["highres_masks"] = {
        _LEGACY_SOURCE_NAMES.get(name, name): value
        for name, value in batch["highres_masks"].items()
    }
    return result


def _load_legacy_state(checkpoint: Path, model: AEFModel) -> None:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_state = state["model"]
        if not isinstance(model_state, dict):
            raise TypeError("model 不是 mapping")
        model.load_state_dict(model_state, strict=True)
    except (KeyError, OSError, RuntimeError, TypeError) as exc:
        raise ReleaseValidationError(f"旧模型状态无法严格加载: {exc}") from exc


def compare_haidian_checkpoint(
    checkpoint: str | Path,
    config: Config,
    batch: dict[str, Any],
    *,
    device: torch.device,
    legacy_reference_root: str | Path,
) -> dict[str, Any]:
    """同一真实 batch 上严格比较原模型与规范命名模型。"""
    from xuannv_embedding.training.cli import build_training_system

    checkpoint_path = Path(checkpoint)
    before_sha256 = _sha256(checkpoint_path)
    canonical = build_training_system(config).model.eval()
    compatibility = load_compatible_checkpoint(
        checkpoint_path,
        canonical,
        profile="haidian_p10c_v1",
        device="cpu",
    )
    legacy = build_legacy_haidian_model(config)
    _load_legacy_state(checkpoint_path, legacy)
    canonical.to(device)
    legacy.to(device)
    moved = _move(batch, device)
    legacy_moved = _legacy_batch(moved)
    with torch.no_grad():
        canonical_output = canonical(*_model_inputs(moved))
        legacy_output = legacy(*_model_inputs(legacy_moved))
    if not torch.equal(canonical_output.embedding_map, legacy_output.embedding_map):
        max_difference = float(
            (canonical_output.embedding_map - legacy_output.embedding_map).abs().max().cpu()
        )
        raise ReleaseValidationError(
            f"原/新 embedding 不精确一致: max_abs_difference={max_difference:.8g}"
        )
    first_temporal = next(iter(moved["source_frames"].values()))
    report = validate_embedding_output(
        canonical_output,
        expected_batch=len(batch["patch_ids"]),
        expected_months=config.model.num_months,
        expected_dim=config.model.embed_dim,
        expected_size=tuple(first_temporal.shape[-2:]),
    )
    after_sha256 = _sha256(checkpoint_path)
    if after_sha256 != before_sha256:
        raise ReleaseValidationError("兼容门禁修改了原 checkpoint")
    report.update(
        {
            "checkpoint_sha256": after_sha256,
            "checkpoint_unchanged": True,
            "consumed_keys": compatibility.consumed_keys,
            "embedding_exact": True,
            "patch_ids": list(batch["patch_ids"]),
            "independent_legacy_reference": validate_independent_legacy_reference(
                checkpoint_path,
                batch,
                canonical_output.embedding_map,
                reference_root=legacy_reference_root,
            ),
        }
    )
    return report


def validate_harbin_missing_modality(
    checkpoint: str | Path,
    config: Config,
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """加载冻结 P10C 并验证真实哈尔滨缺 SAR 的数值合同。"""
    from xuannv_embedding.training.cli import build_training_system

    missing_report = validate_missing_source(batch, "highres_sar")
    model = build_training_system(config).model.eval()
    compatibility = load_compatible_checkpoint(
        checkpoint,
        model,
        profile="haidian_p10c_v1",
        device="cpu",
    )
    model.to(device)
    moved = _move(batch, device)
    with torch.no_grad():
        output = model(*_model_inputs(moved))
    first_temporal = next(iter(moved["source_frames"].values()))
    report = validate_embedding_output(
        output,
        expected_batch=len(batch["patch_ids"]),
        expected_months=config.model.num_months,
        expected_dim=config.model.embed_dim,
        expected_size=tuple(first_temporal.shape[-2:]),
    )
    report.update(
        {
            "consumed_keys": compatibility.consumed_keys,
            "missing_modality": missing_report,
            "patch_ids": list(batch["patch_ids"]),
        }
    )
    return report


def _batch(config: Config, region: str, limit: int) -> dict[str, Any]:
    dataset = RegionRasterDataset(
        config,
        config.data.dataset_for_region(region),
        max_records=limit,
    )
    if len(dataset) < limit:
        raise ReleaseValidationError(f"{region} 记录不足: required={limit}, actual={len(dataset)}")
    return collate_region_batch([dataset[index] for index in range(limit)])


def _environment(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "device": str(device),
    }
    try:
        report["torch_npu"] = importlib.metadata.version("torch-npu")
    except importlib.metadata.PackageNotFoundError:
        report["torch_npu"] = None
    if device.type == "npu":
        report["device_name"] = torch.npu.get_device_name(device)
    return report


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行海淀兼容与哈尔滨缺模态 NPU 门禁")
    parser.add_argument("--haidian-config", type=Path, required=True)
    parser.add_argument("--harbin-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-reference-root", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--haidian-limit", type=int, default=2)
    parser.add_argument("--harbin-limit", type=int, default=1)
    args = parser.parse_args(argv)
    if args.haidian_limit <= 0 or args.harbin_limit <= 0:
        parser.error("limit 必须是正整数")
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(device)
    haidian_config = Config.from_yaml(args.haidian_config)
    harbin_config = Config.from_yaml(args.harbin_config)
    payload = {
        "schema_version": "1",
        "environment": _environment(device),
        "haidian": compare_haidian_checkpoint(
            args.checkpoint,
            haidian_config,
            _batch(haidian_config, "haidian", args.haidian_limit),
            device=device,
            legacy_reference_root=args.legacy_reference_root,
        ),
        "harbin": validate_harbin_missing_modality(
            args.checkpoint,
            harbin_config,
            _batch(harbin_config, "harbin", args.harbin_limit),
            device=device,
        ),
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
