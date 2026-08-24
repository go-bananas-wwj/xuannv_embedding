"""可重复执行的生产模型发布门禁。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

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
