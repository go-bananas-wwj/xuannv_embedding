from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import torch
from torch import nn


class CompatibilityError(ValueError):
    """旧 checkpoint 未通过登记、键映射或严格消费合同。"""


@dataclass(frozen=True)
class CompatibilityReport:
    profile: str
    checkpoint_sha256: str
    consumed_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_artifact_manifest() -> Path:
    return Path(str(resources.files("xuannv_embedding.export").joinpath("artifacts.json")))


def _load_artifact_entry(path: Path, profile: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompatibilityError(f"无法读取 artifact manifest {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != "1":
        raise CompatibilityError("artifact manifest schema_version 必须为 '1'")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, dict) or profile not in artifacts:
        raise CompatibilityError(f"artifact manifest 未登记 profile: {profile!r}")
    entry = artifacts[profile]
    required = {"filename", "sha256", "model_key_count", "compatibility_profile"}
    if not isinstance(entry, dict) or set(entry) != required:
        raise CompatibilityError(f"artifact {profile!r} 字段不符合严格合同")
    if entry["compatibility_profile"] != profile:
        raise CompatibilityError("artifact compatibility_profile 与请求不一致")
    if not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64:
        raise CompatibilityError("artifact sha256 非法")
    if isinstance(entry["model_key_count"], bool) or not isinstance(entry["model_key_count"], int):
        raise CompatibilityError("artifact model_key_count 非法")
    return entry


def remap_haidian_p10c_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """把 P10C 海淀物理 source 键一对一映射到规范 source 键。"""
    mapped: dict[str, torch.Tensor] = {}
    for old_key, tensor in state_dict.items():
        new_key = old_key.replace("highres_optical_haidian", "highres_optical")
        new_key = new_key.replace("highres_sar_haidian", "highres_sar")
        if new_key in mapped:
            raise CompatibilityError(
                f"checkpoint 键映射冲突: {old_key!r} 与另一键均映射到 {new_key!r}"
            )
        mapped[new_key] = tensor
    if len(mapped) != len(state_dict):
        raise CompatibilityError("checkpoint 键映射未一对一消费")
    return mapped


def load_compatible_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    *,
    profile: str,
    artifact_manifest_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> CompatibilityReport:
    """校验登记 SHA 后映射旧键，并严格加载全部模型状态。"""
    if profile != "haidian_p10c_v1":
        raise CompatibilityError(f"不支持的 compatibility profile: {profile!r}")
    checkpoint = Path(checkpoint_path)
    manifest = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else _default_artifact_manifest()
    )
    entry = _load_artifact_entry(manifest, profile)
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != entry["sha256"]:
        raise CompatibilityError(
            f"checkpoint SHA-256 未登记: expected={entry['sha256']}, actual={actual_sha256}"
        )
    try:
        checkpoint_state = torch.load(checkpoint, map_location=device, weights_only=True)
    except (OSError, RuntimeError) as exc:
        raise CompatibilityError(f"无法安全读取 checkpoint: {exc}") from exc
    if not isinstance(checkpoint_state, dict) or not isinstance(
        checkpoint_state.get("model"), dict
    ):
        raise CompatibilityError("checkpoint 缺少 model state_dict")
    old_state = checkpoint_state["model"]
    if len(old_state) != entry["model_key_count"]:
        raise CompatibilityError(
            f"checkpoint 键数不匹配: expected={entry['model_key_count']}, actual={len(old_state)}"
        )
    mapped = remap_haidian_p10c_state_dict(old_state)
    expected_keys = set(model.state_dict())
    mapped_keys = set(mapped)
    missing = tuple(sorted(expected_keys - mapped_keys))
    unexpected = tuple(sorted(mapped_keys - expected_keys))
    if missing or unexpected:
        raise CompatibilityError(
            f"checkpoint 未完整消费: missing={list(missing)}, unexpected={list(unexpected)}"
        )
    try:
        model.load_state_dict(mapped, strict=True)
    except RuntimeError as exc:
        raise CompatibilityError(f"checkpoint 张量形状或类型不兼容: {exc}") from exc
    return CompatibilityReport(
        profile=profile,
        checkpoint_sha256=actual_sha256,
        consumed_keys=len(mapped),
        missing_keys=missing,
        unexpected_keys=unexpected,
    )
