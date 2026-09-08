from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from xuannv_embedding.config import Config
from xuannv_embedding.export.validation import (
    ReleaseValidationError,
    _environment,
    _resolve_device,
    build_legacy_haidian_model,
    model_input_evidence,
    tensor_evidence,
    validate_embedding_output,
    validate_independent_legacy_reference,
    validate_missing_source,
)
from xuannv_embedding.models.model import AEFOutput


def _output(values: torch.Tensor) -> AEFOutput:
    return AEFOutput(
        embedding_map=values,
        embedding=values.mean(dim=(-2, -1)),
        reconstructions={},
    )


def test_embedding_release_contract_reports_shape_finite_and_vmf_norm() -> None:
    values = torch.zeros(2, 6, 64, 128, 128)
    values[:, :, 0] = 1.0

    report = validate_embedding_output(
        _output(values),
        expected_batch=2,
        expected_months=6,
        expected_dim=64,
        expected_size=(128, 128),
    )

    assert report["shape"] == [2, 6, 64, 128, 128]
    assert report["finite"] is True
    assert report["vmf_max_abs_error"] == 0.0


def test_embedding_release_contract_rejects_nonfinite_or_nonunit_values() -> None:
    nonfinite = torch.zeros(1, 1, 2, 16, 16)
    nonfinite[:, :, 0] = 1.0
    nonfinite[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ReleaseValidationError, match="非有限"):
        validate_embedding_output(
            _output(nonfinite),
            expected_batch=1,
            expected_months=1,
            expected_dim=2,
            expected_size=(16, 16),
        )

    nonunit = torch.ones(1, 1, 2, 16, 16)
    with pytest.raises(ReleaseValidationError, match="单位范数"):
        validate_embedding_output(
            _output(nonunit),
            expected_batch=1,
            expected_months=1,
            expected_dim=2,
            expected_size=(16, 16),
        )


def test_missing_source_requires_zero_input_and_reconstruction_masks() -> None:
    batch = {
        "highres_masks": {"highres_sar": torch.zeros(1, 1, 16, 16)},
        "target_masks": {"highres_sar_recon": torch.zeros(1, 6, 16, 16)},
    }
    assert validate_missing_source(batch, "highres_sar") == {
        "source": "highres_sar",
        "availability_nonzero": 0,
        "supervision_nonzero": 0,
    }

    batch["target_masks"]["highres_sar_recon"][0, 0, 0, 0] = 1
    with pytest.raises(ReleaseValidationError, match="重建监督"):
        validate_missing_source(batch, "highres_sar")


def test_legacy_model_builder_only_renames_registered_highres_sources() -> None:
    config = Config.from_yaml("configs/production/haidian_p10c_v1.yaml")
    legacy = build_legacy_haidian_model(config)

    assert legacy.source_roles["s2"] == "temporal"
    assert legacy.source_roles["highres_optical_haidian"] == "highres"
    assert legacy.source_roles["highres_sar_haidian"] == "highres"
    assert "highres_optical_haidian_recon" in legacy.target_heads
    assert "highres_sar_haidian_recon" in legacy.target_heads
    assert legacy.stp_cfg["space_dim"] == asdict(config.model.stp)["space_dim"]


def test_independent_legacy_reference_binds_input_checkpoint_and_exact_output(
    tmp_path: Path,
) -> None:
    batch = {
        "patch_ids": ["p1"],
        "source_frames": {"s2": torch.ones(1, 1, 2, 2, 2)},
        "source_masks": {"s2": torch.ones(1, 1)},
        "timestamps": torch.tensor([[202512]]),
        "highres_frames": {"highres_optical": torch.ones(1, 1, 2, 2)},
        "highres_masks": {"highres_optical": torch.ones(1, 1, 2, 2)},
    }
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    input_path = tmp_path / "input.pt"
    torch.save(batch, input_path)
    embedding = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    output_path = tmp_path / "output.npy"
    np.save(output_path, embedding.numpy(), allow_pickle=False)

    def file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    input_digest, input_document = model_input_evidence(batch)
    output_digest, output_document = tensor_evidence(embedding)
    manifest = {
        "schema_version": "xuannv_independent_legacy_reference_v1",
        "profile": "haidian_p10c_v1",
        "legacy_runtime": {
            "archive_tag": "archive/test",
            "original_git_sha": "1" * 40,
            "sanitized_git_sha": "2" * 40,
            "models_tree_git_sha1": "3" * 40,
        },
        "checkpoint_sha256": file_sha256(checkpoint),
        "input": {
            "path": input_path.name,
            "bytes": input_path.stat().st_size,
            "file_sha256": file_sha256(input_path),
            "tensor_sha256": input_digest,
            "patch_ids": ["p1"],
            "tensor_evidence": input_document,
        },
        "output": {
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "file_sha256": file_sha256(output_path),
            "tensor_sha256": output_digest,
            "tensor_evidence": output_document,
        },
        "environment": {
            "device": "npu:0",
            "device_name": "test",
            "torch": "test",
            "torch_npu": "test",
        },
    }
    manifest_path = tmp_path / "reference.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_independent_legacy_reference(
        checkpoint,
        batch,
        embedding,
        reference_root=tmp_path,
        reference_manifest_path=manifest_path,
    )
    assert report["embedding_exact"] is True
    assert report["input_tensor_sha256"] == input_digest

    with pytest.raises(ReleaseValidationError, match="独立旧运行时.*不一致"):
        validate_independent_legacy_reference(
            checkpoint,
            batch,
            embedding + 1,
            reference_root=tmp_path,
            reference_manifest_path=manifest_path,
        )


def test_gate_device_resolution_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)

    assert _resolve_device(None) == torch.device("cuda:0")
    assert _resolve_device("cuda:1") == torch.device("cuda:1")
    assert selected == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_gate_device_resolution_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _resolve_device(None) == torch.device("cpu")
    assert _resolve_device("cpu") == torch.device("cpu")


def test_gate_environment_records_cuda_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA 门禁证据必须包含设备型号，清单要求核验设备。"""
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA A100-SXM4-40GB")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))

    report = _environment(torch.device("cuda:0"))

    assert report["accelerator"] == "cuda"
    assert report["device"] == "cuda:0"
    assert report["device_name"] == "NVIDIA A100-SXM4-40GB"
    assert report["device_capability"] == "8.0"
    assert report["torch_cuda"] == torch.version.cuda


def test_gate_environment_records_cpu_without_accelerator_fields() -> None:
    report = _environment(torch.device("cpu"))

    assert report["accelerator"] == "cpu"
    assert "device_name" not in report
    assert "torch_cuda" not in report
