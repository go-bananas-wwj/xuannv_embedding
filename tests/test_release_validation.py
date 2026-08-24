from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from xuannv_embedding.config import Config
from xuannv_embedding.export.validation import (
    ReleaseValidationError,
    build_legacy_haidian_model,
    validate_embedding_output,
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
