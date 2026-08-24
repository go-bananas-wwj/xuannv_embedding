from __future__ import annotations

import json
from pathlib import Path

import pytest

from xuannv_embedding.config import InputSourceConfig, RegionDatasetConfig
from xuannv_embedding.data.contracts import (
    RegionContractError,
    canonicalize_record,
    load_region_records,
)
from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest


def _dataset(region: str, manifest_path: Path) -> RegionDatasetConfig:
    return RegionDatasetConfig(
        region=region,
        manifest_path=manifest_path,
        statistics_dir=Path(f"statistics/{region}"),
        patch_grid_path=Path(f"grids/{region}.json"),
        source_map={
            f"s2_{region}": "s2",
            f"optical_{region}": "highres_optical",
            f"sar_{region}": "highres_sar",
        },
        supervised_label_roots={},
        sampling_weight=1.0,
    )


def _inputs() -> dict[str, InputSourceConfig]:
    return {
        "s2": InputSourceConfig(12, "temporal"),
        "highres_optical": InputSourceConfig(3, "highres"),
        "highres_sar": InputSourceConfig(1, "highres"),
    }


def test_canonicalization_uses_same_slots_for_two_regions() -> None:
    haidian = canonicalize_record(
        ManifestRecord(
            "h1",
            "haidian",
            {
                "s2_haidian": ["haidian/s2.tif"],
                "optical_haidian": "haidian/optical.tif",
                "sar_haidian": "haidian/sar.tif",
            },
        ),
        _dataset("haidian", Path("haidian.json")),
        _inputs(),
    )
    harbin = canonicalize_record(
        ManifestRecord(
            "b1",
            "harbin",
            {
                "s2_harbin": ["harbin/s2.tif"],
                "optical_harbin": "harbin/optical.tif",
                "sar_harbin": None,
            },
        ),
        _dataset("harbin", Path("harbin.json")),
        _inputs(),
    )

    assert haidian.sources.keys() == harbin.sources.keys() == _inputs().keys()
    assert haidian.statistics_dir == Path("statistics/haidian")
    assert harbin.statistics_dir == Path("statistics/harbin")
    assert harbin.sources["highres_sar"] is None
    assert harbin.availability["highres_sar"] is False


def test_missing_modality_never_creates_pseudo_observation() -> None:
    canonical = canonicalize_record(
        ManifestRecord(
            "b1",
            "harbin",
            {"s2_harbin": ["s2.tif"], "optical_harbin": "optical.tif"},
        ),
        _dataset("harbin", Path("harbin.json")),
        _inputs(),
    )

    assert canonical.sources["highres_sar"] is None
    assert canonical.availability["highres_sar"] is False


def test_load_region_records_filters_mixed_legacy_manifest(tmp_path: Path) -> None:
    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps(
            [
                {"patch_id": "h1", "region": "haidian", "s2_haidian": ["h/s2.tif"]},
                {"patch_id": "b1", "region": "harbin", "s2_harbin": ["b/s2.tif"]},
            ]
        ),
        encoding="utf-8",
    )

    records = load_region_records(_dataset("harbin", path), expected_months=["2025-12"])

    assert [record.patch_id for record in records] == ["b1"]


def test_v1_manifest_month_conflict_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        [ManifestRecord("h1", "haidian", {"s2_haidian": ["s2.tif"]})],
        months=["2025-12"],
    )

    with pytest.raises(RegionContractError, match="月份冲突"):
        load_region_records(_dataset("haidian", path), expected_months=["2026-01"])


def test_record_region_must_match_dataset() -> None:
    with pytest.raises(RegionContractError, match="区域冲突"):
        canonicalize_record(
            ManifestRecord("x", "harbin", {"s2_haidian": ["s2.tif"]}),
            _dataset("haidian", Path("manifest.json")),
            _inputs(),
        )
