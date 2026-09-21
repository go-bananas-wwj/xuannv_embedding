"""`xuannv diagnose` 的真实栅格回归测试。

P0 诊断是发布门禁里唯一在真实数据上核验月份绑定、排列不变性和缺测回退的入口，
此前没有任何测试覆盖它，任何签名或不变量漂移都只能在整机跑批时才暴露。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import yaml
from rasterio.transform import from_origin

from xuannv_embedding.config import Config
from xuannv_embedding.data_process.pilot_cache import MONTHS, SOURCES, _smoke_config
from xuannv_embedding.training.checkpoint import save_training_checkpoint
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.p0_diagnostics import diagnose, main
from xuannv_embedding.utils.manifest import ManifestRecord, write_manifest

_GIT_SHA = "0" * 40
_HIGHRES = "detail"
_HIGHRES_CHANNELS = 3
_PATCH = 16
_NATIVE = 32
_TEMPORAL = {canonical: channels for canonical, channels in SOURCES.values()}
# s1 故意缺 2021-02：诊断要求真实缺测在该月份产生 availability=0。
_PRESENT = {"s2": ["2021-01", "2021-02"], "s1": ["2021-01"], "landsat": ["2021-01", "2021-02"]}


def _write_tiff(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[1],
        width=values.shape[2],
        count=values.shape[0],
        dtype=values.dtype,
        crs="EPSG:32650",
        transform=from_origin(512000, 3585280, 1280 / values.shape[2], 1280 / values.shape[1]),
    ) as target:
        target.write(values)


def _relative(source: str, month: str) -> str:
    year, number = month.split("-")
    return f"{source}/{year}/{number}/frame.tif"


def _stage(root: Path) -> dict[str, list[str]]:
    generator = np.random.default_rng(7)
    sources: dict[str, list[str]] = {}
    for source, channels in _TEMPORAL.items():
        for month in _PRESENT[source]:
            relative = _relative(source, month)
            values = generator.integers(100, 4000, (channels, _PATCH, _PATCH), dtype=np.uint16)
            _write_tiff(root / relative, values)
            sources.setdefault(source, []).append(relative)
    relative = _relative(_HIGHRES, "2021-01")
    _write_tiff(
        root / relative,
        generator.integers(100, 900, (_HIGHRES_CHANNELS, _NATIVE, _NATIVE), dtype=np.uint16),
    )
    sources[_HIGHRES] = [relative]

    statistics = root / "statistics"
    statistics.mkdir(parents=True, exist_ok=True)
    channel_counts = {**_TEMPORAL, _HIGHRES: _HIGHRES_CHANNELS}
    for source, channels in channel_counts.items():
        (statistics / f"{source}_stats.json").write_text(
            json.dumps({"mean": [1000.0] * channels, "std": [500.0] * channels})
        )
    (root / "observations.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "source_signature": source,
                    "materialized_path": paths[0],
                    "valid_fraction": 1.0,
                    "bounds_error_m": 0.0,
                }
            )
            + "\n"
            for source, paths in sources.items()
        )
    )
    for split in ("train", "validation", "test"):
        write_manifest(
            root / f"{split}.manifest.jsonl",
            [
                ManifestRecord(
                    patch_id=f"{split}-parent", region="national_pilot", sources=dict(sources)
                )
            ],
            months=MONTHS,
            generator_version="test",
        )
    return sources


def _config(root: Path) -> Path:
    config = _smoke_config(root, list(_TEMPORAL), "train")
    config["model"]["embed_dim"] = 8
    config["model"]["input_sources"][_HIGHRES] = {
        "channels": _HIGHRES_CHANNELS,
        "role": "highres",
    }
    config["data"].update(
        patch_size=_PATCH,
        target_months=["2021-01"],
        highres_mode="observations",
        highres_max_observations=2,
    )
    config["data"]["datasets"][0]["source_map"][_HIGHRES] = _HIGHRES
    path = root / "diagnostic.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _checkpoint(config_path: Path, root: Path) -> Path:
    config = Config.from_yaml(config_path)
    system = build_training_system(config)
    path = root / "checkpoint.pt"
    save_training_checkpoint(
        path,
        model=system.model,
        criterion=system.criterion,
        optimizer=torch.optim.AdamW(system.parameters(), lr=1e-4),
        scheduler=None,
        epoch=0,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        git_sha=_GIT_SHA,
        source_schema={key: asdict(value) for key, value in config.model.input_sources.items()},
        regions=[item.region for item in config.data.datasets],
        metrics={"loss": 1.0},
    )
    return path


@pytest.fixture
def staged(tmp_path: Path) -> tuple[Path, Path]:
    _stage(tmp_path)
    config_path = _config(tmp_path)
    return config_path, _checkpoint(config_path, tmp_path)


def test_diagnose_reports_real_data_invariants(staged: tuple[Path, Path]) -> None:
    config_path, checkpoint = staged
    report = diagnose(config_path, checkpoint, torch.device("cpu"))

    assert report["passed"] is True and report["p1_ready"] is False
    assert report["permutation"]["max_abs_difference"] < 1e-5
    assert report["same_weights_empty_fallback"]["max_abs_difference"] == 0
    assert report["masked_pixel_values_invariance"]["max_abs_difference"] == 0
    assert report["no_highres_month_max_difference"] == 0
    assert report["missing_source_reconstruction"] == 0
    assert report["all_invalid_reconstruction"] == 0
    assert report["real_s1_202102_missing"] is True
    assert report["diagnostic_parent"] == "train-parent"
    assert report["diagnostic_active_frames"] == 1
    assert sorted(report["splits"]) == ["test", "train", "validation"]
    assert all(item["finite"] and item["parents"] == 1 for item in report["splits"].values())
    assert sorted(report["sources"]) == sorted([*_TEMPORAL, _HIGHRES])
    assert all(item["units"] == "stored_values_unverified" for item in report["sources"].values())
    for prefix in ("temporal_stem_bank", "highres_encoders", "highres_fusion"):
        assert report["gradients"][prefix]["finite"]
        assert report["gradients"][prefix]["absolute_sum"] > 0
    assert sorted(report["edge_case_backward"]) == [
        "all_invalid",
        "highres_empty",
        "lowres_source_missing",
    ]
    assert all(item["finite_gradients"] for item in report["edge_case_backward"].values())
    # 真实高分观测必须改变 embedding，否则融合分支形同虚设。
    assert report["highres_effect"]["max_abs_difference"] > 0
    assert report["landmark_registration"] == "not_verified"


def test_diagnose_requires_month_bound_observation_mode(staged: tuple[Path, Path]) -> None:
    config_path, checkpoint = staged
    config = yaml.safe_load(config_path.read_text())
    config["data"]["highres_mode"] = "legacy"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="selected months and observation mode"):
        diagnose(config_path, checkpoint, torch.device("cpu"))


def test_diagnose_cli_writes_report(staged: tuple[Path, Path], capsys, tmp_path: Path) -> None:
    config_path, checkpoint = staged
    output = tmp_path / "reports" / "p0.json"
    assert (
        main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"passed": True, "output": str(output)}
    assert json.loads(output.read_text())["checkpoint_sha256"]
