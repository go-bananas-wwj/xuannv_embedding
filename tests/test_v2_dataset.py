from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
import torch
from affine import Affine

from xuannv_embedding.data.v2_dataset import (
    V2LocalZipDataset,
    _load_statistics,
    _read_highres_patch,
    _read_supervised_label,
    _select_highres_candidates,
    _stored_pixel_validity,
)


def _row(year: int, month: int, *, present: bool = True) -> dict[str, object]:
    start = datetime(year, month, 1, tzinfo=UTC)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(year, month + 1, 1, tzinfo=UTC)
    return {
        "year": year,
        "month": month,
        "interval_start": start,
        "interval_end": end,
        "available_at": end,
        "present": present,
    }


def _bare_dataset(selection: str = "random_single") -> V2LocalZipDataset:
    dataset = V2LocalZipDataset.__new__(V2LocalZipDataset)
    dataset.output_selection = selection
    dataset.fixed_output_months = ()
    dataset.random_seed = 17
    dataset.context_days = 365
    dataset.dense_products = ("s2_local", "s1_local", "landsat_local")
    rows = [_row(year, month) for year in (2020, 2021) for month in range(1, 13)]
    dataset.observations = {
        ("p1", product): [dict(row) for row in rows] for product in dataset.dense_products
    }
    return dataset


def test_random_output_selection_is_deterministic_and_single_interval() -> None:
    dataset = _bare_dataset()

    first = dataset._output_rows("p1")
    second = dataset._output_rows("p1")

    assert len(first) == 1
    assert first == second


def test_causal_context_is_padded_to_twelve_monthly_slots() -> None:
    dataset = _bare_dataset()
    output = [_row(2020, 3)]

    context = dataset._context_rows(dataset.observations[("p1", "s2_local")], output)

    assert len(context) == 12
    assert sum(row is None for row in context) == 9
    assert [(row["year"], row["month"]) for row in context if row is not None] == [
        (2020, 1),
        (2020, 2),
        (2020, 3),
    ]


def test_random_output_avoids_month_with_all_dense_products_missing() -> None:
    dataset = _bare_dataset()
    for product in dataset.dense_products:
        for row in dataset.observations[("p1", product)]:
            row["present"] = False
    dataset.observations[("p1", "s1_local")][5]["present"] = True

    output = dataset._output_rows("p1")

    assert [(row["year"], row["month"]) for row in output] == [(2020, 6)]


def test_all_missing_sentinel_still_gets_deterministic_output_interval() -> None:
    dataset = _bare_dataset()
    for product in dataset.dense_products:
        for row in dataset.observations[("p1", product)]:
            row["present"] = False

    first = dataset._output_rows("p1")
    second = dataset._output_rows("p1")

    assert len(first) == 1
    assert first == second


def test_statistics_keep_stored_dn_contract_and_normalize_per_band(tmp_path: Path) -> None:
    statistics = tmp_path / "statistics"
    statistics.mkdir()
    document = {
        "schema_version": "xuannv_v2_band_statistics_v1",
        "product_id": "dense",
        "bands": ["a", "b"],
        "mean": [1000.0, 2000.0],
        "std": [100.0, 200.0],
        "split": "train",
        "representation": "stored_dn",
        "scaling_applied": False,
        "complete_training_split": True,
    }
    (statistics / "dense.json").write_text(json.dumps(document), encoding="utf-8")
    config = SimpleNamespace(
        paths=SimpleNamespace(data_root=tmp_path),
        products={"dense": SimpleNamespace(bands=("a", "b"))},
    )

    mean, std = _load_statistics(config, "dense")

    assert torch.equal(mean[:, 0, 0], torch.tensor([1000.0, 2000.0]))
    assert torch.equal(std[:, 0, 0], torch.tensor([100.0, 200.0]))


def test_production_statistics_reject_incomplete_training_split(tmp_path: Path) -> None:
    statistics = tmp_path / "statistics"
    statistics.mkdir()
    document = {
        "schema_version": "xuannv_v2_band_statistics_v1",
        "product_id": "dense",
        "bands": ["a"],
        "mean": [1.0],
        "std": [1.0],
        "split": "train",
        "representation": "stored_dn",
        "scaling_applied": False,
        "complete_training_split": False,
    }
    (statistics / "dense.json").write_text(json.dumps(document), encoding="utf-8")
    config = SimpleNamespace(
        paths=SimpleNamespace(data_root=tmp_path),
        products={"dense": SimpleNamespace(bands=("a",))},
    )

    import pytest

    with pytest.raises(ValueError, match="完整训练划分"):
        _load_statistics(config, "dense")
    _load_statistics(config, "dense", allow_incomplete=True)


def test_stored_pixel_validity_rejects_zero_and_minus_32768_fill() -> None:
    values = torch.tensor(
        [
            [[1.0, 0.0, -32768.0, 2.0]],
            [[2.0, 0.0, 3.0, float("nan")]],
        ]
    ).numpy()

    assert _stored_pixel_validity(values).tolist() == [[True, False, False, False]]


def test_highres_reader_keeps_native_pixels_and_applies_udm2_clear_mask(tmp_path: Path) -> None:
    image = tmp_path / "scene.tif"
    qa = tmp_path / "qa.tif"
    transform = Affine(2, 0, 0, 0, -2, 8)
    values = np.full((4, 4, 4), 100, dtype=np.uint16)
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        count=4,
        width=4,
        height=4,
        dtype="uint16",
        crs="EPSG:32650",
        transform=transform,
    ) as dataset:
        dataset.write(values)
    quality = np.zeros((8, 4, 4), dtype=np.uint8)
    quality[0] = 1
    quality[5, 0, 0] = 1
    quality[0, 0, 0] = 0
    with rasterio.open(
        qa,
        "w",
        driver="GTiff",
        count=8,
        width=4,
        height=4,
        dtype="uint8",
        crs="EPSG:32650",
        transform=transform,
    ) as dataset:
        dataset.write(quality)
        for index, name in enumerate(
            ("clear", "snow", "shadow", "haze_light", "haze_heavy", "cloud", "confidence", "udm1"),
            start=1,
        ):
            dataset.set_band_description(index, name)
    row = {
        "patch_bounds": [0, 0, 8, 8],
        "image_path": str(image),
        "qa_path": str(qa),
        "qa_present": True,
    }

    frame, mask, geotransform = _read_highres_patch(row, bands=4, stored_gsd_m=2)

    assert frame.shape == (4, 6, 6)
    assert mask.shape == (1, 6, 6)
    assert mask[0, 0, 0] == 0
    assert mask.sum() == 15
    assert geotransform[0] == 2
    assert torch.equal(
        frame[:, :4, :4], torch.from_numpy(values.astype(np.float32)) * mask[:, :4, :4]
    )


def test_supervised_label_reader_reprojects_real_mask_to_output_grid(tmp_path: Path) -> None:
    label_path = tmp_path / "label.tif"
    with rasterio.open(
        label_path,
        "w",
        driver="GTiff",
        count=1,
        width=2,
        height=2,
        dtype="uint8",
        crs="EPSG:32650",
        transform=Affine(10, 0, 0, 0, -10, 20),
    ) as dataset:
        dataset.write(np.array([[[1, 0], [0, 1]]], dtype=np.uint8))

    label, mask = _read_supervised_label(
        [{"label_path": str(label_path)}],
        epsg=32650,
        output_transform=torch.tensor([10.0, 0.0, 0.0, 0.0, -10.0, 20.0]),
        output_size=2,
    )

    assert label.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert mask.sum() == 4


def test_highres_candidate_union_keeps_causal_history_before_scene_limit() -> None:
    rows = [
        {
            "scene_id": "history",
            "acquired_at": "2020-01-15T00:00:00Z",
            "available_at": "2020-01-16T00:00:00Z",
            "intersection_fraction": 0.5,
            "clear_percent": 50,
        }
    ]
    rows.extend(
        {
            "scene_id": f"future-{index:02d}",
            "acquired_at": f"2021-01-{index + 1:02d}T00:00:00Z",
            "available_at": f"2021-01-{index + 2:02d}T00:00:00Z",
            "intersection_fraction": 1.0,
            "clear_percent": 100,
        }
        for index in range(9)
    )
    intervals = torch.tensor(
        [
            [
                datetime(2020, 2, 1, tzinfo=UTC).timestamp() / 86400,
                datetime(2020, 3, 1, tzinfo=UTC).timestamp() / 86400,
            ],
            [
                datetime(2021, 2, 1, tzinfo=UTC).timestamp() / 86400,
                datetime(2021, 3, 1, tzinfo=UTC).timestamp() / 86400,
            ],
        ]
    )

    selected = _select_highres_candidates(
        rows,
        intervals,
        mode="causal_window",
        structure_days=730,
        appearance_days=90,
        structure_max=8,
        appearance_max=4,
    )

    assert "history" in {row["scene_id"] for row in selected}
