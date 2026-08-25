from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from xuannv_embedding.data.v2_dataset import (
    V2LocalZipDataset,
    _load_statistics,
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
    }
    (statistics / "dense.json").write_text(json.dumps(document), encoding="utf-8")
    config = SimpleNamespace(
        paths=SimpleNamespace(data_root=tmp_path),
        products={"dense": SimpleNamespace(bands=("a", "b"))},
    )

    mean, std = _load_statistics(config, "dense")

    assert torch.equal(mean[:, 0, 0], torch.tensor([1000.0, 2000.0]))
    assert torch.equal(std[:, 0, 0], torch.tensor([100.0, 200.0]))


def test_stored_pixel_validity_rejects_zero_and_minus_32768_fill() -> None:
    values = torch.tensor(
        [
            [[1.0, 0.0, -32768.0, 2.0]],
            [[2.0, 0.0, 3.0, float("nan")]],
        ]
    ).numpy()

    assert _stored_pixel_validity(values).tolist() == [[True, False, False, False]]
