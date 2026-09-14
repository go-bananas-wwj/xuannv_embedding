from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from xuannv_embedding.data.contracts import ObservationRef, ProductSpec


def test_product_spec_requires_one_gsd_per_band() -> None:
    with pytest.raises(ValueError, match="波段.*GSD"):
        ProductSpec(
            product_id="broken",
            role="dense",
            bands=("B02", "B03"),
            native_gsd_m=(10.0,),
            stored_gsd_m=10.0,
            dtype="uint16",
            time_precision="month",
            already_resampled=True,
            qa_available=False,
        )


def test_month_observation_must_not_invent_acquisition_time() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2020, 2, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="month.*acquired_at"):
        ObservationRef(
            patch_id="parent_32643:310:3383",
            product_id="s2_local",
            interval_start=start,
            interval_end=end,
            acquired_at=start,
            available_at=end,
            archive_path=Path("/data2/s2.zip"),
            member_name="patch.tif",
            present=True,
            quality_status="legacy_monthly_best_without_qa",
            time_precision="month",
        )


def test_month_observation_becomes_available_after_interval() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2020, 2, 1, tzinfo=UTC)
    ref = ObservationRef(
        patch_id="parent_32643:310:3383",
        product_id="s2_local",
        interval_start=start,
        interval_end=end,
        acquired_at=None,
        available_at=end,
        archive_path=Path("/data2/s2.zip"),
        member_name="patch.tif",
        present=True,
        quality_status="legacy_monthly_best_without_qa",
        time_precision="month",
    )

    assert ref.available_at == ref.interval_end
