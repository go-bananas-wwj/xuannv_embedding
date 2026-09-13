"""Quarterly dense observations with shared, same-calendar-year high-resolution priors."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from xuannv_embedding.data_process.v5_rasters import select_annual


def construct_indexes(
    registry: pd.DataFrame, dense: pd.DataFrame, highres: pd.DataFrame, *, years=(2020, 2021)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if registry.patch_id.duplicated().any() or not set(registry.split).issubset(
        {"train", "val", "test"}
    ):
        raise ValueError("invalid spatial split registry")
    split = registry.set_index("patch_id").split.to_dict()
    if not set(dense.patch_id).issubset(split):
        raise ValueError("dense observations outside registry")
    if not set(dense.quality_status).issubset({"passed", "excluded", "missing"}):
        raise ValueError("unresolved dense quality status; complete quality audit first")
    if dense.duplicated(["patch_id", "product_id", "year", "month"]).any():
        raise ValueError("duplicate dense observation identity")
    if not dense.month.between(1, 12).all():
        raise ValueError("invalid calendar month")
    if not highres.empty:
        if not set(highres.patch_id).issubset(split):
            raise ValueError("high-resolution scenes outside registry")
        if highres.scene_group_id.duplicated().any():
            raise ValueError("high-resolution branches must be grouped into unique scenes first")
        if (
            not np.isfinite(highres.clear_fraction).all()
            or not highres.clear_fraction.between(0, 1).all()
        ):
            raise ValueError("invalid scene quality fraction")
        for row in highres.itertuples():
            if datetime.fromisoformat(row.acquired_at).year != row.year:
                raise ValueError("scene acquisition year disagrees with indexed year")
        candidates = highres.loc[
            highres.year.isin(years)
            & highres.available
            & highres.alignment_status.eq("passed")
            & highres.clear_fraction.gt(0)
        ].copy()
        candidates["split"] = candidates.patch_id.map(split)
    else:
        candidates = highres.reindex(
            columns=list(
                dict.fromkeys(
                    [
                        *highres.columns,
                        "patch_id",
                        "year",
                        "family",
                        "scene_group_id",
                        "acquired_at",
                        "available",
                        "alignment_status",
                        "clear_fraction",
                        "split",
                    ]
                )
            )
        )
    selected_rows = []
    if not candidates.empty:
        for (_, year, _), group in candidates.groupby(["patch_id", "year", "family"], sort=True):
            selected_rows.extend(select_annual(group.to_dict("records"), year=int(year), limit=4))
    selected = pd.DataFrame(selected_rows, columns=candidates.columns)
    dense_ids = {}
    available = dense.loc[
        dense.year.isin(years) & dense.present & dense.quality_status.eq("passed")
    ]
    for row in available.sort_values(["patch_id", "year", "month", "product_id"]).itertuples():
        key = (row.patch_id, int(row.year), (int(row.month) - 1) // 3 + 1)
        dense_ids.setdefault(key, []).append(
            f"{row.product_id}|{row.patch_id}|{row.year}-{row.month:02d}"
        )
    annual_ids = {}
    for row in selected.sort_values(["patch_id", "year", "family", "scene_group_id"]).itertuples():
        annual_ids.setdefault((row.patch_id, int(row.year)), []).append(row.scene_group_id)
    samples = []
    for patch in registry.itertuples():
        for year in years:
            for quarter in (1, 2, 3, 4):
                month = 3 * (quarter - 1) + 1
                start = datetime(year, month, 1, tzinfo=timezone.utc)
                end = (
                    datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                    if quarter == 4
                    else datetime(year, month + 3, 1, tzinfo=timezone.utc)
                )
                references = dense_ids.get((patch.patch_id, year, quarter), [])
                samples.append(
                    {
                        "patch_id": patch.patch_id,
                        "split": patch.split,
                        "year": year,
                        "quarter": quarter,
                        "interval_start": start.isoformat(),
                        "interval_end": end.isoformat(),
                        "dense_observation_ids": references,
                        "highres_scene_ids": annual_ids.get((patch.patch_id, year), []),
                        "base_available": bool(references),
                        "prior_year": year,
                        "prior_mode": "same_calendar_year_retrospective",
                    }
                )
    return pd.DataFrame(samples), candidates.reset_index(drop=True), selected.reset_index(drop=True)
