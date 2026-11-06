"""Quarterly dense observations with shared, same-calendar-year high-resolution priors."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from xuannv_embedding.data_process.v5_rasters import select_annual


def construct_indexes(
    registry: pd.DataFrame, dense: pd.DataFrame, highres: pd.DataFrame, *, years=(2020, 2021)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not years or len(set(years)) != len(years) or any(y not in (2020, 2021) for y in years):
        raise ValueError("invalid sample years")
    if registry.patch_id.duplicated().any() or not set(registry.split).issubset(
        {"train", "val", "test"}
    ):
        raise ValueError("invalid spatial split registry")
    split = registry.set_index("patch_id").split.to_dict()
    if not set(dense.patch_id).issubset(split):
        raise ValueError("dense observations outside registry")
    for table in [dense, highres]:
        if (
            not table.empty
            and "split" in table
            and not np.array_equal(table.split, table.patch_id.map(split))
        ):
            raise ValueError("observation split differs from registry")
    if not set(dense.contract_status).issubset({"verified", "pending", "failed"}):
        raise ValueError("invalid dense product contract status")
    if not set(dense.quality_status).issubset({"passed", "excluded", "missing", "pending"}):
        raise ValueError("unresolved dense quality status; complete quality audit first")
    if dense.duplicated(["patch_id", "product_id", "year", "month"]).any():
        raise ValueError("duplicate dense observation identity")
    if not dense.month.between(1, 12).all() or not (dense.month == np.floor(dense.month)).all():
        raise ValueError("invalid calendar month")
    if not dense.year.isin([2020, 2021]).all() or dense.present.dtype != bool:
        raise ValueError("invalid dense year or presence flag")
    if not highres.empty:
        if any(
            highres[k].dtype != bool
            for k in ["available", "candidate_qualified", "strict_pixel_fusion_candidate"]
        ):
            raise ValueError("invalid high-resolution qualification flags")
        if (highres.strict_pixel_fusion_candidate & ~highres.alignment_status.eq("passed")).any():
            raise ValueError("strict fusion requires reliable alignment evidence")
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
            & highres.candidate_qualified
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
    dense_ids, inventory_ids, observed_months, exclusions = {}, {}, {}, {}
    for row in (
        dense.loc[dense.year.isin(years) & dense.present]
        .sort_values(["patch_id", "year", "month", "product_id"])
        .itertuples()
    ):
        key = (row.patch_id, int(row.year), (int(row.month) - 1) // 3 + 1)
        identity = f"{row.product_id}|{row.patch_id}|{row.year}-{row.month:02d}"
        inventory_ids.setdefault(key, []).append(identity)
        observed_months.setdefault(key, set()).add((row.product_id, int(row.month)))
        reasons = exclusions.setdefault(key, set())
        if row.contract_status != "verified":
            reasons.add(
                "unknown_radiometry" if row.contract_status == "pending" else "failed_radiometry"
            )
        if row.quality_status != "passed":
            reasons.add("quality_" + row.quality_status)
    available = dense.loc[
        dense.year.isin(years)
        & dense.present
        & dense.quality_status.eq("passed")
        & dense.contract_status.eq("verified")
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
                        "dense_inventory_ids": inventory_ids.get(
                            (patch.patch_id, year, quarter), []
                        ),
                        "base_exclusion_reasons": sorted(
                            exclusions.get((patch.patch_id, year, quarter), set())
                        ),
                        "missing_monthly_sources": [
                            f"{product}:{year}-{m:02d}"
                            for product in ["s1_local", "s2_local", "landsat_local"]
                            for m in range(month, month + 3)
                            if (product, m)
                            not in observed_months.get((patch.patch_id, year, quarter), set())
                        ],
                        "target_year": year,
                        "highres_scene_ids": annual_ids.get((patch.patch_id, year), []),
                        "base_available": bool(references),
                        "prior_year": year,
                        "prior_mode": "same_calendar_year_retrospective",
                    }
                )
    return pd.DataFrame(samples), candidates.reset_index(drop=True), selected.reset_index(drop=True)
