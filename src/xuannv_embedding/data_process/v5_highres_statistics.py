"""Materialize training-only high-resolution statistics from immutable QA snapshots."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_highres_reader import HighresQualityReader
from xuannv_embedding.data_process.v5_intraband import _runtime_versions
from xuannv_embedding.data_process.v5_jilin_quality import _digest
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json
from xuannv_embedding.data_process.v5_statistics import StreamingBandStatistics


def training_rows(rows: pd.DataFrame):
    if rows.empty or rows.observation_id.duplicated().any():
        raise ValueError("empty or duplicate observation identities")
    if rows.groupby(["product_id", "sensor", "file_sha256"]).split.nunique().gt(1).any():
        raise ValueError("identical source content has cross-split owners")
    if not rows.year.isin([2020, 2021]).all():
        raise ValueError("unsupported statistics year")
    excluded = rows.loc[~rows.split.eq("train")].copy()
    excluded["reason"] = "non_training_split"
    selected = rows.loc[rows.split.eq("train")].sort_values("observation_id").copy()
    duplicate = selected.duplicated(["product_id", "sensor", "file_sha256"])
    aliases = selected.loc[duplicate].copy()
    aliases["reason"] = "duplicate_content"
    return selected.loc[~duplicate].reset_index(drop=True), pd.concat(
        [excluded, aliases], ignore_index=True
    )


def finish_product(stats: StreamingBandStatistics, bands: list[str]):
    std = np.sqrt(
        np.divide(stats.m2, stats.count, out=np.zeros_like(stats.m2), where=stats.count > 0)
    )
    failed = [
        b
        for i, b in enumerate(bands)
        if stats.count[i] < 2 or not np.isfinite(std[i]) or std[i] <= 1e-12
    ]
    return {
        "band_ids": bands,
        "status": "failed" if failed else "passed",
        "failed_bands": failed,
        "valid_counts": stats.count.tolist(),
        "statistics": None if failed else stats.finish(),
    }


def run_highres_statistics(
    dataset_root: Path,
    report_root: Path,
    family: str,
    quality_root: Path,
    *,
    reservoir_size=4096,
    limit=None,
):
    if reservoir_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("positive reservoir size and pilot limit required")
    reader = HighresQualityReader(dataset_root, family, quality_root)
    rows, excluded = training_rows(reader.inventory)
    available = len(rows)
    if limit is not None:
        rows = rows.head(limit)
    if rows.empty:
        raise ValueError("no training observations")
    code = {
        name: sha256(Path(__file__).with_name(name))
        for name in [
            "v5_highres_reader.py",
            "v5_highres_statistics.py",
            "v5_statistics.py",
            "v5_clear_intraband.py",
            "v5_rasters.py",
            "v5_partial_bands.py",
            "v5_jilin_quality.py",
            "v5_quality.py",
            "v5_intraband.py",
        ]
    }
    fingerprint = {
        "family": family,
        "quality_inputs_sha256": reader.files,
        "quality_configuration": reader.configuration,
        "code_sha256": code,
        "runtime": {**_runtime_versions(), "pandas": pd.__version__},
        "rows_sha256": _digest(rows.to_dict("records")),
        "excluded_sha256": _digest(excluded.to_dict("records")),
        "reservoir_size": reservoir_size,
        "seed": 42,
        "limit": limit,
        "available_training_observations": available,
        "region_rule": "floor(longitude/5),floor(latitude/5):five_degree_cells",
        "alignment_policy": "per-band QA-valid radiometric candidates; no pixel-fusion approval",
    }
    root = dataset_root / "statistics/train/highres" / family / _digest(fingerprint)[:20]
    inputs = root / "inputs.parquet"
    input_lock = root / "inputs.lock.json"
    if inputs.exists():
        if _digest(pd.read_parquet(inputs).to_dict("records")) != fingerprint["rows_sha256"]:
            raise ValueError("statistics input table changed")
    else:
        atomic_parquet(rows, inputs)
    locked = {"fingerprint": fingerprint, "inputs_sha256": sha256(inputs)}
    if input_lock.exists():
        if json.loads(input_lock.read_text()) != locked:
            raise ValueError("statistics input lock changed")
    else:
        write_json(input_lock, locked)
    seal_path = root / "output.lock.json"
    reused = seal_path.exists()
    seal = json.loads(seal_path.read_text()) if reused else None
    if reused and (
        seal["inputs_lock_sha256"] != sha256(input_lock)
        or any(sha256(root / name) != h for name, h in seal["outputs_sha256"].items())
    ):
        raise ValueError("statistics output seal changed")
    stats = {}
    bands = {}
    proofs = []
    contributions = defaultdict(lambda: np.zeros(3, dtype="int64"))
    progress = (
        report_root
        / f"highres_statistics_{family}_{'full' if limit is None else 'pilot_'+str(limit)}.json"
    )
    for row in rows.itertuples():
        frame, proof = reader.read(row)
        if (
            frame.valid.dtype != bool
            or frame.valid.shape != frame.values.shape
            or not np.isfinite(frame.values[frame.valid]).all()
        ):
            raise ValueError("invalid statistics values or masks")
        key = row.product_id + "/" + row.sensor
        if key in bands and bands[key] != list(frame.band_ids):
            raise ValueError("statistics band contract changed within product")
        bands[key] = list(frame.band_ids)
        if not reused:
            if key not in stats:
                stats[key] = StreamingBandStatistics(
                    len(frame.band_ids), reservoir_size=reservoir_size, seed=42
                )
            stats[key].update(frame.values, frame.valid)
        proofs.append(
            {"observation_id": row.observation_id, "source_sha256": row.file_sha256, "proof": proof}
        )
        owner = reader.registry.loc[row.patch_id]
        if not np.isfinite([owner.longitude, owner.latitude]).all():
            raise ValueError("invalid region coordinates")
        region = f"lon5_{int(np.floor(owner.longitude/5))}_lat5_{int(np.floor(owner.latitude/5))}"
        for i, band in enumerate(frame.band_ids):
            n = int(frame.valid[i].sum())
            contributions[(row.product_id, row.sensor, int(row.year), region, band)] += [
                1,
                int(n > 0),
                n,
            ]
        if len(proofs) % 128 == 0 or len(proofs) == len(rows):
            write_json(
                progress,
                {
                    "execution_status": "running",
                    "processed": len(proofs),
                    "selected": len(rows),
                    "reused": reused,
                    "updated_at": now(),
                    "output": str(root),
                    "training_authorized": False,
                },
            )
    reader.verify_unchanged()
    if (
        any(sha256(Path(__file__).with_name(n)) != h for n, h in code.items())
        or sha256(inputs) != locked["inputs_sha256"]
        or json.loads(input_lock.read_text()) != locked
    ):
        raise ValueError("statistics inputs or implementation changed")
    proof_hash = _digest(proofs)
    if reused:
        if proof_hash != seal["source_and_masks_sha256"]:
            raise ValueError("statistics source or masks changed")
        if json.loads(seal_path.read_text()) != seal or any(
            sha256(root / name) != h for name, h in seal["outputs_sha256"].items()
        ):
            raise ValueError("statistics output changed during cache verification")
        result = json.loads((root / "statistics.json").read_text())
    else:
        result = {
            "products": {key: finish_product(value, bands[key]) for key, value in stats.items()},
            "source_and_masks_sha256": proof_hash,
            "units": "native_stored_dn" if family == "gaofen" else "reflectance",
            "normalization_authorized": False,
            "scope": (
                "QA-valid per-band radiometry; " "alignment and visual acceptance remain separate"
            ),
        }
        records = [
            dict(
                product_id=key[0],
                sensor=key[1],
                year=key[2],
                region=key[3],
                band=key[4],
                observations=int(v[0]),
                observations_with_valid_pixels=int(v[1]),
                valid_pixels=int(v[2]),
            )
            for key, v in sorted(contributions.items())
        ]
        # All outputs remain unpublished until their common seal is written last.
        atomic_parquet(excluded, root / "excluded.parquet")
        contribution_table = pd.DataFrame(records)
        totals = contribution_table.groupby(
            ["product_id", "sensor", "band"]
        ).valid_pixels.transform("sum")
        contribution_table["valid_pixel_fraction_of_product_band"] = np.divide(
            contribution_table.valid_pixels,
            totals,
            out=np.zeros(len(totals), dtype="float64"),
            where=totals.to_numpy() > 0,
        )
        atomic_parquet(contribution_table, root / "contributions.parquet")
        atomic_parquet(
            pd.DataFrame(
                [
                    {
                        "observation_id": p["observation_id"],
                        "source_sha256": p["source_sha256"],
                        "proof": json.dumps(p["proof"], sort_keys=True),
                    }
                    for p in proofs
                ]
            ),
            root / "source_and_masks.parquet",
        )
        write_json(root / "statistics.json", result)
        names = [
            "excluded.parquet",
            "contributions.parquet",
            "source_and_masks.parquet",
            "statistics.json",
        ]
        for key, product in result["products"].items():
            name = "products/" + key.replace("/", "__") + ".json"
            write_json(
                root / name,
                {**product, "units": result["units"], "normalization_authorized": False},
            )
            names.append(name)
        seal = {
            "inputs_lock_sha256": sha256(input_lock),
            "source_and_masks_sha256": proof_hash,
            "outputs_sha256": {name: sha256(root / name) for name in names},
        }
        write_json(seal_path, seal)
    summary = {
        "execution_status": "finished",
        "scope": "frozen_available_QA_catalog" if limit is None else "pilot",
        "processed": len(rows),
        "available_training_observations": available,
        "excluded_observations": len(excluded),
        "products": len(result["products"]),
        "failed_products": [k for k, v in result["products"].items() if v["status"] != "passed"],
        "output": str(root),
        "output_lock_sha256": sha256(seal_path),
        "reused": reused,
        "finished_at": now(),
        "training_authorized": False,
        "normalization_authorized": False,
    }
    write_json(progress, summary)
    return summary
