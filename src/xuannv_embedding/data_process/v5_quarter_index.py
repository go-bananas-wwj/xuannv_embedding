"""Seal complete quarterly source references while quarantining unresolved dense contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from xuannv_embedding.data_process.v5_candidate_statistics import CandidateSelection
from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_dense_integrity import BAND_COUNTS, STORED_GRIDS
from xuannv_embedding.data_process.v5_highres_reader import HighresQualityReader
from xuannv_embedding.data_process.v5_jilin_quality import _digest
from xuannv_embedding.data_process.v5_sampling import construct_indexes
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def validate_config(value):
    if (
        set(value) != {"schema", "dense_audit_roots", "highres", "target_reader_root"}
        or value["schema"] != "quarter_source_index_v1"
        or set(value["dense_audit_roots"]) != set(BAND_COUNTS)
        or set(value["highres"]) != {"gaofen", "jilin1"}
        or any(set(v) != {"eligibility_root", "quality_root"} for v in value["highres"].values())
    ):
        raise ValueError("invalid complete quarterly source manifest")


def load_highres_views(dataset_root, config, registry):
    scenes, candidates, selected, files = [], [], [], {}
    owners = registry.set_index("patch_id").split
    for family, value in sorted(config.items()):
        root = Path(value["eligibility_root"])
        reader = HighresQualityReader(dataset_root, family, Path(value["quality_root"]))
        evidence = CandidateSelection(reader, root)
        files.update(evidence.files)
        branch = pd.read_parquet(root / "branches.parquet")
        frames = [
            pd.read_parquet(root / name)
            for name in [
                "scene_groups.parquet",
                "annual_highres_candidates.parquet",
                "annual_highres_default_selection.parquet",
            ]
        ]
        scene, pool, default = frames
        if (
            scene.scene_group_id.duplicated().any()
            or not scene.family.eq(family).all()
            or not scene.patch_id.isin(owners.index).all()
            or not np.array_equal(scene.split, owners.loc[scene.patch_id])
        ):
            raise ValueError("high-resolution scene identity differs from registry")
        grouped = {k: g for k, g in branch.groupby("scene_group_id", sort=False)}
        if set(grouped) != set(scene.scene_group_id):
            raise ValueError("high-resolution scene membership differs from branches")
        for row in scene.itertuples():
            group = grouped[row.scene_group_id]
            valid = group.loc[group.tolerant_reconstruction_candidate]
            if (
                set(row.branch_ids) != set(group.observation_id)
                or set(row.eligible_branch_ids) != set(valid.observation_id)
                or bool(row.available) != (not valid.empty)
                or any(
                    not group[k].eq(getattr(row, k)).all()
                    for k in ["patch_id", "split", "year", "family"]
                )
            ):
                raise ValueError("high-resolution qualified scene members changed")
        expected = scene.loc[scene.available].reset_index(drop=True)
        if _digest(expected.to_dict("records")) != _digest(pool.to_dict("records")):
            raise ValueError("high-resolution annual pool differs from qualified scenes")
        scene = scene.copy()
        scene["candidate_qualified"] = scene.available
        scenes.append(scene)
        candidates.append(pool)
        selected.append(default)
        evidence.verify_unchanged()
    return (
        pd.concat(scenes, ignore_index=True),
        pd.concat(candidates, ignore_index=True),
        pd.concat(selected, ignore_index=True),
        files,
    )


def load_target_view(dataset_root, root):
    root = Path(root)
    lock_path = root / "verification.lock.json"
    lock = json.loads(lock_path.read_text())
    table_path = root / "views.parquet"
    rows = pd.read_parquet(table_path)
    fp = lock["fingerprint"]
    files = {**fp["files_sha256"], **fp["annual_files_sha256"]}
    registry = dataset_root / "registry/national_62000.parquet"
    if (
        lock["summary"]["status"] != "annual_target_reader_verified"
        or files.get(str(registry)) != sha256(registry)
        or _digest(rows.to_dict("records")) != lock["records_sha256"]
        or len(rows) != lock["summary"]["verified_views"]
        or rows.duplicated(["patch_id", "year"]).any()
        or set(rows.year) != {2020, 2021}
        or any(sha256(Path(p)) != h for p, h in files.items())
    ):
        raise ValueError("matching verified annual target view required")
    files.update({str(lock_path): sha256(lock_path), str(table_path): sha256(table_path)})
    return files


def dense_sources(config, registry):
    registry_hash = hashlib.sha256(
        registry[["patch_id", "split", "grid_epsg", "utm_bounds"]].to_json().encode()
    ).hexdigest()
    owners = registry.set_index("patch_id").split
    files, sources, monthly = {}, [], []
    for product, folder in sorted(config.items()):
        for year in [2020, 2021]:
            for month in range(1, 13):
                path = Path(folder) / f"{product}_{year}_{month:02d}.json"
                receipt = json.loads(path.read_text())
                inventory = Path(receipt["inventory_path"])
                source = Path(receipt["source_path"])
                if (
                    receipt.get("status") != "integrity_checked_contract_pending"
                    or receipt.get("failed_tiffs") != 0
                    or receipt.get("source_changed_during_audit")
                    or receipt.get("conflicting_patch_ids")
                    or receipt.get("physical_contract_verified") is not False
                    or (receipt["product_id"], receipt["year"], receipt["month"])
                    != (product, year, month)
                    or receipt["fingerprint"]["registry_sha256"] != registry_hash
                    or sha256(inventory) != receipt["inventory_sha256"]
                    or source.stat().st_size != receipt["fingerprint"]["source_bytes"]
                ):
                    raise ValueError("complete matching dense audit required")
                files.update({str(path): sha256(path), str(inventory): sha256(inventory)})
                frame = pd.read_parquet(inventory)
                if (
                    len(frame) != receipt["tiffs"]
                    or len(frame) != receipt["decoded_tiffs"]
                    or not frame.status.isin(["decoded_contract_pending", "duplicate_equal"]).all()
                    or not frame.patch_id.isin(owners.index).all()
                    or not np.array_equal(frame.split, owners.loc[frame.patch_id])
                    or frame.groupby("patch_id").file_sha256.nunique().gt(1).any()
                    or len(registry) - frame.patch_id.nunique() != receipt["missing_patches"]
                ):
                    raise ValueError("dense audit membership or monthly counts changed")
                grid = STORED_GRIDS[product]
                for encoded in frame.metadata_json:
                    meta = json.loads(encoded)
                    if (
                        meta["shape"] != grid["shape"]
                        or len(meta["dtype"]) != BAND_COUNTS[product]
                        or meta["radiometry_status"] != "unverified"
                    ):
                        raise ValueError("dense stored grid or radiometry audit changed")
                key = f"{product}:{year}-{month:02d}"
                source_row = {
                    "archive_key": key,
                    "product_id": product,
                    "year": year,
                    "month": month,
                    "archive_path": str(source),
                    "archive_sha256": receipt["source_sha256"],
                    "archive_bytes": source.stat().st_size,
                    "inventory_path": str(inventory),
                    "inventory_sha256": receipt["inventory_sha256"],
                    "missing_patches": receipt["missing_patches"],
                    "audit_path": str(path),
                    "audit_sha256": files[str(path)],
                    "physical_contract_verified": False,
                }
                sources.append(source_row)
                retained = frame.sort_values("member_name").drop_duplicates("patch_id").copy()
                retained = retained[["patch_id", "split", "member_name", "file_sha256", "bytes"]]
                retained = retained.assign(
                    archive_key=key,
                    product_id=product,
                    year=year,
                    month=month,
                    present=True,
                    contract_status="pending",
                    quality_status="pending",
                )
                monthly.append(retained)
    return monthly, pd.DataFrame(sources), files


def _normalized_records(frame):
    return frame.sort_values("scene_group_id").to_dict("records")


def build_quarter_index(dataset_root: Path, report_root: Path, input_path: Path):
    config = json.loads(input_path.read_text())
    validate_config(config)
    registry_path = dataset_root / "registry/national_62000.parquet"
    registry = pd.read_parquet(registry_path)
    if registry.empty or registry.patch_id.duplicated().any():
        raise ValueError("nonempty unique registry required")
    files = {str(input_path): sha256(input_path), str(registry_path): sha256(registry_path)}
    scenes, pool, default, highres_files = load_highres_views(
        dataset_root, config["highres"], registry
    )
    files.update(highres_files)
    files.update(load_target_view(dataset_root, config["target_reader_root"]))
    monthly, archives, dense_files = dense_sources(config["dense_audit_roots"], registry)
    files.update(dense_files)
    for name in ["v5_quarter_index.py", "v5_sampling.py"]:
        path = Path(__file__).with_name(name)
        files[str(path)] = sha256(path)
    fingerprint = {
        "schema": "quarter_source_index_v1",
        "files_sha256": files,
        "source_archives": archives.to_dict("records"),
        "target_reader_root": config["target_reader_root"],
        "scope": "complete quarterly source inventory; dense radiometry and QA unresolved",
        "source_validation": (
            "sealed complete decode receipts and current archive sizes; "
            "actual member SHA rechecked by loader"
        ),
        "policy": {
            "years": [2020, 2021],
            "quarters": [1, 2, 3, 4],
            "highres": "same_year_retrospective",
            "unknown_alignment_is_passed": False,
            "unknown_dense_contract_is_usable": False,
        },
    }
    root = dataset_root / "observations/index/quarters" / _digest(fingerprint)[:20]
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "output.lock.json"

    def unchanged():
        if any(sha256(Path(p)) != h for p, h in files.items()):
            raise ValueError("quarter source evidence changed during indexing")
        if any(
            Path(v.archive_path).stat().st_size != v.archive_bytes for v in archives.itertuples()
        ):
            raise ValueError("quarter source archive size changed")

    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["fingerprint"] != fingerprint or any(
            sha256(root / n) != h for n, h in lock["outputs_sha256"].items()
        ):
            raise ValueError("quarter index output seal changed")
        unchanged()
        result = {
            **lock["summary"],
            "reused": True,
            "output_lock_sha256": sha256(lock_path),
            "finished_at": now(),
        }
        write_json(report_root / "quarter_sample_index.json", result)
        return result
    progress = report_root / "quarter_sample_index_progress.json"
    raw = pd.concat(monthly, ignore_index=True)
    atomic_parquet(raw, root / "dense_observations.parquet")
    atomic_parquet(archives, root / "dense_archives.parquet")
    del monthly
    temporary = root / "quarter_samples.parquet.partial"
    writer = None
    count, available, coverage, candidates, defaults = 0, 0, [], [], []
    try:
        for year in [2020, 2021]:
            samples, annual, chosen = construct_indexes(
                registry, raw.loc[raw.year.eq(year)], scenes, years=(year,)
            )
            # The quarterly adapter must preserve the sealed pool and default selection exactly.
            if not pool.empty:
                columns = list(pool.columns)
                if _digest(_normalized_records(annual[columns])) != _digest(
                    _normalized_records(pool.loc[pool.year.eq(year)])
                ) or _digest(_normalized_records(chosen[columns])) != _digest(
                    _normalized_records(default.loc[default.year.eq(year)])
                ):
                    raise ValueError(
                        "quarterly annual selection differs from sealed high-resolution view"
                    )
            samples["target_reader_root"] = config["target_reader_root"]
            samples["training_authorized"] = False
            table = pa.Table.from_pandas(samples, preserve_index=False)
            list_columns = {
                "dense_observation_ids",
                "dense_inventory_ids",
                "base_exclusion_reasons",
                "missing_monthly_sources",
                "highres_scene_ids",
            }
            table = table.cast(
                pa.schema(
                    [
                        pa.field(f.name, pa.list_(pa.string())) if f.name in list_columns else f
                        for f in table.schema
                    ]
                )
            )
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(samples)
            available += int(samples.base_available.sum())
            for (split, quarter), g in samples.groupby(["split", "quarter"], sort=True):
                coverage.append(
                    {
                        "year": year,
                        "split": split,
                        "quarter": quarter,
                        "samples": len(g),
                        "with_raw_base": int(g.dense_inventory_ids.map(len).gt(0).sum()),
                        "with_usable_base": int(g.base_available.sum()),
                        "with_annual_prior": int(g.highres_scene_ids.map(len).gt(0).sum()),
                    }
                )
            candidates.append(annual)
            defaults.append(chosen)
            write_json(
                progress,
                {
                    "status": "running",
                    "quarter_samples": count,
                    "expected": len(registry) * 8,
                    "finished_year": year,
                    "updated_at": now(),
                    "training_authorized": False,
                },
            )
    finally:
        if writer is not None:
            writer.close()
    if count != len(registry) * 8 or available:
        raise ValueError("quarter inventory scope or dense quarantine changed")
    temporary.replace(root / "quarter_samples.parquet")
    atomic_parquet(
        pd.concat(candidates, ignore_index=True), root / "annual_highres_candidates.parquet"
    )
    atomic_parquet(
        pd.concat(defaults, ignore_index=True), root / "annual_highres_default_selection.parquet"
    )
    atomic_parquet(pd.DataFrame(coverage), root / "coverage.parquet")
    unchanged()
    summary = {
        "status": "source_index_complete_quality_pending",
        "output": str(root),
        "positions": len(registry),
        "quarter_samples": count,
        "dense_archives": len(archives),
        "dense_observations": len(raw),
        "usable_base_observations": 0,
        "with_usable_base_samples": 0,
        "annual_candidates": sum(len(v) for v in candidates),
        "annual_default_scenes": sum(len(v) for v in defaults),
        "sample_index_gate_passed": False,
        "pending": [
            "dense_physical_contract",
            "dense_quality",
            "matching_statistics",
            "all_modality_loader",
            "final_source_scope",
        ],
        "training_authorized": False,
        "user_accepted": False,
    }
    names = [
        "dense_observations.parquet",
        "dense_archives.parquet",
        "quarter_samples.parquet",
        "annual_highres_candidates.parquet",
        "annual_highres_default_selection.parquet",
        "coverage.parquet",
    ]
    seal = {
        "fingerprint": fingerprint,
        "outputs_sha256": {n: sha256(root / n) for n in names},
        "summary": summary,
    }
    write_json(lock_path, seal)
    result = {
        **summary,
        "reused": False,
        "output_lock_sha256": sha256(lock_path),
        "finished_at": now(),
    }
    write_json(report_root / "quarter_sample_index.json", result)
    write_json(progress, result)
    return result
