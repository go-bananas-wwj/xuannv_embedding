"""Reuse sealed native-band evidence only when source and valid-mask identity are unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xuannv_embedding.data_process.v5_sources import sha256


def inherit_jilin_audit(root, inputs, audit, historical_files, reader):
    """Metadata/receipt proof, not a new pixel audit or absolute-registration approval."""
    for name, expected in historical_files.items():
        if not Path(name).is_file() or sha256(Path(name)) != expected:
            raise ValueError("historical QA input changed")
    for basename in ["configuration.lock.json", "national_62000.parquet"]:
        old = [h for p, h in historical_files.items() if Path(p).name == basename]
        current = [h for p, h in reader.files.items() if Path(p).name == basename]
        if len(old) != 1 or old != current:
            raise ValueError("inheritance configuration or registry changed")
    prior_paths = [p for p in historical_files if Path(p).name == "observation_quality.parquet"]
    if len(prior_paths) != 1:
        raise ValueError("one historical QA table required for inheritance")
    prior = pd.read_parquet(prior_paths[0]).set_index("observation_id")
    current = reader.inventory.set_index("observation_id")
    if (
        not prior.index.is_unique
        or not current.index.is_unique
        or inputs.observation_id.duplicated().any()
    ):
        raise ValueError("duplicate inheritance source identity")
    sources = inputs.set_index("observation_id")
    evidence = dict(historical_files)
    receipt_cache = {}
    for row in audit.itertuples(index=False):
        oid = row.observation_id
        if oid not in current.index or oid not in prior.index or oid not in reader.qa.table.index:
            raise ValueError("inheritance source observation missing from current snapshot")
        source, target = sources.loc[oid], current.loc[oid]
        for key in ["file_sha256", "path", "patch_id", "sensor", "year", "split"]:
            if source[key] != target[key]:
                raise ValueError("inheritance source identity changed")
        old_qa, new_qa = prior.loc[oid], reader.qa.table.loc[oid]
        for key in ["file_sha256", "patch_id", "sensor", "year", "split"]:
            if source[key] != old_qa[key]:
                raise ValueError("inheritance source disagrees with historical QA")
        for key in ["patch_id", "sensor", "year", "split"]:
            if getattr(row, key) != source[key]:
                raise ValueError("inheritance source disagrees with audit output")
        for key in [
            "file_sha256",
            "patch_id",
            "sensor",
            "year",
            "split",
            "scene_group_id",
            "acquired_at",
            "band_ids",
            "quality_status",
        ]:
            if not np.array_equal(old_qa[key], new_qa[key]):
                raise ValueError("inheritance QA identity or band contract changed")
        scene = new_qa.scene_group_id
        if scene not in receipt_cache:
            receipt_path = reader.qa.root / "receipts" / f"{scene}.json"
            digest = sha256(receipt_path)
            if digest != reader.qa.receipts.get(scene):
                raise ValueError("inheritance receipt changed")
            evidence[str(receipt_path)] = digest
            receipt_cache[scene] = json.loads(receipt_path.read_text())
        expected = receipt_cache[scene]["mask_arrays"].get(oid + "/valid")
        if not expected or row.quality_mask_sha256 != expected:
            raise ValueError("inheritance mask differs from original audited mask")
    result = audit.copy()
    result["inherited_from"] = str(root)
    result["inheritance_scope"] = "source_and_mask_unchanged_across_QA_snapshots"
    return result, evidence
