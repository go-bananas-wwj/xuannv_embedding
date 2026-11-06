"""Bind statistics to the exact sealed annual high-resolution candidate view."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xuannv_embedding.data_process import v5_highres_eligibility as eligibility
from xuannv_embedding.data_process.v5_highres_statistics import training_rows
from xuannv_embedding.data_process.v5_jilin_quality import _digest
from xuannv_embedding.data_process.v5_sources import sha256


class CandidateSelection:
    def __init__(self, reader, root: Path):
        import pandas as pd

        # Check cross-split content before filtering; an excluded alias still has an owner.
        training_rows(reader.inventory)
        inputs, output = root / "inputs.lock.json", root / "output.lock.json"
        locked = json.loads(inputs.read_text())
        seal = json.loads(output.read_text())
        fp = locked["fingerprint"]
        self.files = {str(inputs): sha256(inputs), str(output): sha256(output)}
        if seal["inputs_lock_sha256"] != self.files[str(inputs)]:
            raise ValueError("candidate evidence input seal changed")
        if "branches.parquet" not in seal["outputs_sha256"]:
            raise ValueError("candidate evidence lacks branch table")
        for name, expected in seal["outputs_sha256"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("candidate evidence path escapes version")
            self.files[str(path)] = expected
        self.files.update(fp["files_sha256"])
        self.files.update(
            {str(Path(__file__).with_name(n)): h for n, h in fp["code_sha256"].items()}
        )
        if (
            fp["family"] != reader.family
            or fp["quality_configuration"] != reader.configuration
            or any(fp["files_sha256"].get(p) != h for p, h in reader.files.items())
        ):
            raise ValueError("candidate QA snapshot differs from statistics reader")
        if fp["policy"] != eligibility.POLICY:
            raise ValueError("unsupported candidate selection policy")
        self.verify_unchanged()
        normalized = eligibility.normalize_branches(reader, reader.family)
        if _digest(normalized.to_dict("records")) != fp["rows_sha256"]:
            raise ValueError("candidate source rows changed")
        branches = pd.read_parquet(root / "branches.parquet")
        if branches.observation_id.duplicated().any() or _digest(
            branches[normalized.columns].sort_values("observation_id").to_dict("records")
        ) != _digest(normalized.sort_values("observation_id").to_dict("records")):
            raise ValueError("candidate branch source rows differ from frozen QA")
        branches = branches.set_index("observation_id").loc[reader.inventory.observation_id]
        flag = branches.tolerant_reconstruction_candidate
        if (
            flag.dtype != bool
            or flag.isna().any()
            or not np.array_equal(flag.to_numpy(), branches.exclusion_reasons.map(len).eq(0))
            or branches.loc[flag, "native_alignment_status"].eq("over_limit").any()
            or branches.loc[flag, "valid_pixels_by_band"].map(lambda v: sum(v) <= 0).any()
            or branches.training_authorized.any()
            or branches.strict_pixel_fusion_candidate.any()
        ):
            raise ValueError("inconsistent candidate qualification or authorization")
        selected = flag.to_numpy() & reader.inventory.split.eq("train").to_numpy()
        self.rows = (
            reader.inventory.loc[selected].sort_values("observation_id").reset_index(drop=True)
        )
        if self.rows.duplicated(["product_id", "sensor", "file_sha256"]).any():
            raise ValueError("duplicate eligible training source")
        self.excluded = reader.inventory.loc[~selected].copy()
        self.excluded["reason"] = [
            ";".join(
                (["non_training_split"] if row.split != "train" else [])
                + list(branches.loc[row.observation_id, "exclusion_reasons"])
            )
            for row in self.excluded.itertuples()
        ]
        self.descriptor = {
            "root": str(root),
            "inputs_lock_sha256": self.files[str(inputs)],
            "output_lock_sha256": self.files[str(output)],
            "policy": fp["policy"],
            "selection": "all eligible training observations, once per source; not quarterly reuse",
        }

    def verify_unchanged(self):
        if any(sha256(Path(p)) != h for p, h in self.files.items()):
            raise ValueError("candidate evidence changed")
