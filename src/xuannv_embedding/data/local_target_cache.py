"""Row-verified, versioned derived targets; cached independently from image inputs."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from xuannv_embedding.data.target_contract import LOCAL_TASKS, TARGET_TASKS, target_shape
from xuannv_embedding.data_process.target_audit import digest


def target_row_sha(fields: dict) -> str:
    hasher = hashlib.sha256()
    for name in ("values", "valid", "weight"):
        array = np.ascontiguousarray(fields[name])
        hasher.update(f"{name}/{array.dtype}/{array.shape}".encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


class LocalTargetCache:
    def __init__(self, prepared, preparation, scope, train_ids, source_rows):
        import pyarrow.parquet as pq

        self.root = prepared / "local_targets"
        if digest(self.root / "manifest.json") != preparation["local_manifest_sha256"]:
            raise ValueError("Local target manifest identity mismatch")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if not self.manifest["complete"] or self.manifest["scope"] != scope:
            raise ValueError("Local targets are incomplete or do not cover this scope")
        if digest(self.root / "index.parquet") != self.manifest["index_sha256"]:
            raise ValueError("Local target row receipts changed")
        rows = pq.read_table(self.root / "index.parquet").to_pylist()
        self.rows = {(r["patch_id"], r["year"]): r for r in rows}
        if len(self.rows) != len(train_ids) * 2 or set(self.rows) != {
            (patch, year) for patch in train_ids for year in (2020, 2021)
        }:
            raise ValueError("Local cache does not cover the frozen paired training positions")
        for year in (2020, 2021):
            annual = [r for r in rows if r["year"] == year]
            if sorted(r["cache_row"] for r in annual) != list(range(len(train_ids))):
                raise ValueError("Local cache row identities are not bijective")
        if any(r["target_row"] != source_rows[r["patch_id"]] for r in rows):
            raise ValueError("Local cache and published target rows differ")
        self.group, self.pid = None, None

    def read(self, patch, year, tasks):
        import zarr

        if self.pid != os.getpid():
            self.group = None
            self.pid = os.getpid()
        if self.group is None:
            self.group = zarr.open_group(str(self.root / "targets.zarr"), mode="r")
            if not self.group.attrs.get("complete"):
                raise ValueError("Local cache is incomplete")
        receipt, result = self.rows[(patch, year)], {}
        count = self.manifest["positions"]
        for task in tasks:
            if task not in LOCAL_TASKS:
                raise ValueError("Unsupported local supervision task")
            fields = {}
            for name in ("values", "valid", "weight"):
                array = self.group[f"{year}/{task}/{name}"]
                shape = target_shape(task)
                if name != "values" and TARGET_TASKS[task][2] == "soft_categorical":
                    shape = shape[1:]
                expected_dtype = np.dtype(bool if name == "valid" else np.float32)
                if array.shape != (count, *shape) or array.dtype != expected_dtype:
                    raise ValueError("Local supervision array contract changed")
                fields[name] = np.asarray(array[receipt["cache_row"]])
            if target_row_sha(fields) != receipt[f"{task}_sha256"]:
                raise ValueError("Local supervision row integrity mismatch")
            result[task] = fields
        return result

    def close(self):
        self.group = None

    def __getstate__(self):
        return {**self.__dict__, "group": None, "pid": None}
