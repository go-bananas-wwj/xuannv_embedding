"""Pinned, supervision-only annual labels; no dependency of encoder or export inputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from xuannv_embedding.data.target_contract import (
    LEGACY_TARGET_SCHEMA,
    LOCAL_TASKS,
    SCHEMA_TASKS,
    TRAINING_SCOPE_KEYS,
    training_positions,
)
from xuannv_embedding.data_process.target_audit import WC_CODES, digest, local_path


class AnnualTargetReader:
    def __init__(
        self,
        target_root: Path,
        prepared_root: Path,
        *,
        preparation_sha256: str,
        scope: str = "screen",
        allow_evaluation: bool = False,
    ):
        import pyarrow.parquet as pq

        self.root, self.prepared = target_root, prepared_root
        if scope not in TRAINING_SCOPE_KEYS:
            raise ValueError(
                "Training targets require screen, expanded or allowed experiment scope"
            )
        path = prepared_root / "preparation.json"
        if digest(path) != preparation_sha256:
            raise ValueError("Target preparation identity mismatch")
        self.preparation = json.loads(path.read_text())
        if self.preparation["status"] != "engineering_data_prepared":
            raise ValueError("Target preparation is incomplete")
        for filename, field in (
            ("manifest.json", "target_manifest_sha256"),
            ("index.parquet", "target_index_sha256"),
        ):
            if digest(target_root / filename) != self.preparation[field]:
                raise ValueError(f"Published target identity mismatch: {filename}")
        manifest = json.loads((target_root / "manifest.json").read_text())
        index = pq.read_table(
            target_root / "index.parquet", columns=["patch_id", "target_row"]
        ).to_pylist()
        self.rows = {row["patch_id"]: row["target_row"] for row in index}
        if len(self.rows) != manifest["patches"] or sorted(self.rows.values()) != list(
            range(manifest["patches"])
        ):
            raise ValueError("Target position registry is not bijective")
        split_path = prepared_root / "experiment_splits.json"
        if digest(split_path) != self.preparation["splits_sha256"]:
            raise ValueError("Target experiment split identity mismatch")
        split = json.loads(split_path.read_text())
        self.roles = {
            role: {row["patch_id"] for row in rows} for role, rows in split["roles"].items()
        }
        scope_ids = training_positions(split, scope)
        self.roles["train"] = set(scope_ids)
        self.tasks = SCHEMA_TASKS[self.preparation.get("target_schema", LEGACY_TARGET_SCHEMA)]
        self.allow_evaluation = allow_evaluation
        stats_path = prepared_root / f"train_statistics.{scope}.json"
        if digest(stats_path) != self.preparation["train_statistics_sha256"][scope]:
            raise ValueError("Target train statistics identity mismatch")
        self.statistics = json.loads(stats_path.read_text())
        expected_rows = [self.rows[patch] for patch in scope_ids]
        if (
            hashlib.sha256(json.dumps(expected_rows).encode()).hexdigest()
            != self.statistics["target_rows_sha256"]
        ):
            raise ValueError("Target statistics position identity mismatch")
        if self.statistics["scope"] != scope or self.statistics["unique_positions"] != len(
            self.roles["train"]
        ):
            raise ValueError("Target statistics were not fit on this training scope")
        normalization = self.statistics["targets"]["dem_elevation"]
        self.dem_mean, self.dem_std = normalization["mean"], normalization["population_std"]
        if not np.isfinite([self.dem_mean, self.dem_std]).all() or self.dem_std <= 0:
            raise ValueError("Invalid DSM train normalization")
        self.native = prepared_root / "native_dem"
        if digest(self.native / "manifest.json") != self.preparation["native_manifest_sha256"]:
            raise ValueError("Native DSM manifest identity mismatch")
        native = json.loads((self.native / "manifest.json").read_text())
        if not native["complete"] or native["spacing_m"] != 40:
            raise ValueError("Native DSM cache is incomplete or has unsupported spacing")
        if digest(self.native / "index.parquet") != native["index_sha256"]:
            raise ValueError("Native DSM row receipts changed")
        native_rows = pq.read_table(self.native / "index.parquet").to_pylist()
        self.row_hashes = {row["patch_id"]: row["row_sha256"] for row in native_rows}
        if {row["patch_id"]: row["target_row"] for row in native_rows} != self.rows:
            raise ValueError("Native and published target row identities differ")
        # Hash each WorldCover archive once before fork; child readers reopen handles, not hashes.
        self.receipts = {}
        for entry in manifest["files"]:
            if entry["path"] in {f"static/worldcover_{year}.zarr.zip" for year in (2020, 2021)}:
                path = local_path(target_root, entry["path"])
                if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                    raise ValueError("WorldCover archive integrity mismatch")
                self.receipts[entry["path"]] = (path.stat().st_size, path.stat().st_mtime_ns)
        if len(self.receipts) != 2:
            raise ValueError("Both annual WorldCover archives are required")
        self.groups, self.stores, self.pid = {}, {}, os.getpid()
        self.local_cache = None
        if "local_manifest_sha256" in self.preparation:
            from xuannv_embedding.data.local_target_cache import LocalTargetCache

            self.local_cache = LocalTargetCache(
                prepared_root, self.preparation, scope, self.roles["train"], self.rows
            )

    def _group(self, name: str):
        import zarr

        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        if name != "dem_elevation":
            path = local_path(self.root, f"static/{name}.zarr.zip")
            status = path.stat()
            if (status.st_size, status.st_mtime_ns) != self.receipts[f"static/{name}.zarr.zip"]:
                raise ValueError("WorldCover changed after integrity verification")
        if name not in self.groups:
            if name == "dem_elevation":
                group = zarr.open_group(str(self.native / "dem40m.zarr"), mode="r")
                if (
                    not group.attrs.get("complete")
                    or group["values"].shape != (len(self.rows), 32, 32)
                    or group["valid"].shape != group["values"].shape
                ):
                    raise ValueError("Native DSM array contract changed")
            else:
                relative = f"static/{name}.zarr.zip"
                path = local_path(self.root, relative)
                status = path.stat()
                if (status.st_size, status.st_mtime_ns) != self.receipts[relative]:
                    raise ValueError("WorldCover changed after integrity verification")
                store = zarr.ZipStore(str(path), mode="r")
                self.stores[name] = store
                group = zarr.open_group(store, mode="r")
            self.groups[name] = group
        return self.groups[name]

    def read(self, patch_id: str, year: int, *, role: str = "train", tasks=None) -> dict:
        if isinstance(year, bool) or year not in {2020, 2021}:
            raise ValueError("Targets require an explicit supported observation year")
        if role != "train" and not self.allow_evaluation:
            raise ValueError("Evaluation target access is sealed by default")
        if role not in self.roles or patch_id not in self.roles[role]:
            raise ValueError("Position is outside the frozen target experiment role")
        tasks = ("dem_elevation", "worldcover") if tasks is None else tuple(tasks)
        if not tasks or set(tasks) - self.tasks:
            raise ValueError("Unknown or empty independent target task list")
        row, result = self.rows[patch_id], {}
        if "dem_elevation" in tasks:
            group = self._group("dem_elevation")
            values = np.asarray(group["values"][row], dtype=np.float32)
            valid = np.asarray(group["valid"][row], dtype=bool)
            actual = hashlib.sha256(values.tobytes() + valid.tobytes()).hexdigest()
            if actual != self.row_hashes[patch_id]:
                raise ValueError("Native DSM row integrity mismatch")
            if not np.isfinite(values[valid]).all():
                raise ValueError("Nonfinite valid DSM target")
            result["dem_elevation"] = {"values": np.where(valid, values, 0), "valid": valid}
        if "worldcover" in tasks:
            name = f"worldcover_{year}"
            group = self._group(name)
            raw = np.asarray(group[f"targets/{name}"][row])
            valid = np.asarray(group[f"valid_masks/{name}"][row], dtype=bool)
            if raw.shape != (128, 128) or valid.shape != raw.shape:
                raise ValueError("WorldCover shape mismatch")
            if (valid & ~np.isin(raw, WC_CODES)).any():
                raise ValueError("Illegal valid WorldCover class code")
            values = np.full(raw.shape, -100, dtype=np.int64)
            for code, label in zip(WC_CODES, range(len(WC_CODES))):
                values[valid & (raw == code)] = label
            result["worldcover"] = {"values": values, "valid": valid}
        local_tasks = sorted(set(tasks) & set(LOCAL_TASKS))
        if local_tasks:
            if self.local_cache is None or role != "train":
                raise ValueError("Local targets require a prepared training-only cache")
            result.update(self.local_cache.read(patch_id, year, local_tasks))
        return result

    def close(self):
        for store in self.stores.values():
            store.close()
        self.stores.clear()
        self.groups.clear()
        if self.local_cache is not None:
            self.local_cache.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(groups={}, stores={}, pid=None)
        return state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def collate_targets(samples: list[dict]) -> dict:
    import torch

    if not samples:
        raise ValueError("Target batch cannot be empty")
    tasks = set(samples[0])
    if any(set(sample) != tasks for sample in samples):
        raise ValueError("Target task lists differ across samples")
    fields = {task: set(samples[0][task]) for task in tasks}
    if any(
        fields[task] - {"values", "valid", "weight"} or not {"values", "valid"} <= fields[task]
        for task in tasks
    ):
        raise ValueError("Invalid supervision tensor fields")
    if any(set(sample[task]) != fields[task] for sample in samples for task in tasks):
        raise ValueError("Target tensor fields differ across samples")
    return {
        task: {
            field: torch.from_numpy(np.stack([sample[task][field] for sample in samples]))
            for field in sorted(fields[task])
        }
        for task in sorted(tasks)
    }
