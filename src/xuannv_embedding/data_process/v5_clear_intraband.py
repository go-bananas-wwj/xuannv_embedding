"""Read verified native QA and calibrate relative registration on clear ground only."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from xuannv_embedding.data_process.v5_intraband import (
    CALIBRATION,
    _code_fingerprint,
    _read_row,
    _root,
    _runtime_versions,
    calibrate_family,
    calibrate_texture,
    inspect_intraband,
)
from xuannv_embedding.data_process.v5_jilin_quality import _array_digest, _digest
from xuannv_embedding.data_process.v5_quality import quality_masks
from xuannv_embedding.data_process.v5_rasters import NativeRaster
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def apply_clear_mask(
    frame: NativeRaster, mask: np.ndarray, band_ids: tuple[str, ...] | list[str]
) -> NativeRaster:
    if tuple(band_ids) != frame.band_ids:
        raise ValueError("quality band identities or order disagree")
    if mask.shape != frame.values.shape:
        raise ValueError("quality shape differs from native bands")
    if mask.dtype != np.dtype(bool):
        raise ValueError("quality mask must be boolean")
    return replace(frame, valid=frame.valid & mask)


class NativeQualityReader:
    """Freeze QA metadata and check source identities and actual mask bytes on every read."""

    def __init__(self, dataset_root: Path, family: str, quality_root: Path):
        self.family = family
        self.root = quality_root
        self.files = {}
        registry_path = dataset_root / "registry/national_62000.parquet"
        self.registry = pd.read_parquet(registry_path).set_index("patch_id")
        if not self.registry.index.is_unique:
            raise ValueError("duplicate national patch")
        self.files[str(registry_path)] = sha256(registry_path)

        def read_json(path):
            self.files[str(path)] = sha256(path)
            return json.loads(path.read_text())

        def read_table(path):
            self.files[str(path)] = sha256(path)
            return pd.read_parquet(path)

        if family == "gaofen":
            self.configuration = read_json(quality_root / "source.lock.json")
            if (
                self.configuration.get("limit") is not None
                or self.configuration["registry_sha256"] != sha256(registry_path)
                or self.configuration["input_bands"] != ["red", "green", "nir"]
                or self.configuration["buffer_m"] != 30
            ):
                raise ValueError("unverified complete Gaofen QA contract")
            order = read_table(quality_root / "observation_order.parquet")
            table = read_table(quality_root / "observation_quality.parquet")
            if (
                order.pair_id.duplicated().any()
                or table.pair_id.duplicated().any()
                or order.pair_id.tolist() != sorted(order.pair_id)
                or set(order.pair_id) != set(table.pair_id)
            ):
                raise ValueError("Gaofen QA order or completeness disagreement")
            self.positions = {identity: i for i, identity in enumerate(order.pair_id)}
            self.table = table.set_index("pair_id")
            self.masks = zarr.open_group(str(quality_root / "valid_masks.zarr"), mode="r")
            self.classes = zarr.open_group(str(quality_root / "classes.zarr"), mode="r")
            if (
                self.masks.attrs.get("bitorder") != "little"
                or self.masks.attrs.get("packed_axis") != -1
                or not np.asarray(self.masks["completed"]).all()
                or len(self.masks["completed"]) != len(order)
            ):
                raise ValueError("Gaofen QA is incomplete or has unsupported packing")
            for group in [self.masks, self.classes]:
                if any(group.attrs.get(k) != v for k, v in self.configuration.items()):
                    raise ValueError("Gaofen QA storage contract changed")
        elif family == "jilin1":
            locked = read_json(quality_root / "quality.lock.json")
            snapshot = locked["snapshot"]
            if snapshot.get("limit") is not None:
                raise ValueError("complete frozen Jilin QA snapshot is required")
            catalog_path = Path(snapshot["catalog_path"])
            if sha256(catalog_path) != snapshot["catalog_sha256"]:
                raise ValueError("Jilin QA source catalog changed")
            self.files[str(catalog_path)] = snapshot["catalog_sha256"]
            catalog_lock_path = catalog_path.parent / "catalog.lock.json"
            catalog_lock = read_json(catalog_lock_path)
            if sha256(catalog_lock_path) != snapshot["catalog_lock_sha256"] or catalog_lock[
                "fingerprint"
            ]["registry_sha256"] != sha256(registry_path):
                raise ValueError("Jilin QA source catalog or national registry changed")
            self.root = quality_root.parent.parent
            self.configuration = read_json(self.root / "configuration.lock.json")
            if _digest(self.configuration)[:20] != snapshot["configuration_id"]:
                raise ValueError("Jilin QA configuration changed")
            table = read_table(quality_root / "observation_quality.parquet")
            if (
                sha256(quality_root / "observation_quality.parquet")
                != locked["quality_table_sha256"]
                or len(table) != locked["branches"]
                or table.observation_id.duplicated().any()
                or table.scene_group_id.nunique() != locked["processed_scenes"]
            ):
                raise ValueError("Jilin QA table or completeness changed")
            self.receipts = locked["receipts_sha256"]
            if set(table.scene_group_id) != set(self.receipts):
                raise ValueError("Jilin QA scene receipts disagree")
            self.table = table.set_index("observation_id")
            self.masks = zarr.open_group(str(self.root / "valid_masks.zarr"), mode="r")
            if self.masks.attrs.get("configuration_id") != snapshot["configuration_id"]:
                raise ValueError("Jilin QA mask configuration changed")
        else:
            raise ValueError("unknown native QA family")
        if self.table.empty or not self.table.year.isin([2020, 2021]).all():
            raise ValueError("empty or unsupported annual QA table")
        if not self.table.patch_id.isin(self.registry.index).all():
            raise ValueError("QA outside national registry")
        if not np.array_equal(
            self.table.split.to_numpy(), self.registry.loc[self.table.patch_id, "split"].to_numpy()
        ):
            raise ValueError("QA spatial split disagreement")
        code = self.configuration.get("code_sha256", {})
        if not code or any(sha256(Path(__file__).with_name(n)) != h for n, h in code.items()):
            raise ValueError("QA implementation differs from its locked contract")
        self.code = code

    def verify_unchanged(self):
        if any(sha256(Path(p)) != h for p, h in self.files.items()) or any(
            sha256(Path(__file__).with_name(n)) != h for n, h in self.code.items()
        ):
            raise ValueError("QA inputs changed during clear alignment")

    def read(self, row):
        frame, gsd, reference = _read_row(row, self.family)
        record = self.table.loc[row.observation_id]
        for field in ["patch_id", "sensor", "split", "year"]:
            if record[field] != getattr(row, field):
                raise ValueError("source and QA observation identity disagree")
        if self.family == "gaofen":
            if record.ms_sha256 != row.file_sha256 or Path(record.ms_path) != Path(row.path):
                raise ValueError("Gaofen QA source hash or path changed")
            i = self.positions[row.observation_id]

            def unpack(name):
                return np.unpackbits(
                    np.asarray(self.masks[name][i]), axis=-1, count=160, bitorder="little"
                ).astype(bool)

            data_valid = frame.valid.all(axis=0)
            if not np.array_equal(unpack("data_valid_packed"), data_valid):
                raise ValueError("Gaofen native validity differs from QA")
            expected = quality_masks(np.asarray(self.classes["classes"][i]), data_valid, gsd=gsd)
            clear = unpack("ms_valid_packed")
            if (
                not np.array_equal(clear, expected["valid"])
                or not np.array_equal(unpack("before_buffer_packed"), expected["before_buffer"])
                or int(clear.sum()) != record.ms_valid_pixels
            ):
                raise ValueError("Gaofen stored QA mask differs from cloud classification")
            mask = np.broadcast_to(clear, frame.values.shape).copy()
            bands = frame.band_ids
        else:
            if record.file_sha256 != row.file_sha256:
                raise ValueError("Jilin QA source hash changed")
            scene = record.scene_group_id
            receipt_path = self.root / "receipts" / f"{scene}.json"
            if sha256(receipt_path) != self.receipts[scene]:
                raise ValueError("Jilin QA receipt changed")
            receipt = json.loads(receipt_path.read_text())
            prefix = row.observation_id + "/"
            arrays = {}
            for name in ["data_valid", "valid", "qa_clear"]:
                array = np.asarray(self.masks[prefix + name])
                if _array_digest(array) != receipt["mask_arrays"][prefix + name]:
                    raise ValueError("Jilin QA mask changed")
                arrays[name] = array
            mask, bands = arrays["valid"], tuple(record.band_ids)
            if (
                not np.array_equal(arrays["data_valid"], frame.valid)
                or not np.array_equal(mask, frame.valid & arrays["qa_clear"][None])
                or not np.array_equal(mask.sum(axis=(1, 2)), record.valid_pixels_by_band)
            ):
                raise ValueError("Jilin native validity differs from QA")
            if record.quality_status == "qa_missing" and mask.any():
                raise ValueError("missing Jilin QA contains valid pixels")
        masked = apply_clear_mask(frame, mask, bands)
        provenance = {
            "observation_id": row.observation_id,
            "file_sha256": row.file_sha256,
            "quality_mask_sha256": _array_digest(mask),
            "band_ids": list(bands),
            "crs": frame.crs,
            "transform": list(frame.transform),
            "quality_status": record.quality_status,
            "clear_fraction_by_band": masked.valid.mean(axis=(1, 2)).tolist(),
        }
        return frame, masked, gsd, reference, provenance


def calibrate_clear(
    dataset_root: Path, report_root: Path, family: str, quality_root: Path, *, version="v5"
):
    """Compare the exact existing calibration samples with QA; never select easier replacements."""
    baseline = calibrate_family(dataset_root, report_root, family, version=version)
    if baseline["status"] != "passed":
        raise ValueError("passed original native calibration is required")
    original = _root(dataset_root, family, version)
    inputs = original / "calibration_inputs.parquet"
    rows = pd.read_parquet(inputs)
    if not rows.split.eq("train").all():
        raise ValueError("calibration must use training positions only")
    reader = NativeQualityReader(dataset_root, family, quality_root)
    code = {**_code_fingerprint(), Path(__file__).name: sha256(Path(__file__))}
    fingerprint = {
        "code_sha256": code,
        "runtime": {**_runtime_versions(), "pandas": pd.__version__, "zarr": zarr.__version__},
        "baseline_calibration_sha256": sha256(original / "calibration.json"),
        "baseline_inputs_sha256": sha256(inputs),
        "quality_inputs_sha256": reader.files,
        "family": family,
        "baseline_version": version,
        "calibration": CALIBRATION,
    }
    results = []
    for row in rows.itertuples():
        raw, masked, gsd, reference, provenance = reader.read(row)
        band = masked.band_ids.index(reference)
        results.append(
            {
                **provenance,
                "patch_id": row.patch_id,
                "sensor": row.sensor,
                "split": row.split,
                "known_shift_calibration": calibrate_texture(
                    masked.values[band], masked.valid[band], gsd=gsd
                ),
                "raw_alignment": inspect_intraband(raw, reference_band=reference, gsd=gsd),
                "clear_alignment": inspect_intraband(masked, reference_band=reference, gsd=gsd),
            }
        )
        write_json(
            report_root / f"clear_alignment_calibration_{family}_progress.json",
            {"status": "running", "processed": len(results), "selected": len(rows), "at": now()},
        )
    reader.verify_unchanged()
    if (
        any(sha256(Path(__file__).with_name(n)) != h for n, h in code.items())
        or sha256(inputs) != fingerprint["baseline_inputs_sha256"]
        or sha256(original / "calibration.json") != fingerprint["baseline_calibration_sha256"]
    ):
        raise ValueError("clear calibration code or frozen inputs changed")
    sensors = {
        sensor: dict(
            Counter(
                r["known_shift_calibration"]["status"] for r in results if r["sensor"] == sensor
            )
        )
        for sensor in sorted(rows.sensor.unique())
    }
    failed = any(count.get("failed", 0) for count in sensors.values())
    enough = bool(sensors) and all(
        count.get("passed", 0) >= CALIBRATION["minimum_successful_positions_per_sensor"]
        for count in sensors.values()
    )
    locked = {
        "fingerprint": fingerprint,
        "status": "failed" if failed else "passed" if enough else "insufficient_clear_calibration",
        "sensors": sensors,
        "results": results,
        "scope": "fixed training textures with clear QA; relative spectral alignment only",
        "pixel_fusion_authorized": False,
        "training_authorized": False,
    }
    identity = _digest(
        {"fingerprint": fingerprint, "masks": [r["quality_mask_sha256"] for r in results]}
    )[:20]
    output = (
        dataset_root / "quality/alignment/clear_intraband" / family / identity / "calibration.json"
    )
    if output.exists():
        if json.loads(output.read_text()) != locked:
            raise ValueError("published clear calibration changed")
    else:
        write_json(output, locked)
    result = {
        "status": locked["status"],
        "processed": len(results),
        "sensors": sensors,
        "output": str(output),
        "sha256": sha256(output),
        "finished_at": now(),
        "pixel_fusion_authorized": False,
        "training_authorized": False,
    }
    write_json(report_root / f"clear_alignment_calibration_{family}.json", result)
    return result
