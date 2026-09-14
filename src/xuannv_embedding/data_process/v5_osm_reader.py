"""Read corrected annual OSM labels against actual source-chunk and overlay digests."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from xuannv_embedding.data_process.v5_cli import atomic_parquet
from xuannv_embedding.data_process.v5_jilin_quality import _digest
from xuannv_embedding.data_process.v5_negative_corrections import (
    CorrectedNegativeReader,
    corrected_overlay,
)
from xuannv_embedding.data_process.v5_negative_rules import TASKS
from xuannv_embedding.data_process.v5_osm_corrections import array_digest
from xuannv_embedding.data_process.v5_osm_geometry import CHANNELS, FIELDS, STRUCTURE
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json


def local_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("OSM correction path escapes its version")
    return path


class CorrectedOSMReader:
    def __init__(self, dataset_root: Path, correction_root: Path):
        self.directory = correction_root
        self.input_path = correction_root / "input.lock.json"
        self.output_path = correction_root / "output.lock.json"
        self.input_sha, self.output_sha = sha256(self.input_path), sha256(self.output_path)
        self.inputs = json.loads(self.input_path.read_text())
        self.seal = json.loads(self.output_path.read_text())
        if self.seal["input_lock_sha256"] != self.input_sha:
            raise ValueError("OSM correction input seal changed")
        for name, expected in self.seal["files_sha256"].items():
            if sha256(local_file(correction_root, name)) != expected:
                raise ValueError("OSM correction publication changed")
        for name, expected in self.inputs["inputs_sha256"].items():
            if sha256(Path(name)) != expected:
                raise ValueError("OSM correction source evidence changed")
        self.fp = self.inputs["source_fingerprint"]
        self.registry_path = dataset_root / "registry/national_62000.parquet"
        self.manifest_path = dataset_root / "targets/manifest.parquet"
        if sha256(self.registry_path) != self.fp["registry_sha256"]:
            raise ValueError("OSM dataset registry differs from correction source")
        if sha256(self.manifest_path) != self.fp["manifest_sha256"]:
            raise ValueError("OSM dataset manifest differs from correction source")
        for name, expected in self.inputs["code_sha256"].items():
            if sha256(Path(__file__).with_name(name)) != expected:
                raise ValueError("OSM correction implementation changed")
        self.registry = pd.read_parquet(self.registry_path)
        if self.registry.patch_id.duplicated().any():
            raise ValueError("duplicate OSM spatial identity")
        self.indices = {row.patch_id: i for i, row in enumerate(self.registry.itertuples())}
        manifest = pd.read_parquet(dataset_root / "targets/manifest.parquet")
        paths = manifest.loc[manifest.family.eq("osm"), "path"].unique()
        if len(paths) != 1:
            raise ValueError("one original OSM label source required")
        self.base_path = Path(paths[0])
        self.base = zarr.open_group(str(self.base_path), mode="r")
        self.audit_root = dataset_root / "quality/targets/geometry/osm/v1/full"
        if json.loads((self.audit_root / "input.lock.json").read_text()) != self.fp:
            raise ValueError("OSM original chunk source contract changed")
        self.rows = pd.read_parquet(correction_root / "manifest.parquet")
        keys = ["patch_id", "year", "resolution"]
        if self.rows.duplicated(keys).any() or not self.rows.source_reconstruction_passed.all():
            raise ValueError("invalid OSM corrected view manifest")
        self.lookup = self.rows.set_index(keys)
        self.metadata_files = {
            str(self.registry_path): self.fp["registry_sha256"],
            str(self.manifest_path): self.fp["manifest_sha256"],
            str(self.audit_root / "input.lock.json"): sha256(self.audit_root / "input.lock.json"),
            str(correction_root / "manifest.parquet"): self.seal["files_sha256"][
                "manifest.parquet"
            ],
        }
        self.decoded_chunks = 0
        self._unchanged()

    def _unchanged(self):
        if (
            sha256(self.input_path) != self.input_sha
            or sha256(self.output_path) != self.output_sha
            or sha256(self.base_path / ".zattrs") != self.fp["metadata_sha256"]
            or any(sha256(Path(p)) != h for p, h in self.metadata_files.items())
        ):
            raise ValueError("OSM publication or source metadata changed")

    def _chunk(self, start):
        stop = min(start + 16, len(self.registry))
        block, digest = {}, hashlib.sha256()
        for year in [2020, 2021]:
            for prefix, channels, pixels in [
                ("", CHANNELS, 128),
                ("structure_2p5m/", STRUCTURE, 512),
            ]:
                for field in FIELDS:
                    for channel in channels:
                        name = f"{year}/{prefix}{field}/{channel}"
                        array = np.asarray(self.base[name][start:stop])
                        if array.dtype != np.dtype("u1") or array.shape != (
                            stop - start,
                            pixels,
                            pixels,
                        ):
                            raise ValueError("OSM source chunk shape or type changed")
                        digest.update(array.tobytes())
                        block[name] = array
        receipt = json.loads((self.audit_root / "chunks" / f"{start:06d}.json").read_text())
        expected = {
            **self.fp,
            "start": start,
            "stop": stop,
            "decoded_chunk_sha256": digest.hexdigest(),
        }
        if receipt.get("fingerprint") != expected:
            raise ValueError("OSM source chunk differs from complete audit")
        self.decoded_chunks += 1
        self._unchanged()
        return block

    def iter_views(self, keys):
        keys = list(keys)
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate OSM view requests must be handled explicitly")
        grouped = defaultdict(list)
        for patch, year, resolution in keys:
            if (
                patch not in self.indices
                or year not in [2020, 2021]
                or resolution not in ["10m", "2p5m"]
            ):
                raise ValueError("invalid OSM view key")
            index = self.indices[patch]
            grouped[index // 16 * 16].append((patch, year, resolution))
        for start in sorted(grouped):
            self._unchanged()
            block = self._chunk(start)
            for key in grouped[start]:
                patch, year, resolution = key
                channels = CHANNELS if resolution == "10m" else STRUCTURE
                prefix = "" if resolution == "10m" else "structure_2p5m/"
                offset = self.indices[patch] - start
                arrays = {
                    field: np.stack(
                        [block[f"{year}/{prefix}{field}/{c}"][offset] for c in channels]
                    )
                    for field in FIELDS
                }
                if key in self.lookup.index:
                    row = self.lookup.loc[key]
                    if array_digest(arrays) != row.original_fields_sha256:
                        raise ValueError("original OSM view differs from correction source")
                    if row.correction_file:
                        path = local_file(self.directory, row.correction_file)
                        if sha256(path) != self.seal["files_sha256"].get(row.correction_file):
                            raise ValueError("OSM correction array changed")
                        with np.load(path, allow_pickle=False) as source:
                            arrays = {k: source[k] for k in source.files}
                    if array_digest(arrays) != row.corrected_fields_sha256:
                        raise ValueError("OSM corrected view differs from recorded array digest")
                pixels = 128 if resolution == "10m" else 512
                if (
                    set(arrays) != set(FIELDS)
                    or any(
                        a.shape != (len(channels), pixels, pixels) or a.dtype != np.dtype("u1")
                        for a in arrays.values()
                    )
                    or not np.isin(arrays["states"], [0, 1]).all()
                    or not np.array_equal(arrays["states"] == 1, arrays["targets"] > 0)
                    or not np.array_equal(
                        arrays["confidence"], np.where(arrays["states"] == 1, 255, 0)
                    )
                    or not np.isin(arrays["source_bits"], [0, 1, 2, 3]).all()
                ):
                    raise ValueError("invalid corrected OSM positive/unknown fields")
                yield key, arrays
            del block

    def read(self, patch_id: str, year: int, resolution: str):
        return next(self.iter_views([(patch_id, year, resolution)]))[1]


def verify_negative_view(reader, patch_id, index, year, fields):
    old = reader.evidence.read(index, year)
    overlay, slope, valid = old["overlay"], old["slope"], old["slope_valid"]
    if (patch_id, year) in reader.rows.index:
        overlay = reader.read(patch_id, year)
        dem = reader.dem.read(patch_id)
        slope, valid = dem["slope"], dem["slope_valid"]
    args = (old["worldcover"], old["worldcover_valid"], slope, valid, reader.evidence.parameters)
    baseline = corrected_overlay(old["states"], *args)
    if any(not np.array_equal(baseline[k], overlay[k]) for k in overlay):
        raise ValueError("existing negative overlay differs from its source rules")
    states = {t: fields["states"][CHANNELS.index(t)] for t in TASKS}
    updated = corrected_overlay(states, *args)
    if any(not np.array_equal(updated[k], overlay[k]) for k in overlay):
        raise ValueError("negative overlay requires regeneration for corrected OSM labels")


def verify_osm_reader(dataset_root: Path, report_root: Path, correction_root: Path):
    reader = CorrectedOSMReader(dataset_root, correction_root)
    negative = CorrectedNegativeReader(dataset_root, report_root)
    keys = list(reader.lookup.index)
    affected = set(reader.rows.patch_id)
    controls = []
    for split in ["train", "val", "test"]:
        candidates = reader.registry.loc[
            reader.registry.split.eq(split) & ~reader.registry.patch_id.isin(affected)
        ]
        ranked = sorted(
            candidates.patch_id,
            key=lambda p: hashlib.sha256(("osm-reader-v1:" + p).encode()).hexdigest(),
        )
        controls.extend(ranked[:100])
    keys += [(p, y, res) for p in controls for y in [2020, 2021] for res in ["10m", "2p5m"]]
    fingerprint = {
        "correction_input_sha256": reader.input_sha,
        "correction_output_sha256": reader.output_sha,
        "negative_audit_sha256": negative.evidence.audit_sha,
        "negative_correction_lock_sha256": sha256(negative.directory / "corrections.lock.json"),
        "dem_correction_lock_sha256": sha256(negative.demlock),
        "code_sha256": sha256(Path(__file__)),
        "keys_sha256": _digest(keys),
    }
    directory = dataset_root / "quality/targets/osm_reader" / _digest(fingerprint)[:20]
    directory.mkdir(parents=True, exist_ok=True)
    records, negatives = [], 0
    progress = report_root / "osm_reader_progress.json"
    for key, fields in reader.iter_views(keys):
        patch, year, resolution = key
        index = reader.indices[patch]
        corrected = key in reader.lookup.index and bool(reader.lookup.loc[key].correction_file)
        if corrected and resolution == "10m":
            verify_negative_view(negative, patch, index, year, fields)
            negatives += 1
        records.append(
            {
                "patch_id": patch,
                "year": int(year),
                "resolution": resolution,
                "split": reader.registry.iloc[index].split,
                "corrected": corrected,
                "fields_sha256": array_digest(fields),
                "negative_checked": corrected and resolution == "10m",
            }
        )
        if len(records) % 40 == 0:
            write_json(
                progress,
                {
                    "status": "running",
                    "verified_views": len(records),
                    "selected_views": len(keys),
                    "decoded_chunks": reader.decoded_chunks,
                    "updated_at": now(),
                    "training_authorized": False,
                },
            )
    reader._unchanged()
    if (
        sha256(negative.evidence.audit_path) != fingerprint["negative_audit_sha256"]
        or sha256(negative.directory / "corrections.lock.json")
        != fingerprint["negative_correction_lock_sha256"]
        or sha256(negative.demlock) != fingerprint["dem_correction_lock_sha256"]
        or sha256(Path(__file__)) != fingerprint["code_sha256"]
    ):
        raise ValueError("OSM reader verification evidence changed during execution")
    summary = {
        "status": "corrected_OSM_reader_and_negative_dependencies_verified",
        "verified_views": len(records),
        "affected_positions": len(affected),
        "control_positions": len(controls),
        "negative_views_verified": negatives,
        "decoded_source_chunks": reader.decoded_chunks,
        "output": str(directory),
        "scope": (
            "all correction positions plus up to 100 unmodified positions per split; "
            "annual OSM labels only"
        ),
        "training_authorized": False,
        "user_accepted": False,
    }
    lock_path = directory / "verification.lock.json"
    seal = {"fingerprint": fingerprint, "records_sha256": _digest(records), "summary": summary}
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != seal:
            raise ValueError("OSM reader verification replay changed")
        saved = pd.read_parquet(directory / "views.parquet").to_dict("records")
        if _digest(saved) != seal["records_sha256"]:
            raise ValueError("OSM reader verification output changed")
    else:
        atomic_parquet(pd.DataFrame(records), directory / "views.parquet")
        write_json(lock_path, seal)
    result = {**summary, "verification_lock_sha256": sha256(lock_path), "finished_at": now()}
    write_json(report_root / "osm_reader_verification.json", result)
    write_json(progress, result)
    return result
