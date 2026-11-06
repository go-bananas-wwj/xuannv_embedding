"""Read source-audited annual maps and assemble corrected positive/unknown/negative targets."""

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
from xuannv_embedding.data_process.v5_negative_corrections import CorrectedNegativeReader
from xuannv_embedding.data_process.v5_negative_rules import TASKS
from xuannv_embedding.data_process.v5_osm_geometry import CHANNELS
from xuannv_embedding.data_process.v5_osm_reader import CorrectedOSMReader
from xuannv_embedding.data_process.v5_sources import now, sha256, write_json
from xuannv_embedding.data_process.v5_target_geometry import FAMILIES, POLICY

AUDITS = {
    "clcd": "target_geometry_clcd_full.json",
    "nightlights": "target_geometry_nightlights_full.json",
    "worldcover": "target_geometry_worldcover_v2_full.json",
}


def checked_keys(registry, keys):
    keys = list(keys)
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate annual target keys")
    if any(p not in registry.index or y not in [2020, 2021] for p, y in keys):
        raise ValueError("invalid annual target key")
    return keys


class AnnualTargetReader:
    def __init__(self, dataset_root: Path, report_root: Path, *, audits=None):
        audits = AUDITS if audits is None else audits
        if not audits or not set(audits).issubset(FAMILIES):
            raise ValueError("invalid annual target families")
        registry_path = dataset_root / "registry/national_62000.parquet"
        manifest_path = dataset_root / "targets/manifest.parquet"
        self.registry = pd.read_parquet(registry_path).set_index("patch_id")
        if self.registry.index.has_duplicates or self.registry.empty:
            raise ValueError("invalid annual target registry")
        self.indices = {p: i for i, p in enumerate(self.registry.index)}
        manifest = pd.read_parquet(manifest_path)
        self.files = {str(p): sha256(p) for p in [registry_path, manifest_path]}
        self.sources, self.audits, self.roots, self.source_indexes = {}, {}, {}, {}
        for family, name in audits.items():
            path = report_root / name
            audit = json.loads(path.read_text())
            fp = audit["fingerprint"]
            if (
                audit.get("status") != "target_geometry_audit_finished"
                or audit.get("family") != family
                or audit.get("scope") != "full"
                or audit.get("failed_targets") != 0
                or audit.get("processed_targets") != len(self.registry) * 2
                or audit.get("selected_targets") != len(self.registry) * 2
                or fp["registry_sha256"] != self.files[str(registry_path)]
                or fp["manifest_sha256"] != self.files[str(manifest_path)]
                or fp["policy"] != POLICY
            ):
                raise ValueError("complete matching annual target audit required")
            # Keep the original producer fingerprint, including legacy v1 code identity.
            # Reading an audited cache does not assert a rerun with today's producer code.
            table_path = Path(audit["output"])
            table = pd.read_parquet(table_path)
            expected = {(p, y) for p in self.registry.index for y in [2020, 2021]}
            if (
                table.duplicated(["patch_id", "year"]).any()
                or set(zip(table.patch_id, table.year, strict=True)) != expected
                or not table.status.eq("passed").all()
                or not np.array_equal(table.split, self.registry.loc[table.patch_id, "split"])
                or not table.target.eq(family + "_" + table.year.astype(str)).all()
            ):
                raise ValueError("annual target audit membership changed")
            self.files.update({str(path): sha256(path), str(table_path): sha256(table_path)})
            self.audits[family] = audit
            source_locks = {s["year"]: s for s in audit["source_locks"]}
            if set(source_locks) != {2020, 2021} or len(audit["source_locks"]) != 2:
                raise ValueError("annual target source years incomplete")
            for year in [2020, 2021]:
                source = source_locks[year]
                source_path = Path(source["path"])
                if (
                    str(source_path) in self.sources
                    and self.sources[str(source_path)] != source["sha256"]
                ):
                    raise ValueError("conflicting annual source versions")
                self.sources[str(source_path)] = source["sha256"]
                index_path = table_path.parent / f"source_{year}.json"
                if json.loads(index_path.read_text()) != source:
                    raise ValueError("annual target source index changed")
                self.files[str(index_path)] = sha256(index_path)
                self.source_indexes[(family, year)] = hashlib.sha256(
                    json.dumps(source, sort_keys=True).encode()
                ).hexdigest()
                rows = manifest.loc[
                    manifest.family.eq("static") & manifest.array.eq(f"targets/{family}_{year}")
                ]
                if (
                    len(rows) != 1
                    or rows.iloc[0].year != year
                    or not rows.iloc[0].registry_order_verified
                ):
                    raise ValueError("annual target manifest contract changed")
                baseline = Path(rows.iloc[0].path)
                root = zarr.open_group(str(baseline), mode="r")
                if list(root.attrs["patch_ids"]) != list(self.registry.index):
                    raise ValueError("annual target position order changed")
                self.files[str(baseline / ".zattrs")] = sha256(baseline / ".zattrs")
                self.roots[(family, year)] = root
        self.decoded_chunks = 0
        self.verify_unchanged()

    def verify_unchanged(self, *, include_sources=True):
        files = {**self.files, **self.sources} if include_sources else self.files
        if any(sha256(Path(p)) != h for p, h in files.items()):
            raise ValueError("annual target evidence changed")

    def iter_views(self, keys):
        keys = checked_keys(self.registry, keys)
        grouped = defaultdict(list)
        for p, year in keys:
            grouped[(year, self.indices[p] // 32 * 32)].append(p)
        for (year, start), patches in sorted(grouped.items()):
            self.verify_unchanged(include_sources=False)
            stop = min(start + 32, len(self.registry))
            block = {}
            for family, audit in self.audits.items():
                root = self.roots[(family, year)]
                values = np.asarray(root[f"targets/{family}_{year}"][start:stop])
                raw_mask = np.asarray(root[f"valid_masks/{family}_{year}"][start:stop])
                if (
                    values.shape != (stop - start, 128, 128)
                    or raw_mask.shape != values.shape
                    or not np.isin(raw_mask, [0, 1]).all()
                ):
                    raise ValueError("annual target chunk shape or mask changed")
                valid = raw_mask.astype(bool)
                digest = hashlib.sha256(values.tobytes() + valid.tobytes()).hexdigest()
                receipt = Path(audit["output"]).parent / "chunks" / f"{year}_{start:06d}.json"
                expected = {
                    **audit["fingerprint"],
                    "year": year,
                    "start": start,
                    "stop": stop,
                    "source_index_sha256": self.source_indexes[(family, year)],
                    "target_chunk_sha256": digest,
                }
                if json.loads(receipt.read_text())["fingerprint"] != expected:
                    raise ValueError("annual target chunk differs from source audit")
                if not np.isfinite(values[valid]).all():
                    raise ValueError("nonfinite annual target")
                allowed = (
                    list(range(1, 10))
                    if family == "clcd"
                    else [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
                )
                if family != "nightlights" and not np.isin(values[valid], allowed).all():
                    raise ValueError("annual categorical target outside class contract")
                block[family] = (np.where(valid, values, 0), valid)
                self.decoded_chunks += 1
            for patch in patches:
                offset = self.indices[patch] - start
                yield (patch, year), {
                    family: {"values": values[offset].copy(), "valid": valid[offset].copy()}
                    for family, (values, valid) in block.items()
                }


def merge_osm_fields(positive, negative):
    if set(positive) != {"targets", "states", "confidence", "source_bits"} or set(negative) != {
        "states",
        "confidence",
    }:
        raise ValueError("incomplete OSM fields")
    if any(
        a.shape != (30, 128, 128) or a.dtype != np.dtype("u1") for a in positive.values()
    ) or any(a.shape != (4, 128, 128) or a.dtype != np.dtype("u1") for a in negative.values()):
        raise ValueError("invalid OSM field shape or type")
    if (
        not np.isin(positive["states"], [0, 1]).all()
        or not np.isin(negative["states"], [0, 3]).all()
    ):
        raise ValueError("invalid positive/negative state contract")
    result = {k: a.copy() for k, a in positive.items()}
    for i, task in enumerate(TASKS):
        index = CHANNELS.index(task)
        valid = negative["states"][i] == 3
        if (valid & (positive["states"][index] != 0)).any():
            raise ValueError("OSM positive and negative evidence conflict")
        result["states"][index, valid] = 3
        result["confidence"][index, valid] = negative["confidence"][i, valid]
    return result


class ValidatedTargetReader:
    """Join audited annual maps with source-corrected OSM, DEM and reliable negatives."""

    def __init__(self, dataset_root: Path, report_root: Path, correction_root: Path):
        self.osm = CorrectedOSMReader(dataset_root, correction_root)
        self.negative = CorrectedNegativeReader(dataset_root, report_root)
        report = json.loads((report_root / "osm_reader_verification.json").read_text())
        proof_root = Path(report["output"])
        proof_path = proof_root / "verification.lock.json"
        proof = json.loads(proof_path.read_text())
        expected = {
            "correction_input_sha256": self.osm.input_sha,
            "correction_output_sha256": self.osm.output_sha,
            "negative_audit_sha256": self.negative.evidence.audit_sha,
            "negative_correction_lock_sha256": sha256(
                self.negative.directory / "corrections.lock.json"
            ),
            "dem_correction_lock_sha256": sha256(self.negative.demlock),
            "code_sha256": sha256(Path(__file__).with_name("v5_osm_reader.py")),
        }
        records_path = proof_root / "views.parquet"
        records = pd.read_parquet(records_path)
        keys = ["patch_id", "year", "resolution"]
        if (
            any(proof["fingerprint"].get(k) != v for k, v in expected.items())
            or sha256(proof_path) != report["verification_lock_sha256"]
            or proof["summary"]["status"]
            != "corrected_OSM_reader_and_negative_dependencies_verified"
            or _digest(records.to_dict("records")) != proof["records_sha256"]
            or records.duplicated(keys).any()
            or proof["summary"]["verified_views"] != len(records)
        ):
            raise ValueError("current complete OSM negative reconciliation required")
        lookup = records.set_index(keys)
        for key, row in self.osm.lookup.iterrows():
            changed = bool(row.correction_file)
            if (
                key not in lookup.index
                or bool(lookup.loc[key].corrected) != changed
                or bool(lookup.loc[key].negative_checked) != (changed and key[2] == "10m")
            ):
                raise ValueError("OSM reconciliation omits corrected target dependencies")
        self.files = {
            str(p): sha256(p)
            for p in [
                proof_path,
                records_path,
                self.negative.evidence.audit_path,
                self.negative.directory / "corrections.lock.json",
                self.negative.directory / "manifest.parquet",
                self.negative.demlock,
                self.negative.dem.directory / "manifest.parquet",
            ]
        }
        self.files.update(self.osm.metadata_files)
        self.files.update(
            {
                str(self.osm.input_path): self.osm.input_sha,
                str(self.osm.output_path): self.osm.output_sha,
            }
        )
        for name in [
            "v5_target_reader.py",
            "v5_osm_reader.py",
            "v5_negative_corrections.py",
            "v5_negative_rules.py",
            "v5_dem_corrections.py",
        ]:
            path = Path(__file__).with_name(name)
            self.files[str(path)] = sha256(path)
        self.annual = AnnualTargetReader(dataset_root, report_root)
        self.registry, self.indices = self.annual.registry, self.annual.indices
        if list(self.registry.index) != self.osm.registry.patch_id.tolist():
            raise ValueError("annual and OSM target registries differ")
        self.fingerprint = {
            "version": "validated_annual_targets_v1",
            "files_sha256": self.files,
            "annual_files_sha256": self.annual.files,
            "annual_source_sha256": self.annual.sources,
            "policy": {
                "years": [2020, 2021],
                "native_shapes": True,
                "unknown_is_negative": False,
                "fine_negative_upsampling": False,
                "training_authorized": False,
            },
        }
        self.verify_unchanged(include_sources=False)

    def verify_unchanged(self, *, include_sources=True):
        if any(sha256(Path(p)) != h for p, h in self.files.items()):
            raise ValueError("annual composite target evidence changed")
        self.osm._unchanged()
        self.annual.verify_unchanged(include_sources=include_sources)

    def iter_views(self, keys):
        keys = checked_keys(self.registry, keys)
        grouped = defaultdict(list)
        for key in keys:
            grouped[self.indices[key[0]] // 16].append(key)
        for _, batch in sorted(grouped.items()):
            self.verify_unchanged(include_sources=False)
            annual = dict(self.annual.iter_views(batch))
            osm_keys = [(p, y, r) for p, y in batch for r in ["10m", "2p5m"]]
            # At most one original 16-position OSM chunk and its selected native views.
            views = iter(self.osm.iter_views(osm_keys))
            for patch, year in batch:
                low_key, low = next(views)
                fine_key, fine = next(views)
                if low_key != (patch, year, "10m") or fine_key != (patch, year, "2p5m"):
                    raise ValueError("OSM composite view order changed")
                negative = self.negative.read(patch, year)
                yield (patch, year), {
                    **annual[(patch, year)],
                    "dem": self.negative.dem.read(patch),
                    "osm_10m": merge_osm_fields(low, negative),
                    "osm_2p5m": fine,
                }
            # Exhaust the iterator before releasing the block, including future final checks.
            if next(views, None) is not None:
                raise ValueError("unexpected extra OSM composite view")


def fields_digest(fields):
    digest = hashlib.sha256()
    for name, values in sorted(fields.items()):
        digest.update(name.encode())
        digest.update(values.dtype.str.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def verify_target_reader(dataset_root: Path, report_root: Path, correction_root: Path):
    reader = ValidatedTargetReader(dataset_root, report_root, correction_root)
    affected = set(reader.osm.rows.patch_id)
    dem_rows = reader.negative.dem.rows
    affected.update(dem_rows.index[dem_rows.correction_file.ne("")])
    controls = []
    for split in ["train", "val", "test"]:
        candidates = reader.registry.index[
            reader.registry.split.eq(split) & ~reader.registry.index.isin(affected)
        ]
        controls.extend(
            sorted(
                candidates,
                key=lambda p: hashlib.sha256(("annual-target-reader-v1:" + p).encode()).hexdigest(),
            )[:100]
        )
    positions = sorted(affected | set(controls), key=reader.indices.__getitem__)
    keys = [(p, y) for p in positions for y in [2020, 2021]]
    fingerprint = {**reader.fingerprint, "keys_sha256": _digest(keys)}
    directory = dataset_root / "quality/targets/reader" / _digest(fingerprint)[:20]
    directory.mkdir(parents=True, exist_ok=True)
    progress = report_root / "target_reader_progress.json"
    records = []
    for (patch, year), view in reader.iter_views(keys):
        record = {"patch_id": patch, "year": year, "split": reader.registry.loc[patch, "split"]}
        for family, fields in view.items():
            record[family + "_sha256"] = fields_digest(fields)
            if family.startswith("osm_"):
                for state, label in [(0, "unknown"), (1, "positive"), (3, "negative")]:
                    record[family + "_" + label] = int((fields["states"] == state).sum())
            else:
                for field, values in fields.items():
                    if values.dtype == bool:
                        record[family + "_" + field + "_pixels"] = int(values.sum())
        records.append(record)
        if len(records) % 20 == 0:
            write_json(
                progress,
                {
                    "status": "running",
                    "verified_views": len(records),
                    "selected_views": len(keys),
                    "updated_at": now(),
                    "training_authorized": False,
                },
            )
    reader.verify_unchanged()
    summary = {
        "status": "annual_target_reader_verified",
        "verified_views": len(records),
        "affected_positions": len(affected),
        "control_positions": len(controls),
        "selected_positions": len(positions),
        "output": str(directory),
        "annual_chunks": reader.annual.decoded_chunks,
        "osm_chunks": reader.osm.decoded_chunks,
        "scope": (
            "all OSM/DEM correction positions and up to 100 controls per split; "
            "annual targets only"
        ),
        "training_authorized": False,
        "user_accepted": False,
    }
    seal = {"fingerprint": fingerprint, "records_sha256": _digest(records), "summary": summary}
    lock_path = directory / "verification.lock.json"
    if lock_path.exists():
        if (
            json.loads(lock_path.read_text()) != seal
            or _digest(pd.read_parquet(directory / "views.parquet").to_dict("records"))
            != seal["records_sha256"]
        ):
            raise ValueError("annual target reader replay or output changed")
    else:
        atomic_parquet(pd.DataFrame(records), directory / "views.parquet")
        write_json(lock_path, seal)
    result = {**summary, "verification_lock_sha256": sha256(lock_path), "finished_at": now()}
    write_json(report_root / "target_reader_verification.json", result)
    write_json(progress, result)
    return result
