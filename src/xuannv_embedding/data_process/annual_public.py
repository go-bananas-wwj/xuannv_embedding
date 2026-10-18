"""Quality-aware public monthly means on an explicitly registered trailing-year grid."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS

QUALITY = {"s2": "SCL", "landsat": "qa_pixel"}
S2_VALID_SCL = {4, 5, 6, 7, 11}


def verified(reference):
    if set(reference) != {"path", "sha256"} or sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Annual observation input identity changed")
    return Path(reference["path"])


def reference(path):
    return {"path": str(path), "sha256": sha(path)}


def validate_months(months):
    if not isinstance(months, list):
        raise ValueError("Annual observation months must be a list")
    try:
        dates = [dt.datetime.strptime(m, "%Y-%m") for m in months]
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid annual observation months") from error
    numbers = [d.year * 12 + d.month - 1 for d in dates]
    if (
        len(months) != 12
        or any(m != d.strftime("%Y-%m") for m, d in zip(months, dates))
        or numbers != list(range(numbers[0], numbers[0] + 12))
    ):
        raise ValueError("Exactly twelve consecutive months are required")


def geography_matches(crs, bounds, record):
    return CRS.from_user_input(crs) == CRS.from_user_input(record["crs"]) and np.allclose(
        bounds, record["bounds"], rtol=0, atol=1e-5
    )


def read_observation(observation, record, source, size, *, trust_companion=False):
    if trust_companion and observation["companion_mask"] is None:
        raise ValueError("Validated legacy input requires a companion mask")
    with rasterio.open(verified(observation["raster"])) as raster:
        if (raster.height, raster.width) != (size, size) or not geography_matches(
            raster.crs, raster.bounds, record
        ):
            raise ValueError("Annual raster geography differs from the registered tile")
        names = list(raster.descriptions)
        expected = [*PUBLIC_BANDS[source], *([QUALITY[source]] if source in QUALITY else [])]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Annual raster must declare its physical and quality band names")
        values = raster.read(out_dtype="float32")
        positions = [names.index(b) for b in PUBLIC_BANDS[source]]
        physical = values[positions]
        valid = (raster.dataset_mask() > 0) & np.isfinite(physical).all(0)
        if raster.nodata is not None:
            valid &= np.all(physical != raster.nodata, axis=0)
        if source in QUALITY and not trust_companion:
            qa = values[names.index(QUALITY[source])]
            integral = np.isfinite(qa) & (qa == np.rint(qa)) & (qa >= 0) & (qa <= 65535)
            clean = np.where(integral, qa, 0).astype(np.uint16)
            usable = np.isin(clean, list(S2_VALID_SCL)) if source == "s2" else (clean & 31) == 0
            valid &= integral & usable
    if observation["companion_mask"] is not None:
        with rasterio.open(verified(observation["companion_mask"])) as mask:
            if (
                mask.count != 1
                or (mask.height, mask.width) != (size, size)
                or not geography_matches(mask.crs, mask.bounds, record)
            ):
                raise ValueError("Annual companion mask geography differs")
            values = mask.read(1)
            valid &= (mask.dataset_mask() > 0) & np.isfinite(values) & (values > 0)
    return physical, valid


def monthly_composites(record, source, months, size, *, trust_companion=False):
    channels = len(PUBLIC_BANDS[source])
    sums = np.zeros((12, channels, size, size), np.float64)
    counts = np.zeros((12, size, size), np.uint32)
    seen = set()
    for observation in record["observations"]:
        if set(observation) != {"date", "raster", "companion_mask"}:
            raise ValueError("Invalid annual observation registration")
        try:
            date = dt.datetime.strptime(observation["date"], "%Y%m%d")
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid annual observation date") from error
        month = date.strftime("%Y-%m")
        key = (observation["date"], observation["raster"]["path"])
        if month not in months or key in seen:
            raise ValueError("Duplicate observation or date outside the registered months")
        seen.add(key)
        i = months.index(month)
        values, valid = read_observation(
            observation, record, source, size, trust_companion=trust_companion
        )
        sums[i] += np.where(valid[None], values, 0)
        counts[i] += valid
    frames = (sums / np.maximum(counts[:, None], 1)).astype(np.float32)
    if not np.isfinite(frames).all():
        raise ValueError("Nonfinite annual monthly composite")
    return frames, counts > 0


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    if (
        not {"schema", "index", "catalogues", "months", "bands", "size", "output"}.issubset(spec)
        or set(spec)
        - {
            "schema",
            "index",
            "catalogues",
            "months",
            "bands",
            "size",
            "output",
            "validated_companions",
        }
        or spec["schema"] != "annual-public-observations-v1"
        or type(spec["size"]) is not int
        or not 1 <= spec["size"] <= 512
        or spec["bands"] != PUBLIC_BANDS
        or set(spec["catalogues"]) != set(PUBLIC_BANDS)
    ):
        raise ValueError("Invalid annual public observation specification")
    companions = spec.get("validated_companions", {})
    if not isinstance(companions, dict) or set(companions) - set(QUALITY):
        raise ValueError("Invalid companion validation proof sources")
    for source, proof_ref in companions.items():
        proof = json.loads(verified(proof_ref).read_text())
        if (
            proof.get("state") != "complete"
            or proof.get("source") != source
            or proof.get("companion_validated") is not True
            or proof.get("mask_semantics") != "quality_validity_before_category_write"
            or type(proof.get("pixels_checked")) is not int
            or proof["pixels_checked"] < 1
        ):
            raise ValueError("Missing or failed companion validation proof")
    validate_months(spec["months"])
    index = json.loads(verified(spec["index"]).read_text())
    records = index["records"]
    if not records or len({r["patch_id"] for r in records}) != len(records):
        raise ValueError("Annual observation grid requires unique registered tiles")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", r["patch_id"]) for r in records):
        raise ValueError("Unsafe annual observation tile identifier")
    catalogues = {}
    for source, ref in spec["catalogues"].items():
        c = json.loads(verified(ref).read_text())
        if (
            set(c) != {"schema", "months", "records", "unmatched_registered_tiles"}
            or c["schema"] != "native-annual-observation-catalogue-v1"
            or c["months"] != spec["months"]
            or c["unmatched_registered_tiles"]
            or len(c["records"]) != len(records)
        ):
            raise ValueError("Annual catalogue does not cover the registered geographic grid")
        for original, row in zip(records, c["records"]):
            target = {**original, "crs": index.get("crs", original.get("crs"))}
            if (
                set(row)
                != {
                    "patch_id",
                    "native_patch_id",
                    "crs",
                    "bounds",
                    "shape",
                    "bands",
                    "observations",
                }
                or row["patch_id"] != original["patch_id"]
                or not geography_matches(row["crs"], row["bounds"], target)
                or row["shape"][1:] != [spec["size"], spec["size"]]
            ):
                raise ValueError("Annual catalogue geography differs from the registered tile")
        catalogues[source] = c["records"]
    return spec, index, catalogues


def export(path):
    spec, index, catalogues = load_spec(path)
    root = Path(spec["output"])
    for name in ["samples", "receipts"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    identity = {"spec_sha256": sha(path), "implementation_sha256": sha(__file__)}
    header = root / "identity.json"
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError("Annual observation recipe identity changed")
    dump(header, identity)
    outputs, totals = [], {s: np.zeros(12, np.int64) for s in PUBLIC_BANDS}
    for i, record in enumerate(index["records"]):
        target = root / "samples" / (record["patch_id"] + ".npz")
        receipt = root / "receipts" / (record["patch_id"] + ".json")
        if receipt.exists():
            item = json.loads(receipt.read_text())
            if item["identity"] != identity or sha(target) != item["output"]["sha256"]:
                raise ValueError("Annual observation cached sample identity changed")
        else:
            stored, counts = {}, {}
            for source in PUBLIC_BANDS:
                frames, valid = monthly_composites(
                    catalogues[source][i],
                    source,
                    spec["months"],
                    spec["size"],
                    trust_companion=source in spec.get("validated_companions", {}),
                )
                stored[source], stored[source + "_valid"] = frames, valid
                counts[source] = valid.sum(axis=(1, 2)).tolist()
            pending = target.with_suffix(".partial.npz")
            np.savez_compressed(pending, **stored)
            pending.replace(target)
            item = {
                "identity": identity,
                "output": reference(target),
                "monthly_valid_pixels": counts,
            }
            dump(receipt, item)
        for source in PUBLIC_BANDS:
            totals[source] += item["monthly_valid_pixels"][source]
        outputs.append(
            {
                "patch_id": record["patch_id"],
                "bounds": record["bounds"],
                "crs": index.get("crs", record.get("crs")),
                "input": item["output"],
            }
        )
        dump(
            root / "status.json",
            {"state": "preparing", "completed": i + 1, "total": len(index["records"])},
        )
    dump(root / "inputs.json", {"records": outputs})
    dump(
        root / "coverage.json",
        {
            "months": spec["months"],
            "monthly_valid_pixels": {s: v.tolist() for s, v in totals.items()},
            "quality": (
                "named SCL/QA bits or independently validated companion, "
                "intersect finite physical bands"
            ),
            "validated_companions": spec.get("validated_companions", {}),
            "missing_months": "zero values and false validity, never duplicated acquisitions",
            "aggregation": "per-pixel monthly mean of valid observations in archive units",
            "labels_used": False,
        },
    )
    dump(
        root / "status.json",
        {
            "state": "complete",
            "records": len(outputs),
            "months": spec["months"],
            "inputs_sha256": sha(root / "inputs.json"),
            "map_labels_used": False,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xuannv experiment annual-public-observations")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.spec)
    return 0
