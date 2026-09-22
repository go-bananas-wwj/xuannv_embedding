"""Audit cache grids against identity-bound physical raster reference metadata."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rasterio

from xuannv_embedding.downstream import multitask_features
from xuannv_embedding.downstream import paired_multitask as contracts
from xuannv_embedding.export.context import dump, sha

PROTOCOL = "input-grid-audit-v1"


def audit(spec_path):
    spec = contracts._load(spec_path)
    if (
        set(spec) != {"protocol", "caches", "official_manifest", "splits", "references", "output"}
        or spec["protocol"] != PROTOCOL
        or not isinstance(spec["caches"], dict)
        or not spec["caches"]
        or any(not isinstance(k, str) or not k for k in spec["caches"])
        or not isinstance(spec["splits"], list)
        or not spec["splits"]
        or len(set(spec["splits"])) != len(spec["splits"])
        or set(spec["splits"]) - {"train", "validation", "test", "buffer"}
    ):
        raise ValueError("invalid input-grid audit specification")
    caches = {name: contracts._registered(record) for name, record in spec["caches"].items()}
    first = next(iter(caches.values()))
    layout = None
    for cache in caches.values():
        _, current = multitask_features._grid(
            cache, {"records": cache["records"], "split": cache["split"]}
        )
        if layout is not None and current != layout:
            raise ValueError("registered cache grids or splits differ")
        layout = current
    indices = [i for split in spec["splits"] for i in first["split"][split]]
    if not indices:
        raise ValueError("input-grid audit selected no tiles")
    selected = {first["records"][i]["patch_id"] for i in indices}
    if not isinstance(spec["references"], dict) or set(spec["references"]) != selected:
        raise ValueError("reference set must exactly match the selected canonical tiles")
    official = contracts._registered(spec["official_manifest"])
    records = official["records"]
    official_by_id = {r["patch_id"]: r for r in records}
    if len(official_by_id) != len(records) or not selected <= set(official_by_id):
        raise ValueError("official reference tile identities are missing or duplicated")
    stage = Path(spec["output"])
    stage.mkdir(parents=True, exist_ok=False)
    tic, checked, unavailable = time.monotonic(), [], 0
    dump(stage / "status.json", {"state": "running", "checked_tiles": 0})
    try:
        for i in indices:
            tile = first["records"][i]
            patch = tile["patch_id"]
            declared = official_by_id[patch]["reference_grid"]
            reference = spec["references"][patch]
            if (
                set(reference) != {"path", "sha256"}
                or reference["sha256"] != declared["sha256"]
                or sha(Path(reference["path"])) != reference["sha256"]
            ):
                raise ValueError("physical reference identity differs from the official source")
            unavailable += not Path(declared["path"]).exists()
            size = first["data"]["patch_size"]
            with rasterio.open(reference["path"]) as ds:
                transform = list(ds.transform)[:6]
                if (
                    ds.crs is None
                    or ds.crs != rasterio.crs.CRS.from_user_input(declared["crs"])
                    or list(ds.shape) != [size, size]
                    or list(ds.shape) != declared["shape"]
                    or len(declared["transform"]) != 6
                    or not np.allclose(transform, declared["transform"], rtol=0, atol=1e-9)
                    or not np.allclose(ds.bounds, tile["bounds"], rtol=0, atol=1e-5)
                    or not np.allclose(ds.bounds, declared["bounds"], rtol=0, atol=1e-5)
                ):
                    raise ValueError("physical reference grid differs from cache or official grid")
                checked.append(
                    {
                        "index": i,
                        "patch_id": patch,
                        "reference_path": reference["path"],
                        "reference_sha256": reference["sha256"],
                        "crs": str(ds.crs),
                        "linear_units": ds.crs.linear_units if ds.crs.is_projected else None,
                        "transform": transform,
                        "bounds": list(ds.bounds),
                        "shape": list(ds.shape),
                        "resolution": list(ds.res),
                    }
                )
            if len(checked) % 25 == 0:
                dump(stage / "status.json", {"state": "running", "checked_tiles": len(checked)})
        dump(stage / "records.json", checked)
        result = {
            "state": "verified",
            "protocol": PROTOCOL,
            "spec_sha256": sha(Path(spec_path)),
            "cache_sha256": {k: v["sha256"] for k, v in spec["caches"].items()},
            "cache_months": {k: v["data"]["months"] for k, v in caches.items()},
            "official_manifest_sha256": spec["official_manifest"]["sha256"],
            "official_year": official.get("year"),
            "splits": spec["splits"],
            "checked_indices": indices,
            "checked_tiles": len(checked),
            "all_cache_tiles_covered": set(indices) == set(range(len(first["records"]))),
            "test_tiles_accessed": bool(set(indices) & set(first["split"]["test"])),
            "pixel_arrays_read": False,
            "original_reference_paths_unavailable": unavailable,
            "crs": sorted({r["crs"] for r in checked}),
            "reference_resolution": [
                list(v) for v in sorted({tuple(r["resolution"]) for r in checked})
            ],
            "records_sha256": sha(stage / "records.json"),
            "elapsed_seconds": time.monotonic() - tic,
            "implementation": {
                "audit": sha(Path(__file__)),
                "cache_grid": sha(Path(multitask_features.__file__)),
                "contracts": sha(Path(contracts.__file__)),
            },
            "runtime": {"rasterio": rasterio.__version__, "numpy": np.__version__},
            "scope": (
                "selected reference raster metadata and cache/official grid identity only; "
                "not corrected embedding value verification, label provenance "
                "or final model binding"
            ),
        }
        dump(stage / "verification.json", result)
        dump(stage / "status.json", {"state": "complete", "checked_tiles": len(checked)})
        return result
    except BaseException as exc:
        dump(stage / "partial_records.json", checked)
        dump(
            stage / "status.json",
            {"state": "failed", "checked_tiles": len(checked), "error": repr(exc)},
        )
        raise
