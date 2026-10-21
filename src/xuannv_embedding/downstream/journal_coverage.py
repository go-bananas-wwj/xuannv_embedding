"""Geographic coverage domains for frozen whole-source observation interventions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.training.frozen_fusion import verified
from xuannv_embedding.training.manuscript_followup import _outside_metrics


def coverage_domains(coverage):
    coverage = np.asarray(coverage)
    if (
        coverage.ndim != 3
        or not np.isfinite(coverage).all()
        or (coverage < 0).any()
        or (coverage > 1).any()
    ):
        raise ValueError("Invalid aligned coverage proportions")
    covered = coverage >= 0.5
    hi = maximum_filter(covered, size=(1, 3, 3), mode="nearest")
    lo = minimum_filter(covered, size=(1, 3, 3), mode="nearest")
    return {"covered": covered, "uncovered": ~covered, "boundary": hi != lo}


def run(evaluation, raster_reference, output):
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    evaluation, output = Path(evaluation), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datafile = next((evaluation / "projection").rglob("data.json"))
    data = json.loads(datafile.read_text())
    manifest = json.loads(verified(raster_reference).read_text())
    lookup = {r["patch_id"]: r for r in manifest["records"]}
    n, size = len(data["records"]), data["size"]
    coverage = np.zeros((n, size, size), np.float32)
    for i, record in enumerate(data["records"]):
        if record["patch_id"] not in lookup:
            raise ValueError("Coverage source tile missing")
        raw = lookup[record["patch_id"]]
        with rasterio.open(verified(raw["raster"])) as source:
            mask = (source.dataset_mask() > 0).astype("float32")
            reproject(
                mask,
                coverage[i],
                src_crs=source.crs,
                src_transform=source.transform,
                dst_crs=data["crs"],
                dst_transform=from_bounds(*record["bounds"], size, size),
                resampling=Resampling.average,
                dst_nodata=0,
            )
    domains = coverage_domains(coverage)
    np.save(output / "coverage.npy", coverage)
    original = json.loads((evaluation / "result.json").read_text())
    rows, predictions = [], []
    for row in original["rows"]:
        p = evaluation / row["head"] / "predictions/candidate" / (row["key"] + ".npz")
        predictions.append({"path": str(p), "sha256": sha(p)})
        with np.load(p) as z:
            positions = z["valid_indices"]
            if not np.array_equal(
                np.asarray(data["indices"])[positions // (size * size)], z["tiles"]
            ):
                raise ValueError("Prediction coordinates do not match geographic grid")
            values = {}
            for name, domain in domains.items():
                keep = domain.ravel()[positions]
                values[name] = _outside_metrics(
                    z["truth"][keep], z["scores"][keep], row["metrics"]["threshold"]
                )
        rows.append({"head": row["head"], "task": row["task"], "domains": values})
    result = dict(
        state="complete",
        rasters=raster_reference,
        data={"path": str(datafile), "sha256": sha(datafile)},
        evaluation={
            "path": str(evaluation / "result.json"),
            "sha256": sha(evaluation / "result.json"),
        },
        predictions=predictions,
        coverage={"path": str(output / "coverage.npy"), "sha256": sha(output / "coverage.npy")},
        definition=(
            "Observed fraction >=0.5 on reference cells; "
            "boundary is 3x3 binary range, nearest tile padding"
        ),
        scope=(
            "All-region reference cells; same saved thresholds; "
            "optical composite valid footprint"
        ),
        rows=rows,
        code_sha256=sha(__file__),
    )
    dump(output / "result.json", result)
    return result
