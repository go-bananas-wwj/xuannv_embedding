import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.downstream.input_grid_audit import audit
from xuannv_embedding.export.context import sha


def fixture(tmp_path):
    records, official, references = [], [], {}
    for i in range(3):
        transform = from_origin(i * 40, 40, 10, 10)
        bounds = [i * 40, 0, (i + 1) * 40, 40]
        record = {"patch_id": f"p{i}", "bounds": bounds}
        records.append(record)
        path = tmp_path / f"p{i}.tif"
        if i < 2:
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=4,
                width=4,
                count=1,
                dtype="uint8",
                crs="EPSG:3857",
                transform=transform,
            ) as ds:
                ds.write(np.zeros((1, 4, 4), np.uint8))
            references[f"p{i}"] = {"path": str(path), "sha256": sha(path)}
        official.append(
            {
                "patch_id": f"p{i}",
                "reference_grid": {
                    "path": f"/unavailable/staging/{i}.tif",
                    "sha256": sha(path) if i < 2 else "0" * 64,
                    "crs": "EPSG:3857",
                    "shape": [4, 4],
                    "transform": list(transform)[:6],
                    "bounds": bounds,
                },
            }
        )
    cache = {
        "data": {"patch_size": 4, "months": ["2026-05"]},
        "records": records,
        "split": {"train": [0], "validation": [1], "test": [2], "buffer": []},
    }

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": sha(path)}

    spec = {
        "protocol": "input-grid-audit-v1",
        "caches": {"public": write("public.json", cache), "monthly": write("monthly.json", cache)},
        "official_manifest": write("official.json", {"year": 2025, "records": official}),
        "splits": ["train", "validation"],
        "references": references,
        "output": str(tmp_path / "audit"),
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path, spec, cache


def test_actual_raster_metadata_restores_archived_reference_identity_without_test_file(tmp_path):
    path, spec, _ = fixture(tmp_path)
    result = audit(path)
    assert result["state"] == "verified" and result["checked_tiles"] == 2
    assert result["checked_indices"] == [0, 1] and result["all_cache_tiles_covered"] is False
    assert result["pixel_arrays_read"] is False
    assert result["original_reference_paths_unavailable"] == 2
    assert result["crs"] == ["EPSG:3857"]
    assert result["reference_resolution"] == [[10.0, 10.0]]
    assert result["test_tiles_accessed"] is False
    assert not (tmp_path / "p2.tif").exists()
    with pytest.raises(FileExistsError):
        audit(path)


@pytest.mark.parametrize(
    "change",
    [
        "cache_bounds",
        "cache_size",
        "reference_sha",
        "official_crs",
        "official_transform",
        "official_shape",
        "official_hash",
        "duplicate_patch",
        "splits",
        "reference_set",
    ],
)
def test_inconsistent_geographic_contracts_are_rejected(tmp_path, change):
    path, spec, cache = fixture(tmp_path)
    if change.startswith("cache"):
        if change == "cache_bounds":
            cache["records"][0]["bounds"][0] += 1
        else:
            cache["data"]["patch_size"] = 8
        p = tmp_path / "monthly.json"
        p.write_text(json.dumps(cache))
        spec["caches"]["monthly"]["sha256"] = sha(p)
    elif change.startswith("official") or change == "duplicate_patch":
        p = tmp_path / "official.json"
        value = json.loads(p.read_text())
        g = value["records"][0]["reference_grid"]
        if change == "official_crs":
            g["crs"] = "EPSG:4326"
        if change == "official_transform":
            g["transform"][2] += 10
        if change == "official_shape":
            g["shape"] = [5, 5]
        if change == "official_hash":
            g["sha256"] = "f" * 64
        if change == "duplicate_patch":
            value["records"][1]["patch_id"] = "p0"
        p.write_text(json.dumps(value))
        spec["official_manifest"]["sha256"] = sha(p)
    elif change == "reference_sha":
        spec["references"]["p0"]["sha256"] = "0" * 64
    elif change == "splits":
        spec["splits"] = ["train", "train"]
    else:
        spec["references"]["p2"] = {"path": str(tmp_path / "p2.tif"), "sha256": "0" * 64}
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        audit(path)


def test_stored_metadata_alone_cannot_hide_wrong_physical_geography(tmp_path):
    path, spec, _ = fixture(tmp_path)
    p = tmp_path / "p0.tif"
    with rasterio.open(p, "r+") as ds:
        ds.transform = from_origin(20, 40, 10, 10)
    spec["references"]["p0"]["sha256"] = sha(p)
    official = tmp_path / "official.json"
    s = json.loads(official.read_text())
    s["records"][0]["reference_grid"]["sha256"] = sha(p)
    official.write_text(json.dumps(s))
    spec["official_manifest"]["sha256"] = sha(official)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="grid"):
        audit(path)
    state = json.loads((tmp_path / "audit/status.json").read_text())
    assert state["state"] == "failed"
