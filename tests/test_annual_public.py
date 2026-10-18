import copy
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from xuannv_embedding.data_process import annual_public
from xuannv_embedding.export.context import dump, sha
from xuannv_embedding.export.public_statistics import PUBLIC_BANDS


def reference(path):
    return {"path": str(path), "sha256": sha(path)}


def write_raster(path, bands, values, bounds=(0, 0, 20, 20)):
    array = np.asarray(values, np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=len(bands),
        dtype="float32",
        crs="EPSG:32650",
        transform=from_bounds(*bounds, 2, 2),
        nodata=-9999,
    ) as dst:
        dst.write(array)
        dst.descriptions = tuple(bands)
    return reference(path)


def fixture_spec(tmp_path):
    bounds = [0, 0, 20, 20]
    index = tmp_path / "index.json"
    dump(index, {"crs": "EPSG:32650", "records": [{"patch_id": "tile_a", "bounds": bounds}]})
    months = [f"2024-{m:02d}" for m in range(1, 13)]
    catalogues = {}
    for source, bands in PUBLIC_BANDS.items():
        names = bands + ({"s2": ["SCL"], "landsat": ["qa_pixel"]}.get(source, []))
        values = np.full((len(names), 2, 2), 2, np.float32)
        if source == "s2":
            values[-1] = 4
        if source == "landsat":
            values[-1] = 0
        path = tmp_path / f"{source}.tif"
        raster = write_raster(path, names, values)
        item = {
            "patch_id": "tile_a",
            "native_patch_id": "different_archived_id",
            "crs": "EPSG:32650",
            "bounds": bounds,
            "shape": list(values.shape),
            "bands": names,
            "observations": [{"date": "20240101", "raster": raster, "companion_mask": None}],
        }
        path = tmp_path / f"{source}.json"
        dump(
            path,
            {
                "schema": "native-annual-observation-catalogue-v1",
                "months": months,
                "records": [item],
                "unmatched_registered_tiles": [],
            },
        )
        catalogues[source] = reference(path)
    spec = {
        "schema": "annual-public-observations-v1",
        "index": reference(index),
        "catalogues": catalogues,
        "months": months,
        "bands": copy.deepcopy(PUBLIC_BANDS),
        "size": 2,
        "output": str(tmp_path / "output"),
    }
    path = tmp_path / "spec.json"
    dump(path, spec)
    return path, spec


def test_annual_public_means_exclude_clouds_masks_and_keep_missing_months_explicit(tmp_path):
    path, spec = fixture_spec(tmp_path)
    cpath = spec["catalogues"]["s2"]["path"]
    catalogue = json.loads(Path(cpath).read_text())
    names = list(reversed(PUBLIC_BANDS["s2"])) + ["SCL"]
    values = np.full((12, 2, 2), 4, np.float32)
    values[names.index("B02")] = 6
    values[-1] = 4
    values[-1, 0, 0] = 9
    ref = write_raster(tmp_path / "second.tif", names, values)
    mask = np.ones((1, 2, 2), np.float32)
    mask[0, 0, 1] = 0
    mref = write_raster(tmp_path / "mask.tif", ["valid"], mask)
    catalogue["records"][0]["observations"].append(
        {"date": "20240102", "raster": ref, "companion_mask": mref}
    )
    dump(cpath, catalogue)
    spec["catalogues"]["s2"] = reference(cpath)
    dump(path, spec)
    annual_public.export(path)
    output = json.loads((tmp_path / "output/inputs.json").read_text())
    with np.load(output["records"][0]["input"]["path"]) as data:
        assert data["s2"].shape == (12, 11, 2, 2)
        np.testing.assert_array_equal(data["s2"][0, 0], [[2, 2], [4, 4]])
        assert not data["s2_valid"][1:].any()
        assert np.count_nonzero(data["s2"][1:]) == 0
    annual_public.export(path)
    assert json.loads((tmp_path / "output/status.json").read_text())["state"] == "complete"


def test_annual_public_never_joins_archived_ids_without_checking_geography(tmp_path):
    path, spec = fixture_spec(tmp_path)
    cpath = spec["catalogues"]["s1"]["path"]
    catalogue = json.loads(Path(cpath).read_text())
    catalogue["records"][0]["bounds"] = [20, 0, 40, 20]
    dump(cpath, catalogue)
    spec["catalogues"]["s1"] = reference(cpath)
    dump(path, spec)
    with pytest.raises(ValueError, match="geograph"):
        annual_public.export(path)


def test_annual_public_keeps_entirely_missing_source_and_quality_bits_out(tmp_path):
    path, spec = fixture_spec(tmp_path)
    for source in ["s1", "landsat"]:
        cpath = spec["catalogues"][source]["path"]
        catalogue = json.loads(Path(cpath).read_text())
        if source == "s1":
            catalogue["records"][0]["observations"] = []
        else:
            names = PUBLIC_BANDS[source] + ["qa_pixel"]
            values = np.ones((7, 2, 2), np.float32)
            values[-1] = 8
            catalogue["records"][0]["observations"][0]["raster"] = write_raster(
                tmp_path / "cloudy_landsat.tif", names, values
            )
        dump(cpath, catalogue)
        spec["catalogues"][source] = reference(cpath)
    dump(path, spec)
    annual_public.export(path)
    with np.load(tmp_path / "output/samples/tile_a.npz") as data:
        assert not data["s1_valid"].any()
        assert not data["landsat_valid"].any()
        assert data["landsat"].shape[1] == 6


def test_annual_public_rejects_unknown_fields_nonconsecutive_months_and_source_changes(tmp_path):
    path, spec = fixture_spec(tmp_path)
    bad = {**spec, "evaluation": "must-not-be-read"}
    dump(path, bad)
    with pytest.raises(ValueError, match="specification"):
        annual_public.export(path)
    bad = copy.deepcopy(spec)
    bad["months"][1] = bad["months"][0]
    dump(path, bad)
    with pytest.raises(ValueError, match="month"):
        annual_public.export(path)
    dump(path, spec)
    with (tmp_path / "s2.tif").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="identity"):
        annual_public.export(path)


def test_annual_public_rejects_dates_outside_window_and_corrupted_cached_samples(tmp_path):
    path, spec = fixture_spec(tmp_path)
    cpath = spec["catalogues"]["s2"]["path"]
    catalogue = json.loads(Path(cpath).read_text())
    catalogue["records"][0]["observations"][0]["date"] = "20250101"
    dump(cpath, catalogue)
    spec["catalogues"]["s2"] = reference(cpath)
    dump(path, spec)
    with pytest.raises(ValueError, match="outside"):
        annual_public.export(path)
    catalogue["records"][0]["observations"][0]["date"] = "20240101"
    dump(cpath, catalogue)
    spec["catalogues"]["s2"] = reference(cpath)
    spec["output"] = str(tmp_path / "valid_output")
    dump(path, spec)
    annual_public.export(path)
    with (tmp_path / "valid_output/samples/tile_a.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="cached sample identity"):
        annual_public.export(path)


def test_annual_public_accepts_only_proven_companion_masks_for_legacy_quality(tmp_path):
    path, spec = fixture_spec(tmp_path)
    proof = tmp_path / "proof.json"
    dump(
        proof,
        {
            "state": "complete",
            "source": "s2",
            "companion_validated": True,
            "mask_semantics": "quality_validity_before_category_write",
            "pixels_checked": 4,
        },
    )
    values = np.full((12, 2, 2), 2, np.float32)
    values[-1] = -9999
    raster = write_raster(tmp_path / "legacy_scl.tif", PUBLIC_BANDS["s2"] + ["SCL"], values)
    mask = np.ones((1, 2, 2), np.float32)
    mask[0, 0, 1] = 0
    valid = write_raster(tmp_path / "known_mask.tif", ["valid"], mask)
    cpath = spec["catalogues"]["s2"]["path"]
    catalogue = json.loads(Path(cpath).read_text())
    catalogue["records"][0]["observations"][0].update(raster=raster, companion_mask=valid)
    dump(cpath, catalogue)
    spec["catalogues"]["s2"] = reference(cpath)
    spec["validated_companions"] = {"s2": reference(proof)}
    dump(path, spec)
    annual_public.export(path)
    with np.load(tmp_path / "output/samples/tile_a.npz") as data:
        np.testing.assert_array_equal(data["s2_valid"][0], [[True, False], [True, True]])
        assert data["s2"][0, 0, 1, 1] == 2


def test_annual_public_rejects_unverified_or_absent_companion_override(tmp_path):
    path, spec = fixture_spec(tmp_path)
    proof = tmp_path / "bad_proof.json"
    dump(
        proof,
        {
            "state": "mismatch",
            "source": "s2",
            "companion_validated": False,
            "mask_semantics": "quality_validity_before_category_write",
            "pixels_checked": 4,
        },
    )
    spec["validated_companions"] = {"s2": reference(proof)}
    dump(path, spec)
    with pytest.raises(ValueError, match="proof"):
        annual_public.export(path)
    dump(
        proof,
        {
            "state": "complete",
            "source": "s2",
            "companion_validated": True,
            "mask_semantics": "quality_validity_before_category_write",
            "pixels_checked": 4,
        },
    )
    spec["validated_companions"] = {"s2": reference(proof)}
    dump(path, spec)
    with pytest.raises(ValueError, match="requires.*companion"):
        annual_public.export(path)
