import numpy as np
import pytest

from xuannv_embedding.data_process.v5_target_geometry import (
    categorical_validity,
    compare_target,
    member_matches_year,
)


def test_annual_member_contract_rejects_wrong_year_and_ambiguous_products():
    assert member_matches_year("worldcover", 2020, "ESA_WorldCover_10m_2020_v100_N15E072_Map.tif")
    assert not member_matches_year(
        "worldcover", 2020, "ESA_WorldCover_10m_2021_v200_N15E072_Map.tif"
    )
    assert member_matches_year("clcd", 2021, "CLCD_v01_2021_albert.tif")
    assert not member_matches_year("clcd", 2020, "CLCD_v01_2021_albert.tif")
    assert member_matches_year(
        "nightlights", 2020, "VNL_v2_npp_2020_global_vcmslcfg_c202101211500.average.tif"
    )
    assert not member_matches_year(
        "nightlights", 2020, "VNL_v2_npp_2020_global_vcmslcfg_c202101211500.average_masked.tif"
    )
    with pytest.raises(ValueError):
        member_matches_year("unknown", 2020, "file.tif")


def test_category_boundary_remains_unknown_and_nodata_never_becomes_a_class():
    values = np.array([[10, 10, 20, 20], [10, 10, 20, 0]])
    valid = values != 0
    result = categorical_validity(values, valid)
    np.testing.assert_equal(result, [[True, False, False, True], [True, False, False, False]])
    assert not result[1, 3]


def test_target_comparison_detects_spatial_or_mask_changes_even_when_histograms_match():
    expected = np.arange(16, dtype="f4").reshape(4, 4)
    valid = np.ones((4, 4), bool)
    assert (
        compare_target(expected, valid, expected.copy(), valid.copy(), categorical=True)["status"]
        == "passed"
    )
    shifted = np.roll(expected, 1, axis=1)
    assert (
        compare_target(expected, valid, shifted, valid, categorical=True)["value_mismatch_pixels"]
        == 16
    )
    altered = valid.copy()
    altered[0, 0] = False
    assert (
        compare_target(expected, valid, expected, altered, categorical=True)["mask_mismatch_pixels"]
        == 1
    )
    with pytest.raises(ValueError):
        compare_target(expected, valid, expected[:, :2], valid[:, :2], categorical=True)


def test_source_reconstruction_detects_wrong_grid_and_keeps_categorical_boundaries(tmp_path):
    import zipfile

    import rasterio
    from rasterio.transform import from_origin

    from xuannv_embedding.data_process.v5_target_geometry import AnnualRasterArchive

    image = tmp_path / "CLCD_v01_2020_albert.tif"
    original = np.ones((160, 160), "u1")
    original[:, 80:] = 2
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        count=1,
        width=160,
        height=160,
        dtype="uint8",
        crs="EPSG:32650",
        transform=from_origin(300000, 4000000, 10, 10),
        nodata=0,
    ) as dst:
        dst.write(original, 1)
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.write(image, image.name)
    reader = AnnualRasterArchive(archive, "clcd", 2020)
    try:
        fresh, valid, members = reader.reconstruct(32650, [300160, 3998560, 301440, 3999840])
        assert fresh.shape == (128, 128) and len(members) == 1
        assert not valid[:, 63:65].any()
        assert valid[:, :63].all() and valid[:, 65:].all()
        shifted, shifted_valid, _ = reader.reconstruct(32650, [300170, 3998560, 301450, 3999840])
        result = compare_target(fresh, valid, shifted, shifted_valid, categorical=True)
        assert result["status"] == "failed" and result["mask_mismatch_pixels"] > 0
    finally:
        reader.close()
    with pytest.raises(ValueError, match="year-specific"):
        AnnualRasterArchive(archive, "clcd", 2021)


def test_geometry_cli_requires_family_and_keeps_pilot_explicit(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_cli, v5_target_geometry

    calls = []
    monkeypatch.setattr(v5_cli, "lock_source", lambda _: {"manifest_sha256": {}})
    monkeypatch.setattr(v5_cli, "input_lock", lambda *args: None)
    monkeypatch.setattr(
        v5_target_geometry, "audit_target_geometry", lambda *a, **k: calls.append((a, k))
    )
    argv = ["--stage", "target-geometry", "--max-patches", "32"]
    for key in ["source-root", "dataset-root", "report-root", "base-root"]:
        argv += ["--" + key, str(tmp_path / key)]
    with pytest.raises(SystemExit):
        v5_cli.main(argv)
    assert v5_cli.main(argv + ["--target-family", "clcd"]) == 0
    assert calls == [
        ((tmp_path / "dataset-root", tmp_path / "report-root", "clcd"), {"max_patches": 32})
    ]


def test_full_annual_reconstruction_reuses_chunks_and_detects_changed_target_bytes(tmp_path):
    import hashlib
    import zipfile

    import pandas as pd
    import rasterio
    import zarr
    from rasterio.transform import from_origin

    from xuannv_embedding.data_process.v5_sources import sha256, write_json
    from xuannv_embedding.data_process.v5_target_geometry import audit_target_geometry

    data, report = tmp_path / "data", tmp_path / "report"
    (data / "registry").mkdir(parents=True)
    (data / "targets").mkdir()
    report.mkdir()
    pd.DataFrame(
        [
            {
                "patch_id": "p",
                "grid_epsg": 32650,
                "utm_bounds": [300000, 3998720, 301280, 4000000],
                "split": "train",
            }
        ]
    ).to_parquet(data / "registry/national_62000.parquet", index=False)
    label_path = tmp_path / "labels.zarr"
    root = zarr.open_group(str(label_path), mode="w")
    root.attrs["patch_ids"] = ["p"]
    targets, masks = root.create_group("targets"), root.create_group("valid_masks")
    references, metadata, audits = [], [], []
    for year, category in [(2020, 1), (2021, 2)]:
        values = np.full((1, 128, 128), category, "u1")
        valid = np.ones_like(values, bool)
        name = f"clcd_{year}"
        targets.array(name, values, chunks=(1, 128, 128))
        masks.array(name, valid, chunks=(1, 128, 128))
        for prefix, array in [("targets", values), ("valid_masks", valid)]:
            audits.append(
                {
                    "family": "static",
                    "array": f"{prefix}/{name}",
                    "decoded_values_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
            )
        metadata.append(
            {
                "family": "static",
                "array": f"targets/{name}",
                "year": year,
                "registry_order_verified": True,
                "path": str(label_path),
            }
        )
        image = tmp_path / f"CLCD_v01_{year}_albert.tif"
        with rasterio.open(
            image,
            "w",
            driver="GTiff",
            count=1,
            width=128,
            height=128,
            dtype="uint8",
            crs="EPSG:32650",
            transform=from_origin(300000, 4000000, 10, 10),
            nodata=0,
        ) as dst:
            dst.write(values)
        archive = tmp_path / f"static_clcd_china_{year}.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.write(image, image.name)
        references.append(
            {"family": "static", "path": str(archive), "actual_sha256": sha256(archive)}
        )
    pd.DataFrame(metadata).to_parquet(data / "targets/manifest.parquet", index=False)
    pd.DataFrame(audits).to_parquet(report / "target_value_audit.parquet", index=False)
    write_json(
        report / "target_source_audit.json",
        {"status": "source_audit_finished", "failed_sources": 0, "sources": references},
    )
    first = audit_target_geometry(data, report, "clcd")
    assert first["processed_targets"] == 2 and first["failed_targets"] == 0
    assert first["scope"] == "full" and first["reused_targets"] == 0
    cached = audit_target_geometry(data, report, "clcd")
    assert cached["reused_targets"] == 2
    # Same histogram / source files cannot hide a changed per-pixel label value.
    root["targets/clcd_2020"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="changed after completed value audit"):
        audit_target_geometry(data, report, "clcd")
    pilot = audit_target_geometry(data, report, "clcd", max_patches=1)
    assert pilot["scope"] == "pilot_1" and pilot["failed_targets"] == 1
    assert pilot["training_authorized"] is False
