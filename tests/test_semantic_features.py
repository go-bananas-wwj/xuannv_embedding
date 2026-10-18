import numpy as np
import pytest
import torch

from xuannv_embedding.export.semantic_features import (
    align_patch_features,
    read_aerial_window,
    restore_rgb,
    stitch_patch_windows,
    window_starts,
)


def test_registered_raster_inputs_match_order_bounds_and_identity(tmp_path):
    import json

    from xuannv_embedding.export.context import sha
    from xuannv_embedding.export.semantic_features import registered_rasters

    raster = tmp_path / "tile.tif"
    raster.write_bytes(b"registered-source")
    record = {"patch_id": "tile", "bounds": [0, 0, 10, 10], "crs": "EPSG:32650"}
    manifest = tmp_path / "rasters.json"
    value = {
        "schema": "semantic-raster-inputs-v1",
        "records": [{**record, "raster": {"path": str(raster), "sha256": sha(raster)}}],
    }
    manifest.write_text(json.dumps(value))
    spec = {"rasters": {"path": str(manifest), "sha256": sha(manifest)}}
    assert registered_rasters(spec, {"records": [record]}) == [str(raster)]
    with pytest.raises(ValueError, match="geograph"):
        registered_rasters(spec, {"records": [{**record, "patch_id": "other"}]})
    raster.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"):
        registered_rasters(spec, {"records": [record]})


def test_external_raster_output_shape_requires_positive_integer_pair():
    from xuannv_embedding.export.semantic_features import external_output_shape

    assert external_output_shape({"output_shape": [128, 128], "rasters": {}}) == (128, 128)
    for shape in [[0, 128], [128.0, 128], [128], [128, 128, 3]]:
        with pytest.raises(ValueError, match="output shape"):
            external_output_shape({"output_shape": shape, "rasters": {}})
    with pytest.raises(ValueError, match="external"):
        external_output_shape({"output_shape": [128, 128]})


def test_rgb_normalization_uses_physical_values_and_satellite_statistics():
    image = torch.zeros(3, 16, 16)
    mean = [51.0, 102.0, 153.0]
    std = [10.0, 10.0, 10.0]
    output = restore_rgb(image, torch.ones(1, 16, 16), mean, std, [0.2, 0.4, 0.6], [0.1, 0.2, 0.3])
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
    image[:, 0, 0] = 100
    output = restore_rgb(image, torch.zeros(1, 16, 16), mean, std, [0.2, 0.4, 0.6], [0.1, 0.2, 0.3])
    assert torch.count_nonzero(output) == 0


def test_patch_alignment_accounts_for_padding_without_stretching_geography():
    # Three width-wise tokens centered at 8, 24 and 40 source pixels.
    values = torch.tensor([8.0, 24.0, 40.0]).expand(1, 1, 2, 3).clone()
    result = align_patch_features(values, (30, 46), (15, 23), patch=16)
    assert result.shape == (1, 1, 15, 23)
    # Output cell 8 is centered at source x=17, not padded-width x=17.739.
    assert result[0, 0, 7, 8].item() == pytest.approx(17.0, abs=1e-5)
    with pytest.raises(ValueError):
        align_patch_features(values, (32, 49), (15, 23), patch=16)


def test_patch_alignment_is_finite_for_tiny_output():
    values = torch.ones(1, 3, 2, 2)
    assert np.isfinite(align_patch_features(values, (31, 31), (1, 1), patch=16).numpy()).all()


def test_window_starts_cover_actual_image_on_patch_grid():
    assert window_starts(1280, 512) == [0, 384, 768]
    assert window_starts(432, 432) == [0]
    assert window_starts(640, 512) == [0, 128]


def test_overlapping_window_stitching_keeps_native_feature_centers():
    full = torch.arange(80).float()[None, None, :].expand(1, 80, 80).clone()
    pieces = [
        (y, x, full[:, y // 16 : y // 16 + 32, x // 16 : x // 16 + 32])
        for y in [0, 384, 768]
        for x in [0, 384, 768]
    ]
    result = stitch_patch_windows(pieces, (80, 80), patch=16)
    assert torch.allclose(result, full, atol=1e-5)
    with pytest.raises(ValueError, match="coverage"):
        stitch_patch_windows(pieces[:1], (80, 80), patch=16)


def test_native_aerial_crop_obeys_crs_pixel_bounds_and_explicit_mask(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "native.tif"
    values = np.broadcast_to(np.arange(20, dtype=np.uint8), (3, 20, 20)).copy()
    valid = np.full((20, 20), 255, np.uint8)
    valid[8, 4] = 0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=20,
        height=20,
        count=3,
        dtype="uint8",
        crs="EPSG:2193",
        transform=from_origin(100, 120, 1, 1),
    ) as dst:
        dst.write(values)
        dst.write_mask(valid)
    record = {"crs": "EPSG:2193", "bounds": [104, 104, 112, 112]}
    image, mask = read_aerial_window(path, record)
    assert image.shape == (3, 8, 8)
    assert image[0, 0, 0] == 4
    assert mask[0, 0, 0] == 0
    assert mask.sum() == 63
    with pytest.raises(ValueError, match="aligned"):
        read_aerial_window(path, {**record, "bounds": [104.5, 104, 112, 112]})
    with pytest.raises(ValueError, match="CRS"):
        read_aerial_window(path, {**record, "crs": "EPSG:4326"})


def test_extraction_phase_defers_pca_until_all_native_features_are_available(tmp_path, monkeypatch):
    import json

    from xuannv_embedding.export import semantic_features as semantic

    path = tmp_path / "spec.json"
    path.write_text("{}")
    root = tmp_path / "features"
    spec = {
        "output": str(root),
        "seed": 41,
        "fit_samples": 64,
        "components": 64,
        "weights": {"sha256": "0" * 64},
        "raster": {"path": "not-read"},
        "output_shape": [2, 2],
    }
    cache = {"records": [{"patch_id": str(i)} for i in range(4)], "months": ["2024-01"]}
    monkeypatch.setattr(semantic, "read_spec", lambda _: (spec, cache))

    def extracted(*args, **kwargs):
        np.save(
            root / "native.npy",
            np.random.default_rng(41).normal(size=(4, 1024, 4, 4)).astype(np.float16),
        )
        np.save(root / "native-mask.npy", np.ones((4, 1, 4, 4), np.float32))
        (root / "native-status.json").write_text(
            json.dumps({"stage": "complete", "completed": 4, "source_shape": [64, 64]})
        )

    monkeypatch.setattr(semantic, "extract_native", extracted)
    assert semantic.main(["--spec", str(path), "--phase", "extract", "--device", "cpu"]) == 0
    assert not (root / "annual.npy").exists()
    assert not (root / "projection.npz").exists()
    assert semantic.main(["--spec", str(path), "--phase", "project", "--device", "cpu"]) == 0
    assert np.load(root / "annual.npy").shape == (4, 64, 2, 2)
    assert json.loads((root / "status.json").read_text())["label_data_used"] is False


def test_projection_phase_never_starts_a_new_extraction(tmp_path, monkeypatch):
    from xuannv_embedding.export import semantic_features as semantic

    path = tmp_path / "spec.json"
    path.write_text("{}")
    monkeypatch.setattr(
        semantic, "read_spec", lambda _: ({"output": str(tmp_path / "empty")}, {"records": [{}]})
    )
    monkeypatch.setattr(
        semantic, "extract_native", lambda *a, **k: pytest.fail("project phase extracted data")
    )
    with pytest.raises(ValueError, match="extract"):
        semantic.main(["--spec", str(path), "--phase", "project", "--device", "cpu"])


def test_completed_projection_reuses_checked_output_after_native_cache_removal(
    tmp_path, monkeypatch
):
    import json

    from xuannv_embedding.export import semantic_features as semantic
    from xuannv_embedding.export.context import sha

    path = tmp_path / "spec.json"
    path.write_text("{}")
    root = tmp_path / "complete"
    root.mkdir()
    np.save(root / "annual.npy", np.ones((1, 64, 2, 2), np.float16))
    (root / "status.json").write_text(
        json.dumps({"stage": "complete", "annual_sha256": sha(root / "annual.npy")})
    )
    monkeypatch.setattr(semantic, "read_spec", lambda _: ({"output": str(root)}, {"records": [{}]}))
    monkeypatch.setattr(
        semantic, "extract_native", lambda *a, **k: pytest.fail("finished output re-extracted")
    )
    assert semantic.main(["--spec", str(path), "--phase", "project"]) == 0
