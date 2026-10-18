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
