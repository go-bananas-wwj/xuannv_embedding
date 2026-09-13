import numpy as np
from rasterio.transform import from_origin

from xuannv_embedding.data_process.v5_quality import quality_masks, transfer_invalid


def test_low_clear_fraction_keeps_clear_pixels():
    classes = np.ones((10, 10), dtype="uint8")
    classes[:4] = 0
    result = quality_masks(classes, np.ones((10, 10), dtype=bool), gsd=5, buffer_m=0)
    assert not result["strict_scene_qualified"]
    assert result["valid"].sum() == 40
    assert result["clear_fraction"] == 0.4


def test_only_clouds_are_buffered_and_nodata_is_separate():
    classes = np.zeros((10, 10), dtype="uint8")
    valid = np.ones((10, 10), dtype=bool)
    valid[5, 5] = False
    result = quality_masks(classes, valid, gsd=5, buffer_m=30)
    assert result["valid"].sum() == 99
    classes[0, 0] = 3
    result = quality_masks(classes, valid, gsd=5, buffer_m=5)
    assert result["valid"].sum() == 95


def test_cloud_on_fine_pixel_invalidates_coarse_pixel():
    invalid = np.zeros((4, 4), dtype=bool)
    invalid[0, 0] = True
    result = transfer_invalid(
        invalid,
        src_transform=from_origin(0, 20, 5, 5),
        src_crs="EPSG:32643",
        dst_transform=from_origin(0, 20, 10, 10),
        dst_crs="EPSG:32643",
        shape=(2, 2),
    )
    assert result.tolist() == [[True, False], [False, False]]


def test_outside_cloud_coverage_is_invalid():
    result = transfer_invalid(
        np.zeros((2, 2), dtype=bool),
        src_transform=from_origin(0, 10, 5, 5),
        src_crs="EPSG:32643",
        dst_transform=from_origin(0, 20, 5, 5),
        dst_crs="EPSG:32643",
        shape=(4, 4),
    )
    assert result.sum() == 12
