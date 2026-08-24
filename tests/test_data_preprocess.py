from __future__ import annotations

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window

from xuannv_embedding.data_process import preprocess as MODULE


def test_landsat_scaling_preserves_qa_pixel_bits() -> None:
    arr = np.full((7, 2, 2), 10000, dtype=np.float32)
    arr[-1] = np.array([[0, 31], [8, 16]], dtype=np.float32)
    scaled = MODULE._scale_landsat_reflectance_preserving_qa(arr, nodata=0.0)

    assert np.isclose(scaled[0, 0, 0], 0.075)
    assert np.array_equal(scaled[-1], np.array([[0, 31], [8, 16]], dtype=np.float32))
    mask = MODULE._compute_valid_mask(scaled, "landsat", 0.0)
    assert mask.tolist() == [[1, 0], [0, 0]]


def test_landsat_qa_resampling_is_nearest_neighbor() -> None:
    source = np.full((7, 2, 2), 10000, dtype=np.float32)
    source[-1] = np.array([[0, 31], [8, 16]], dtype=np.float32)
    output = MODULE._extract_patch_bounded(
        arr=source,
        src_transform=from_origin(0, 40, 20, 20),
        src_crs=CRS.from_epsg(32650),
        master_transform=from_origin(0, 40, 10, 10),
        master_crs=CRS.from_epsg(32650),
        window=Window(0, 0, 4, 4),
        nodata=-9999.0,
        source="landsat",
    )
    assert set(np.unique(output[-1]).tolist()).issubset({0.0, 8.0, 16.0, 31.0})
