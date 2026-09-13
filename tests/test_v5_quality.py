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


def test_jilin_cloud_pipeline_uses_rgnir_and_keeps_low_clear_scene(tmp_path):
    import pandas as pd
    import rasterio

    from xuannv_embedding.data_process.v5_cloud import process_jilin_cloud

    source = tmp_path / "scene.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=256,
        height=256,
        count=6,
        dtype="int16",
        nodata=-28672,
        crs="EPSG:32643",
        transform=from_origin(0, 1280, 5, 5),
    ) as ds:
        ds.write(np.stack([np.full((256, 256), i * 10, dtype="int16") for i in range(1, 7)]))
        ds.descriptions = tuple(f"B{i}(0.5)" for i in range(1, 7))
        ds.scales = (0.0001,) * 6
    dataset = tmp_path / "dataset"
    catalog = dataset / "observations/highres/jilin1/files.parquet"
    catalog.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "path": str(source),
                "year": 2020,
                "product_id": "jilin1_ms_5m",
                "observation_id": "o1",
                "scene_group_id": "s1",
                "patch_id": "p1",
                "sensor": "JL1GP01",
                "split": "train",
                "acquired_at": "2020-12-31",
            }
        ]
    ).to_parquet(catalog)
    models = tmp_path / "models"
    models.mkdir()
    for i in (0, 1):
        (models / f"ocm_v4_model_{i}_96_910b4.om").write_bytes(b"test-double")

    class Predictor:
        def __init__(self, **kwargs):
            pass

        def predict_batch(self, arrays):
            np.testing.assert_allclose(arrays[0][:, 0, 0], [0.005, 0.004, 0.006])
            classes = np.zeros((1, 256, 256), dtype="uint8")
            classes[:, :170] = 1
            return classes, np.ones(classes.shape, dtype="float32")

        def close(self):
            pass

    result = process_jilin_cloud(dataset, tmp_path / "reports", models, predictor_factory=Predictor)
    qa = pd.read_parquet(dataset / "quality/cloud/jilin1/observation_quality.parquet")
    assert qa.iloc[0].available
    assert not qa.iloc[0].strict_scene_qualified
    assert result["processed_scenes"] == 1 and not result["acceptance_passed"]
