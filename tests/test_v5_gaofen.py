import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process.v5_gaofen import process_gaofen


def test_gaofen_recomputes_qa_without_rejecting_partial_clear_scene(tmp_path):
    dataset = tmp_path / "dataset"
    registry = dataset / "registry/national_62000.parquet"
    registry.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"patch_id": "p1", "grid_epsg": 32643, "utm_bounds": [0, 0, 1280, 1280], "split": "train"}]
    ).to_parquet(registry)
    paths = {}
    for name, count, size, gsd in [("ms", 4, 160, 8), ("pan", 1, 640, 2)]:
        paths[name] = tmp_path / (name + ".tif")
        with rasterio.open(
            paths[name],
            "w",
            driver="GTiff",
            count=count,
            width=size,
            height=size,
            dtype="uint16",
            nodata=0,
            crs="EPSG:32643",
            transform=from_origin(0, 1280, gsd, gsd),
        ) as ds:
            ds.write(
                np.stack(
                    [np.full((size, size), 10 * i, dtype="uint16") for i in range(1, count + 1)]
                )
            )
    catalog = tmp_path / "source.parquet"
    pd.DataFrame(
        [
            {
                "pair_id": "pair1",
                "patch_id": "p1",
                "sensor": "GF6",
                "acquired_at": "2020-12-31T00:00:00Z",
                "ms_path": str(paths["ms"]),
                "pan_path": str(paths["pan"]),
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
            np.testing.assert_allclose(arrays[0][:, 0, 0], [30, 20, 40])
            labels = np.zeros((1, 160, 160), dtype="u1")
            labels[:, :110] = 1
            return labels, np.ones(labels.shape, dtype="f4")

        def close(self):
            pass

    result = process_gaofen(
        dataset, tmp_path / "reports", catalog, models, predictor_factory=Predictor
    )
    frame = pd.read_parquet(dataset / "quality/cloud/gaofen/observation_quality.parquet")
    assert result["processed_scenes"] == 1
    assert not frame.iloc[0].strict_scene_qualified
    assert frame.iloc[0].available
    assert frame.iloc[0].pan_valid_pixels > 0
