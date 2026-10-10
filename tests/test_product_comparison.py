import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from xuannv_embedding.downstream.product_comparison import choose_candidate, svm_scores


def test_official_reprojection_decodes_before_interpolating(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from xuannv_embedding.downstream.product_comparison import official_raster

    path = tmp_path / "source.tif"
    raw = np.zeros((64, 2, 2), dtype="int8")
    raw[0] = [[40, 80], [40, 80]]
    raw[1] = 40
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=64,
        dtype="int8",
        crs="EPSG:32650",
        transform=from_origin(0, 20, 10, 10),
        nodata=-128,
    ) as ds:
        ds.write(raw)
    record = {
        "source_assets": [{"cache_path": str(path)}],
        "reference_grid": {
            "shape": [1, 1],
            "bounds": [0, 0, 20, 20],
            "crs": "EPSG:32650",
            "transform": [20, 0, 0, 0, -20, 20],
        },
    }
    z, valid = official_raster(record)
    expected = np.array([4000, 1600], dtype=float)
    expected /= np.linalg.norm(expected)
    assert valid.all()
    np.testing.assert_allclose(z[0, 0, :2], expected, atol=1e-6)


def test_validation_ties_choose_stronger_regularization():
    assert choose_candidate([0.6, 0.6, 0.5], [10, 1, 0.1], "ridge") == 0
    assert choose_candidate([0.6, 0.6, 0.5], [0.1, 1, 10], "svm") == 0


def test_vectorized_svm_matches_sklearn_without_refitting_scaler():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(35, 7))
    y = np.arange(35) % 2
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    fits = [SVC(C=c, class_weight="balanced").fit(z, y) for c in (0.1, 1, 10)]
    q = rng.normal(size=(51, 7))
    got = svm_scores(fits, scaler.transform(q), z)
    for i, fit in enumerate(fits):
        np.testing.assert_allclose(
            got[:, i], fit.decision_function(scaler.transform(q)), atol=1e-10
        )
