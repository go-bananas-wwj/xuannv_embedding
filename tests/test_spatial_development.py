import numpy as np
import pytest

from xuannv_embedding.downstream.spatial_development import (
    fit_readout,
    metric_summary,
    predict_readout,
    sample_indices,
    spatial_folds,
)


def test_spatial_folds_union_macro_and_component_and_keep_parent_years_together():
    rows = [
        {"patch_id": f"p{i}", "macro_id": f"m{i}", "spatial_component_id": f"c{i}"}
        for i in range(12)
    ]
    rows[1]["macro_id"] = rows[0]["macro_id"]
    rows[2]["spatial_component_id"] = rows[1]["spatial_component_id"]
    assignments = spatial_folds(rows)
    assert assignments["p0"] == assignments["p1"] == assignments["p2"]
    assert set(row["fold"] for row in assignments.values()) == set(range(5))
    assert assignments == spatial_folds(list(reversed(rows)))
    with pytest.raises(ValueError):
        spatial_folds(rows[:3])


def test_fixed_readout_fits_only_given_training_statistics_and_known_classes():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(500, 4))
    target = 3 * x[:, 0] - 2 * x[:, 1] + 100
    fit = fit_readout(x, target)
    np.testing.assert_allclose(fit["mean"], x.mean(0))
    assert np.mean((predict_readout(fit, x)[:, 0] - target) ** 2) < 1e-4
    labels = (x[:, 0] > 0).astype(np.int64)
    classifier = fit_readout(x, labels, classes=3)
    assert not classifier["active_classes"][2]
    assert np.isneginf(predict_readout(classifier, x)[:, 2]).all()
    assert (predict_readout(classifier, x).argmax(1) == labels).mean() > 0.95
    with pytest.raises(ValueError):
        fit_readout(x, target, regularization=float("nan"))
    with pytest.raises(ValueError):
        fit_readout(x, np.ones(500, dtype=int) * -1, classes=3)


def test_pixel_sampling_is_deterministic_valid_only_and_caps_each_record():
    valid = np.ones((32, 32), bool)
    valid[::2] = False
    a = sample_indices(valid, "fixed-parent/2020", 128)
    np.testing.assert_array_equal(a, sample_indices(valid, "fixed-parent/2020", 128))
    assert len(a) == 128 and valid.ravel()[a].all()
    assert len(sample_indices(valid, "other", 1024)) == valid.sum()


def test_metrics_preserve_class_and_native_dsm_support_without_fake_zero_variance_r2():
    confusion = np.zeros((11, 11), np.int64)
    confusion[0, 0] = 3
    confusion[7, 7] = 2
    confusion[7, 0] = 1
    row = {
        "confusion": confusion.tolist(),
        "dem_count": 2,
        "dem_sse": 2.0,
        "dem_sae": 2.0,
        "dem_sum_y": 4.0,
        "dem_sum_y2": 10.0,
    }
    result = metric_summary([row])
    assert result["dem_RMSE_m"] == 1
    assert result["dem_R2"] == 0
    assert result["worldcover_water_F1"] == pytest.approx(0.8)
    assert result["worldcover_valid_pixels"] == 6
    assert result["worldcover_class_IoU"][1] is None
    row["dem_sum_y2"] = 8.0
    assert metric_summary([row])["dem_R2"] is None


def test_buffered_folds_exclude_nearby_whole_training_groups():
    pytest.importorskip("shapely")
    pytest.importorskip("pyproj")
    from xuannv_embedding.downstream.spatial_development import buffered_folds

    rows = []
    for i, x in enumerate([0, 1280, 20000, 40000, 60000, 80000]):
        rows.append(
            {
                "patch_id": f"p{i}",
                "macro_id": f"m{i}",
                "spatial_component_id": f"c{i}",
                "grid_epsg": 32650,
                "utm_bounds": [500000 + x, 4000000, 501280 + x, 4001280],
            }
        )
    assignments = spatial_folds(rows)
    folds = buffered_folds(rows, assignments)
    for fold in folds:
        assert fold["minimum_train_test_boundary_m"] >= 1290
        if "p0" in fold["test"] and "p1" not in fold["test"]:
            assert "p1" not in fold["train"]


def test_spatial_probe_end_to_end_fits_separate_folds_and_native_grid_tasks(tmp_path):
    import json
    from types import SimpleNamespace

    from xuannv_embedding.data_process.target_audit import digest, write_json
    from xuannv_embedding.downstream.spatial_development import probe

    rng = np.random.default_rng(5)
    geometry = [
        {"patch_id": f"p{i}", "macro_id": f"m{i}", "spatial_component_id": f"c{i}"}
        for i in range(5)
    ]
    assignments = spatial_folds(geometry)
    folds = [
        {
            "fold": i,
            "train": [r["patch_id"] for r in geometry if assignments[r["patch_id"]]["fold"] != i],
            "test": [r["patch_id"] for r in geometry if assignments[r["patch_id"]]["fold"] == i],
        }
        for i in range(5)
    ]
    plan = tmp_path / "plan.json"
    write_json(
        plan,
        {
            "role": "development",
            "assignments": assignments,
            "folds": folds,
            "protocol": {
                "folds": 5,
                "worldcover_train_pixels_per_record": 1024,
                "dem_train_cells_per_record": 256,
                "regularization": 0.001,
            },
        },
    )
    feature_root = tmp_path / "features" / "original_exact"
    feature_root.mkdir(parents=True)
    label_root = tmp_path / "labels"
    label_root.mkdir()
    feature_rows = []
    label_rows = []
    for row in geometry:
        f10 = rng.normal(size=(64, 128, 128)).astype(np.float16)
        f40 = f10.astype(np.float32).reshape(64, 32, 4, 32, 4).mean((2, 4)).astype(np.float16)
        wc = np.where(f10[0] > 0, 7, 0).astype(np.int16)
        dem = 100 + 100 * f40[0].astype(np.float32)
        for year in [2020, 2021]:
            name = f"{row['patch_id']}.{year}.npz"
            f = feature_root / name
            label_path = label_root / name
            np.savez_compressed(
                f, f10=f10, f40=f40, v10=np.ones((128, 128), bool), v40=np.ones((32, 32), bool)
            )
            np.savez_compressed(
                label_path,
                wc=wc,
                wc_valid=np.ones_like(wc, bool),
                dem=dem,
                dem_valid=np.ones_like(dem, bool),
            )
            feature_rows.append(
                {"patch_id": row["patch_id"], "year": year, "path": str(f), "sha256": digest(f)}
            )
            label_rows.append(
                {
                    "patch_id": row["patch_id"],
                    "year": year,
                    "path": str(label_path),
                    "sha256": digest(label_path),
                }
            )
    write_json(
        feature_root / "features.json",
        {"arm": "original_exact", "plan_sha256": digest(plan), "records": feature_rows},
    )
    write_json(
        tmp_path / "labels.json",
        {"role": "development", "plan_sha256": digest(plan), "records": label_rows},
    )
    probe(SimpleNamespace(plan=plan, arm="original_exact"))
    result = json.loads((tmp_path / "probes" / "original_exact" / "metrics.json").read_text())
    assert len(result["per_record"]) == 10 and len(result["fits"]) == 10
    assert result["metrics"]["worldcover_mIoU"] > 0.95 and result["metrics"]["dem_R2"] > 0.99
    assert not result["confirmation_opened"] and not result["validation_opened"]
    assert all(row["training_records"] == 8 and row["test_records"] == 2 for row in result["fits"])
