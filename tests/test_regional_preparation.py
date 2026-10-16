import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data_process.regional_preparation import materialize_region


def test_preparation_keeps_single_static_observation_and_quality_masks(tmp_path: Path):
    spec = {
        "root": str(tmp_path),
        "bounds": [0, 0, 20, 20],
        "crs": "EPSG:2193",
        "months": ["2024-01"],
        "patch_size_m": 20,
        "min_free_gib": 0,
        "max_gib": 1,
    }
    for source, count in [("s2", 11), ("s1", 2), ("landsat", 6), ("aerial", 3), ("worldcover", 1)]:
        folder = tmp_path / ("monthly/" + source if source in {"s2", "s1", "landsat"} else "static")
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / ("2024-01.tif" if source in {"s2", "s1", "landsat"} else source + ".tif")
        data = np.ones((count, 2, 2), dtype="float32") * (10 if source == "worldcover" else 1)
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=count,
            dtype="float32",
            crs="EPSG:2193",
            transform=from_origin(0, 20, 10, 10),
            nodata=-9999,
        ) as f:
            f.write(data)
            f.write_mask(np.array([[255, 0], [255, 255]], dtype="uint8"))
    (tmp_path / "weak_labels").mkdir()
    (tmp_path / "static/worldcover.tif").rename(tmp_path / "weak_labels/worldcover_2021.tif")
    mask = tmp_path / "static/worldcover.tif.msk"
    if mask.exists():
        mask.rename(tmp_path / "weak_labels/worldcover_2021.tif.msk")
    result = materialize_region(spec)
    records = json.loads((tmp_path / "processed/manifest.json").read_text())
    assert len(records) == 1
    assert len(records[0]["sources"]["highres_optical"]) == 1
    assert "static_2024" in records[0]["sources"]["highres_optical"][0]
    p = tmp_path / records[0]["sources"]["s2"][0]
    with rasterio.open(p) as f:
        assert f.count == 12
        assert f.read(12)[0, 0] == 0
    with rasterio.open(p.with_name(p.stem + "_mask.tif")) as f:
        assert f.read(1).tolist() == [[1, 0], [1, 1]]
    assert result["training_started"] is False
    from xuannv_embedding.data_process.regional_preparation import audit_geography_and_panels

    audit = audit_geography_and_panels(spec, result["coverage"])
    assert audit["max_bounds_difference_m"] < 1e-6
    assert audit["header_checks"] == 5


def test_weak_semantics_does_not_treat_unmapped_built_area_as_building_background(tmp_path):
    from pyproj import Transformer

    from xuannv_embedding.data_process.regional_preparation import prepare_weak_semantics

    (tmp_path / "weak_labels").mkdir()
    with rasterio.open(
        tmp_path / "weak_labels/worldcover_2021.tif",
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 40, 10, 10),
    ) as f:
        f.write(np.full((4, 4), 50, dtype="uint8"), 1)
    inverse = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    geom = [
        dict(zip(["lon", "lat"], inverse.transform(x, y)))
        for x, y in [(0, 30), (10, 30), (10, 40), (0, 40), (0, 30)]
    ]
    (tmp_path / "weak_labels/osm_snapshot.json").write_text(
        json.dumps({"elements": [{"type": "way", "tags": {"building": "yes"}, "geometry": geom}]})
    )
    report = prepare_weak_semantics(
        {"root": str(tmp_path), "crs": "EPSG:3857", "bounds": [0, 0, 40, 40], "patch_size_m": 40}
    )
    assert report["tasks"]["osm_building"]["positive_pixels"] > 0
    with rasterio.open(
        tmp_path / "processed/labels/osm_building/valid_masks/patch_000000.tif"
    ) as f:
        valid = f.read(1)
    assert valid.sum() == report["tasks"]["osm_building"]["positive_pixels"]
    assert valid[-1, -1] == 0


def test_evaluation_targets_keep_height_and_old_building_vintages_separate(tmp_path):
    from xuannv_embedding.data_process.regional_preparation import prepare_evaluation_targets

    (tmp_path / "static").mkdir()
    (tmp_path / "evaluation_sources").mkdir()
    for source, value in [("dem", 10), ("dsm", 15)]:
        with rasterio.open(
            tmp_path / "static" / f"{source}.tif",
            "w",
            driver="GTiff",
            width=20,
            height=20,
            count=1,
            dtype="float32",
            crs="EPSG:3857",
            transform=from_origin(0, 20, 1, 1),
            nodata=-9999,
        ) as f:
            f.write(np.full((20, 20), value, dtype="float32"), 1)
    polygon = {"type": "Polygon", "coordinates": [[[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]]}
    for filename, properties in [
        ("lcdb_v6_generalized15m.geojson", {"Class_2023": 1, "Name_2023": "built"}),
        ("buildings_mixed_2017_2022.geojson", {"building_id": 1}),
    ]:
        (tmp_path / "evaluation_sources" / filename).write_text(
            json.dumps({"features": [{"geometry": polygon, "properties": properties}]})
        )
    report = prepare_evaluation_targets(
        {"root": str(tmp_path), "bounds": [0, 0, 20, 20], "crs": "EPSG:3857", "patch_size_m": 20}
    )
    with rasterio.open(tmp_path / "evaluation/lidar_2024_targets.tif") as f:
        assert np.all(f.read(3) == 5)
        assert np.all(f.read(2) == 0)
    assert report["buildings"]["eligible_for_claims_about_2024_building_accuracy"] is False
    assert report["training_inputs"] is False
