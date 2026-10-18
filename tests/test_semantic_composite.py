import numpy as np
import pytest

from xuannv_embedding.data_process.semantic_composite import composite_rgb


def test_temporal_rgb_median_excludes_missing_observations_and_reorders_bands():
    images = np.array([[[[10]], [[20]], [[30]]], [[[250]], [[250]], [[250]]]], np.float32)
    masks = np.array([[[True]], [[False]]])
    rgb, valid = composite_rgb(images, masks, bands=[3, 2, 1], low=0, high=255)
    np.testing.assert_array_equal(rgb[:, 0, 0], [30, 20, 10])
    assert valid.all()


def test_temporal_rgb_scaling_is_fixed_and_does_not_invent_missing_pixels():
    images = np.full((2, 3, 2, 2), 1500, np.float32)
    masks = np.ones((2, 2, 2), bool)
    masks[:, 0, 0] = False
    rgb, valid = composite_rgb(images, masks, bands=[1, 2, 3], low=0, high=3000)
    assert not valid[0, 0]
    assert (rgb[:, 0, 0] == 0).all()
    assert (rgb[:, 1, 1] == 127.5).all()


@pytest.mark.parametrize("bands", [[1, 1, 2], [0, 1, 2], [1, 2, 4]])
def test_temporal_rgb_rejects_invalid_band_mapping(bands):
    with pytest.raises(ValueError):
        composite_rgb(np.ones((1, 3, 2, 2)), np.ones((1, 2, 2), bool), bands=bands, low=0, high=1)


def test_prepared_raster_preserves_geography_masks_and_native_dependency(tmp_path):
    import json

    import rasterio
    from rasterio.transform import from_origin

    from xuannv_embedding.data_process.semantic_composite import prepare
    from xuannv_embedding.export.context import sha
    from xuannv_embedding.export.semantic_features import read_aerial_window, registered_rasters

    def ref(path):
        return {"path": str(path), "sha256": sha(path)}

    record = {"patch_id": "tile", "bounds": [0, 0, 16, 16], "crs": "EPSG:32650"}
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [record]}))
    source, mask = tmp_path / "source.tif", tmp_path / "mask.tif"
    profile = dict(
        driver="GTiff",
        width=8,
        height=8,
        dtype="uint8",
        crs=record["crs"],
        transform=from_origin(0, 16, 2, 2),
    )
    with rasterio.open(source, "w", count=3, **profile) as dst:
        dst.write(np.full((3, 8, 8), 128, np.uint8))
    valid = np.ones((8, 8), np.uint8)
    valid[:4, :4] = 0
    with rasterio.open(mask, "w", count=1, **profile) as dst:
        dst.write(valid, 1)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "schema": "temporal-semantic-observations-v1",
                "records": [
                    {
                        "patch_id": "tile",
                        "observations": [{"raster": ref(source), "mask": ref(mask)}],
                    }
                ],
            }
        )
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema": "temporal-semantic-rgb-v1",
                "index": ref(index),
                "inputs": ref(inputs),
                "rgb": {"bands": [1, 2, 3], "low": 0, "high": 255},
                "encoder_resolution_m": 0.5,
                "output": str(tmp_path / "prepared"),
            }
        )
    )
    manifest = prepare(spec)
    paths = registered_rasters({"rasters": ref(manifest)}, {"records": [record]})
    image, coverage = read_aerial_window(paths[0], record)
    assert image.shape == (3, 32, 32)
    assert coverage[0, 0, 0] == 0
    assert coverage[0, -1, -1] == 1
    assert image[0, -1, -1] == pytest.approx(128)
    native = tmp_path / "prepared/000000.tif"
    native.write_bytes(b"changed")
    with pytest.raises(ValueError, match="dependency"):
        registered_rasters({"rasters": ref(manifest)}, {"records": [record]})


def test_fixed_fusion_export_does_not_require_benchmark_or_labels(tmp_path):
    import json

    from xuannv_embedding.export.context import sha
    from xuannv_embedding.training.representation_search import export_recipe

    rng = np.random.default_rng(41)
    sources = {}
    for name in ["base", "semantic"]:
        path = tmp_path / f"{name}.npy"
        np.save(path, rng.normal(size=(2, 64, 8, 8)).astype(np.float32))
        sources[name] = {"path": str(path), "sha256": sha(path)}
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [{"patch_id": "a"}, {"patch_id": "b"}]}))
    spec = {
        "sources": {"public": sources["base"], "semantic": sources["semantic"]},
        "index": {"path": str(index), "sha256": sha(index)},
        "seed": 41,
        "fit_samples": 128,
    }
    recipe = {"kind": "blocks", "sources": ["public", "semantic"], "dimensions": [40, 24]}
    output = tmp_path / "fusion"
    export_recipe(spec, recipe, output)
    result = np.load(output / "annual.npy")
    assert result.shape == (2, 64, 8, 8)
    np.testing.assert_allclose(np.linalg.norm(result.astype(np.float32), axis=1), 1, atol=0.001)
    assert json.loads((output / "status.json").read_text())["labels_used_for_projection"] is False
