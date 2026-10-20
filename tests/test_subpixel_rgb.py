from pathlib import Path

import numpy as np
import pytest

from xuannv_embedding.data_process import subpixel_rgb as rgb


def test_subpixel_packing_preserves_every_channel_and_pixel_position():
    image = np.arange(3 * 10 * 10, dtype=np.float32).reshape(3, 10, 10) % 256
    packed, mask = rgb.pack_subpixels(image, np.ones((1, 10, 10)), factor=5)
    assert packed.shape == (75, 2, 2) and mask.shape == (25, 2, 2)
    for c in range(3):
        for y in range(5):
            for x in range(5):
                np.testing.assert_array_equal(packed[c * 25 + y * 5 + x], image[c, y::5, x::5])


def test_pool_uses_actual_observations_once_and_tracks_fractional_coverage():
    one = np.full((3, 5, 5), 10.0)
    two = np.full((3, 5, 5), 30.0)
    first_mask = np.ones((1, 5, 5))
    first_mask[:, 0, 0] = 0
    one[:, 0, 0] = np.nan
    values, coverage = rgb.pool_rgb([(one, first_mask), (two, np.ones((1, 5, 5)))], (5, 5))
    assert values[0, 0, 0] == 30 and values[0, 1, 1] == 20
    assert coverage[0, 0, 0] == 0.5 and coverage[0, 1, 1] == 1


def test_no_visual_observations_produce_explicit_zero_values_and_coverage():
    image, coverage = rgb.pool_rgb([], (10, 10))
    values, mask = rgb.pack_subpixels(image, coverage, factor=5)
    assert values.shape == (75, 2, 2) and not values.any() and not mask.any()


@pytest.mark.parametrize("mask_value", [np.nan, -1, 1.1])
def test_invalid_rgb_coverage_is_rejected(mask_value):
    with pytest.raises(ValueError, match="coverage"):
        rgb.pack_subpixels(np.ones((3, 5, 5)), np.full((1, 5, 5), mask_value), factor=5)


def test_packing_rejects_unknown_observed_values_or_unaligned_shapes():
    with pytest.raises(ValueError, match="observed"):
        rgb.pack_subpixels(np.full((3, 5, 5), np.nan), np.ones((1, 5, 5)), factor=5)
    with pytest.raises(ValueError, match="geometry"):
        rgb.pack_subpixels(np.ones((3, 6, 5)), np.ones((1, 6, 5)), factor=5)


def cache_fixture(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from test_public_visual_subspace import fixture_spec

    from xuannv_embedding.export.context import dump, sha

    _, old = fixture_spec(tmp_path)
    geometry = old["views"][0]["geometry"]
    records = rgb.read_json(geometry["path"])["records"]
    manifests = []
    for i, record in enumerate(records):
        path = tmp_path / f"image{i}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=40,
            height=40,
            count=3,
            dtype="uint8",
            crs=record["crs"],
            transform=from_origin(record["bounds"][0], 80, 2, 2),
        ) as ds:
            ds.write(np.full((3, 40, 40), 10 + i, np.uint8))
        manifests.append({**record, "raster": {"path": str(path), "sha256": sha(path)}})
    raster_manifest = tmp_path / "rasters.json"
    dump(raster_manifest, {"schema": "semantic-raster-inputs-v1", "records": manifests})
    raster_ref = {"path": str(raster_manifest), "sha256": sha(raster_manifest)}
    producer = tmp_path / "producer.json"
    dump(producer, {"rasters": raster_ref, "cache": geometry, "source": "highres_optical"})
    spec = {
        "schema": "subpixel-rgb-cache-v1",
        "index": old["index"],
        "public": old["public"],
        "views": [
            {
                "slot": 0,
                "producer_spec": {"path": str(producer), "sha256": sha(producer)},
                "rasters": raster_ref,
                "geometry": geometry,
            }
        ],
        "time_contract": old["time_contract"],
        "working_pixel_m": 2,
        "output_pixel_m": 10,
        "factor": 5,
        "source_shape": [40, 40],
        "output_shape": [8, 8],
        "output": str(tmp_path / "pixel_cache"),
        "reserve_gib": 0,
    }
    path = tmp_path / "pixel-spec.json"
    dump(path, spec)
    return path, spec


def test_cache_streaming_manifest_and_reload_are_complete_and_label_free(tmp_path):
    path, spec = cache_fixture(tmp_path)
    result = rgb.prepare(path)
    assert result["state"] == "complete" and result["registered_tiles"] == 2
    assert result["primary_or_weak_maps_loaded"] is False
    values = np.load(result["rgb"]["path"])
    masks = np.load(result["coverage"]["path"])
    assert values.shape == (2, 75, 8, 8) and masks.shape == (2, 25, 8, 8)
    np.testing.assert_array_equal(values[0], 10)
    np.testing.assert_array_equal(values[1], 11)
    assert masks.min() == masks.max() == 1
    assert rgb.prepare(path) == result
    assert not list(Path(spec["output"]).glob("*.partial.npy"))


def test_individual_frames_reproduce_existing_mean_without_copies(tmp_path):
    path, _ = cache_fixture(tmp_path)
    spec, index = rgb.specification(path)
    sources = rgb.Sources(spec, index)
    frames = list(sources.frames(0))
    assert len(frames) == 1
    image, coverage = rgb.pool_rgb(frames, spec["source_shape"])
    expected = rgb.pack_subpixels(image, coverage, factor=5)
    actual = sources.tile(0)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_cache_rejects_changed_raster_payload_and_geographic_transform(tmp_path):
    import rasterio

    from xuannv_embedding.export.context import dump, sha

    path, spec = cache_fixture(tmp_path)
    manifest_path = Path(spec["views"][0]["rasters"]["path"])
    manifest = rgb.read_json(manifest_path)
    first = Path(manifest["records"][0]["raster"]["path"])
    with rasterio.open(first, "r+") as ds:
        ds.transform = rasterio.Affine(3, 0, ds.bounds.left, 0, -3, ds.bounds.top)
    with pytest.raises(ValueError, match="identity"):
        rgb.prepare(path)
    manifest["records"][0]["raster"]["sha256"] = sha(first)
    dump(manifest_path, manifest)
    spec["views"][0]["rasters"]["sha256"] = sha(manifest_path)
    producer = Path(spec["views"][0]["producer_spec"]["path"])
    source = rgb.read_json(producer)
    source["rasters"] = spec["views"][0]["rasters"]
    dump(producer, source)
    spec["views"][0]["producer_spec"]["sha256"] = sha(producer)
    spec["output"] = str(tmp_path / "changed-pixel-cache")
    dump(path, spec)
    with pytest.raises(ValueError, match="resolution|geometry"):
        rgb.prepare(path)


def test_specification_rejects_unknown_fields_and_incompatible_pixel_grids(tmp_path):
    _, spec = cache_fixture(tmp_path)
    for extra in [{**spec, "_base_": "x"}, {**spec, "factor": 4}]:
        with pytest.raises(ValueError, match="specification"):
            rgb.validate_specification(extra)


def test_cache_resumes_only_completed_tiles_without_changing_outputs(tmp_path):
    path, spec = cache_fixture(tmp_path)
    partial = rgb.prepare(path, max_tiles=1)
    assert partial["state"] == "checkpointed" and partial["processed_tiles"] == 1
    assert not (Path(spec["output"]) / "manifest.json").exists()
    final = rgb.prepare(path)
    assert final["state"] == "complete" and final["registered_tiles"] == 2
    assert np.load(final["rgb"]["path"])[1].mean() == 11


def test_recovery_rejects_out_of_range_progress(tmp_path):
    from xuannv_embedding.export.context import dump

    path, spec = cache_fixture(tmp_path)
    rgb.prepare(path, max_tiles=1)
    progress = Path(spec["output"]) / "progress.json"
    record = rgb.read_json(progress)
    record["next_tile"] = 99
    dump(progress, record)
    with pytest.raises(ValueError, match="recovery|Recovery"):
        rgb.prepare(path)


def test_recovery_rejects_mutated_completed_pixel_prefix(tmp_path):
    path, spec = cache_fixture(tmp_path)
    rgb.prepare(path, max_tiles=1)
    values = np.lib.format.open_memmap(Path(spec["output"]) / "rgb.partial.npy", mode="r+")
    values[0, 0, 0, 0] += 1
    values.flush()
    with pytest.raises(ValueError, match="recovery|Recovery"):
        rgb.prepare(path)


@pytest.mark.parametrize(
    "field,value", [("path", None), ("path", "relative.npy"), ("sha256", "z" * 64)]
)
def test_specification_rejects_malformed_public_array_reference(tmp_path, field, value):
    _, spec = cache_fixture(tmp_path)
    spec["public"]["array"][field] = value
    with pytest.raises(ValueError, match="specification"):
        rgb.validate_specification(spec)


def test_tiny_float32_display_range_roundoff_is_clipped():
    values = np.full((3, 5, 5), 100, np.float32)
    values[0, 0, 0] = np.nextafter(np.float32(255), np.float32(np.inf))
    values[1, 0, 0] = -np.finfo(np.float32).eps
    packed, _ = rgb.pack_subpixels(values, np.ones((1, 5, 5)), factor=5)
    assert packed[0, 0, 0] == 255
    assert packed[25, 0, 0] == 0


@pytest.mark.parametrize("value", [-1.0, 256.0, 1000.0])
def test_actual_rgb_radiometric_range_errors_still_fail(value):
    with pytest.raises(ValueError, match="observed"):
        rgb.pack_subpixels(np.full((3, 5, 5), value), np.ones((1, 5, 5)), factor=5)


def test_values_inside_rgb_range_keep_exact_original_bits():
    rng = np.random.default_rng(41)
    values = rng.uniform(0, 255, size=(3, 10, 10)).astype(np.float32)
    packed, _ = rgb.pack_subpixels(values, np.ones((1, 10, 10)), factor=5)
    expected = values.reshape(3, 2, 5, 2, 5).transpose(0, 2, 4, 1, 3).reshape(75, 2, 2)
    np.testing.assert_array_equal(packed, expected)
