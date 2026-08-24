from __future__ import annotations

import json

import numpy as np

from xuannv_embedding.data_process import materialize as MODULE


def test_patch_bounds_reconstruct_the_grid_cell() -> None:
    patch = MODULE._patch_from_record(
        {"patch_id": "p", "grid_epsg": 32643, "grid_col": 312, "grid_row": 3346}
    )
    assert patch.bounds == (399360.0, 4282880.0, 400640.0, 4284160.0)


def test_s2_cloud_mask_rejects_cloud_and_keeps_land() -> None:
    stack = np.ones((12, 2, 2), dtype=np.float32)
    stack[-1] = np.array([[4, 8], [6, 3]], dtype=np.float32)
    assert MODULE._scene_valid_mask("s2", stack).tolist() == [[True, False], [True, False]]


def test_landsat_qa_mask_uses_bits_without_rescaling() -> None:
    stack = np.ones((7, 2, 2), dtype=np.float32)
    stack[-1] = np.array([[0, 1], [16, 32]], dtype=np.float32)
    assert MODULE._scene_valid_mask("landsat", stack).tolist() == [[True, False], [False, True]]


def test_cdse_s3_href_uses_gdal_s3_transport() -> None:
    href = "s3://eodata/Sentinel-2/example.jp2"
    assert MODULE._remote_href(href) == "/vsis3/eodata/Sentinel-2/example.jp2"


def test_s1_composite_excludes_invalid_scene_values() -> None:
    invalid = np.zeros((2, 2, 2), dtype=np.float32)
    valid = np.full((2, 2, 2), 2.0, dtype=np.float32)
    image, mask, fractions = MODULE._composite("s1", [invalid, valid])
    assert fractions == [0.0, 1.0]
    assert mask.tolist() == [[1, 1], [1, 1]]
    assert np.all(image == 2.0)


def test_load_available_scenes_skips_bad_candidate(monkeypatch) -> None:
    candidates = [
        {"id": "bad", "assets": {"vv": {"href": "bad"}, "vh": {"href": "bad-vh"}}},
        {
            "id": "first-good",
            "assets": {"vv": {"href": "first-good"}, "vh": {"href": "first-good-vh"}},
        },
        {
            "id": "second-good",
            "assets": {"vv": {"href": "second-good"}, "vh": {"href": "second-good-vh"}},
        },
    ]

    def fake_quality(href, _patch, _categorical, _cache=None, attempts=3):
        if href == "bad":
            raise ValueError("bbox edge")
        return np.ones((128, 128), dtype=np.float32)

    def fake_load(_source, item, _patch, _asset_workers=1, _reader_cache=None, _quality_array=None):
        if item["id"] == "bad":
            raise ValueError("bbox edge")
        return np.ones((2, 128, 128), dtype=np.float32)

    monkeypatch.setattr(MODULE, "_read_asset_to_patch", fake_quality)
    monkeypatch.setattr(MODULE, "_load_scene", fake_load)
    selected, scenes, rejected = MODULE._load_available_scenes(
        "s1", candidates, object(), 2, 1, None
    )
    assert [item["id"] for item in selected] == ["first-good", "second-good"]
    assert len(scenes) == 2
    assert rejected[0]["item_id"] == "bad"


def test_cloudy_scene_is_rejected_before_full_band_read(monkeypatch) -> None:
    item = {
        "id": "cloudy",
        "assets": {
            "B02": {"href": "B02"},
            "B03": {"href": "B03"},
            "B04": {"href": "B04"},
            "B05": {"href": "B05"},
            "B06": {"href": "B06"},
            "B07": {"href": "B07"},
            "B08": {"href": "B08"},
            "B8A": {"href": "B8A"},
            "B09": {"href": "B09"},
            "B11": {"href": "B11"},
            "B12": {"href": "B12"},
            "SCL": {"href": "SCL"},
        },
    }
    full_reads = []

    def fake_quality(href, _patch, _categorical, _cache=None, attempts=3):
        assert href == "SCL"
        return np.full((128, 128), 8, dtype=np.float32)

    def fake_load(*args, **kwargs):
        full_reads.append(args[1]["id"])
        raise AssertionError("cloudy image must not load full bands")

    monkeypatch.setattr(MODULE, "_read_asset_to_patch", fake_quality)
    monkeypatch.setattr(MODULE, "_load_scene", fake_load)
    selected, scenes, rejected = MODULE._load_available_scenes("s2", [item], object(), 0, 1, None)
    assert selected == [] and scenes == [] and full_reads == []
    assert rejected == [{"item_id": "cloudy", "reason": "quality_rejected", "clear_fraction": 0.0}]


def test_catalog_index_returns_only_intersecting_items(tmp_path) -> None:
    path = tmp_path / "items.jsonl"
    rows = [
        {"id": "near", "bbox": [100.0, 30.0, 101.0, 31.0], "assets": {}},
        {"id": "far", "bbox": [110.0, 30.0, 111.0, 31.0], "assets": {}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    index = MODULE.CatalogIndex(path)
    assert [item["id"] for item in index.query((100.4, 30.4, 100.6, 30.6))] == ["near"]


def test_select_items_filters_missing_assets(tmp_path) -> None:
    path = tmp_path / "items.jsonl"
    required = {
        name: {"href": "https://example.test/x.tif"} for name in MODULE.SOURCES["s1"]["assets"]
    }
    rows = [
        {"id": "usable", "bbox": [100.0, 30.0, 101.0, 31.0], "assets": required, "properties": {}},
        {
            "id": "incomplete",
            "bbox": [100.0, 30.0, 101.0, 31.0],
            "assets": {"vv": {}},
            "properties": {},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    patch = MODULE.Patch("p", 32647, (0, 0, 1, 1), (100.4, 30.4, 100.6, 30.6))
    assert [
        item["id"] for item in MODULE._select_items(MODULE.CatalogIndex(path), patch, "s1", 4)
    ] == ["usable"]


def test_load_catalogs_builds_requested_source_month_pairs(tmp_path) -> None:
    for source in MODULE.SOURCES:
        path = tmp_path / source / "2025-04"
        path.mkdir(parents=True)
        (path / "items.jsonl").write_text("", encoding="utf-8")
    catalogs = MODULE.load_catalogs(tmp_path, ["2025-04"])
    assert set(catalogs) == {(source, "2025-04") for source in MODULE.SOURCES}


def test_load_scene_can_read_assets_with_bounded_threads(monkeypatch) -> None:
    item = {"assets": {"vv": {"href": "vv"}, "vh": {"href": "vh"}}}
    monkeypatch.setattr(
        MODULE,
        "_read_asset_to_patch",
        lambda href, _patch, _categorical, _cache=None: np.full(
            (128, 128), 1 if href == "vv" else 2, dtype=np.float32
        ),
    )
    result = MODULE._load_scene("s1", item, object(), asset_workers=2)
    assert result.shape == (2, 128, 128)
    assert result[0, 0, 0] == 1 and result[1, 0, 0] == 2


def test_asset_reader_cache_opens_each_href_once(monkeypatch) -> None:
    opened = []

    class FakeReader:
        def close(self):
            return None

    monkeypatch.setattr(MODULE.planetary_computer, "sign", lambda href: href)
    monkeypatch.setattr(MODULE.rasterio, "open", lambda href: opened.append(href) or FakeReader())
    cache = MODULE.AssetReaderCache(2)
    assert cache.get("a") is cache.get("a")
    assert opened == ["a"]
    cache.close()


def test_remote_href_routes_signed_asset_through_optional_gateway(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.planetary_computer, "sign", lambda href: f"https://blob.test/a.tif?token={href}"
    )
    monkeypatch.setenv("CHINA_V1_COG_GATEWAY", "http://127.0.0.1:8787/cog")
    routed = MODULE._remote_href("raw")
    assert routed.startswith("http://127.0.0.1:8787/cog/aHR0cHM6Ly9ibG9iLnRlc3QvYS50aWY")


def test_scene_centric_reads_each_asset_once_for_multiple_patches(monkeypatch) -> None:
    item = {"id": "same-scene", "assets": {"vv": {"href": "vv"}, "vh": {"href": "vh"}}}
    calls = []

    def fake_select(_catalog, _patch, _source, candidate_limit):
        assert candidate_limit == 0
        return [item]

    def fake_read(href, patch_pairs, _categorical):
        calls.append((href, [index for index, _ in patch_pairs]))
        value = 1.0 if href == "vv" else 2.0
        return (
            {index: np.full((128, 128), value, dtype=np.float32) for index, _ in patch_pairs},
            {},
        )

    monkeypatch.setattr(MODULE, "_select_items", fake_select)
    monkeypatch.setattr(MODULE, "_read_asset_for_patches", fake_read)
    points = [
        MODULE.Patch("p0", 32643, (0, 0, 1, 1), (0, 0, 1, 1)),
        MODULE.Patch("p1", 32643, (1, 0, 2, 1), (1, 0, 2, 1)),
    ]
    images, masks, records = MODULE._scene_centric_source_month(
        source="s1",
        catalog=object(),
        points=points,
        max_clean_scenes=0,
    )
    assert calls == [("vv", [0, 1]), ("vh", [0, 1])]
    assert all(mask.all() for mask in masks)
    assert all(record["selected_items"] == ["same-scene"] for record in records)
    assert all(np.all(image[0] == 1.0) and np.all(image[1] == 2.0) for image in images)


def test_scene_centric_keeps_valid_composite_when_another_scene_fails(monkeypatch) -> None:
    items = [
        {
            "id": "broken-scene",
            "assets": {"vv": {"href": "broken-vv"}, "vh": {"href": "broken-vh"}},
        },
        {"id": "good-scene", "assets": {"vv": {"href": "good-vv"}, "vh": {"href": "good-vh"}}},
    ]

    monkeypatch.setattr(MODULE, "_select_items", lambda *_args, **_kwargs: items)

    def fake_read(href, patch_pairs, _categorical):
        indexes = [index for index, _ in patch_pairs]
        if href.startswith("broken"):
            return {}, {index: "HTTP 403" for index in indexes}
        value = 1.0 if href.endswith("vv") else 2.0
        return ({index: np.full((128, 128), value, dtype=np.float32) for index in indexes}, {})

    monkeypatch.setattr(MODULE, "_read_asset_for_patches", fake_read)
    point = MODULE.Patch("p0", 32643, (0, 0, 1, 1), (0, 0, 1, 1))
    images, masks, records = MODULE._scene_centric_source_month(
        source="s1",
        catalog=object(),
        points=[point],
        max_clean_scenes=0,
    )

    assert masks[0].all()
    assert records[0]["status"] == "ok"
    assert records[0]["selected_items"] == ["good-scene"]
    assert records[0]["rejected_items"][0]["item_id"] == "broken-scene"
    assert np.all(images[0][0] == 1.0) and np.all(images[0][1] == 2.0)


def test_scene_centric_scene_limit_is_applied_per_patch(monkeypatch) -> None:
    items = [
        {"id": "scene-1", "assets": {"vv": {"href": "scene-1-vv"}, "vh": {"href": "scene-1-vh"}}},
        {"id": "scene-2", "assets": {"vv": {"href": "scene-2-vv"}, "vh": {"href": "scene-2-vh"}}},
    ]
    calls = []
    monkeypatch.setattr(MODULE, "_select_items", lambda *_args, **_kwargs: items)

    def fake_read(href, patch_pairs, _categorical):
        calls.append((href, [index for index, _ in patch_pairs]))
        value = 1.0 if href.endswith("vv") else 2.0
        return (
            {index: np.full((128, 128), value, dtype=np.float32) for index, _ in patch_pairs},
            {},
        )

    monkeypatch.setattr(MODULE, "_read_asset_for_patches", fake_read)
    points = [
        MODULE.Patch("p0", 32643, (0, 0, 1, 1), (0, 0, 1, 1)),
        MODULE.Patch("p1", 32643, (1, 0, 2, 1), (1, 0, 2, 1)),
    ]
    _images, masks, records = MODULE._scene_centric_source_month(
        source="s1",
        catalog=object(),
        points=points,
        max_clean_scenes=1,
    )

    assert all(mask.all() for mask in masks)
    assert all(record["selected_items"] == ["scene-1"] for record in records)
    assert all("scene-2" not in href for href, _ in calls)


def test_scene_limit_skips_only_patches_that_reached_the_cap(monkeypatch) -> None:
    items = [
        {"id": "partial", "assets": {"vv": {"href": "partial-vv"}, "vh": {"href": "partial-vh"}}},
        {"id": "shared", "assets": {"vv": {"href": "shared-vv"}, "vh": {"href": "shared-vh"}}},
    ]
    monkeypatch.setattr(
        MODULE,
        "_select_items",
        lambda _catalog, patch, *_args, **_kwargs: items if patch.patch_id == "p0" else items[1:],
    )

    def fake_read(href, patch_pairs, _categorical):
        values = {
            index: np.full((128, 128), 1.0 if href.endswith("vv") else 2.0, dtype=np.float32)
            for index, _ in patch_pairs
        }
        return values, {}

    monkeypatch.setattr(MODULE, "_read_asset_for_patches", fake_read)
    points = [
        MODULE.Patch("p0", 32643, (0, 0, 1, 1), (0, 0, 1, 1)),
        MODULE.Patch("p1", 32643, (1, 0, 2, 1), (1, 0, 2, 1)),
    ]
    _images, masks, records = MODULE._scene_centric_source_month(
        source="s1",
        catalog=object(),
        points=points,
        max_clean_scenes=1,
    )

    assert all(mask.all() for mask in masks)
    assert records[0]["selected_items"] == ["partial"]
    assert records[1]["selected_items"] == ["shared"]


def test_scene_centric_treats_stac_raster_extent_mismatch_as_permanent(monkeypatch) -> None:
    item = {"id": "bad-extent", "assets": {"vv": {"href": "vv"}, "vh": {"href": "vh"}}}
    monkeypatch.setattr(MODULE, "_select_items", lambda *_args, **_kwargs: [item])
    monkeypatch.setattr(
        MODULE,
        "_read_asset_for_patches",
        lambda _href, patch_pairs, _categorical: (
            {},
            {index: "WindowError: Intersection is empty Window(...)" for index, _ in patch_pairs},
        ),
    )
    point = MODULE.Patch("p0", 32643, (0, 0, 1, 1), (0, 0, 1, 1))
    _images, masks, records = MODULE._scene_centric_source_month(
        source="s1",
        catalog=object(),
        points=[point],
        max_clean_scenes=1,
    )

    assert not masks[0].any()
    assert records[0]["status"] == "no_valid_pixels"
    assert records[0]["rejected_items"][0]["item_id"] == "bad-extent"
