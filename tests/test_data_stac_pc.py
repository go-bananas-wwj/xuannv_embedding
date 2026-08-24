from __future__ import annotations

from xuannv_embedding.data_process import stac_pc as MODULE


def test_months_cover_requested_thirteen_month_archive() -> None:
    assert MODULE._months() == [
        "2025-04",
        "2025-05",
        "2025-06",
        "2025-07",
        "2025-08",
        "2025-09",
        "2025-10",
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
    ]


def test_compact_feature_keeps_only_requested_assets() -> None:
    feature = {
        "id": "item",
        "collection": "demo",
        "bbox": [1, 2, 3, 4],
        "geometry": None,
        "properties": {"datetime": "2025-04-01T00:00:00Z", "eo:cloud_cover": 4},
        "assets": {
            "B02": {"href": "https://example/B02.tif", "type": "image/tiff"},
            "thumbnail": {"href": "https://example.png"},
        },
    }
    compact = MODULE._compact_feature(feature, ["B02"])
    assert compact["assets"] == {"B02": {"href": "https://example/B02.tif", "type": "image/tiff"}}
    assert compact["properties"]["eo:cloud_cover"] == 4
