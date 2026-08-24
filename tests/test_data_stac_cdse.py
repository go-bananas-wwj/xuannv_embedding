"""Unit tests for CDSE STAC asset normalization."""

from xuannv_embedding.data_process import stac_cdse as MODULE


def test_s2_assets_are_normalized_to_materializer_names() -> None:
    assets = {
        name: {"href": f"s3://eodata/example/{name}.jp2"}
        for name in MODULE.SOURCE_CONFIG["s2"]["asset_map"].values()
    }
    result = MODULE._compact_feature({"id": "s2", "assets": assets, "properties": {}}, "s2")
    assert result is not None
    assert set(result["assets"]) == set(MODULE.SOURCE_CONFIG["s2"]["asset_map"])
    assert result["assets"]["SCL"]["href"].endswith("SCL_20m.jp2")


def test_feature_without_a_required_s3_asset_is_skipped() -> None:
    assert MODULE._compact_feature({"id": "s2", "assets": {}, "properties": {}}, "s2") is None
