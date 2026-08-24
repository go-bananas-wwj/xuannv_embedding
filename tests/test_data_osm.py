from __future__ import annotations

from xuannv_embedding.data_process import osm as MODULE


def test_categories_merge_related_osm_values_without_background_assumption() -> None:
    assert {"road"} <= MODULE.categories_for_tags({"highway": "residential"})
    assert {"road", "construction"} <= MODULE.categories_for_tags({"highway": "construction"})
    assert {"education"} <= MODULE.categories_for_tags({"amenity": "university"})
    assert {"sports"} <= MODULE.categories_for_tags({"leisure": "pitch"})
    assert MODULE.categories_for_tags({}) == set()
