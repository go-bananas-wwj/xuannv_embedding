"""Audit broad OSM weak-semantic tags before rasterizing national labels.

The result is a tag-frequency audit over rasterizable ways and relations, not a
negative-label map. In particular, the absence of an OSM feature is unknown and
is never counted as background. Untagged point nodes are intentionally skipped:
they cannot form polygon/line weak labels and make a nationwide PBF scan far
slower without improving the training target.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

CATEGORY_RULES: dict[str, tuple[tuple[str, set[str] | None], ...]] = {
    "building": (("building", None),),
    "road": (
        (
            "highway",
            {
                "motorway",
                "trunk",
                "primary",
                "secondary",
                "tertiary",
                "unclassified",
                "residential",
                "service",
                "living_street",
                "pedestrian",
                "track",
                "path",
                "footway",
                "cycleway",
                "steps",
                "bridleway",
                "construction",
            },
        ),
    ),
    "rail": (("railway", {"rail", "light_rail", "subway", "tram", "narrow_gauge", "monorail"}),),
    "water": (
        ("natural", {"water", "wetland"}),
        ("water", None),
        ("waterway", {"riverbank", "canal", "river", "stream", "drain"}),
        ("landuse", {"reservoir", "basin"}),
    ),
    "green": (
        ("leisure", {"park", "garden", "nature_reserve", "common"}),
        ("landuse", {"forest", "grass", "meadow", "recreation_ground", "village_green"}),
        ("natural", {"wood", "grassland", "scrub", "heath"}),
    ),
    "agriculture": (
        (
            "landuse",
            {
                "farmland",
                "farmyard",
                "orchard",
                "vineyard",
                "plant_nursery",
                "greenhouse_horticulture",
            },
        ),
    ),
    "education": (("amenity", {"school", "university", "college", "kindergarten"}),),
    "sports": (
        ("leisure", {"pitch", "stadium", "sports_centre", "playground", "track"}),
        ("amenity", {"sports_centre"}),
    ),
    "industrial": (("landuse", {"industrial", "commercial", "retail"}),),
    "construction": (
        ("landuse", {"construction", "quarry", "landfill"}),
        ("highway", {"construction"}),
    ),
    "airport_port": (
        ("aeroway", {"aerodrome", "runway", "taxiway", "apron", "terminal"}),
        ("man_made", {"pier", "breakwater"}),
        ("harbour", None),
    ),
}
TRACKED_KEYS = (
    "building",
    "highway",
    "railway",
    "natural",
    "water",
    "waterway",
    "landuse",
    "leisure",
    "amenity",
    "aeroway",
    "man_made",
    "harbour",
)


def categories_for_tags(tags: dict[str, str]) -> set[str]:
    matched: set[str] = set()
    for category, alternatives in CATEGORY_RULES.items():
        for key, values in alternatives:
            value = tags.get(key)
            if value is not None and (values is None or value in values):
                matched.add(category)
                break
    return matched


def _scan(path: Path) -> dict[str, object]:
    try:
        import osmium
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional 'osmium' package before running this audit."
        ) from exc

    class Handler(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.object_counts: Counter[str] = Counter()
            self.category_counts: dict[str, Counter[str]] = defaultdict(Counter)
            self.tag_counts: dict[str, Counter[str]] = defaultdict(Counter)

        def _record(self, obj_type: str, tags: Iterable[object]) -> None:
            tag_map = {tag.k: tag.v for tag in tags}
            self.object_counts[obj_type] += 1
            for category in categories_for_tags(tag_map):
                self.category_counts[category][obj_type] += 1
            for key in TRACKED_KEYS:
                if key in tag_map:
                    self.tag_counts[key][tag_map[key]] += 1

        def way(self, way: object) -> None:
            self._record("way", way.tags)

        def relation(self, relation: object) -> None:
            self._record("relation", relation.tags)

    handler = Handler()
    handler.apply_file(str(path), locations=False)
    return {
        "objects_scanned": dict(sorted(handler.object_counts.items())),
        "category_counts_by_object_type": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(handler.category_counts.items())
        },
        "tracked_tag_values": {
            key: dict(counter.most_common()) for key, counter in sorted(handler.tag_counts.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-lock", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.snapshot_lock.read_text(encoding="utf-8"))
    if Path(lock["artifact_path"]).resolve() != args.pbf.resolve():
        raise ValueError("snapshot lock artifact_path does not match --pbf")
    audit = _scan(args.pbf)
    report = {
        "schema_version": "china_v1_osm_weak_semantics_audit_v1",
        "pbf": str(args.pbf.resolve()),
        "pbf_sha256": lock["sha256"],
        "category_rules": {
            category: [[key, sorted(values) if values else None] for key, values in rules]
            for category, rules in CATEGORY_RULES.items()
        },
        "absence_policy": "OSM absence is unknown and must be masked out of weak-semantic loss.",
        **audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
