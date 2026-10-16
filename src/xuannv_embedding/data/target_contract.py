"""Versioned supervision grids and readout contracts; no optional data dependencies."""

LEGACY_TARGET_SCHEMA = "annual-dem40m-worldcover-v1"
MULTISOURCE_TARGET_SCHEMA = "annual-local-multisource-v2"
OSM_CHANNELS = ("building", "road_all", "water_area", "waterway")
STRUCTURE_CHANNELS = ("building", "road_all", "waterway")
LOCAL_TASKS = ("dem_slope", "clcd", "nightlights", "osm_semantic", "osm_structure")

# channels, area-pooling factor on the public5m embedding, objective
TARGET_TASKS = {
    "dem_elevation": (1, 8, "continuous"),
    "dem_slope": (1, 8, "continuous"),
    "worldcover": (11, 2, "categorical"),
    "clcd": (9, 8, "soft_categorical"),
    "nightlights": (1, 128, "continuous"),
    "osm_semantic": (len(OSM_CHANNELS), 2, "binary"),
    "osm_structure": (len(STRUCTURE_CHANNELS), 1, "binary"),
}
SCHEMA_TASKS = {
    LEGACY_TARGET_SCHEMA: frozenset(("dem_elevation", "worldcover")),
    MULTISOURCE_TARGET_SCHEMA: frozenset(TARGET_TASKS),
}


def target_shape(task: str) -> tuple[int, ...]:
    channels, pool, kind = TARGET_TASKS[task]
    size = 256 // pool
    return (channels, size, size) if kind in {"binary", "soft_categorical"} else (size, size)
