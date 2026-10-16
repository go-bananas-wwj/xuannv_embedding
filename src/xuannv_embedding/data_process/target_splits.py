"""Freeze paired annual experiments using coordinates and availability, never label values."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path


def read_seen_patches(paths: list[Path]) -> set[str]:
    result = set()

    def visit(value):
        if isinstance(value, str) and re.fullmatch(r"parent_\d+:\d+:\d+", value):
            result.add(value)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(key)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in paths:
        visit(json.loads(path.read_text()))
    return result


def geometry_record(patch_id: str) -> dict:
    epsg, col, row = map(int, patch_id.removeprefix("parent_").split(":"))
    return {
        "patch_id": patch_id,
        "grid_epsg": epsg,
        "utm_bounds": [col * 1280, row * 1280, (col + 1) * 1280, (row + 1) * 1280],
        "macro_id": f"utm{epsg % 100}n_c{col // 10}_r{row // 10}",
    }


def projected_geometries(rows: list[dict], epsg: int):
    import numpy as np
    import shapely
    from pyproj import Transformer

    coordinates = np.empty((len(rows), 5, 2), dtype=np.float64)
    centers = np.empty((len(rows), 2), dtype=np.float64)
    by_crs = defaultdict(list)
    for i, row in enumerate(rows):
        x0, y0, x1, y1 = row["utm_bounds"]
        coordinates[i] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        centers[i] = [(x0 + x1) / 2, (y0 + y1) / 2]
        by_crs[row["grid_epsg"]].append(i)
    for source, indices in by_crs.items():
        if source == epsg:
            continue
        transformer = Transformer.from_crs(source, epsg, always_xy=True)
        x, y = transformer.transform(coordinates[indices, :, 0], coordinates[indices, :, 1])
        coordinates[indices] = np.stack([x, y], axis=-1)
        x, y = transformer.transform(centers[indices, 0], centers[indices, 1])
        centers[indices] = np.stack([x, y], axis=-1)
    return shapely.polygons(coordinates), shapely.points(centers)


def rank(seed: int, scope: str, patch: str) -> str:
    return hashlib.sha256(f"{seed}/{scope}/{patch}".encode()).hexdigest()


def proportional_sample(rows: list[dict], total: int, *, seed: int, scope: str) -> list[dict]:
    """Largest-remainder quotas preserve geographic/HR availability proportions."""
    if not 0 <= total <= len(rows):
        raise ValueError("Requested paired training sample exceeds eligible positions")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["grid_epsg"], row["highres_supported"])].append(row)
    quotas = {key: total * len(group) // len(rows) for key, group in groups.items()}
    remaining = total - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(total * len(groups[key]) % len(rows)), key))
    for key in order[:remaining]:
        quotas[key] += 1
    selected = []
    for key, group in sorted(groups.items()):
        selected.extend(
            sorted(group, key=lambda row: rank(seed, scope, row["patch_id"]))[: quotas[key]]
        )
    return sorted(selected, key=lambda row: row["target_row"])


def freeze_splits(
    index: list[dict],
    memberships: list[dict],
    seen: set[str],
    *,
    seed: int = 42,
    development: int = 256,
    validation: int = 128,
    confirmation: int = 256,
    buffer_m: float = 1280,
    screen_positions: int = 8192,
    expanded_positions: int = 16384,
) -> dict:
    import numpy as np
    import shapely
    from shapely.strtree import STRtree

    identity_fields = (
        "patch_id",
        "target_row",
        "grid_epsg",
        "utm_bounds",
        "macro_id",
        "spatial_component_id",
    )
    lookup = {row["patch_id"]: {key: row[key] for key in identity_fields} for row in index}
    original = defaultdict(set)
    years, support = defaultdict(set), defaultdict(bool)
    for record in memberships:
        patch = record["patch_id"]
        original[record["split"]].add(patch)
        years[patch].add(record["year"])
        support[patch] |= record["highres_supported"]
    if original["train"] & (original["validation"] | original["test"]):
        raise ValueError("Pretraining positions cross current evaluation splits")
    baseline_ids = sorted(original["train"] | seen)
    baseline = [lookup.get(patch, geometry_record(patch)) for patch in baseline_ids]
    excluded_macro = {row["macro_id"] for row in baseline}
    excluded_component = {
        row["spatial_component_id"] for row in baseline if row.get("spatial_component_id")
    }
    candidates = []
    for split in ("validation", "test"):
        for patch in sorted(original[split]):
            row = lookup[patch]
            if (
                years[patch] != {2020, 2021}
                or row["macro_id"] in excluded_macro
                or row["spatial_component_id"] in excluded_component
            ):
                continue
            candidates.append({**row, "original_split": split, "highres_supported": support[patch]})
    eligible = []
    for epsg in sorted({row["grid_epsg"] for row in candidates}):
        reference_polygons, reference_centers = projected_geometries(baseline, epsg)
        polygons, centers = STRtree(reference_polygons), STRtree(reference_centers)
        local = [row for row in candidates if row["grid_epsg"] == epsg]
        query_polygons, query_centers = projected_geometries(local, epsg)
        nearest_boundary, nearest_center = polygons.nearest(query_polygons), centers.nearest(
            query_centers
        )
        boundary_distances = shapely.distance(query_polygons, reference_polygons[nearest_boundary])
        center_distances = shapely.distance(query_centers, reference_centers[nearest_center])
        for i, row in enumerate(local):
            if boundary_distances[i] < buffer_m + 10:
                continue
            eligible.append(
                {
                    **row,
                    "nearest_baseline_boundary_m": float(boundary_distances[i]),
                    "nearest_baseline_center_m": float(center_distances[i]),
                    "nearest_boundary_patch": baseline_ids[int(nearest_boundary[i])],
                    "nearest_center_patch": baseline_ids[int(nearest_center[i])],
                }
            )
    selected, used_macro, used_component = [], set(), set()
    used_geometries = defaultdict(list)
    target_epsgs = sorted({row["grid_epsg"] for row in candidates})
    roles = {}
    for role, split, count in (
        ("development", "validation", development),
        ("validation", "validation", validation),
        ("confirmation", "test", confirmation),
    ):
        strata = defaultdict(list)
        for row in eligible:
            if row["original_split"] == split:
                strata[(row["grid_epsg"], row["highres_supported"])].append(row)
        queues = {
            key: deque(sorted(group, key=lambda row: rank(seed, role, row["patch_id"])))
            for key, group in sorted(strata.items())
        }
        chosen = []
        while len(chosen) < count and any(queues.values()):
            for queue in queues.values():
                if not queue or len(chosen) >= count:
                    continue
                row = queue.popleft()
                if row["macro_id"] in used_macro or row["spatial_component_id"] in used_component:
                    continue
                epsg = row["grid_epsg"]
                polygon = projected_geometries([row], epsg)[0][0]
                previous = used_geometries[epsg]
                if (
                    previous
                    and float(shapely.distance(polygon, np.asarray(previous)).min()) < buffer_m + 10
                ):
                    continue
                chosen.append(row)
                selected.append(row)
                used_macro.add(row["macro_id"])
                used_component.add(row["spatial_component_id"])
                for target in target_epsgs:
                    used_geometries[target].append(projected_geometries([row], target)[0][0])
        if len(chosen) != count:
            raise ValueError(
                f"Insufficient spatially independent {role} positions: {len(chosen)}/{count}"
            )
        roles[role] = chosen
    allowed = [
        {**lookup[patch], "highres_supported": support[patch]}
        for patch in sorted(original["train"] - seen)
        if years[patch] == {2020, 2021}
    ]
    screen = proportional_sample(allowed, screen_positions, seed=seed, scope="screen")
    screen_ids = {row["patch_id"] for row in screen}
    expansion = proportional_sample(
        [row for row in allowed if row["patch_id"] not in screen_ids],
        expanded_positions - screen_positions,
        seed=seed,
        scope="expansion",
    )
    expanded = sorted(screen + expansion, key=lambda row: row["target_row"])
    for role, rows in roles.items():
        other = [row for other_role, group in roles.items() if other_role != role for row in group]
        for epsg in target_epsgs:
            local = [row for row in rows if row["grid_epsg"] == epsg]
            if not local or not other:
                continue
            other_polygons, _ = projected_geometries(other, epsg)
            tree = STRtree(other_polygons)
            polygons, _ = projected_geometries(local, epsg)
            distance = shapely.distance(polygons, other_polygons[tree.nearest(polygons)])
            for row, value in zip(local, distance):
                row["nearest_other_role_boundary_m"] = float(value)
                if value < buffer_m:
                    raise ValueError("Cross-role geometry buffer failed")
    return {
        "schema": "xuannv.target-experiment-splits.v1",
        "seed": seed,
        "selection_uses_label_values": False,
        "selection_inputs": "v6split, paired-year availability, EPSG, HR availability, "
        "geometry, prior-seen IDs",
        "buffer_m": buffer_m,
        "projection_guard_m": 10,
        "distance_definition": "Nearest full 1280m footprint edge and parent center "
        "in each candidate's UTM CRS; "
        "all reference polygons, including adjacent UTM zones, are transformed before queries.",
        "original_pretraining_positions": len(original["train"]),
        "prior_seen_positions": sorted(seen),
        "eligible_paired_train_positions": len(allowed),
        "allowed_train_patch_ids": sorted(row["patch_id"] for row in allowed),
        "roles": roles,
        "screen_patch_ids": [row["patch_id"] for row in screen],
        "expanded_patch_ids": [row["patch_id"] for row in expanded],
        "annual_records": {"screen": 2 * len(screen), "expanded": 2 * len(expanded)},
        "confirmation_status": "Frozen identities only; no embeddings, task scores "
        "or new labels inspected",
        "limitations": [
            "Spatial independence is conditional on the documented original pretraining manifests",
            "Current product labels are not independent manual5mtruth",
            "Class support must be reported after identity freezing, "
            "without replacing difficult samples",
        ],
    }
