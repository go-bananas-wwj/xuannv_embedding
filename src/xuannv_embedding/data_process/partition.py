"""Create an auditable, exact-count tenfold partition of the China parent grid."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import CRS, Transformer

TOTAL_SHAPES = 5_785_781
BASE_SHARD_SIZE = TOTAL_SHAPES // 10
PARTITION_CRS = CRS.from_proj4(
    "+proj=lcc +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"
)


def region_bounds_feature(summary: dict[str, object]) -> dict[str, object]:
    """Build a lightweight WGS84 range index, deliberately separate from exact shapes."""
    shard_id = int(summary["shard_id"])
    longitude_range = summary["longitude_range"]
    latitude_range = summary["latitude_range"]
    if not isinstance(longitude_range, list) or not isinstance(latitude_range, list):
        raise ValueError("Region summary must contain longitude and latitude ranges")
    lon_min, lon_max = (float(value) for value in longitude_range)
    lat_min, lat_max = (float(value) for value in latitude_range)
    centroid = summary["centroid_wgs84"]
    if not isinstance(centroid, dict):
        raise ValueError("Region summary must contain a centroid")
    return {
        "type": "Feature",
        "properties": {
            "shard_id": f"shard_{shard_id:02d}",
            "shape_count": int(summary["shape_count"]),
            "center_longitude": float(centroid["longitude"]),
            "center_latitude": float(centroid["latitude"]),
            "min_longitude": lon_min,
            "max_longitude": lon_max,
            "min_latitude": lat_min,
            "max_latitude": lat_max,
            "geometry_role": "center_coordinate_bbox_index",
            "exact_membership": (
                "Use the GeoParquet geometry and parent_key files in the shard directory."
            ),
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min],
                ]
            ],
        },
    }


def write_region_indices(
    output_root: Path, summaries: list[dict[str, object]] | None = None
) -> None:
    """Write root and per-shard WGS84 range indexes without touching GeoParquet data."""
    manifest_path = output_root / "tenfold_partition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if summaries is None:
        raw_summaries = manifest.get("shards")
        if not isinstance(raw_summaries, list):
            raise ValueError(f"Invalid shard summaries in {manifest_path}")
        summaries = raw_summaries
    region_features = [region_bounds_feature(summary) for summary in summaries]
    region_collection = {
        "type": "FeatureCollection",
        "name": "china_tenfold_shard_regions",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "metadata": {
            "geometry_role": "center_coordinate_bbox_index",
            "exact_membership": (
                "Use the GeoParquet geometry and parent_key files in each shard directory."
            ),
        },
        "features": region_features,
    }
    (output_root / "china_tenfold_shard_regions.geojson").write_text(
        json.dumps(region_collection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for feature in region_features:
        shard_id = str(feature["properties"]["shard_id"])
        shard_path = output_root / "shards" / shard_id
        if not shard_path.is_dir():
            raise FileNotFoundError(f"Shard directory does not exist: {shard_path}")
        (shard_path / "region_bounds.geojson").write_text(
            json.dumps(feature, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    manifest["region_bounds_index"] = {
        "path": "china_tenfold_shard_regions.geojson",
        "geometry_role": "center_coordinate_bbox_index",
        "exact_membership": (
            "Use the GeoParquet geometry and parent_key files in each shard directory."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def add_region_index_readme_notes(output_root: Path, shard_ids: list[int]) -> None:
    """Document lightweight boundary indexes in a delivery already written to disk."""
    root_readme = output_root / "README.md"
    root_note = (
        "\n## 范围索引\n\n"
        "`china_tenfold_shard_regions.geojson` 提供十个分片的 WGS84 轻量范围索引。"
        "其中矩形由 Shape 中心经纬度范围生成，只用于检索和预览；"
        "精确成员必须以各分片 GeoParquet 的 `geometry` 和 `parent_key` 为准。\n"
    )
    root_text = root_readme.read_text(encoding="utf-8")
    if "china_tenfold_shard_regions.geojson" not in root_text:
        root_readme.write_text(root_text.rstrip() + root_note, encoding="utf-8")
    shard_note = (
        "\n## 范围索引\n\n"
        "`region_bounds.geojson` 是该片的中心经纬度外接矩形，仅用于检索和预览。"
        "精确成员关系以目录内 GeoParquet 的 `geometry` 和 `parent_key` 为准。\n"
    )
    for shard_id in shard_ids:
        shard_readme = output_root / "shards" / f"shard_{shard_id:02d}" / "README.md"
        shard_text = shard_readme.read_text(encoding="utf-8")
        if "region_bounds.geojson" not in shard_text:
            shard_readme.write_text(shard_text.rstrip() + shard_note, encoding="utf-8")


def partition_equal_capacity(
    x: np.ndarray,
    y: np.ndarray,
    shard_capacities: list[int],
) -> np.ndarray:
    """Assign each point exactly once using deterministic, compact recursive cuts."""
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays of the same shape")
    if sum(shard_capacities) != len(x) or any(size <= 0 for size in shard_capacities):
        raise ValueError("Shard capacities must be positive and sum to the number of points")

    assignment = np.zeros(len(x), dtype=np.int16)

    def split(
        indices: np.ndarray,
        capacities: list[int],
        bounds: tuple[float, float, float, float],
    ) -> None:
        if len(capacities) == 1:
            assignment[indices] = len(leaves) + 1
            leaves.append(indices)
            return
        left_leaf_count = len(capacities) // 2
        left_capacity = sum(capacities[:left_leaf_count])
        x0, x1, y0, y1 = bounds
        values = x[indices] if (x1 - x0) >= (y1 - y0) else y[indices]
        order = np.argsort(values, kind="stable")
        ordered = indices[order]
        cut = float((values[order[left_capacity - 1]] + values[order[left_capacity]]) / 2.0)
        if (x1 - x0) >= (y1 - y0):
            left_bounds = (x0, cut, y0, y1)
            right_bounds = (cut, x1, y0, y1)
        else:
            left_bounds = (x0, x1, y0, cut)
            right_bounds = (x0, x1, cut, y1)
        split(ordered[:left_capacity], capacities[:left_leaf_count], left_bounds)
        split(ordered[left_capacity:], capacities[left_leaf_count:], right_bounds)

    leaves: list[np.ndarray] = []
    split(
        np.arange(len(x), dtype=np.int64),
        shard_capacities,
        (float(x.min()), float(x.max()), float(y.min()), float(y.max())),
    )
    return assignment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--write-region-indices-only",
        action="store_true",
        help=(
            "Add GeoJSON range indexes to an existing tenfold delivery "
            "without rewriting GeoParquet."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    if args.write_region_indices_only:
        if not output_root.is_dir():
            raise FileNotFoundError(f"Existing delivery directory does not exist: {output_root}")
        write_region_indices(output_root)
        manifest = json.loads(
            (output_root / "tenfold_partition_manifest.json").read_text(encoding="utf-8")
        )
        add_region_index_readme_notes(
            output_root,
            [int(summary["shard_id"]) for summary in manifest["shards"]],
        )
        return
    if args.source_root is None:
        raise ValueError("--source-root is required unless --write-region-indices-only is set")
    source_root = args.source_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    paths = sorted((source_root / "all").glob("utm*/part-*.parquet"))
    counts = [pq.ParquetFile(path).metadata.num_rows for path in paths]
    if sum(counts) != TOTAL_SHAPES:
        raise ValueError(f"Expected {TOTAL_SHAPES} source Shapes, got {sum(counts)}")

    longitude = np.empty(TOTAL_SHAPES, dtype=np.float64)
    latitude = np.empty(TOTAL_SHAPES, dtype=np.float64)
    segments: list[tuple[Path, int, int]] = []
    start = 0
    for path, count in zip(paths, counts):
        table = pq.read_table(path, columns=["longitude", "latitude"])
        end = start + count
        longitude[start:end] = table["longitude"].to_numpy(zero_copy_only=False)
        latitude[start:end] = table["latitude"].to_numpy(zero_copy_only=False)
        segments.append((path, start, end))
        start = end
    transformer = Transformer.from_crs(4326, PARTITION_CRS, always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    capacities = [BASE_SHARD_SIZE] * 9 + [BASE_SHARD_SIZE + 1]
    assignment = partition_equal_capacity(np.asarray(x), np.asarray(y), capacities)
    actual_counts = {
        f"shard_{index:02d}": int(np.count_nonzero(assignment == index)) for index in range(1, 11)
    }
    if list(actual_counts.values()) != capacities:
        raise RuntimeError(f"Capacity validation failed: {actual_counts}")

    output_root.mkdir(parents=True)
    shard_root = output_root / "shards"
    by_shard_zone: dict[int, Counter[str]] = defaultdict(Counter)
    for path, start, end in segments:
        table = pq.read_table(path)
        local_assignment = assignment[start:end]
        zone = str(table["grid_id"][0].as_py())
        for shard_id in np.unique(local_assignment):
            rows = np.flatnonzero(local_assignment == shard_id)
            selected = table.take(pa.array(rows, type=pa.int64())).append_column(
                "shard_id", pa.array([int(shard_id)] * len(rows), type=pa.int8())
            )
            destination = shard_root / f"shard_{int(shard_id):02d}" / zone / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(selected, destination, compression="zstd")
            by_shard_zone[int(shard_id)][zone] += len(rows)

    summaries = []
    for shard_id in range(1, 11):
        mask = assignment == shard_id
        summaries.append(
            {
                "shard_id": shard_id,
                "shape_count": actual_counts[f"shard_{shard_id:02d}"],
                "centroid_wgs84": {
                    "longitude": float(longitude[mask].mean()),
                    "latitude": float(latitude[mask].mean()),
                },
                "longitude_range": [float(longitude[mask].min()), float(longitude[mask].max())],
                "latitude_range": [float(latitude[mask].min()), float(latitude[mask].max())],
                "counts_by_grid_id": dict(sorted(by_shard_zone[shard_id].items())),
            }
        )
    manifest = {
        "schema_version": "china_full_grid_tenfold_spatial_partition_v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_parent_count": TOTAL_SHAPES,
        "method": "joint_recursive_capacity_balanced_partition_v1",
        "partition_crs": PARTITION_CRS.to_string(),
        "validation": {
            "shard_counts": actual_counts,
            "unassigned_shape_count": int(np.count_nonzero(assignment == 0)),
        },
        "region_bounds_index": {
            "path": "china_tenfold_shard_regions.geojson",
            "geometry_role": "center_coordinate_bbox_index",
            "exact_membership": (
                "Use the GeoParquet geometry and parent_key files in each shard directory."
            ),
        },
        "shards": summaries,
    }
    (output_root / "tenfold_partition_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_region_indices(output_root, summaries)
    readme_lines = [
        "# 中国全国父网格十等分交付包",
        "",
        "本包将全国父网格的全部 5,785,781 个 Shape 按空间位置分为十个互不重叠的 GeoParquet 集合。",
        "每个原始 `parent_key` 恰好出现一次；分片 01--09 各含 578,578 个 Shape，",
        "分片 10 含 578,579 个。",
        "",
        "## 使用方式",
        "",
        "每个 `shards/shard_XX/` 目录可独立用于后续数据收集、预处理和 embedding 生产。",
        "`tenfold_partition_manifest.json` 是机器可读总清单；各分片目录中的",
        "`README.md` 提供范围和 UTM 分区统计。",
        "`china_tenfold_shard_regions.geojson` 是十个分片的轻量范围索引（WGS84）。",
        "其矩形仅由 Shape 中心经纬度范围生成，不能作为精确成员判定；精确范围以各分片",
        "GeoParquet 的 `geometry` 和 `parent_key` 为准。",
        "",
        "## 分片范围概览",
        "",
        "| 分片 | Shape 数 | 经度范围 | 纬度范围 |",
        "| --- | ---: | --- | --- |",
    ]
    for summary in summaries:
        shard_id = int(summary["shard_id"])
        lon_min, lon_max = summary["longitude_range"]
        lat_min, lat_max = summary["latitude_range"]
        readme_lines.append(
            f"| shard_{shard_id:02d} | {summary['shape_count']:,} | "
            f"{lon_min:.4f} – {lon_max:.4f} | {lat_min:.4f} – {lat_max:.4f} |"
        )
        shard_readme = [
            f"# shard_{shard_id:02d}",
            "",
            f"- Shape 数：{summary['shape_count']:,}",
            "- 中心点（WGS84）："
            f"{summary['centroid_wgs84']['longitude']:.4f}, "
            f"{summary['centroid_wgs84']['latitude']:.4f}",
            f"- 经度范围：{lon_min:.4f} – {lon_max:.4f}",
            f"- 纬度范围：{lat_min:.4f} – {lat_max:.4f}",
            "- 范围索引：`region_bounds.geojson`（中心经纬度外接矩形，仅用于检索/预览）。",
            "- 该目录内 GeoParquet 保留原始父网格字段与 geometry，并追加 `shard_id`。",
            "  精确成员关系必须以 GeoParquet 的 `geometry` 和 `parent_key` 为准。",
            "",
            "## UTM 分区计数",
            "",
        ]
        shard_readme.extend(
            f"- {zone}: {count:,}" for zone, count in summary["counts_by_grid_id"].items()
        )
        (shard_root / f"shard_{shard_id:02d}" / "README.md").write_text(
            "\n".join(shard_readme) + "\n", encoding="utf-8"
        )
    (output_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    qa_report = [
        "# 十等分质量检查",
        "",
        f"- 输入 Shape 数：{TOTAL_SHAPES:,}",
        f"- 已分配 Shape 数：{sum(actual_counts.values()):,}",
        "- 未分配 Shape 数：0",
        "- 分片重叠规则：每条输入记录只在一次写入循环中分配给一个 `shard_id`。",
        "- 详细计数和范围见 `tenfold_partition_manifest.json`。",
        "- 范围索引文件：`china_tenfold_shard_regions.geojson`，每片也有 `region_bounds.geojson`。",
        "  它们是 Shape 中心经纬度的外接矩形；精确成员关系以 GeoParquet 为准。",
    ]
    (output_root / "QA.md").write_text("\n".join(qa_report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
