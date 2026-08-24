"""Create a locally reviewable renumbering of an existing China tenfold delivery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SHARD_COUNT = 10


def build_shard_remap(old_shard_order: list[int]) -> dict[int, int]:
    """Map each old shard id to its new one-based position."""
    expected = list(range(1, SHARD_COUNT + 1))
    if sorted(old_shard_order) != expected:
        raise ValueError(f"Shard order must be a permutation of {expected}: {old_shard_order}")
    return {
        old_shard_id: new_shard_id for new_shard_id, old_shard_id in enumerate(old_shard_order, 1)
    }


def reorder_summary(summary: dict[str, object], new_shard_id: int) -> dict[str, object]:
    """Preserve a shard summary while recording the source identity for auditability."""
    result = dict(summary)
    result["source_shard_id"] = int(summary["shard_id"])
    result["shard_id"] = new_shard_id
    return result


def region_bounds_feature(summary: dict[str, object]) -> dict[str, object]:
    """Create the deliberately coarse center-coordinate bbox search index."""
    lon_min, lon_max = (float(value) for value in summary["longitude_range"])
    lat_min, lat_max = (float(value) for value in summary["latitude_range"])
    centroid = summary["centroid_wgs84"]
    assert isinstance(centroid, dict)
    shard_id = int(summary["shard_id"])
    return {
        "type": "Feature",
        "properties": {
            "shard_id": f"shard_{shard_id:02d}",
            "source_shard_id": int(summary["source_shard_id"]),
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


def write_readmes(output_root: Path, summaries: list[dict[str, object]]) -> None:
    """Write root and shard documentation with the old-to-new mapping visible."""
    root_lines = [
        "# 中国全国父网格十等分交付包（本地重排预览 V4）",
        "",
        "本包不修改任何 Shape、parent_key 或 geometry；它仅重新编号 V3 的十个已分配分片，",
        "用于本地 QGIS 审阅。",
        "",
        "## 新旧编号映射",
        "",
        "| 新分片 | 原分片 | Shape 数 | 经度范围 | 纬度范围 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for summary in summaries:
        shard_id = int(summary["shard_id"])
        source_shard_id = int(summary["source_shard_id"])
        lon_min, lon_max = summary["longitude_range"]
        lat_min, lat_max = summary["latitude_range"]
        root_lines.append(
            f"| shard_{shard_id:02d} | shard_{source_shard_id:02d} | "
            f"{int(summary['shape_count']):,} | {float(lon_min):.4f} – {float(lon_max):.4f} | "
            f"{float(lat_min):.4f} – {float(lat_max):.4f} |"
        )
        shard_root = output_root / "shards" / f"shard_{shard_id:02d}"
        shard_lines = [
            f"# shard_{shard_id:02d}",
            "",
            f"- 原分片：shard_{source_shard_id:02d}",
            f"- Shape 数：{int(summary['shape_count']):,}",
            "- 中心点（WGS84）："
            f"{summary['centroid_wgs84']['longitude']:.4f}, "
            f"{summary['centroid_wgs84']['latitude']:.4f}",
            f"- 经度范围：{float(lon_min):.4f} – {float(lon_max):.4f}",
            f"- 纬度范围：{float(lat_min):.4f} – {float(lat_max):.4f}",
            "- `region_bounds.geojson` 是中心经纬度外接矩形，仅用于检索/预览。",
            "- 精确成员关系必须以该目录 GeoParquet 的 `geometry` 与 `parent_key` 为准。",
            "",
            "## UTM 分区计数",
            "",
        ]
        shard_lines.extend(
            f"- {zone}: {count:,}" for zone, count in summary["counts_by_grid_id"].items()
        )
        (shard_root / "README.md").write_text("\n".join(shard_lines) + "\n", encoding="utf-8")
    root_lines.extend(
        [
            "",
            "## 范围索引",
            "",
            "`china_tenfold_shard_regions.geojson` 是十区中心经纬度外接矩形，允许互相重叠。",
            "它不是精确的分片边界；在 QGIS 中请加载 `shards/shard_XX/` 下的 GeoParquet",
            "查看实际 Shape。",
        ]
    )
    (output_root / "README.md").write_text("\n".join(root_lines) + "\n", encoding="utf-8")


def write_overview(
    output_root: Path,
    samples: dict[int, tuple[np.ndarray, np.ndarray]],
    summaries: list[dict[str, object]],
) -> None:
    """Draw a lightweight, label-correct overview for visual inspection."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(14, 10), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for shard_id in range(1, SHARD_COUNT + 1):
        longitude, latitude = samples[shard_id]
        axes.scatter(
            longitude, latitude, s=0.15, color=palette(shard_id - 1), label=f"shard_{shard_id:02d}"
        )
    for summary in summaries:
        shard_id = int(summary["shard_id"])
        centroid = summary["centroid_wgs84"]
        axes.text(
            centroid["longitude"],
            centroid["latitude"],
            str(shard_id),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
        )
    axes.set_title("China full-grid tenfold reorder V4 (new shard numbering)")
    axes.set_xlabel("Longitude (WGS84)")
    axes.set_ylabel("Latitude (WGS84)")
    axes.set_aspect("equal", adjustable="box")
    axes.legend(loc="lower left", ncols=2, markerscale=12)
    figure.savefig(output_root / "china_tenfold_partition_overview.png", dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--old-shard-order",
        type=int,
        nargs=SHARD_COUNT,
        default=[8, 1, 2, 3, 4, 5, 6, 7, 9, 10],
        help="Old shard ids in the desired new shard_01 through shard_10 order.",
    )
    parser.add_argument("--overview-sample-stride", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    if args.overview_sample_stride <= 0:
        raise ValueError("--overview-sample-stride must be positive")
    remap = build_shard_remap(args.old_shard_order)
    source_manifest = json.loads((source_root / "tenfold_partition_manifest.json").read_text())
    source_summaries = {int(summary["shard_id"]): summary for summary in source_manifest["shards"]}
    if sorted(source_summaries) != list(range(1, SHARD_COUNT + 1)):
        raise ValueError("Source delivery does not contain exactly ten shard summaries")

    output_root.mkdir(parents=True)
    output_shards = output_root / "shards"
    summaries: list[dict[str, object]] = []
    samples: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    counts = Counter()
    for old_shard_id in args.old_shard_order:
        new_shard_id = remap[old_shard_id]
        source_shard = source_root / "shards" / f"shard_{old_shard_id:02d}"
        target_shard = output_shards / f"shard_{new_shard_id:02d}"
        target_shard.mkdir(parents=True)
        sample_lon: list[np.ndarray] = []
        sample_lat: list[np.ndarray] = []
        for source_path in sorted(source_shard.glob("utm*/part-*.parquet")):
            table = pq.read_table(source_path)
            shard_index = table.schema.get_field_index("shard_id")
            if shard_index < 0:
                raise ValueError(f"Missing shard_id column: {source_path}")
            target_table = table.set_column(
                shard_index,
                "shard_id",
                pa.array([new_shard_id] * len(table), type=pa.int8()),
            )
            target_path = target_shard / source_path.relative_to(source_shard)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(target_table, target_path, compression="zstd")
            counts[new_shard_id] += len(target_table)
            sample_lon.append(target_table["longitude"].to_numpy()[:: args.overview_sample_stride])
            sample_lat.append(target_table["latitude"].to_numpy()[:: args.overview_sample_stride])
        samples[new_shard_id] = (np.concatenate(sample_lon), np.concatenate(sample_lat))
        summaries.append(reorder_summary(source_summaries[old_shard_id], new_shard_id))

    expected_counts = [int(summary["shape_count"]) for summary in summaries]
    actual_counts = [counts[index] for index in range(1, SHARD_COUNT + 1)]
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Shard count mismatch: actual={actual_counts}, expected={expected_counts}"
        )
    features = [region_bounds_feature(summary) for summary in summaries]
    region_index = {
        "type": "FeatureCollection",
        "name": "china_tenfold_shard_regions_reordered_v4",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "metadata": {
            "geometry_role": "center_coordinate_bbox_index",
            "exact_membership": (
                "Use the GeoParquet geometry and parent_key files in each shard directory."
            ),
        },
        "features": features,
    }
    for feature in features:
        shard_id = feature["properties"]["shard_id"]
        (output_shards / shard_id / "region_bounds.geojson").write_text(
            json.dumps(feature, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema_version": "china_full_grid_tenfold_spatial_partition_v4_reordered",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_delivery": source_root.name,
        "source_schema_version": source_manifest["schema_version"],
        "new_to_old_shard_order": args.old_shard_order,
        "old_to_new_shard_map": {str(old): new for old, new in sorted(remap.items())},
        "source_parent_count": int(source_manifest["source_parent_count"]),
        "method": "label_reorder_only_v1",
        "validation": {"shard_counts": dict(zip(range(1, SHARD_COUNT + 1), actual_counts))},
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
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "china_tenfold_shard_regions.geojson").write_text(
        json.dumps(region_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_readmes(output_root, summaries)
    write_overview(output_root, samples, summaries)
    qa_lines = [
        "# 十等分重排 V4 质量检查",
        "",
        f"- 来源交付包：`{source_root.name}`",
        f"- 新到旧编号顺序：{args.old_shard_order}",
        f"- Shape 总数：{sum(actual_counts):,}",
        "- 几何与 parent_key 未修改；仅 shard_id 与文件目录编号重排。",
        "- 每区计数与来源分区一致。",
    ]
    (output_root / "QA.md").write_text("\n".join(qa_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
