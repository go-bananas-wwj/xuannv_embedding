"""Visualize downloaded observations and references with explicit provenance and missingness."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

WC = {
    10: ("树木", "#006400"),
    20: ("灌丛", "#ffbb22"),
    30: ("草地", "#ffff4c"),
    40: ("耕地", "#f096ff"),
    50: ("建成区", "#fa0000"),
    60: ("裸地", "#b4b4b4"),
    70: ("冰雪", "#f0f0f0"),
    80: ("水体", "#0064c8"),
    90: ("草本湿地", "#0096a0"),
    95: ("红树林", "#00cf75"),
    100: ("苔藓／地衣", "#fae6a0"),
}


def rgb_band_order(source: str) -> list[int]:
    if source == "aerial":
        return [1, 2, 3]
    if source in {"s2", "landsat"}:
        return [3, 2, 1]
    raise ValueError("source does not declare a true-color band ordering")


def _read(path: Path, bands: list[int], max_side: int = 900, nearest=False):
    with rasterio.open(path) as src:
        scale = min(1, max_side / max(src.height, src.width))
        height = max(1, round(src.height * scale))
        width = max(1, round(src.width * scale))
        values = src.read(
            bands,
            out_shape=(len(bands), height, width),
            masked=True,
            resampling=Resampling.nearest if nearest else Resampling.average,
        )
    valid = ~np.ma.getmaskarray(values).any(axis=0)
    data = values.astype("float32").filled(np.nan)
    valid &= np.isfinite(data).all(axis=0)
    return data, valid


def read_rgb_preview(path: Path, source: str, max_side=900):
    values, valid = _read(path, rgb_band_order(source), max_side)
    image = values.transpose(1, 2, 0)
    if valid.any():
        low, high = np.percentile(image[valid], [2, 98])
        image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    image[~valid] = [0.94, 0.95, 0.96]
    return image, valid


def source_inventory(spec: dict) -> list[dict]:
    from xuannv_embedding.data_process.regional_acquisition import LANDSAT_QUALITY_VERSION

    root = Path(spec["root"])
    rows = []
    for source in ["s2", "s1", "landsat"]:
        for month in spec["months"]:
            path = root / "monthly" / source / (month + ".tif")
            available = path.exists() and path.with_suffix(".json").exists()
            if available and source == "landsat":
                try:
                    available = (
                        json.loads(path.with_suffix(".json").read_text()).get(
                            "quality_mask_version"
                        )
                        == LANDSAT_QUALITY_VERSION
                    )
                except (OSError, json.JSONDecodeError):
                    available = False
            rows.append(
                {
                    "source": source,
                    "period": month,
                    "path": str(path),
                    "available": available,
                    "note": "10 m grid; S1 effective resolution 20–22 m, Landsat native 30 m",
                }
            )
    static = [
        ("aerial", "2024 Mar–May", "static/aerial.tif", "static RGB, nominal 3 m"),
        ("dem", "2024 Apr–Jun", "static/dem.tif", "1 m, NZVD2016"),
        ("dsm", "2024 Apr–Jun", "static/dsm.tif", "1 m, NZVD2016"),
        ("worldcover", "2021", "weak_labels/worldcover_2021.tif", "common weak supervision"),
        ("osm", "2024-12-31", "weak_labels/osm_snapshot_20241231.json", "common weak supervision"),
        (
            "lcdb",
            "2023/24",
            "evaluation_sources/lcdb_v6_generalized15m.geojson",
            "v6, geometry simplified 15 m",
        ),
        (
            "buildings",
            "2017;2022",
            "evaluation_sources/buildings_mixed_2017_2022.geojson",
            "historical reference, not 2024 truth",
        ),
    ]
    for source, period, relative, note in static:
        path = root / relative
        rows.append(
            {
                "source": source,
                "period": period,
                "path": str(path),
                "available": path.exists()
                and (
                    path.with_suffix(".json").exists()
                    if source in {"aerial", "dem", "dsm"}
                    else True
                ),
                "note": note,
            }
        )
    return rows


def _lcdb(root: Path, bounds, shape):
    from rasterio.features import rasterize

    features = json.loads((root / "evaluation_sources/lcdb_v6_generalized15m.geojson").read_text())[
        "features"
    ]
    names = {}
    polygons = []
    for feature in features:
        code = feature["properties"].get("Class_2023")
        if code is None or not feature.get("geometry"):
            continue
        code = int(code)
        names[code] = feature["properties"].get("Name_2023", str(code))
        polygons.append((feature["geometry"], code))
    labels = rasterize(
        polygons,
        out_shape=shape,
        transform=from_bounds(*bounds, shape[1], shape[0]),
        dtype="uint16",
        fill=0,
    )
    return labels, names


def _density(root: Path, bounds):
    from shapely.geometry import shape

    features = json.loads(
        (root / "evaluation_sources/buildings_mixed_2017_2022.geojson").read_text()
    )["features"]
    coordinates = []
    for feature in features:
        if not feature.get("geometry"):
            continue
        polygon = shape(feature["geometry"])
        if polygon.is_empty or polygon.area <= 0:
            continue
        point = polygon.centroid
        coordinates.append((point.x, point.y))
    coordinates = np.asarray(coordinates)
    nx = math.ceil((bounds[2] - bounds[0]) / 200)
    ny = math.ceil((bounds[3] - bounds[1]) / 200)
    counts, _, _ = np.histogram2d(
        coordinates[:, 1],
        coordinates[:, 0],
        bins=[ny, nx],
        range=[[bounds[1], bounds[3]], [bounds[0], bounds[2]]],
    )
    cell_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / (nx * ny)
    return np.flipud(counts) * 1e6 / cell_area


def render_data_visualizations(spec: dict, *, require_complete=False, label="研究区") -> dict:
    import fcntl

    output = Path(spec["root"]) / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".render.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        return _render_data_visualizations(spec, require_complete=require_complete, label=label)


def _render_data_visualizations(spec: dict, *, require_complete=False, label="研究区") -> dict:
    rows = source_inventory(spec)
    missing = [r["source"] + ":" + r["period"] for r in rows if not r["available"]]
    if require_complete and missing:
        raise ValueError("downloaded data are incomplete")
    root = Path(spec["root"])
    out = root / "visualizations"
    out.mkdir(exist_ok=True)
    with (out / "data_paths.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "period", "path", "available", "note"])
        writer.writeheader()
        writer.writerows(rows)
    month = spec["months"][min(3, len(spec["months"]) - 1)]
    needed = [root / "monthly" / s / (month + ".tif") for s in ["s2", "s1", "landsat"]]
    needed += [
        root / "static/dem.tif",
        root / "static/dsm.tif",
        root / "weak_labels/worldcover_2021.tif",
        root / "evaluation_sources/lcdb_v6_generalized15m.geojson",
    ]
    if any(not p.exists() for i, p in enumerate(needed) if i != 2):
        raise ValueError("preview inputs are incomplete")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Patch

    font_file = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
    font = FontProperties(fname=str(font_file)) if font_file.exists() else FontProperties()
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False})

    def title(ax, text):
        ax.set_title(text, fontproperties=font, fontsize=12, pad=8)

    bounds = spec["bounds"]
    extent = [bounds[0] / 1000, bounds[2] / 1000, bounds[1] / 1000, bounds[3] / 1000]
    rgb, _ = read_rgb_preview(needed[0], "s2")
    landsat_available = next(
        r["available"] for r in rows if r["source"] == "landsat" and r["period"] == month
    )
    landsat = read_rgb_preview(needed[2], "landsat")[0] if landsat_available else None
    radar, radar_valid = _read(needed[1], [1, 2])
    radar[:, ~radar_valid] = np.nan
    radar_db = 10 * np.log10(np.maximum(radar, 1e-8))
    dem, dem_valid = _read(needed[3], [1])
    dsm, dsm_valid = _read(needed[4], [1])
    dem = dem[0]
    dsm = dsm[0]
    height = np.maximum(dsm - dem, 0)
    height[~(dem_valid & dsm_valid)] = np.nan
    wc, wc_valid = _read(needed[5], [1], nearest=True)
    wc = wc[0]
    lcdb, names = _lcdb(root, bounds, rgb.shape[:2])
    wc_codes = list(WC)
    wc_values = np.ma.masked_all(wc.shape)
    for index, code in enumerate(wc_codes):
        wc_values[wc_valid & (wc == code)] = index
    wc_cmap = ListedColormap([WC[k][1] for k in wc_codes])
    wc_cmap.set_bad("#f0f2f4")
    lc_codes = sorted(names)
    lc_values = np.ma.masked_all(lcdb.shape)
    colors = plt.get_cmap("tab20", max(len(lc_codes), 1))
    for index, code in enumerate(lc_codes):
        lc_values[lcdb == code] = index
    lc_cmap = ListedColormap([colors(i) for i in range(len(lc_codes))])
    lc_cmap.set_bad("#f0f2f4")
    complete = not missing
    panels = []
    if complete:
        aerial, _ = read_rgb_preview(root / "static/aerial.tif", "aerial")
        panels.append(("高分航拍 RGB｜2024年3–5月", aerial, None, None, None))
    panels += [
        (f"Sentinel-2 RGB｜{month}", rgb, None, None, None),
        (
            (f"Landsat RGB｜{month}", landsat, None, None, None)
            if landsat_available
            else (f"Sentinel-1 VH｜{month}", radar_db[1], "gray", (-32, -5), "dB，gamma0")
        ),
        (f"Sentinel-1 VV｜{month}", radar_db[0], "gray", (-25, 0), "dB，gamma0"),
    ]
    if complete:
        panels.append((f"Sentinel-1 VH｜{month}", radar_db[1], "gray", (-32, -5), "dB，gamma0"))
    if complete:
        panels.append(("WorldCover｜2021", wc_values, wc_cmap, (0, len(wc_codes) - 1), None))
    elevation_range = np.nanpercentile(
        np.concatenate([dem[np.isfinite(dem)], dsm[np.isfinite(dsm)]]), [1, 99]
    )
    panels += [
        ("DEM 地面高程｜2024", dem, "terrain", elevation_range, "m"),
        ("DSM 地表高程｜2024", dsm, "terrain", elevation_range, "m"),
        ("表面高差｜DSM − DEM，显示用", height, "viridis", (0, 35), "m；含植被和建筑"),
    ]
    if not complete:
        panels.append(("WorldCover｜2021", wc_values, wc_cmap, (0, len(wc_codes) - 1), None))
    panels.append(
        ("LCDB v6｜2023/24；边界简化15 m", lc_values, lc_cmap, (0, len(lc_codes) - 1), None)
    )
    if complete:
        panels.append(
            (
                "LINZ 参考轮廓计数密度｜2017/2022",
                _density(root, bounds),
                "magma",
                (0, 5000),
                "个/km²；历史参考",
            )
        )
    panels.append(
        (
            f'研究区网格｜{len(range(0,int(bounds[2]-bounds[0]),spec["patch_size_m"]))*len(range(0,int(bounds[3]-bounds[1]),spec["patch_size_m"]))}个图块',
            rgb,
            None,
            None,
            "grid",
        )
    )
    nrows = math.ceil(len(panels) / 3)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5.5 * nrows), squeeze=False)
    for ax, (text, values, cmap, limits, unit) in zip(axes.flat, panels):
        kwargs = {"cmap": cmap, "extent": extent, "origin": "upper", "interpolation": "nearest"}
        if cmap is not None and limits is not None:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        image = ax.imshow(values, **kwargs)
        title(ax, text)
        ax.set_facecolor("#f0f2f4")
        ax.tick_params(labelsize=8)
        if unit == "grid":
            lines = [
                [(x / 1000, extent[2]), (x / 1000, extent[3])]
                for x in np.arange(bounds[0], bounds[2] + 1, spec["patch_size_m"])
            ]
            lines += [
                [(extent[0], y / 1000), (extent[1], y / 1000)]
                for y in np.arange(bounds[1], bounds[3] + 1, spec["patch_size_m"])
            ]
            ax.add_collection(LineCollection(lines, colors="white", linewidths=0.35, alpha=0.8))
        elif unit:
            bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025, extend="max")
            bar.ax.set_ylabel(unit, fontproperties=font, fontsize=9)
            bar.ax.tick_params(labelsize=8)
        ax.set_xlabel("东向坐标 / km", fontproperties=font, fontsize=8)
        ax.set_ylabel("北向坐标 / km", fontproperties=font, fontsize=8)
    for ax in list(axes.flat)[len(panels) :]:
        ax.axis("off")
    pending = []
    if any(x.startswith("aerial:") for x in missing):
        pending.append("高分航拍")
    if any(x.startswith("landsat:") for x in missing):
        pending.append("Landsat质量复核")
    state = "完整下载数据" if complete else "已有数据预览｜待完成：" + "、".join(pending)
    fig.suptitle(label + " · " + state, fontproperties=font, fontsize=20, y=0.99)
    fig.text(
        0.5,
        0.01,
        "EPSG:2193；全区概览显示时降采样，原始文件不变。浅灰为无有效观测。\n"
        "此图展示输入及参考数据，不是模型预测；表面高差包含植被。建筑轮廓为历史参考。",
        ha="center",
        fontproperties=font,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    overview = out / ("data_overview_complete.png" if complete else "data_overview_available.png")
    fig.savefig(overview, dpi=170)
    plt.close(fig)
    legends, axes = plt.subplots(1, 2, figsize=(14, max(7, len(lc_codes) * 0.28)))
    for ax in axes:
        ax.axis("off")
    axes[0].legend(
        handles=[Patch(color=WC[k][1], label=f"{k} {WC[k][0]}") for k in wc_codes],
        title="WorldCover 2021",
        loc="upper left",
        prop=font,
        frameon=False,
    )
    axes[1].legend(
        handles=[Patch(color=colors(i), label=f"{k} {names[k]}") for i, k in enumerate(lc_codes)],
        title="LCDB 2023/24（展示配色）",
        loc="upper left",
        prop=font,
        frameon=False,
    )
    legend_path = out / "landcover_legends.png"
    legends.tight_layout()
    legends.savefig(legend_path, dpi=150)
    plt.close(legends)
    monthly, axes = plt.subplots(3, 4, figsize=(14, 15))
    for ax, period in zip(axes.flat, spec["months"]):
        path = root / "monthly/s2" / (period + ".tif")
        if path.exists():
            frame, valid = read_rgb_preview(path, "s2", max_side=500)
            ax.imshow(frame)
            coverage = valid.mean()
            receipt = path.with_suffix(".json")
            if receipt.exists():
                coverage = json.loads(receipt.read_text()).get("valid_fraction", coverage)
            title(ax, f"{period}｜有效覆盖 {coverage:.1%}")
        else:
            title(ax, period + "｜未完成")
        ax.axis("off")
    monthly.suptitle(label + " · Sentinel-2 全年月度影像", fontproperties=font, fontsize=18)
    monthly.tight_layout(rect=(0, 0, 1, 0.97))
    monthly_path = out / "sentinel2_monthly_2024.png"
    monthly.savefig(monthly_path, dpi=160)
    plt.close(monthly)
    detail_path = _render_highres_details(spec, root, out, month, font) if complete else None
    report = {
        "state": "complete_source_visualization" if complete else "available_data_preview",
        "detail_panels": detail_path,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "overview": str(overview),
        "monthly_s2": str(monthly_path),
        "landcover_legends": str(legend_path),
        "path_inventory": str(out / "data_paths.csv"),
        "missing_sources": missing,
        "sources": rows,
        "display_notes": {
            "rgb": "joint 2–98 percentile stretch; aerial RGB, public satellites B04/B03/B02",
            "height": "difference of downsampled DSM/DEM for display, not a building-only target",
            "grid": "display coordinates in km; EPSG:2193",
            "labels": "LCDB geometry simplified 15 m; LINZ outlines from 2017/2022",
        },
        "training_started": False,
    }
    (
        out / ("visualization_complete.json" if complete else "visualization_available.json")
    ).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _render_highres_details(spec, root, output, month, font):
    from matplotlib import pyplot as plt
    from rasterio.windows import from_bounds as pixel_window

    from xuannv_embedding.data_process.regional_acquisition import regular_grid

    grid = regular_grid(tuple(spec["bounds"]), spec["patch_size_m"])
    with rasterio.open(root / "weak_labels/worldcover_2021.tif") as src:
        wc = src.read(1)
    with rasterio.open(root / "static/aerial.tif") as src:
        availability = src.read_masks(1, out_shape=wc.shape) > 0
    n = int(spec["patch_size_m"] / 10)
    selected = []
    used = set()
    for code, name in [(50, "城镇"), (40, "农田"), (10, "林地"), (80, "水岸／水体")]:
        candidates = []
        for cell in grid:
            row = cell["row"] * n
            col = cell["col"] * n
            cover = wc[row : row + n, col : col + n]
            valid = availability[row : row + n, col : col + n]
            if valid.mean() >= 0.8 and cell["patch_id"] not in used:
                candidates.append((float((cover == code).mean()), cell))
        if not candidates:
            continue
        fraction, cell = max(candidates, key=lambda pair: pair[0])
        used.add(cell["patch_id"])
        selected.append({"landscape": name, "worldcover_code": code, "fraction": fraction, **cell})
    if not selected:
        return None

    def read(path, bands, bounds):
        with rasterio.open(path) as src:
            window = pixel_window(*bounds, transform=src.transform)
            values = src.read(
                bands,
                window=window,
                out_shape=(len(bands), 512, 512),
                masked=True,
                resampling=Resampling.bilinear,
            )
        valid = ~np.ma.getmaskarray(values).any(axis=0)
        return values.astype("float32").filled(np.nan), valid

    fig, axes = plt.subplots(len(selected), 3, figsize=(13, 4.2 * len(selected)), squeeze=False)
    for row, cell in enumerate(selected):
        for col, (path, source) in enumerate(
            [(root / "static/aerial.tif", "aerial"), (root / "monthly/s2" / (month + ".tif"), "s2")]
        ):
            values, valid = read(path, rgb_band_order(source), cell["bounds"])
            rgb = values.transpose(1, 2, 0)
            if valid.any():
                lo, hi = np.percentile(rgb[valid], [2, 98])
                rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
            rgb[~valid] = [0.94, 0.95, 0.96]
            axes[row, col].imshow(rgb)
        dem, valid_dem = read(root / "static/dem.tif", [1], cell["bounds"])
        dsm, valid_dsm = read(root / "static/dsm.tif", [1], cell["bounds"])
        difference = np.maximum(dsm[0] - dem[0], 0)
        difference[~(valid_dem & valid_dsm)] = np.nan
        image = axes[row, 2].imshow(difference, cmap="viridis", vmin=0, vmax=35)
        fig.colorbar(image, ax=axes[row, 2], fraction=0.04, pad=0.02, label="m")
        for col, text in enumerate(
            ["航拍RGB｜3 m输入", f"S2 RGB｜{month}，原生10 m", "表面高差｜显示用"]
        ):
            axes[row, col].set_title(
                cell["landscape"] + " · " + text, fontproperties=font, fontsize=11
            )
            axes[row, col].axis("off")
    fig.suptitle("同一位置的输入细节｜每行范围1.28 km × 1.28 km", fontproperties=font, fontsize=17)
    fig.text(
        0.5,
        0.01,
        "按WorldCover 2021类别占比选择；不是模型预测。高差包含植被与建筑，浅灰为无有效观测。",
        ha="center",
        fontproperties=font,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    path = output / "highres_details_complete.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    (output / "detail_sample_locations.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2)
    )
    return str(path)
