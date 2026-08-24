"""对齐多源 NetCDF 时序数据到统一 10 m UTM 网格并切分为 patches。

核心改进：
- 一次性加载整时相到内存；
- 按 AOI 主网格对齐（整数像素偏移或 warp）；
- 批量整数窗口切片并补 nodata；
- 每源独立有效掩膜；
- 按 (nc_file, time_idx) 多进程并行。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds, from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds as window_from_bounds

from xuannv_embedding.utils.geo import make_patch_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TOLERANCE = 1e-3
MASTER_RES = 10.0

# S2 SCL 有效类别：植被、裸土、水、未分类、雪/冰
S2_VALID_SCL = {4, 5, 6, 7, 11}


def _find_data_variable(ds: xr.Dataset) -> str:
    """从 Dataset 中找出具有 (time, band, y, x) 维度的数据变量。"""
    target_dims = {"time", "band", "y", "x"}
    candidates = [name for name, var in ds.data_vars.items() if set(var.dims) == target_dims]
    if not candidates:
        raise ValueError(
            f"NetCDF 中未找到维度为 (time, band, y, x) 的数据变量，"
            f"实际变量: {list(ds.data_vars.keys())}"
        )
    return candidates[0]


def _extract_epsg(ds: xr.Dataset) -> int:
    """从 Dataset 坐标或属性中提取 EPSG 代码。"""
    if "epsg" in ds.coords:
        epsg = int(ds.coords["epsg"].values)
        if epsg > 0:
            return epsg
    if "proj:code" in ds.coords:
        code = str(ds.coords["proj:code"].values)
        if code.upper().startswith("EPSG:"):
            return int(code.split(":", 1)[1])
    for key in ("epsg", "crs"):
        if key in ds.attrs:
            val = ds.attrs[key]
            if isinstance(val, int):
                return val
            if isinstance(val, str) and val.upper().startswith("EPSG:"):
                return int(val.split(":", 1)[1])
    raise ValueError(f"无法从 NetCDF 提取 EPSG: {list(ds.coords.keys())}, {dict(ds.attrs)}")


def _build_src_transform(x: np.ndarray, y: np.ndarray) -> tuple[Affine, bool]:
    """根据 xarray 的 x/y 坐标构建 rasterio 仿射变换。

    返回:
        (src_transform, needs_vertical_flip)
    """
    x_vals = np.asarray(x, dtype=np.float64)
    y_vals = np.asarray(y, dtype=np.float64)

    x_res = float(np.abs(np.diff(x_vals).mean()))
    y_res = float(np.abs(np.diff(y_vals).mean()))

    left = float(x_vals.min()) - x_res / 2.0
    top = float(y_vals.max()) + y_res / 2.0

    src_transform = from_origin(left, top, x_res, y_res)
    needs_vertical_flip = bool(y_vals[1] > y_vals[0])
    return src_transform, needs_vertical_flip


def load_config(path: Path) -> dict[str, Any]:
    """加载 JSON 配置文件。"""
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("min_valid_ratio", 0.3)
    config.setdefault("workers", 8)
    config.setdefault("nodata", 0.0)
    config.setdefault("aoi_path", f"configs/regions/{config['region']}.geojson")
    # 默认优先使用参考项目中的 patches_meta JSON（如存在）
    default_patch_grid = f"configs/regions/{config['region']}_patches.json"
    if Path(default_patch_grid).exists():
        config.setdefault("patch_grid_path", default_patch_grid)
    return config


def _load_patch_grid_from_json(
    patch_grid_path: str | Path,
    crs_obj: CRS,
) -> tuple[gpd.GeoDataFrame, Affine]:
    """从 patches_meta JSON 加载预定义 patch 列表并构建主变换。

    JSON 格式支持：
      - 列表：[{"patch_id": "...", "bounds": [left,bottom,right,top]}, ...]
      - 字典：{"patches": [...], ...}

    主变换以所有 patch 最小 bounds 为原点、10 m 分辨率构建，保证每个 patch
    在 master grid 上对应整数像素窗口。
    """
    with open(patch_grid_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    patches = data if isinstance(data, list) else data.get("patches", [])
    if not patches:
        raise ValueError(f"patch_grid 文件为空或格式错误: {patch_grid_path}")

    records: list[dict[str, Any]] = []
    lefts, bottoms, rights, tops = [], [], [], []
    for p in patches:
        pbounds = tuple(p["bounds"])
        lefts.append(pbounds[0])
        bottoms.append(pbounds[1])
        rights.append(pbounds[2])
        tops.append(pbounds[3])
        records.append(
            {
                "patch_id": p["patch_id"],
                "geometry": shapely.geometry.box(*pbounds),
                "bounds": pbounds,
            }
        )

    left = math.floor(min(lefts) / MASTER_RES) * MASTER_RES
    bottom = math.floor(min(bottoms) / MASTER_RES) * MASTER_RES
    right = math.ceil(max(rights) / MASTER_RES) * MASTER_RES
    top = math.ceil(max(tops) / MASTER_RES) * MASTER_RES

    width = int(round((right - left) / MASTER_RES))
    height = int(round((top - bottom) / MASTER_RES))
    master_transform = from_bounds(left, bottom, right, top, width=width, height=height)

    patch_gdf = gpd.GeoDataFrame(records, crs=crs_obj)
    return patch_gdf, master_transform


def generate_patch_grid(
    aoi_path_or_file: str | Path | None,
    patch_size_m: float,
    crs: str | CRS,
    patch_grid_path: str | Path | None = None,
) -> tuple[gpd.GeoDataFrame, Affine]:
    """生成 patch 网格与主变换。

    若提供 ``patch_grid_path``，则使用预定义 patch 列表；否则从 AOI 生成规则网格。
    返回 (patch_gdf, master_transform)，其中 master_transform 的 a=10, e=-10。
    """
    crs_obj = CRS.from_string(crs) if isinstance(crs, str) else crs

    if patch_grid_path is not None:
        return _load_patch_grid_from_json(patch_grid_path, crs_obj)

    if aoi_path_or_file is None:
        raise ValueError("必须提供 aoi_path 或 patch_grid_path 之一")

    aoi = gpd.read_file(aoi_path_or_file)
    aoi = aoi.to_crs(crs_obj)

    left, bottom, right, top = aoi.total_bounds

    # 将边界向外吸附到 10 m 网格，确保 transform 严格为 10 m
    left = math.floor(left / MASTER_RES) * MASTER_RES
    bottom = math.floor(bottom / MASTER_RES) * MASTER_RES
    right = math.ceil(right / MASTER_RES) * MASTER_RES
    top = math.ceil(top / MASTER_RES) * MASTER_RES

    width = int(round((right - left) / MASTER_RES))
    height = int(round((top - bottom) / MASTER_RES))

    master_transform = from_bounds(left, bottom, right, top, width=width, height=height)

    raw_patches = make_patch_grid((left, bottom, right, top), patch_size_m)
    n_rows = int(np.ceil((top - bottom) / patch_size_m))

    records: list[dict[str, Any]] = []
    for idx, pbounds in enumerate(raw_patches):
        col = idx // n_rows
        row = idx % n_rows
        patch_id = f"p{col:03d}_r{row:03d}"
        records.append(
            {"patch_id": patch_id, "geometry": shapely.geometry.box(*pbounds), "bounds": pbounds}
        )

    patch_gdf = gpd.GeoDataFrame(records, crs=crs_obj)
    return patch_gdf, master_transform


def _compute_valid_mask(
    slice_arr: np.ndarray, source: str, nodata: float, num_bands: int | None = None
) -> np.ndarray:
    """计算 uint8 有效像素掩膜。"""
    if num_bands is None:
        num_bands = slice_arr.shape[0]
    ref = slice_arr[0]
    valid = np.isfinite(ref) & (ref != nodata)

    if source == "s2":
        scl = slice_arr[-1]
        scl_uint8 = np.nan_to_num(scl, nan=0.0).astype(np.uint8)
        valid = valid & np.isin(scl_uint8, list(S2_VALID_SCL))
    elif source == "landsat" and num_bands == 7:
        # 仅当包含 qa_pixel 波段时才做 QA 掩膜；缺失时退化为 nodata/finite 掩膜
        qa = np.nan_to_num(slice_arr[-1], nan=0.0).astype(np.uint16)
        valid = valid & ((qa & 0b11111) == 0)

    return valid.astype(np.uint8)


def _scale_landsat_reflectance_preserving_qa(arr: np.ndarray, nodata: float) -> np.ndarray:
    """Scale Landsat reflectance bands while preserving final QA_PIXEL bit values."""
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError("Landsat array must contain reflectance bands plus QA_PIXEL")
    reflectance = arr[:-1]
    valid = reflectance != nodata
    reflectance[valid] = reflectance[valid] * 0.0000275 - 0.2
    return arr


def _slice_with_padding(
    arr: np.ndarray,
    row_off: int,
    col_off: int,
    win_h: int,
    win_w: int,
    nodata: float,
) -> np.ndarray:
    """从数组中切出指定窗口，越界部分用 nodata 填充。"""
    src_h, src_w = arr.shape[-2:]
    pad_top = max(0, -row_off)
    pad_left = max(0, -col_off)
    pad_bottom = max(0, row_off + win_h - src_h)
    pad_right = max(0, col_off + win_w - src_w)

    slice_arr = arr[
        :,
        max(0, row_off) : min(row_off + win_h, src_h),
        max(0, col_off) : min(col_off + win_w, src_w),
    ]

    if pad_top or pad_left or pad_bottom or pad_right:
        slice_arr = np.pad(
            slice_arr,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=nodata,
        )
    return slice_arr


def _extract_patch_bounded(
    arr: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    master_transform: Affine,
    master_crs: CRS,
    window: rasterio.windows.Window,
    nodata: float,
    source: str,
) -> np.ndarray:
    """按 master 网格窗口提取一个 patch，不分配完整 AOI 数组。

    对 S2/S1 等已匹配 10 m 网格的数据直接整数像素切片；对 Landsat 等
    分辨率不同的数据，先切出覆盖该 patch 的源区域再重采样到 128x128。
    """
    bands = arr.shape[0]
    patch_h = int(round(window.height))
    patch_w = int(round(window.width))
    patch_transform = rasterio.windows.transform(window, master_transform)

    src_res = abs(src_transform.a)
    src_y_res = abs(src_transform.e)
    same_crs = src_crs == master_crs
    res_match = abs(src_res - MASTER_RES) < TOLERANCE and abs(src_y_res - MASTER_RES) < TOLERANCE
    no_skew = abs(src_transform.b) < TOLERANCE and abs(src_transform.d) < TOLERANCE

    # 计算覆盖 patch 地理范围的源窗口
    src_window = window_from_bounds(
        *rasterio.windows.bounds(window, master_transform),
        transform=src_transform,
    )
    row_off = int(round(src_window.row_off))
    col_off = int(round(src_window.col_off))
    win_h = int(round(src_window.height))
    win_w = int(round(src_window.width))

    slice_arr = _slice_with_padding(arr, row_off, col_off, win_h, win_w, nodata)

    if same_crs and res_match and no_skew and slice_arr.shape[-2:] == (patch_h, patch_w):
        return slice_arr

    # 子数组必须使用对应窗口的 transform，否则 reproject 会按整景坐标错位。
    slice_transform = rasterio.windows.transform(
        rasterio.windows.Window(col_off, row_off, win_w, win_h),
        src_transform,
    )

    dst = np.full((bands, patch_h, patch_w), nodata, dtype=arr.dtype)
    # SCL and QA_PIXEL are categorical/bitmask bands.  Bilinear interpolation
    # invents class values and corrupts the Landsat QA bits used below.
    if source in {"s2", "landsat"} and bands >= 2:
        reproject(
            source=slice_arr[:-1],
            destination=dst[:-1],
            src_transform=slice_transform,
            src_crs=src_crs,
            dst_transform=patch_transform,
            dst_crs=master_crs,
            resampling=Resampling.bilinear,
            dst_nodata=nodata,
        )
        reproject(
            source=slice_arr[-1],
            destination=dst[-1],
            src_transform=slice_transform,
            src_crs=src_crs,
            dst_transform=patch_transform,
            dst_crs=master_crs,
            resampling=Resampling.nearest,
            dst_nodata=nodata,
        )
    else:
        reproject(
            source=slice_arr,
            destination=dst,
            src_transform=slice_transform,
            src_crs=src_crs,
            dst_transform=patch_transform,
            dst_crs=master_crs,
            resampling=Resampling.bilinear,
            dst_nodata=nodata,
        )
    return dst


def _init_patch_accumulator(
    slice_arr: np.ndarray,
    source: str,
    nodata: float,
) -> dict[str, Any]:
    """为单个 patch 创建聚合缓冲区。"""
    acc: dict[str, Any] = {
        "sum": np.zeros_like(slice_arr, dtype=np.float64),
        "count": np.zeros_like(slice_arr, dtype=np.uint16),
        "valid_count": np.zeros(slice_arr.shape[1:], dtype=np.uint8),
    }
    if source in {"s2", "landsat"}:
        # SCL / QA_PIXEL are categorical bit fields, never arithmetic targets.
        key = "scl" if source == "s2" else "qa"
        acc[key] = np.full(slice_arr.shape[1:], nodata, dtype=np.float32)
    return acc


def process_one_date(args: tuple[Path, str, list[int], dict[str, Any]]) -> int:
    """处理单个 NetCDF 的单个日期（可能含多景/多 tile），返回写入的 patch 数量。"""
    nc_path, date_str, time_idxs, config = args

    try:
        source = config["source"]
        nodata = float(config["nodata"])
        min_valid_ratio = float(config["min_valid_ratio"])
        master_transform = Affine(*config["master_transform"])
        master_crs = CRS.from_string(config["crs"])
        patches = config["patches"]

        out_dir = Path(config["output_root"]) / "patches" / source
        out_dir.mkdir(parents=True, exist_ok=True)

        # 为每个 patch 预分配聚合缓冲区
        patch_acc: dict[str, dict[str, Any]] = {}
        for patch in patches:
            patch_id = patch["patch_id"]
            window = window_from_bounds(*patch["bounds"], transform=master_transform)
            patch_acc[patch_id] = {
                "bounds": patch["bounds"],
                "window": window,
                "acc": None,
            }

        band_names: list[str] = []
        with xr.open_dataset(nc_path, chunks={"time": 1}) as ds:
            data_var = _find_data_variable(ds)
            band_names = [str(b) for b in ds.band.values]
            y_coords = np.asarray(ds.y.values, dtype=np.float64)
            x_coords = np.asarray(ds.x.values, dtype=np.float64)
            src_transform, needs_vertical_flip = _build_src_transform(x_coords, y_coords)
            src_crs = CRS.from_epsg(_extract_epsg(ds))

            for time_idx in time_idxs:
                arr = ds[data_var].isel(time=time_idx).values.astype(np.float32)

                # Landsat Collection 2 Level-2 DN -> surface reflectance.
                # QA_PIXEL is the final band and must remain an unscaled bitmask.
                if source == "landsat":
                    arr = _scale_landsat_reflectance_preserving_qa(arr, nodata)

                if needs_vertical_flip:
                    arr = arr[:, ::-1, :]

                for patch_id, pacc in patch_acc.items():
                    slice_arr = _extract_patch_bounded(
                        arr=arr,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        master_transform=master_transform,
                        master_crs=master_crs,
                        window=pacc["window"],
                        nodata=nodata,
                        source=source,
                    )
                    valid_mask = _compute_valid_mask(
                        slice_arr, source, nodata, num_bands=slice_arr.shape[0]
                    )

                    if pacc["acc"] is None:
                        pacc["acc"] = _init_patch_accumulator(slice_arr, source, nodata)
                    acc = pacc["acc"]

                    vm = valid_mask[None, :, :]
                    if source in {"s2", "landsat"}:
                        ref = slice_arr[:-1]
                        categorical = slice_arr[-1]
                        acc["sum"][:-1] += np.where(vm, ref, 0.0)
                        acc["count"][:-1] += vm.astype(np.uint16)
                        key = "scl" if source == "s2" else "qa"
                        empty = (acc[key] == nodata) & valid_mask
                        acc[key][empty] = categorical[empty]
                        acc["valid_count"] += valid_mask.astype(np.uint8)
                    else:
                        acc["sum"] += np.where(vm, slice_arr, 0.0)
                        acc["count"] += vm.astype(np.uint16)
                        acc["valid_count"] += valid_mask.astype(np.uint8)

        base_profile = {
            "driver": "GTiff",
            "height": config["patch_size_px"],
            "width": config["patch_size_px"],
            "crs": master_crs,
            "nodata": nodata,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 128,
            "blockysize": 128,
        }

        written = 0
        for patch_id, pacc in patch_acc.items():
            acc = pacc["acc"]
            if acc is None:
                continue

            valid_mask = (acc["valid_count"] > 0).astype(np.uint8)
            if float(valid_mask.mean()) < min_valid_ratio:
                continue

            out_path = out_dir / f"{source}_{date_str}_{patch_id}.tif"
            mask_path = out_dir / f"{source}_{date_str}_{patch_id}_mask.tif"
            if out_path.exists() and not config.get("overwrite", False):
                continue

            avg = np.divide(
                acc["sum"],
                acc["count"],
                out=np.full_like(acc["sum"], nodata, dtype=np.float64),
                where=acc["count"] > 0,
            ).astype(np.float32)
            if source == "s2":
                avg[-1] = acc["scl"]
            elif source == "landsat":
                avg[-1] = acc["qa"]

            patch_transform = rasterio.windows.transform(pacc["window"], master_transform)
            profile = {
                **base_profile,
                "count": avg.shape[0],
                "dtype": avg.dtype,
                "transform": patch_transform,
            }
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(avg)
                for idx, name in enumerate(band_names, start=1):
                    dst.set_band_description(idx, name)

            mask_profile = {
                **base_profile,
                "count": 1,
                "dtype": "uint8",
                "nodata": None,
                "transform": patch_transform,
            }
            with rasterio.open(mask_path, "w", **mask_profile) as dst:
                dst.write(valid_mask, 1)

            written += 1

        return written
    except Exception as exc:  # noqa: BLE001
        logger.exception("处理失败: nc=%s date=%s - %s", nc_path, date_str, exc)
        return 0


def process_file(nc_path: Path, config: dict[str, Any], max_times: int | None = None) -> int:
    """处理单个 NetCDF 文件的所有时间步（按日期聚合）。"""
    with xr.open_dataset(nc_path, chunks={"time": 1}) as ds:
        n_times = ds.sizes["time"]

    n_times = min(n_times, max_times) if max_times is not None else n_times
    with xr.open_dataset(nc_path, chunks={"time": 1}) as ds:
        times = pd.to_datetime(ds["time"].values[:n_times])

    date_groups: dict[str, list[int]] = defaultdict(list)
    for idx, timestamp in enumerate(times):
        date_groups[timestamp.strftime("%Y%m%d")].append(idx)

    tasks = [(nc_path, date, idxs, config) for date, idxs in date_groups.items()]
    total_written = 0
    with ProcessPoolExecutor(max_workers=config["workers"]) as executor:
        for date_str, written in zip(date_groups.keys(), executor.map(process_one_date, tasks)):
            total_written += written
            if written == 0:
                logger.warning(
                    "%s: date=%s 未写入任何 patch（可能处理异常或全部被过滤）",
                    nc_path.name,
                    date_str,
                )

    return total_written


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="将 NetCDF 时序数据对齐到统一 10 m 网格并切分为 patches",
    )
    parser.add_argument("--config", required=True, type=Path, help="JSON 配置文件路径")
    parser.add_argument("--source", default=None, help="仅处理指定 source（调试用）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 patch 文件")
    parser.add_argument("--max-files", type=int, default=None, help="最多处理的 NetCDF 文件数")
    parser.add_argument("--max-times", type=int, default=None, help="每个文件最多处理的时间步数")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    region = config["region"]
    raw_root = Path(config["raw_root"])
    output_root = Path(config["output_root"])
    crs = CRS.from_string(config["crs"])
    patch_size_m = float(config["patch_size_m"])
    sources = config["sources"]

    if args.source is not None:
        sources = [args.source]

    aoi_path = Path(config["aoi_path"])
    patch_grid_path = config.get("patch_grid_path")
    patch_gdf, master_transform = generate_patch_grid(
        aoi_path, patch_size_m, crs, patch_grid_path=patch_grid_path
    )

    # 从 snapped bounds 计算主网格形状
    left, bottom, right, top = patch_gdf.total_bounds
    master_shape = (
        int(round((top - bottom) / MASTER_RES)),
        int(round((right - left) / MASTER_RES)),
    )

    config["master_transform"] = tuple(master_transform)
    config["master_shape"] = master_shape
    # 多进程传递：转换为纯 Python 结构
    config["patches"] = [
        {"patch_id": row["patch_id"], "bounds": row.geometry.bounds}
        for _, row in patch_gdf.iterrows()
    ]

    logger.info(
        "开始预处理: region=%s raw_root=%s output_root=%s crs=%s patches=%d shape=%s",
        region,
        raw_root,
        output_root,
        crs,
        len(config["patches"]),
        master_shape,
    )

    total_written = 0
    t0 = time.time()
    for source in sources:
        source_dir = raw_root / source
        if not source_dir.exists():
            logger.warning("source 目录不存在，跳过: %s", source_dir)
            continue

        nc_files = sorted(source_dir.glob("*.nc"))
        if not nc_files:
            logger.warning("未找到 NetCDF 文件: %s", source_dir)
            continue

        if args.max_files is not None:
            nc_files = nc_files[: args.max_files]

        config["source"] = source
        config["overwrite"] = args.overwrite

        with xr.open_dataset(nc_files[0], chunks={"time": 1}) as ds:
            config["band_names"] = [str(b) for b in ds.band.values]

        written = 0
        for nc_path in nc_files:
            logger.info("%s: 处理 %s", source, nc_path.name)
            written += process_file(nc_path, config, args.max_times)

        total_written += written
        logger.info("%s: 写入 %d 个 patches", source, written)

    elapsed = time.time() - t0
    logger.info(
        "预处理完成: 共写入 %d 个 patches, 耗时 %.1f s (%.2f patches/s)",
        total_written,
        elapsed,
        total_written / elapsed if elapsed > 0 else 0.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
