from __future__ import annotations

# 地理空间工具模块：提供栅格 bounds / CRS 读取、patch 网格生成、重投影等通用能力。
from pathlib import Path
from typing import overload

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject


def read_bounds(path: Path) -> tuple[float, float, float, float]:
    """读取 GeoTIFF 的地理边界。

    Args:
        path: 栅格文件路径。

    Returns:
        (left, bottom, right, top) 边界元组。
    """
    with rasterio.open(path) as src:
        return tuple(src.bounds)  # type: ignore[return-value]


def get_crs(path: Path) -> CRS:
    """读取 GeoTIFF 的坐标参考系统。

    Args:
        path: 栅格文件路径。

    Returns:
        栅格的 :class:`rasterio.crs.CRS` 对象。
    """
    with rasterio.open(path) as src:
        crs = src.crs
        if crs is None:
            raise ValueError(f"文件缺少 CRS: {path}")
        return crs


def make_patch_grid(
    bounds: tuple[float, float, float, float],
    patch_size_m: float,
) -> list[tuple[float, float, float, float]]:
    """根据边界和 patch 边长生成不重叠的 patch 列表。

    从左上角 ``(left, top)`` 开始，自左向右、自下向上（先沿 y 方向）遍历，
    生成 ``(left, bottom, right, top)`` 列表。边界不足一个 patch 时取到边界为止。

    Args:
        bounds: ``(left, bottom, right, top)``，单位为米或 CRS 本身单位。
        patch_size_m: patch 边长，单位与 bounds 一致。

    Returns:
        patch 边界列表，每个元素为 ``(left, bottom, right, top)``。
    """
    if patch_size_m <= 0:
        raise ValueError(f"patch_size_m 必须大于 0，得到 {patch_size_m}")

    left, bottom, right, top = bounds
    patches: list[tuple[float, float, float, float]] = []
    x = left
    while x < right:
        y = bottom
        while y < top:
            patches.append((x, y, min(x + patch_size_m, right), min(y + patch_size_m, top)))
            y += patch_size_m
        x += patch_size_m
    return patches


def reproject_array(
    src_array: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    dst_shape: tuple[int, int],
    dst_transform: Affine,
    dst_crs: CRS,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """将源数组重投影到目标网格。

    Args:
        src_array: 源数组，形状通常为 ``(height, width)`` 或 ``(bands, height, width)``。
        src_transform: 源数组的仿射变换。
        src_crs: 源 CRS。
        dst_shape: 目标数组形状 ``(height, width)``。
        dst_transform: 目标仿射变换。
        dst_crs: 目标 CRS。
        resampling: 重采样方法，默认为双线性。

    Returns:
        重投影后的目标数组。
    """
    # 对于多波段数组，warp.reproject 会自动处理 band 维度
    dst_array = np.empty(dst_shape, dtype=src_array.dtype)
    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    return dst_array


@overload
def reproject_raster_to_match(
    src_path: Path,
    dst_path: Path,
    dst_path_or_transform: Path,
    dst_crs: CRS | None = None,
    dst_shape: tuple[int, int] | None = None,
) -> None: ...


@overload
def reproject_raster_to_match(
    src_path: Path,
    dst_path: Path,
    dst_path_or_transform: Affine,
    dst_crs: CRS,
    dst_shape: tuple[int, int],
) -> None: ...


def reproject_raster_to_match(
    src_path: Path,
    dst_path: Path,
    dst_path_or_transform: Path | Affine,
    dst_crs: CRS | None = None,
    dst_shape: tuple[int, int] | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """将源栅格对齐到目标栅格或指定网格，并写出为新的 GeoTIFF。

    当 ``dst_path_or_transform`` 为 :class:`pathlib.Path` 时，
    会从该文件读取 ``dst_transform``、``dst_crs`` 和 ``dst_shape``，
    此时 ``dst_crs`` 与 ``dst_shape`` 应为 ``None``。
    当 ``dst_path_or_transform`` 为 :class:`rasterio.transform.Affine` 时，
    必须同时传入 ``dst_crs`` 与 ``dst_shape``。

    Args:
        src_path: 源栅格路径。
        dst_path: 输出栅格路径。
        dst_path_or_transform: 目标栅格路径或目标仿射变换。
        dst_crs: 目标 CRS，当 ``dst_path_or_transform`` 为路径时忽略。
        dst_shape: 目标形状，当 ``dst_path_or_transform`` 为路径时忽略。
        resampling: 重采样方法，默认为双线性。
    """
    with rasterio.open(src_path) as src:
        src_array = src.read()
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
        src_dtype = src.dtypes[0]
        src_count = src.count

        if isinstance(dst_path_or_transform, Path):
            if dst_crs is not None or dst_shape is not None:
                raise ValueError(
                    "当 dst_path_or_transform 为 Path 时，dst_crs 和 dst_shape 必须为 None"
                )
            with rasterio.open(dst_path_or_transform) as dst_template:
                dst_transform = dst_template.transform
                dst_crs_read = dst_template.crs
                dst_shape_read = (dst_template.height, dst_template.width)
            _dst_crs = dst_crs_read
            _dst_shape = dst_shape_read
        else:
            if dst_crs is None or dst_shape is None:
                raise ValueError(
                    "当 dst_path_or_transform 为 Affine 时，必须提供 dst_crs 和 dst_shape"
                )
            dst_transform = dst_path_or_transform
            _dst_crs = dst_crs
            _dst_shape = dst_shape

        dst_array = np.empty((src_count, *_dst_shape), dtype=src_dtype)
        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,  # type: ignore[has-type]
            dst_crs=_dst_crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=src_nodata,
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            dst_path,
            "w",
            driver="GTiff",
            height=_dst_shape[0],
            width=_dst_shape[1],
            count=src_count,
            dtype=dst_array.dtype,
            crs=_dst_crs,
            transform=dst_transform,  # type: ignore[has-type]
            nodata=src_nodata,
        ) as dst:
            dst.write(dst_array)
