"""Local recovery of projection-induced polygon defects, with three-way raster agreement."""

from __future__ import annotations

import hashlib

import numpy as np
import shapely
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import transform

from xuannv_embedding.data_process.v5_osm_geometry import SpatialIndex


def recover_polygon(raw, projected, epsg, bounds, padding_m=0):
    crs = CRS.from_epsg(int(epsg))
    if not crs.is_projected or any(a.unit_conversion_factor != 1 for a in crs.axis_info):
        raise ValueError("local recovery requires a metric projected grid")
    if (
        raw.geom_type not in ["Polygon", "MultiPolygon"]
        or not raw.is_valid
        or raw.is_empty
        or not np.isfinite(raw.bounds).all()
    ):
        raise ValueError("invalid original source polygon cannot be recovered")
    if not np.isfinite(padding_m) or padding_m < 0:
        raise ValueError("invalid local repair padding")
    b = np.asarray(bounds, dtype=float)
    if b.shape != (4,) or not np.isfinite(b).all() or not np.allclose(b[2:] - b[:2], 1280):
        raise ValueError("invalid local repair footprint")
    margin = 100 + padding_m
    wgs = transform_bounds(crs, 4326, *(b + [-margin, -margin, margin, margin]), densify_pts=21)
    clipped = raw.intersection(box(*wgs))
    local = transform(Transformer.from_crs(4326, crs, always_xy=True).transform, clipped)
    variants = {
        "source_clipped": local,
        "linework": shapely.make_valid(projected, method="linework"),
        "structure": shapely.make_valid(projected, method="structure"),
    }
    arrays = {}
    for key, geom in variants.items():
        if geom.is_empty or not geom.is_valid or geom.geom_type not in ["Polygon", "MultiPolygon"]:
            raise ValueError("local repair did not produce valid polygon evidence")
        arrays[key] = rasterize(
            [(geom, 1)],
            out_shape=(512, 512),
            transform=from_bounds(*b, 512, 512),
            all_touched=True,
            dtype="uint8",
            skip_invalid=False,
        )
    differences = {
        key: int(np.count_nonzero(arrays[key] != arrays["source_clipped"]))
        for key in ["linework", "structure"]
    }
    if any(differences.values()):
        raise ValueError("local repair methods disagree inside target footprint")
    return local, {
        "epsg": int(epsg),
        "bounds": b.tolist(),
        "clip_padding_m": margin,
        "source_wkb_sha256": hashlib.sha256(raw.wkb).hexdigest(),
        "local_wkb_sha256": hashlib.sha256(local.wkb).hexdigest(),
        "comparison_pixels": 512 * 512,
        "source_modified": False,
        **{key + "_different_pixels": count for key, count in differences.items()},
    }


class RecoveringSpatialIndex(SpatialIndex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repair_records = []

    def query(self, epsg, bounds, padding_m=0):
        rows = super().query(epsg, bounds, padding_m)
        for row in rows:
            if not row["geometry"].is_valid:
                blob = self.connection.execute(
                    "SELECT wkb FROM features WHERE feature_id=?", (row["feature_id"],)
                ).fetchone()[0]
                row["geometry"], proof = recover_polygon(
                    shapely.from_wkb(bytes(blob)), row["geometry"], epsg, bounds, padding_m
                )
                self.repair_records.append(
                    {"index_path": str(self.path), "feature_id": row["feature_id"], **proof}
                )
        return rows
