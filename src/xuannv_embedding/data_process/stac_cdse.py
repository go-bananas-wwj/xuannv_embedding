"""Cache CDSE STAC metadata for national S2/S1 materialization.

Only compact STAC metadata is stored under the data disk.  Image access stays
on Copernicus Data Space's authenticated S3 endpoint at materialization time;
no credentials are accepted by this program or written to its outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from calendar import monthrange
from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests

CATALOG_URL = "https://stac.dataspace.copernicus.eu/v1"
SOURCE_CONFIG: dict[str, dict[str, Any]] = {
    "s2": {
        "collection": "sentinel-2-l2a",
        "asset_map": {
            "B02": "B02_10m",
            "B03": "B03_10m",
            "B04": "B04_10m",
            "B05": "B05_20m",
            "B06": "B06_20m",
            "B07": "B07_20m",
            "B08": "B08_10m",
            "B8A": "B8A_20m",
            "B09": "B09_60m",
            "B11": "B11_20m",
            "B12": "B12_20m",
            "SCL": "SCL_20m",
        },
        "query": {"eo:cloud_cover": {"lt": 90}},
    },
    "s1": {
        "collection": "sentinel-1-grd",
        "asset_map": {"vv": "vv", "vh": "vh"},
        "query": {},
    },
}


def _months() -> list[str]:
    return [
        f"{year:04d}-{month:02d}"
        for year, start, end in ((2025, 4, 12), (2026, 1, 4))
        for month in range(start, end + 1)
    ]


def _interval(month: str) -> str:
    year, number = (int(value) for value in month.split("-"))
    first_day = date(year, number, 1).isoformat()
    last_day = date(year, number, monthrange(year, number)[1]).isoformat()
    return f"{first_day}T00:00:00Z/{last_day}T23:59:59Z"


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _compact_feature(feature: dict[str, Any], source: str) -> dict[str, Any] | None:
    config = SOURCE_CONFIG[source]
    original_assets = feature.get("assets", {})
    assets: dict[str, dict[str, Any]] = {}
    for target_name, cdse_name in config["asset_map"].items():
        asset = original_assets.get(cdse_name)
        if not asset or not str(asset.get("href", "")).startswith("s3://eodata/"):
            return None
        assets[target_name] = {
            key: asset[key] for key in ("href", "type", "roles", "title") if key in asset
        }
    properties = feature.get("properties", {})
    return {
        "id": feature["id"],
        "collection": feature.get("collection"),
        "bbox": feature.get("bbox"),
        "geometry": feature.get("geometry"),
        "properties": {
            key: properties.get(key)
            for key in (
                "datetime",
                "start_datetime",
                "end_datetime",
                "eo:cloud_cover",
                "proj:code",
                "proj:epsg",
                "sat:orbit_state",
                "s1:instrument_configuration_ID",
            )
            if key in properties
        },
        "assets": assets,
    }


def _next_request(
    payload: dict[str, Any], page: dict[str, Any]
) -> tuple[str, str, dict[str, Any] | None] | None:
    for link in page.get("links", []):
        if link.get("rel") != "next":
            continue
        method = str(link.get("method", "GET")).upper()
        return method, str(link["href"]), link.get("body", payload) if method == "POST" else None
    return None


def _request_with_retry(
    session: requests.Session, method: str, url: str, body: dict[str, Any] | None, attempts: int = 6
) -> requests.Response:
    for attempt in range(attempts):
        try:
            response = session.request(method, url, json=body, timeout=(15, 120))
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response
            delay = min(60.0, 2.0**attempt)
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            delay = min(60.0, 2.0**attempt)
        time.sleep(delay)
    raise RuntimeError("unreachable retry loop")


def cache_month(
    session: requests.Session,
    source: str,
    month: str,
    bbox: list[float],
    output_root: Path,
    limit: int,
) -> dict[str, Any]:
    config = SOURCE_CONFIG[source]
    output_dir = output_root / source / month
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "collections": [config["collection"]],
        "bbox": bbox,
        "datetime": _interval(month),
        "query": config["query"],
        "limit": limit,
        # CDSE caps the heavy Sentinel-2 collection at 200 items unless a
        # projection is used.  Requesting only the geometry, date/cloud
        # metadata and bands consumed by the materializer keeps every page
        # compact and makes the pagination limit explicit.
        "fields": {
            "include": [
                "id",
                "bbox",
                "geometry",
                "properties.datetime",
                "properties.start_datetime",
                "properties.end_datetime",
                "properties.eo:cloud_cover",
                "properties.proj:code",
                "properties.proj:epsg",
                "properties.sat:orbit_state",
                "properties.s1:instrument_configuration_ID",
                *[f"assets.{name}" for name in config["asset_map"].values()],
            ]
        },
    }
    query_hash = _sha256({"catalog": CATALOG_URL, **payload, "asset_map": config["asset_map"]})
    status_path, items_path = output_dir / "status.json", output_dir / "items.jsonl"
    if status_path.exists() and items_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("query_sha256") == query_hash and status.get("complete"):
            return status
    temporary = items_path.with_suffix(".jsonl.partial")
    request: tuple[str, str, dict[str, Any] | None] | None = (
        "POST",
        f"{CATALOG_URL}/search",
        payload,
    )
    count = pages = skipped = 0
    started = time.monotonic()
    with temporary.open("w", encoding="utf-8") as handle:
        while request is not None:
            method, url, body = request
            page = _request_with_retry(session, method, url, body).json()
            for feature in page.get("features", []):
                compact = _compact_feature(feature, source)
                if compact is None:
                    skipped += 1
                    continue
                handle.write(json.dumps(compact, ensure_ascii=False) + "\n")
                count += 1
            pages += 1
            request = _next_request(payload, page)
    temporary.replace(items_path)
    status = {
        "schema_version": "china_v1_cdse_stac_cache_v1",
        "complete": True,
        "source": source,
        "collection": config["collection"],
        "month": month,
        "bbox": bbox,
        "query_sha256": query_hash,
        "item_count": count,
        "skipped_incomplete_assets": skipped,
        "page_count": pages,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "catalog_url": CATALOG_URL,
    }
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCE_CONFIG), required=True)
    parser.add_argument("--country", type=Path, required=True)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Optional small-area smoke-test extent in EPSG:4326.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--months", nargs="+", default=None)
    args = parser.parse_args()
    if args.limit <= 0 or args.limit > 200:
        raise ValueError("CDSE STAC page limit must be in 1..200")
    country = gpd.read_file(args.country).to_crs("EPSG:4326")
    months = args.months or _months()
    invalid = sorted(set(months) - set(_months()))
    if invalid:
        raise ValueError(f"unsupported months outside the frozen archive: {invalid}")
    session = requests.Session()
    session.trust_env = False
    bbox = list(args.bbox) if args.bbox else [float(x) for x in country.total_bounds]
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("bbox must be west < east and south < north")
    results = [
        cache_month(session, args.source, month, bbox, args.output_root, args.limit)
        for month in months
    ]
    print(
        json.dumps(
            {
                "source": args.source,
                "months": len(results),
                "items": sum(row["item_count"] for row in results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
