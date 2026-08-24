"""Cache minimal Planetary Computer STAC metadata for China V1 by source/month.

This is metadata acquisition only. It deliberately stores raw unsigned asset
URLs; short-lived signed URLs are created only by a later shard materializer.
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

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SOURCE_CONFIG = {
    "s2": {
        "collection": "sentinel-2-l2a",
        "assets": [
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
            "SCL",
        ],
        "query": {"eo:cloud_cover": {"lt": 90}},
    },
    "s1": {"collection": "sentinel-1-rtc", "assets": ["vv", "vh"], "query": {}},
    "landsat": {
        "collection": "landsat-c2-l2",
        "assets": ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"],
        "query": {"eo:cloud_cover": {"lt": 90}},
    },
}


def _months() -> list[str]:
    return [
        f"{year:04d}-{month:02d}"
        for year, start, end in ((2025, 4, 12), (2026, 1, 4))
        for month in range(start, end + 1)
    ]


def _interval(month: str) -> str:
    year, month_number = (int(value) for value in month.split("-"))
    last_day = monthrange(year, month_number)[1]
    start = date(year, month_number, 1).isoformat()
    end = date(year, month_number, last_day).isoformat()
    return f"{start}T00:00:00Z/{end}T23:59:59Z"


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _compact_feature(feature: dict[str, Any], asset_names: list[str]) -> dict[str, Any]:
    assets = feature.get("assets", {})
    return {
        "id": feature["id"],
        "collection": feature.get("collection"),
        "bbox": feature.get("bbox"),
        "geometry": feature.get("geometry"),
        "properties": {
            key: feature.get("properties", {}).get(key)
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
            if key in feature.get("properties", {})
        },
        "assets": {
            name: {
                key: assets[name].get(key)
                for key in ("href", "type", "roles", "title", "raster:bands")
                if key in assets[name]
            }
            for name in asset_names
            if name in assets
        },
    }


def _next_request(
    payload: dict[str, Any], page: dict[str, Any]
) -> tuple[str, str, dict[str, Any] | None] | None:
    for link in page.get("links", []):
        if link.get("rel") != "next":
            continue
        method = str(link.get("method", "GET")).upper()
        if method == "POST":
            return method, str(link["href"]), link.get("body", payload)
        return method, str(link["href"]), None
    return None


def _request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    body: dict[str, Any] | None,
    *,
    attempts: int = 6,
) -> requests.Response:
    """Fetch one STAC page with bounded backoff for transient catalogue errors."""
    for attempt in range(attempts):
        try:
            response = session.request(method, url, json=body, timeout=(15, 120))
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response
            retry_after = response.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else min(60.0, 2.0**attempt)
            )
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            delay = min(60.0, 2.0**attempt)
        if attempt == attempts - 1:
            response.raise_for_status()
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
        "fields": {
            "include": [
                "id",
                "collection",
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
                "assets",
            ]
        },
    }
    query_hash = _sha256(payload)
    status_path = output_dir / "status.json"
    items_path = output_dir / "items.jsonl"
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
    count = 0
    pages = 0
    started = time.monotonic()
    with temporary.open("w", encoding="utf-8") as handle:
        while request is not None:
            method, url, body = request
            response = _request_with_retry(session, method, url, body)
            page = response.json()
            for feature in page.get("features", []):
                handle.write(
                    json.dumps(_compact_feature(feature, config["assets"]), ensure_ascii=False)
                    + "\n"
                )
                count += 1
            pages += 1
            request = _next_request(payload, page)
    temporary.replace(items_path)
    status = {
        "schema_version": "china_v1_stac_cache_v1",
        "complete": True,
        "source": source,
        "collection": config["collection"],
        "month": month,
        "bbox": bbox,
        "query_sha256": query_hash,
        "item_count": count,
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--months",
        nargs="+",
        default=None,
        help="Optional YYYY-MM subset; separate workers may own disjoint months.",
    )
    args = parser.parse_args()
    country = gpd.read_file(args.country).to_crs("EPSG:4326")
    bbox = [float(value) for value in country.total_bounds]
    session = requests.Session()
    session.trust_env = False  # No proxy / tunnel for China V1 transfer.
    months = args.months or _months()
    invalid = sorted(set(months) - set(_months()))
    if invalid:
        raise ValueError(f"unsupported months outside the frozen archive: {invalid}")
    results = [
        cache_month(session, args.source, month, bbox, args.output_root, args.limit)
        for month in months
    ]
    print(
        json.dumps(
            {
                "source": args.source,
                "months": len(results),
                "items": sum(item["item_count"] for item in results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
