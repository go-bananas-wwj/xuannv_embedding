from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from xuannv_embedding.utils.manifest import (
    ManifestError,
    ManifestRecord,
    load_legacy_manifest,
    load_manifest,
    manifest_meta_path,
    write_manifest,
)


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_manifest_v1_round_trip_with_verified_sidecar(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"manifest{suffix}"
    records = [
        ManifestRecord(
            patch_id="patch_000001",
            region="haidian",
            sources={
                "s2_haidian": ["patches/s2/202512/patch_000001.tif"],
                "sar_haidian": None,
            },
            source_patch_id="legacy_1",
            quality={"cloud_fraction": 0.1},
            provenance={"generator": "test"},
        )
    ]

    meta = write_manifest(path, records, months=["2025-12"], generator_version="test-1")
    document = load_manifest(path)

    assert meta.schema_version == "1"
    assert meta.record_count == 1
    assert document.meta.sha256 == meta.sha256
    assert document.records == records
    assert manifest_meta_path(path).exists()


def test_manifest_rejects_absolute_source_paths(tmp_path: Path) -> None:
    record = ManifestRecord(
        patch_id="patch_000001",
        region="haidian",
        sources={"s2": ["/data/xuannv_embedding/secret.tif"]},
    )

    with pytest.raises(ManifestError, match="相对路径"):
        write_manifest(tmp_path / "manifest.json", [record], months=["2025-12"])


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        [ManifestRecord("patch_1", "haidian", {"s2": ["s2/patch_1.tif"]})],
        months=["2025-12"],
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace("patch_1", "patch_2"),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="SHA-256"):
        load_manifest(path)


def test_manifest_rejects_duplicate_region_patch_identity_on_write_and_load(
    tmp_path: Path,
) -> None:
    duplicate = [
        ManifestRecord("p1", "haidian", {"s2": "s2/a.tif"}),
        ManifestRecord("p1", "haidian", {"s2": "s2/b.tif"}),
    ]
    with pytest.raises(ManifestError, match="重复.*region.*patch_id"):
        write_manifest(tmp_path / "write.jsonl", duplicate, months=["2025-12"])

    path = tmp_path / "load.jsonl"
    payload = b"{" + b'"patch_id":"p1","region":"haidian","sources":{"s2":"s2/a.tif"}}\n'
    payload += b"{" + b'"patch_id":"p1","region":"haidian","sources":{"s2":"s2/b.tif"}}\n'
    path.write_bytes(payload)
    manifest_meta_path(path).write_text(
        json.dumps(
            {
                "schema_version": "1",
                "months": ["2025-12"],
                "record_count": 2,
                "generator_version": "test",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="重复.*region.*patch_id"):
        load_manifest(path)


def test_manifest_requires_patch_region_and_sources(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([{"patch_id": "p1", "sources": {}}]), encoding="utf-8")
    manifest_meta_path(path).write_text(
        json.dumps(
            {
                "schema_version": "1",
                "months": [],
                "record_count": 1,
                "generator_version": "test",
                "sha256": "invalid",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError):
        load_manifest(path, verify_sha256=False)


def test_legacy_adapter_is_explicit_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            [
                {
                    "patch_id": "p001_r001",
                    "s2": ["../haidian/s2/s2_20251201_p001_r001.tif"],
                    "highres_sar": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_legacy_manifest(path, region="haidian")

    assert records[0].region == "haidian"
    assert records[0].sources["s2"] == ["haidian/s2/s2_20251201_p001_r001.tif"]
    assert records[0].sources["highres_sar"] is None
    with pytest.raises(ManifestError, match="sidecar"):
        load_manifest(path)
