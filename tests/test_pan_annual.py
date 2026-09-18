from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data.annual_dataset import AnnualObservationDataset
from xuannv_embedding.data_process.pan_annual import (
    audit_pan,
    merge_annual,
    scene_identity,
    scene_owners,
)
from xuannv_embedding.utils.manifest import ManifestRecord, load_manifest, write_manifest


def make_pan(root: Path, parent: str, split: str, scene: str, value_offset=0) -> dict:
    _, col, row = map(int, parent.split(":"))
    name = f"2020_GF6_PMS_20200105_PAN_2m_GF6_PMS_E110.0_N30.0_20200105_L1A{scene}-PAN.tif"
    path = root / parent / name
    path.parent.mkdir(parents=True, exist_ok=True)
    values = (np.arange(640 * 640).reshape(640, 640) % 1000 + 1 + value_offset).astype("uint16")
    values[0, 0] = 0
    values[0, 1] = 65535
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=1,
        width=640,
        height=640,
        dtype="uint16",
        crs="EPSG:32650",
        transform=from_origin(col * 1280, (row + 1) * 1280, 2, 2),
    ) as raster:
        raster.write(values, 1)
    return {
        "observation_id": hashlib.sha256(str(path).encode()).hexdigest(),
        "archive_member": f"patch/GF6/{name}",
        "parent_key": parent,
        "split": split,
        "source_signature": "GF6_PAN_c1",
        "platform": "GF6",
        "acquisition_time": "2020-01-05",
        "issues": [],
        "materialized_path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_pan_payload_audit_preserves_grid_and_excludes_sentinels(tmp_path):
    catalog, output = tmp_path / "catalog", tmp_path / "out"
    r = make_pan(catalog, "32650:400:2800", "train", "123")
    audited = audit_pan(r, catalog, output, tmp_path)
    assert audited["status"] == "numeric_candidate"
    assert audited["band_counts"] == [640 * 640 - 2]
    assert audited["cloud_shadow_QA"] == "unverified"
    with rasterio.open(tmp_path / audited["mask"]) as mask:
        assert mask.shape == (640, 640)
        assert not mask.read(1)[0, :2].any()
    r["sha256"] = "0" * 64
    assert audit_pan(r, catalog, output, tmp_path)["reasons"] == ["payload_checksum_mismatch"]


def test_date_conflict_and_scene_priority(tmp_path):
    r = make_pan(tmp_path, "32650:400:2800", "train", "123")
    assert scene_identity(r).endswith("L1A123")
    r["acquisition_time"] = "2021-01-05"
    with pytest.raises(ValueError, match="scene_date_conflict"):
        scene_identity(r)
    assert scene_owners([("a", "train"), ("a", "test"), ("a", "validation")]) == {"a": "test"}


def test_annual_merge_filters_scene_leakage_and_keeps_release_gate(tmp_path):
    catalog, output, base = (tmp_path / name for name in ("catalog", "out", "base"))
    (output / "pan_parts").mkdir(parents=True)
    (base / "statistics").mkdir(parents=True)
    (base / "sources.json").write_text("{}")
    (base / "summary.json").write_text(
        json.dumps(
            {
                "data_root": str(tmp_path),
                "training_ready": False,
                "remaining_gates": ["cloud_shadow_QA"],
            }
        )
    )
    train, test = "32650:400:2800", "32650:450:2800"
    originals = [
        make_pan(catalog, train, "train", "123"),
        make_pan(catalog, test, "test", "123"),
        make_pan(catalog, train, "train", "456", 100),
    ]
    audited = [audit_pan(r, catalog, output, tmp_path) for r in originals]
    assert all(r["status"] == "numeric_candidate" for r in audited)
    (output / "pan_parts" / "one.jsonl").write_text("".join(json.dumps(r) + "\n" for r in audited))
    for split, parent in (("train", train), ("test", test)):
        record = ManifestRecord(
            patch_id="parent_" + parent,
            region="annual_2020",
            sources={},
            grid={"parent_key": parent},
            quality={},
            provenance={"year": 2020, "split": split, "observations": {}},
        )
        write_manifest(
            base / f"2020.{split}.manifest.jsonl",
            [record],
            months=[f"2020-{m:02}" for m in range(1, 13)],
            generator_version="test",
        )
    report = merge_annual(base, output, [{"counts": {"numeric_candidate": 3}, "reasons": {}}])
    assert report["training_ready"] is False
    manifest = output / "2020.train.manifest.jsonl"
    obs = load_manifest(manifest).records[0].provenance["observations"]["GF6_PAN_c1"]
    assert len(obs) == 1 and obs[0]["scene_id"].endswith("L1A456")
    with pytest.raises(ValueError, match="scientific quality"):
        AnnualObservationDataset(manifest)
    dataset = AnnualObservationDataset(manifest, allow_candidates=True)
    sample = dataset[0]
    assert sample["highres_observations"][0]["values"].shape == (1, 640, 640)
    stats = json.loads((output / "statistics" / "GF6_PAN_c1_stats.json").read_text())
    assert stats["num_files"] == 1 and stats["fit_split"] == "train"
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "scene_ids": [audited[1]["scene_id"]],
                "reason": "visually suspected cloud",
            }
        )
    )
    revised = merge_annual(base, output, [], review)
    assert revised["pan2m_supported_samples"]["2020.test.manifest.jsonl"] == 0
    assert revised["pan2m_supported_samples"]["2020.train.manifest.jsonl"] == 1
    assert revised["training_ready"] is False
    assert (
        revised["pan_review_exclusions"]["sha256"]
        == hashlib.sha256(review.read_bytes()).hexdigest()
    )
