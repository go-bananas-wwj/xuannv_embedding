from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pytest
from affine import Affine
from rasterio.io import MemoryFile

from xuannv_embedding.data.contracts import ProductSpec
from xuannv_embedding.data.local_archives import (
    ArchiveContractError,
    LocalArchive,
    audit_raster_member,
    index_local_archive,
    month_bounds,
    validate_archive_sidecar,
    verify_archive_lock,
)


def _tiff_bytes(*, count: int = 2, width: int = 8, height: int = 8) -> bytes:
    values = np.arange(count * width * height, dtype=np.uint16).reshape(count, height, width)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            count=count,
            width=width,
            height=height,
            dtype="uint16",
            crs="EPSG:32650",
            transform=Affine(10, 0, 500000, 0, -10, 4400000),
        ) as dataset:
            dataset.write(values)
        return memory.read()


def _archive(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _spec() -> ProductSpec:
    return ProductSpec(
        product_id="s1_local",
        role="dense",
        bands=("vh", "vv"),
        native_gsd_m=(20.0, 20.0),
        stored_gsd_m=10.0,
        dtype="uint16",
        time_precision="month",
        already_resampled=True,
        qa_available=False,
    )


def test_month_archive_index_preserves_missing_mask_and_honest_time(tmp_path: Path) -> None:
    path = tmp_path / "pc-s1_2020_01.zip"
    member = "pc-s1/2020/01/parent_32650:1:2.tif"
    _archive(path, {member: _tiff_bytes()})
    descriptor = LocalArchive("s1_local", 2020, 1, path)

    result = index_local_archive(
        descriptor,
        _spec(),
        expected_patch_ids={"parent_32650:1:2", "parent_32650:1:3"},
    )

    assert result.member_count == 1
    assert result.extra_patch_ids == ()
    assert [item.patch_id for item in result.observations if item.present] == ["parent_32650:1:2"]
    missing = next(item for item in result.observations if not item.present)
    assert missing.patch_id == "parent_32650:1:3"
    assert missing.member_name == ""
    assert missing.acquired_at is None
    assert missing.available_at == datetime(2020, 2, 1, tzinfo=timezone.utc)
    assert month_bounds(2020, 12)[1] == datetime(2021, 1, 1, tzinfo=timezone.utc)


def test_archive_index_blocks_patch_ids_outside_frozen_grid(tmp_path: Path) -> None:
    path = tmp_path / "pc-s1_2020_01.zip"
    _archive(path, {"pc-s1/2020/01/unexpected.tif": _tiff_bytes()})

    with pytest.raises(ArchiveContractError, match="多余 patch"):
        index_local_archive(LocalArchive("s1_local", 2020, 1, path), _spec(), set())


def test_pixel_audit_distinguishes_resampled_monthly_patch(tmp_path: Path) -> None:
    path = tmp_path / "pc-s1_2020_01.zip"
    member = "pc-s1/2020/01/parent_32650:1:2.tif"
    _archive(path, {member: _tiff_bytes()})

    audit = audit_raster_member(path, member, _spec(), expected_shape=(8, 8))

    assert audit.passed is True
    assert audit.processing_state == "monthly_patch_already_resampled"
    assert audit.channels == 2
    assert audit.crs == "EPSG:32650"
    assert audit.finite_fraction == 1.0


def test_pixel_audit_rejects_wrong_channel_contract(tmp_path: Path) -> None:
    path = tmp_path / "pc-s1_2020_01.zip"
    member = "pc-s1/2020/01/parent_32650:1:2.tif"
    _archive(path, {member: _tiff_bytes(count=1)})

    audit = audit_raster_member(path, member, _spec(), expected_shape=(8, 8))

    assert audit.passed is False
    assert "channels" in audit.failures


def test_local_archive_rejects_remote_uri() -> None:
    with pytest.raises(ArchiveContractError, match="本地路径"):
        LocalArchive("s2_local", 2020, 1, Path("https:/example.test/archive.zip"))


def test_sidecar_locks_real_band_order_without_inventing_metadata(tmp_path: Path) -> None:
    path = tmp_path / "pc-s1_2020_01.zip"
    _archive(path, {"pc-s1/2020/01/parent_32650:1:2.tif": _tiff_bytes()})
    sidecar = path.with_suffix(".zip.txt")
    sidecar.write_text("波段/通道: vh, vv\n单张尺寸: 128 x 128 像素\n", encoding="utf-8")

    digest = validate_archive_sidecar(LocalArchive("s1_local", 2020, 1, path), _spec())

    assert len(digest) == 64
    sidecar.write_text("波段/通道: vv, vh\n单张尺寸: 128 x 128 像素\n", encoding="utf-8")
    with pytest.raises(ArchiveContractError, match="波段顺序"):
        validate_archive_sidecar(LocalArchive("s1_local", 2020, 1, path), _spec())


def test_archive_lock_recomputes_current_sha_and_rejects_incomplete_lock(tmp_path: Path) -> None:
    path = tmp_path / "pc-s1_2020_01.zip"
    _archive(path, {"pc-s1/2020/01/p.tif": _tiff_bytes()})
    lock = tmp_path / "lock.jsonl"
    lock.write_text(
        '{"archive_path":"%s","size_bytes":%d,"sha256":null}\n' % (path, path.stat().st_size),
        encoding="utf-8",
    )

    with pytest.raises(ArchiveContractError, match="完整 SHA256"):
        verify_archive_lock(lock, expected_count=1)

    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lock.write_text(
        '{"archive_path":"%s","size_bytes":%d,"sha256":"%s"}\n'
        % (path, path.stat().st_size, digest),
        encoding="utf-8",
    )
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ArchiveContractError, match="size|SHA256"):
        verify_archive_lock(lock, expected_count=1)
