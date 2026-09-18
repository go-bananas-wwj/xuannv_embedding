from __future__ import annotations

import io
import json
import tarfile
import zipfile

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from xuannv_embedding.data.raster_dataset import _month
from xuannv_embedding.data_process.finalize_catalog import finalize_catalog
from xuannv_embedding.data_process.highres_catalog import (
    ArchiveRow,
    build_catalog,
    observation_record,
    parent_key,
    read_archive_index,
    safe_member,
    select_parents,
)
from xuannv_embedding.data_process.observation_raster import inspect_raster, merge_statistics
from xuannv_embedding.data_process.pilot_cache import (
    SOURCES,
    _monthly_archive,
    month_gap,
    prepare_lowres,
)


def raster_payload(channels=1, offset=0, invalid=False):
    values = np.arange(channels * 16, dtype=np.uint16).reshape(channels, 4, 4) + 1
    if invalid:
        values[:] = 0
    else:
        values[0, 0, 0] = 0
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            count=channels,
            width=4,
            height=4,
            dtype="uint16",
            crs="EPSG:32650",
            transform=from_origin(512000 + offset, 3585280, 320, 320),
            nodata=0,
        ) as raster:
            raster.write(values)
        return memory.read()


def parent_record():
    return {
        "archive": "one.tar.gz",
        "patch_id": "ownerfix_epsg32650_c400_r2800",
        "parent_key": "32650:400:2800",
        "split": "train",
        "materialize": True,
    }


def test_parent_identity_dates_and_product_conflicts():
    assert parent_key("preview_utm50n_c40_r280_c0_r0") == "32650:400:2800"
    assert parent_key("ownerfix_epsg32650_c400_r2800") == "32650:400:2800"
    record = observation_record(parent_record(), "owner/GF6/2021_GF6_20230107_PAN_2m.tif")
    assert record["acquisition_time"] == "2023-01-07"
    assert record["product_type"] == "PAN"
    assert set(record["issues"]) == {"declared_year_conflict", "outside_pilot_years"}
    assert (
        observation_record(parent_record(), "owner/ZY/2021_IMG_2m_scene-FWD.tif")["product_type"]
        == "FWD"
    )
    assert (
        observation_record(parent_record(), "owner/GF/20200230_PAN.tif")["acquisition_time"] is None
    )
    assert (
        observation_record(parent_record(), "owner/GF/20200101_20200202_PAN.tif")[
            "acquisition_time"
        ]
        is None
    )


@pytest.mark.parametrize(
    "name", ["../bad.tar", "/absolute.tif", "parent/../../bad", "parent\\bad.tif"]
)
def test_unsafe_archive_names_rejected(name):
    with pytest.raises(ValueError):
        safe_member(name)


def test_sampling_is_order_independent_nested_and_spatially_grouped():
    rows = [
        ArchiveRow("one.tar.gz", f"ownerfix_epsg32650_c{column}_r{row}", f"32650:{column}:{row}")
        for column in range(250, 550, 20)
        for row in range(2000, 3500, 50)
    ]
    small = select_parents(rows, 64, 16, 42)
    larger = select_parents(list(reversed(rows)), 128, 16, 42)
    assert small == larger[:64]
    assert len({record["spatial_block"] for record in small}) > 10
    split_by_block = {}
    for record in larger:
        assert (
            split_by_block.setdefault(record["spatial_block"], record["split"]) == record["split"]
        )
    assert sum(record["materialize"] for record in larger) == 16


def test_index_rejects_duplicate_parent_aliases(tmp_path):
    path = tmp_path / "index.tsv"
    path.write_text(
        "archive\tpatchid\none.tar.gz\tpreview_utm50n_c40_r280_c0_r0\none.tar.gz\townerfix_epsg32650_c400_r2800\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_archive_index(path)


def test_raster_preserves_pixels_masks_and_unknown_semantics(tmp_path):
    payload = raster_payload(2)
    output = tmp_path / "native.tif"
    record = inspect_raster(payload, "32650:400:2800", pixels=True, destination=output)
    assert output.read_bytes() == payload
    assert record["grid_matches_parent"]
    assert record["valid_fraction"] == 15 / 16
    assert record["band_counts"] == [15, 15]
    assert record["native_gsd_m"] is None
    assert record["band_names"] == [None, None]
    assert merge_statistics([record])["band_counts"] == [15, 15]
    inspect_raster(payload, "32650:400:2800", pixels=True, destination=output)
    shifted = inspect_raster(
        raster_payload(offset=10),
        "32650:400:2800",
        pixels=True,
        destination=tmp_path / "shifted.tif",
    )
    assert not shifted["grid_matches_parent"]
    assert not (tmp_path / "shifted.tif").exists()


def test_month_paths_preserve_month_precision_and_legacy_dates():
    assert _month("lowres/s2/2020/01/parent_32650:400:2800.tif") == 202001
    assert _month("legacy/s2_20251225_patch.tif") == 202512
    assert _month("lowres/2020/13/patch.tif") == 0
    assert month_gap("2020-02-29", "2020-02") == 0
    assert month_gap("2020-03-01", "2020-02") == 1
    with pytest.raises(ValueError, match="冲突"):
        _month("root/2020/01/2021/02/patch.tif")


def test_corrupt_month_is_not_silent_missing_observation(tmp_path):
    root = tmp_path / "source"
    path = root / "pc-s1/2021/02/pc-s1_2021_02.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"broken zip")
    availability, observations = _monthly_archive(
        tmp_path, root, "pc-s1", "2021-02", [parent_record()]
    )
    assert availability[0]["status"] == "archive_unreadable"
    assert observations == []


def test_catalog_and_lowres_loader_roundtrip(tmp_path):
    parent = parent_record()
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    index = tmp_path / "index.tsv"
    index.write_text(f"archive\tpatchid\none.tar.gz\t{parent['patch_id']}\n")
    payload = raster_payload()
    with tarfile.open(archive_root / "one.tar.gz", "w:gz") as archive:
        for name in ["2020_GF6_20200115_PAN_2m.tif", "2021_GF6_20230115_PAN_2m.tif"]:
            member = tarfile.TarInfo(f"{parent['patch_id']}/GF6/{name}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    output = tmp_path / "catalog"
    result = build_catalog(
        archive_root=archive_root,
        index_path=index,
        output_root=output,
        parent_limit=1,
        materialize_parents=1,
        workers=1,
    )
    assert result["observation_count"] == 2
    assert result["materialized_tiffs"] == 1
    assert result["issues"]["outside_pilot_years"] == 1
    assert (output / "checksums.json").exists()
    with pytest.raises(FileExistsError):
        build_catalog(archive_root=archive_root, index_path=index, output_root=output)
    lowres = tmp_path / "monthly"
    stage = tmp_path / "paired.partial"
    stage.mkdir()
    (stage / "selected_parents.jsonl").write_text(json.dumps(parent) + "\n")
    for source, (_, channels) in SOURCES.items():
        path = lowres / source / "2020/01" / f"{source}_2020_01.zip"
        path.parent.mkdir(parents=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                f"{source}/2020/01/parent_{parent['parent_key']}.tif", raster_payload(channels)
            )
    report = prepare_lowres(stage, lowres, [parent], [])
    assert report["cached_tiffs"] == 3
    assert report["loader"]["available_source_months"] == 3
    assert report["loader"]["passed"]

    (stage / "fingerprint.json").write_text(
        json.dumps({"version": "observation-pilot-v1", "parent_limit": 1}) + "\n"
    )
    (stage / "catalog_parts").mkdir()
    (stage / "catalog_parts/one.summary.json").write_text(
        json.dumps({"observations": 2, "materialized": 1}) + "\n"
    )
    final = tmp_path / "paired"
    finalized = finalize_catalog(
        stage,
        final,
        train_samples=1,
        validation_samples=1,
        test_samples=1,
        workers=1,
    )
    assert finalized["status"] == "training_data_prepared"
    assert finalized["lowres"]["loader"]["parents_checked"] == 1
    assert (final / "summary.json").is_file()
