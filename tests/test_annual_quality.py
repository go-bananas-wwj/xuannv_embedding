from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from xuannv_embedding.data.annual_dataset import (
    AnnualObservationDataset,
    collate_annual_observations,
)
from xuannv_embedding.data.annual_dataset import (
    main as check_annual,
)
from xuannv_embedding.data_process.annual_quality import (
    add_moments,
    buffered_parents,
    build,
    numeric_reasons,
    select_seasons,
    verify_observation,
)
from xuannv_embedding.data_process.observation_raster import inspect_raster


def observation():
    return {
        "grid_matches_parent": True,
        "pixel_check": "decoded",
        "channels": 2,
        "source_signature": "s1",
        "valid_fraction": 1.0,
        "band_mean": [1.0, 2.0],
        "band_variance": [1.0, 2.0],
        "band_counts": [4, 4],
        "zero_fractions": [0.0, 0.0],
        "dtype_max_fractions": [None, None],
        "band_min_max": [[0.1, 3.0], [0.1, 4.0]],
    }


def test_conservative_filter_never_equates_finite_with_clear_sky():
    record = observation()
    assert numeric_reasons(record, highres=False) == []
    record["dtype_max_fractions"] = [0.01, 0]
    assert "integer_max_review" in numeric_reasons(record, highres=False)
    record["zero_fractions"] = [1, 1]
    record["band_variance"] = [0, 0]
    assert "zero_value_review" in numeric_reasons(record, highres=False)
    assert "constant_image_review" in numeric_reasons(record, highres=False)
    record["grid_matches_parent"] = False
    assert "grid_mismatch" in numeric_reasons(record, highres=False)


def test_spatial_buffer_crosses_utm_and_preserves_test_priority():
    parents = [
        {"parent_key": "a", "longitude": 120.0, "latitude": 40.0, "split": "train"},
        {"parent_key": "b", "longitude": 120.01, "latitude": 40.0, "split": "test"},
        {"parent_key": "c", "longitude": 123.0, "latitude": 40.0, "split": "train"},
    ]
    assert buffered_parents(parents) == {"a"}
    assert buffered_parents(list(reversed(parents))) == {"a"}


def test_float_fill_is_rejected_without_rejecting_legitimate_negative_values():
    record = observation()
    record["band_min_max"] = [[-30.0, 1.0], [-40.0, 1.0]]
    assert numeric_reasons(record, highres=False) == []
    record["band_min_max"][0][0] = -32768.0
    assert numeric_reasons(record, highres=False) == ["undeclared_negative_fill_review"]
    record.pop("band_min_max")
    assert "missing_numeric_extrema" in numeric_reasons(record, highres=False)
    record["band_min_max"] = [None, None]
    assert "missing_numeric_extrema" in numeric_reasons(record, highres=False)


def test_season_selection_retains_temporal_diversity_and_deduplicates():
    observations = [
        {
            "date": f"2020-{month:02}-01",
            "path": str(month),
            "sha256": str(month),
            "valid_fraction": 1 - month / 100,
        }
        for month in (1, 2, 3, 4, 7, 10)
    ]
    observations.append(dict(observations[0], path="duplicate"))
    selected = select_seasons(observations)
    assert [item["date"] for item in selected] == [
        "2020-01-01",
        "2020-04-01",
        "2020-07-01",
        "2020-10-01",
    ]
    assert select_seasons(list(reversed(observations))) == selected


def test_stable_statistics_merge():
    state = {}
    add_moments(state, {"band_counts": [2], "band_mean": [2], "band_variance": [1]})
    add_moments(state, {"band_counts": [2], "band_mean": [6], "band_variance": [1]})
    assert state["mean"] == pytest.approx([4])
    assert state["moment"] / state["counts"] == pytest.approx([5])


def test_build_filters_bad_observation_and_reads_native_annual_data(tmp_path):
    lowres, highres, output = (tmp_path / name for name in ("lowres", "highres", "output"))
    lowres.mkdir()
    (highres / "parts").mkdir(parents=True)
    (highres / "sources.json").write_text("{}")
    parent = {
        "parent_key": "32650:400:2800",
        "longitude": 117.0,
        "latitude": 32.0,
        "split": "train",
    }
    (lowres / "selected_parents.jsonl").write_text(json.dumps(parent) + "\n")
    records = []
    for month in (1, 2, 4, 5, 7, 8, 10, 11):
        path = lowres / f"s1/2020/{month:02}/parent.tif"
        path.parent.mkdir(parents=True)
        values = np.arange(2 * 128 * 128, dtype=np.float32).reshape(2, 128, 128) + 1
        if month == 10:
            values[:] = 0
        if month == 11:
            values[:, :64] = -32768
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            count=2,
            dtype="float32",
            width=128,
            height=128,
            crs="EPSG:32650",
            transform=from_origin(512000, 3585280, 10, 10),
        ) as raster:
            raster.write(values)
        record = inspect_raster(
            path.read_bytes(), parent["parent_key"], pixels=True, destination=path
        )
        record.update(
            parent_key=parent["parent_key"],
            source_signature="s1",
            split="train",
            month=f"2020-{month:02}",
            materialized_path=path.relative_to(lowres).as_posix(),
            mask_path=path.with_name("parent_mask.tif").relative_to(lowres).as_posix(),
        )
        records.append(record)
    (lowres / "lowres_observations.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records)
    )
    report = build(lowres, highres, output, verify_per_source=1)
    assert report["training_ready"] is False
    assert report["catalog_counts"]["excluded_records"] == 2
    assert report["exclusion_reasons_nonexclusive"]["undeclared_negative_fill_review"] == 1
    assert report["manifests"] == {"2020.train.manifest.jsonl": 1}
    with pytest.raises(ValueError, match="scientific quality"):
        AnnualObservationDataset(output / "2020.train.manifest.jsonl")
    dataset = AnnualObservationDataset(output / "2020.train.manifest.jsonl", allow_candidates=True)
    sample = dataset[0]
    assert sample["source_frames"]["s1"].shape == (12, 2, 128, 128)
    assert sample["source_masks"]["s1"].sum() == 6
    assert sample["highres_observations"] == []
    assert sample["output_grid"]["crs"] == "EPSG:32650"
    batch = collate_annual_observations([sample, sample])
    assert batch["source_frames"]["s1"].shape == (2, 12, 2, 128, 128)
    assert batch["highres_observations"] == [[], []]
    assert (
        check_annual(
            [
                "--dataset",
                str(output),
                "--samples-per-manifest",
                "1",
                "--output",
                str(tmp_path / "loader-check.json"),
            ]
        )
        == 0
    )
    assert not sample["source_pixel_masks"]["s1"][9].any()
    assert not sample["source_pixel_masks"]["s1"][10].any()
    record = dataset.records[0].provenance["observations"]["s1"][0]
    target = tmp_path / record["path"]
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_observation(tmp_path, record)


def test_annual_dataset_import_disables_gdal_directory_probing():
    """单目录十万量级文件时，sidecar 探测让 open 贵一个数量级；三条读栅格路径都要关。"""
    import os

    import xuannv_embedding.data.annual_dataset  # noqa: F401

    assert os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
    assert os.environ["GDAL_PAM_ENABLED"] == "NO"
