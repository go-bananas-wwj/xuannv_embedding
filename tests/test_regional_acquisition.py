from pathlib import Path

import numpy as np
import pytest

from xuannv_embedding.data_process.regional_acquisition import (
    Budget,
    optical_valid,
    regular_grid,
    select_optical_items,
)


def test_grid_retains_full_area_and_exact_shared_edges():
    cells = regular_grid((0, 0, 20480, 25600), 1280)
    assert len(cells) == 320
    assert cells[0]["bounds"] == [0, 24320, 1280, 25600]
    assert cells[-1]["bounds"] == [19200, 0, 20480, 1280]
    with pytest.raises(ValueError):
        regular_grid((0, 0, 1281, 2560), 1280)


def test_optical_masks_reject_cloud_shadow_snow_and_landsat_saturation():
    assert optical_valid("s2", np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])).tolist() == [
        False,
        False,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    qa = np.array([0, 1, 2, 4, 8, 16, 32, 64], dtype=np.uint16)
    valid = optical_valid("landsat", qa, np.array([0, 0, 0, 0, 0, 0, 0, 1]))
    assert valid.tolist() == [True, False, False, False, False, False, False, False]


def test_scene_selection_keeps_spatial_tiles_and_distinct_dates():
    items = [
        {"id": str(i), "properties": {"datetime": d, "eo:cloud_cover": c, "s2:mgrs_tile": t}}
        for i, (d, c, t) in enumerate(
            [
                ("2024-01-01", 1, "A"),
                ("2024-01-02", 2, "A"),
                ("2024-01-01", 3, "B"),
                ("2024-01-03", 4, "B"),
            ]
        )
    ]
    assert len(select_optical_items(items, per_tile=1)) == 2


def test_budget_enforces_reserve_and_peak(tmp_path: Path):
    guard = Budget(tmp_path, max_bytes=8, reserve_bytes=0)
    (tmp_path / "payload").write_bytes(b"12345678")
    with pytest.raises(RuntimeError, match="budget"):
        guard.check(1)


def test_monthly_composite_uses_only_valid_observations(tmp_path: Path):
    import rasterio
    from rasterio.transform import from_origin

    from xuannv_embedding.data_process.regional_acquisition import monthly_optical

    paths = []
    for i, values in enumerate([[[1, -9999], [-9999, 2]], [[3, 4], [-9999, 6]]]):
        path = tmp_path / f"scene{i}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float32",
            crs="EPSG:2193",
            transform=from_origin(0, 20, 10, 10),
            nodata=-9999,
        ) as dst:
            dst.write(np.array(values, dtype=np.float32), 1)
        paths.append(path)
    output = tmp_path / "month.tif"
    receipt = monthly_optical(paths, output, {}, Budget(tmp_path, 1024**3, 0))
    with rasterio.open(output) as src:
        assert src.read(1).tolist() == [[2, 4], [-9999, 4]]
        assert src.read_masks(1).tolist() == [[255, 255], [0, 255]]
    assert receipt["valid_fraction"] == 0.75


def test_window_mosaic_preserves_uncovered_area(tmp_path: Path):
    import rasterio
    from rasterio.transform import from_origin

    from xuannv_embedding.data_process.regional_acquisition import window_mosaic

    source = tmp_path / "source.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=2,
        width=1,
        count=1,
        dtype="uint8",
        crs="EPSG:2193",
        transform=from_origin(0, 20, 10, 10),
        nodata=0,
    ) as dst:
        dst.write(np.ones((2, 1), dtype=np.uint8) * 7, 1)
    target = tmp_path / "mosaic.tif"
    report = window_mosaic(
        [str(source)],
        target,
        {"bounds": [0, 0, 20, 20], "crs": "EPSG:2193"},
        10,
        1,
        "uint8",
        Budget(tmp_path, 1024**3, 0),
    )
    assert report["valid_fraction"] == 0.5
    with rasterio.open(target) as src:
        assert src.read(1).tolist() == [[7, 0], [7, 0]]


def test_budget_uses_unreserved_available_space(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    import xuannv_embedding.data_process.regional_acquisition as acquisition

    monkeypatch.setattr(
        acquisition.os, "statvfs", lambda _: SimpleNamespace(f_bavail=99, f_frsize=1, f_bfree=999)
    )
    with pytest.raises(RuntimeError, match="reserve"):
        Budget(tmp_path, max_bytes=1000, reserve_bytes=100).check()


def test_acquisition_rejects_training_jobs_and_path_escape(tmp_path: Path):
    from xuannv_embedding.data_process.regional_acquisition import validate_jobs

    with pytest.raises(ValueError, match="kind"):
        validate_jobs([{"id": "train", "kind": "train", "output": "model.pt"}], tmp_path)
    with pytest.raises(ValueError, match="output"):
        validate_jobs([{"id": "data", "kind": "vector", "output": "../outside"}], tmp_path)


def test_mosaic_honors_alpha_and_preserves_valid_black_pixels(tmp_path: Path):
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.transform import from_origin

    from xuannv_embedding.data_process.regional_acquisition import window_mosaic

    p = tmp_path / "rgba.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        width=2,
        height=1,
        count=4,
        dtype="uint8",
        crs="EPSG:2193",
        transform=from_origin(0, 10, 10, 10),
    ) as s:
        s.write(np.array([[[0, 50]], [[0, 50]], [[0, 50]], [[255, 0]]], dtype="uint8"))
        s.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    r = window_mosaic(
        [str(p)],
        tmp_path / "target.tif",
        {"bounds": [0, 0, 20, 10], "crs": "EPSG:2193"},
        10,
        3,
        "uint8",
        Budget(tmp_path, 1024**3, 0),
    )
    assert r["valid_fraction"] == 0.5
    with rasterio.open(tmp_path / "target.tif") as s:
        assert s.read_masks(1).tolist() == [[255, 0]]


def test_radar_acquisition_preserves_linear_gamma0_and_masks_nonpositive(
    tmp_path: Path, monkeypatch
):
    import rasterio

    import xuannv_embedding.data_process.regional_acquisition as acq

    array = np.ma.array([[0.01, 0], [0.02, 0.03]], mask=[[False, False], [True, False]])
    monkeypatch.setattr(acq, "read_optical_asset", lambda *args, **kwargs: array)
    item = {"id": "radar", "properties": {"datetime": "2024-01-01T00:00:00Z"}}
    dest = tmp_path / "s1.tif"
    acq.acquire_radar_item(
        item, dest, {"bounds": [0, 0, 20, 20], "crs": "EPSG:2193"}, Budget(tmp_path, 1024**3, 0)
    )
    with rasterio.open(dest) as src:
        assert src.read(1)[0, 0] == pytest.approx(0.01)
        assert src.read_masks(1).tolist() == [[255, 0], [0, 255]]


def test_long_vector_queries_use_post_instead_of_exceeding_server_url_limits(monkeypatch):
    from types import SimpleNamespace

    import xuannv_embedding.data_process.regional_acquisition as acquisition

    calls = []

    def post(url, data, timeout):
        calls.append(data)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"features": []})

    monkeypatch.setattr(acquisition.requests, "post", post)
    result = acquisition.fetch_json(
        "https://example.org/query", {"objectIds": ",".join(map(str, range(500)))}
    )
    assert result == {"features": []}
    assert len(calls) == 1


def test_acquisition_rejects_unknown_job_configuration(tmp_path: Path):
    from xuannv_embedding.data_process.regional_acquisition import validate_jobs

    with pytest.raises(ValueError, match="unknown"):
        validate_jobs(
            [
                {
                    "id": "sample",
                    "kind": "vector",
                    "output": "ref.json",
                    "url": "https://example.org",
                    "unkown": True,
                }
            ],
            tmp_path,
        )


def test_raster_jobs_execute_in_isolated_processes(tmp_path):
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    import rasterio
    from rasterio.transform import from_origin

    from xuannv_embedding.data_process.regional_acquisition import _execute_job

    source = tmp_path / "source.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:2193",
        transform=from_origin(0, 20, 10, 10),
    ) as f:
        f.write(np.full((2, 2), 7, dtype="uint8"), 1)
    spec = {
        "root": str(tmp_path),
        "bounds": [0, 0, 20, 20],
        "crs": "EPSG:2193",
        "max_gib": 1,
        "min_free_gib": 0,
    }
    jobs = [
        {
            "id": f"job_{i}",
            "kind": "raster",
            "output": f"output_{i}.tif",
            "assets": [str(source)],
            "resolution": 10,
            "count": 1,
            "dtype": "uint8",
        }
        for i in range(2)
    ]
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_execute_job, [(spec, job) for job in jobs]))
    assert all(r["state"] == "complete" for r in results)
    assert all(r["worker_pid"] != os.getpid() for r in results)
    for i in range(2):
        with rasterio.open(tmp_path / f"output_{i}.tif") as f:
            assert np.all(f.read(1) == 7)


def test_raster_reads_allocate_cache_in_bytes(tmp_path, monkeypatch):
    import contextlib

    import planetary_computer
    import rasterio
    from rasterio.env import get_gdal_config
    from rasterio.transform import from_origin

    import xuannv_embedding.data_process.regional_acquisition as acquisition

    source = tmp_path / "source.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:2193",
        transform=from_origin(0, 20, 10, 10),
    ) as dst:
        dst.write(np.full((2, 2), 7, dtype="uint8"), 1)
    original_env = rasterio.Env

    @contextlib.contextmanager
    def checked_env(**kwargs):
        with original_env(**kwargs):
            assert get_gdal_config("GDAL_CACHEMAX") >= 128 * 1024**2
            yield

    monkeypatch.setattr(acquisition.rasterio, "Env", checked_env)
    monkeypatch.setattr(planetary_computer, "sign", lambda url: url)
    result = acquisition.read_optical_asset(
        {"assets": {"band": {"href": str(source)}}},
        "band",
        {"bounds": [0, 0, 20, 20], "crs": "EPSG:2193"},
    )
    assert np.all(result == 7)


def test_same_job_cannot_write_concurrently(tmp_path, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import xuannv_embedding.data_process.regional_acquisition as acquisition

    active = maximum = 0
    mutex = threading.Lock()

    def execute(payload):
        nonlocal active, maximum
        with mutex:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with mutex:
            active -= 1
        return {"state": "complete"}

    monkeypatch.setattr(acquisition, "_execute_job_unlocked", execute)
    payload = ({"root": str(tmp_path)}, {"id": "same_job"})
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(acquisition._execute_job, [payload, payload]))
    assert all(r["state"] == "complete" for r in results)
    assert maximum == 1
