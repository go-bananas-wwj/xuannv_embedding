import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pandas as pd
import pytest

from xuannv_embedding.data_process import v5_cli, v5_sources


def setup_run(tmp_path, monkeypatch, *, count=8, download_hook=None, validate_hook=None):
    args = SimpleNamespace(
        source_root=tmp_path / "source",
        dataset_root=tmp_path / "data",
        report_root=tmp_path / "report",
        stage="ingest",
        limit=None,
        download_route="direct",
    )
    args.source_root.mkdir()
    specs = [
        v5_sources.ArchiveSpec(f"p{i}.tar.gz", 3, hashlib.sha256(b"abc").hexdigest(), 1)
        for i in range(1, count + 1)
    ]
    source = {"archives": [s.__dict__ for s in specs], "revision": "fixed", "manifest_sha256": {}}

    def download(spec, directory, revision, **kwargs):
        assert revision == "fixed" and kwargs["route"] == "direct"
        if download_hook:
            download_hook(spec.archive)
        return {**spec.__dict__, "status": "complete", "actual_bytes": 3, "retries": 0}

    monkeypatch.setattr(v5_cli, "download_chunked", download)
    monkeypatch.setattr(v5_cli, "extract_archive", lambda *a: {"tiff_count": 1})
    monkeypatch.setattr(
        v5_cli, "validate_package", lambda a, s: validate_hook(s.archive) if validate_hook else None
    )
    return args, source


def test_ingest_prefetches_only_next_pair_while_processing_and_waits_for_final_validation(
    tmp_path, monkeypatch
):
    processing = threading.Event()
    release = threading.Event()
    next_pair = threading.Event()
    started = set()
    lock = threading.Lock()

    def download(name):
        with lock:
            started.add(name)
            if {"p5.tar.gz", "p6.tar.gz"} <= started:
                next_pair.set()

    def validate(name):
        if name == "p3.tar.gz":
            processing.set()
            assert release.wait(10)

    args, source = setup_run(tmp_path, monkeypatch, download_hook=download, validate_hook=validate)
    with ThreadPoolExecutor(max_workers=1) as runner:
        future = runner.submit(v5_cli.acquire, args, source)
        try:
            assert processing.wait(5)
            assert next_pair.wait(2), "next pair blocked behind current extraction"
            assert not future.done()
            with lock:
                assert "p7.tar.gz" not in started and "p8.tar.gz" not in started
            assert not (args.report_root / "archive_integrity.json").exists()
        finally:
            release.set()
        future.result(timeout=10)
    state = json.loads((args.report_root / "acquisition_progress.json").read_text())
    assert state["status"] == "complete"
    assert state["downloaded_archives"] == state["validated_archives"] == 8
    assert state["maximum_download_packages"] == 2
    assert state["maximum_prefetch_pairs"] == 1
    assert state["training_authorized"] is False
    assert json.loads((args.report_root / "archive_integrity.json").read_text())["fully_acquired"]


def test_ingest_does_not_prefetch_beyond_initial_pilot_until_both_packages_validate(
    tmp_path, monkeypatch
):
    validating_second = threading.Event()
    release = threading.Event()
    started = set()

    def validate(name):
        if name == "p2.tar.gz":
            validating_second.set()
            assert release.wait(10)

    args, source = setup_run(
        tmp_path, monkeypatch, count=4, download_hook=started.add, validate_hook=validate
    )
    with ThreadPoolExecutor(max_workers=1) as runner:
        future = runner.submit(v5_cli.acquire, args, source)
        try:
            assert validating_second.wait(5)
            assert started == {"p1.tar.gz", "p2.tar.gz"}
        finally:
            release.set()
        future.result(timeout=10)


@pytest.mark.parametrize("failed_stage", ["download", "validate"])
def test_ingest_failure_drains_existing_work_without_starting_later_pairs_or_claiming_complete(
    tmp_path, monkeypatch, failed_stage
):
    started = set()

    def download(name):
        started.add(name)
        if failed_stage == "download" and name == "p5.tar.gz":
            raise PermissionError("private credentials must not appear in pipeline state")

    def validate(name):
        if failed_stage == "validate" and name == "p3.tar.gz":
            raise ValueError("private source path must not appear in pipeline state")

    args, source = setup_run(tmp_path, monkeypatch, download_hook=download, validate_hook=validate)
    with pytest.raises(RuntimeError if failed_stage == "download" else ValueError):
        v5_cli.acquire(args, source)
    assert "p7.tar.gz" not in started and "p8.tar.gz" not in started
    state = json.loads((args.report_root / "acquisition_progress.json").read_text())
    assert state["status"] == "failed" and state["training_authorized"] is False
    assert state["validated_archives"] < 8
    assert "private" not in json.dumps(state)
    assert not (args.report_root / "archive_integrity.json").exists()
    statuses = pd.read_parquet(args.source_root / "manifests/download_status.parquet")
    if failed_stage == "download":
        assert statuses.set_index("archive").loc["p5.tar.gz", "status"] == "failed"


def test_ingest_limited_selection_is_not_full_source_acquisition(tmp_path, monkeypatch):
    args, source = setup_run(tmp_path, monkeypatch, count=4)
    args.limit = 1
    v5_cli.acquire(args, source)
    result = json.loads((args.report_root / "archive_integrity.json").read_text())
    assert result["selected_archives"] == 1
    assert result["total_archives"] == 4
    assert result["fully_acquired"] is False
    state = json.loads((args.report_root / "acquisition_progress.json").read_text())
    assert state["validated_archives"] == 1


def test_ingest_invalidates_stale_completion_but_preserves_its_history(tmp_path, monkeypatch):
    def broken(_):
        raise ValueError("bad source")

    args, source = setup_run(tmp_path, monkeypatch, validate_hook=broken)
    old = {"selected_archives": 8, "total_archives": 8, "fully_acquired": True}
    path = args.report_root / "archive_integrity.json"
    v5_sources.write_json(path, old)
    old_digest = v5_sources.sha256(path)
    with pytest.raises(ValueError):
        v5_cli.acquire(args, source)
    assert not path.exists()
    assert (
        json.loads((args.report_root / "acquisition_history" / f"{old_digest}.json").read_text())
        == old
    )


def test_ingest_runs_real_extraction_and_full_tiff_decode_for_every_selected_package(
    tmp_path, monkeypatch
):
    import io
    import tarfile

    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    args = SimpleNamespace(
        source_root=tmp_path / "source",
        report_root=tmp_path / "report",
        stage="ingest",
        limit=None,
        download_route="direct",
    )
    packages = args.source_root / "packages"
    packages.mkdir(parents=True)
    specs = []
    index = []
    for i in range(1, 5):
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff",
                width=8,
                height=8,
                count=1,
                dtype="uint16",
                crs="EPSG:32650",
                transform=from_origin(400000, 4000000, 5, 5),
            ) as dst:
                dst.write(np.full((1, 8, 8), i, dtype="uint16"))
            data = memory.read()
        archive = packages / f"p{i}.tar.gz"
        with tarfile.open(archive, "w:gz") as target:
            member = tarfile.TarInfo(f"patch{i}/image.tif")
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))
        specs.append(
            v5_sources.ArchiveSpec(
                archive.name, archive.stat().st_size, v5_sources.sha256(archive), 1
            )
        )
        index.append({"archive": archive.name, "patchid": f"patch{i}"})
    (args.source_root / "manifests").mkdir()
    pd.DataFrame(index).to_csv(
        args.source_root / "manifests/ARCHIVE_INDEX.tsv", sep="\t", index=False
    )
    monkeypatch.setattr(
        v5_sources,
        "session",
        lambda: pytest.fail("verified local archives must not access network"),
    )
    v5_cli.acquire(
        args, {"archives": [s.__dict__ for s in specs], "revision": "fixed", "manifest_sha256": {}}
    )
    for i, spec in enumerate(specs, 1):
        marker = json.loads(
            (args.report_root / "integrity_shards" / (spec.archive + ".json")).read_text()
        )
        assert marker["decoded_tiffs"] == marker["expected_tiffs"] == 1
        assert marker["status"] == "complete" and marker["failures"] == []
        assert marker["sha256"] == spec.sha256
        with rasterio.open(args.source_root / f"extracted/patch{i}/image.tif") as image:
            assert (image.read() == i).all()
