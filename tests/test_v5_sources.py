import hashlib
import io
import tarfile

import pytest

from xuannv_embedding.data_process.v5_sources import (
    ArchiveSpec,
    download_archive,
    extract_archive,
    parse_manifests,
)


def test_manifest_rejects_hash_disagreement():
    archives = "archive\tbytes\tsha256\ttiff_count\npart.tar.gz\t3\t" + "a" * 64 + "\t1\n"
    with pytest.raises(ValueError, match="checksum"):
        parse_manifests(archives, "b" * 64 + "  packages/part.tar.gz\n")


def test_manifest_rejects_duplicate_archive():
    row = "part.tar.gz\t3\t" + "a" * 64 + "\t1\n"
    with pytest.raises(ValueError, match="duplicate"):
        parse_manifests(
            "archive\tbytes\tsha256\ttiff_count\n" + row * 2, "a" * 64 + "  packages/part.tar.gz\n"
        )


def _archive(tmp_path, name, kind=None):
    path = tmp_path / "part.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        if kind:
            member.type = kind
            member.linkname = "/outside"
        else:
            member.size = 3
        archive.addfile(member, None if kind else io.BytesIO(b"abc"))
    return path, ArchiveSpec(
        path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest(), 1
    )


@pytest.mark.parametrize(
    "name,kind",
    [("../outside.tif", None), ("/outside.tif", None), ("patch/link.tif", tarfile.SYMTYPE)],
)
def test_extract_rejects_unsafe_members(tmp_path, name, kind):
    path, spec = _archive(tmp_path, name, kind)
    with pytest.raises(ValueError, match="unsafe"):
        extract_archive(path, tmp_path / "out", spec)
    assert not (tmp_path / "outside.tif").exists()


def test_extract_checks_count_and_is_repeatable(tmp_path):
    path, spec = _archive(tmp_path, "patch/image.tif")
    first = extract_archive(path, tmp_path / "out", spec)
    second = extract_archive(path, tmp_path / "out", spec)
    assert first["tiff_count"] == second["tiff_count"] == 1
    assert (tmp_path / "out/patch/image.tif").read_bytes() == b"abc"


class Response:
    def __init__(self, status, data, headers=None):
        self.status_code = status
        self.data = data
        self.headers = headers or {}

    def iter_content(self, chunk_size):
        yield self.data

    def close(self):
        pass


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.ranges = []

    def get(self, url, **kwargs):
        self.ranges.append(kwargs.get("headers", {}).get("Range"))
        return next(self.responses)


def test_download_resumes_only_correct_range(tmp_path):
    content = b"abcdef"
    spec = ArchiveSpec("p.tar.gz", 6, hashlib.sha256(content).hexdigest(), 1)
    (tmp_path / "p.tar.gz.partial").write_bytes(b"abc")
    session = Session([Response(206, b"def", {"Content-Range": "bytes 3-5/6"})])
    result = download_archive(spec, tmp_path, "revision", session=session)
    assert result["status"] == "complete"
    assert session.ranges == ["bytes=3-"]
    assert (tmp_path / "p.tar.gz").read_bytes() == content


def test_download_range_ignored_restarts_without_appending(tmp_path):
    content = b"abcdef"
    spec = ArchiveSpec("p.tar.gz", 6, hashlib.sha256(content).hexdigest(), 1)
    (tmp_path / "p.tar.gz.partial").write_bytes(b"abc")
    session = Session([Response(200, content)])
    download_archive(spec, tmp_path, "revision", session=session)
    assert (tmp_path / "p.tar.gz").read_bytes() == content


def test_download_rejects_wrong_range_and_auth_without_retries(tmp_path):
    spec = ArchiveSpec("p.tar.gz", 6, hashlib.sha256(b"abcdef").hexdigest(), 1)
    (tmp_path / "p.tar.gz.partial").write_bytes(b"abc")
    session = Session([Response(206, b"def", {"Content-Range": "bytes 2-4/6"})])
    with pytest.raises(ValueError, match="Content-Range"):
        download_archive(spec, tmp_path, "revision", session=session)
    assert not (tmp_path / "p.tar.gz").exists()
    session = Session([Response(401, b"")])
    with pytest.raises(PermissionError):
        download_archive(spec, tmp_path, "revision", session=session)
    assert len(session.ranges) == 1


def test_chunk_download_validates_range_and_checksum(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_sources
    from xuannv_embedding.data_process.v5_transfer import download_chunked

    data = b"abcdef"

    class RangeSession:
        def get(self, url, **kwargs):
            header = kwargs["headers"]["Range"]
            start, end = (int(x) for x in header.removeprefix("bytes=").split("-"))
            return Response(
                206, data[start : end + 1], {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
            )

        def close(self):
            pass

    monkeypatch.setattr(v5_sources, "session", RangeSession)
    spec = ArchiveSpec("p.tar.gz", len(data), hashlib.sha256(data).hexdigest(), 1)
    result = download_chunked(spec, tmp_path, "revision", workers=2, chunk_bytes=2)
    assert result["status"] == "complete"
    assert (tmp_path / "p.tar.gz").read_bytes() == data
    assert not (tmp_path / "p.tar.gz.parts").exists()


def test_chunk_retry_resumes_written_bytes_and_preserves_range_validation(tmp_path, monkeypatch):
    import requests

    from xuannv_embedding.data_process import v5_sources, v5_transfer

    class Interrupted(Response):
        def iter_content(self, chunk_size):
            yield b"abc"
            raise requests.ConnectionError("private signed URL must not be logged")

    session = Session(
        [
            Interrupted(206, b"", {"Content-Range": "bytes 0-5/6"}),
            Response(206, b"def", {"Content-Range": "bytes 3-5/6"}),
        ]
    )
    session.close = lambda: None
    monkeypatch.setattr(v5_sources, "session", lambda: session)
    monkeypatch.setattr(v5_transfer.time, "sleep", lambda _: None)
    spec = ArchiveSpec("p.tar.gz", 6, hashlib.sha256(b"abcdef").hexdigest(), 1)
    result = v5_transfer.download_chunked(spec, tmp_path, "revision", workers=1)
    assert session.ranges == ["bytes=0-5", "bytes=3-5"]
    assert result["retries"] == 1
    assert (tmp_path / "p.tar.gz").read_bytes() == b"abcdef"


def test_chunk_failures_record_safe_diagnostics_and_bound_retries(tmp_path, monkeypatch):
    import json

    from xuannv_embedding.data_process import v5_sources, v5_transfer

    session = Session([Response(503, b"secret response") for _ in range(4)])
    session.close = lambda: None
    monkeypatch.setattr(v5_sources, "session", lambda: session)
    monkeypatch.setattr(v5_transfer.time, "sleep", lambda _: None)
    spec = ArchiveSpec("p.tar.gz", 6, hashlib.sha256(b"abcdef").hexdigest(), 1)
    with pytest.raises(RuntimeError):
        v5_transfer.download_chunked(spec, tmp_path, "revision", workers=1)
    events = [
        json.loads(line)
        for line in (tmp_path / "p.tar.gz.parts/000000000000.events.jsonl").read_text().splitlines()
    ]
    assert len(session.ranges) == len(events) == 4
    assert all(e["http_status"] == 503 and e["error_type"] == "ConnectionError" for e in events)
    assert [e["attempt"] for e in events] == [1, 2, 3, 4]
    assert "secret" not in json.dumps(events)


def test_chunk_auth_failure_stops_queued_ranges(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_sources, v5_transfer

    session = Session([Response(403, b"private response")])
    session.close = lambda: None
    monkeypatch.setattr(v5_sources, "session", lambda: session)
    spec = ArchiveSpec("p.tar.gz", 6, hashlib.sha256(b"abcdef").hexdigest(), 1)
    with pytest.raises(PermissionError):
        v5_transfer.download_chunked(spec, tmp_path, "revision", workers=1, chunk_bytes=2)
    assert len(session.ranges) == 1


def test_chunk_resume_rejects_wrong_range_without_appending(tmp_path, monkeypatch):
    from xuannv_embedding.data_process import v5_sources, v5_transfer

    session = Session([Response(206, b"abc", {"Content-Range": "bytes 0-2/6"})])
    session.close = lambda: None
    monkeypatch.setattr(v5_sources, "session", lambda: session)
    parts = tmp_path / "p.tar.gz.parts"
    parts.mkdir()
    partial = parts / "000000000000.partial"
    partial.write_bytes(b"abc")
    spec = ArchiveSpec("p.tar.gz", 6, hashlib.sha256(b"abcdef").hexdigest(), 1)
    # Production partials always have a pinned-source marker.
    v5_sources.write_json(
        parts / "source.json",
        {"revision": "revision", **spec.__dict__, "chunk_bytes": 16 * 1024 * 1024},
    )
    with pytest.raises(ValueError, match="Content-Range"):
        v5_transfer.download_chunked(spec, tmp_path, "revision", workers=1)
    assert partial.read_bytes() == b"abc"
    assert session.ranges == ["bytes=3-5"]
