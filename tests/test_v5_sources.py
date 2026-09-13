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
