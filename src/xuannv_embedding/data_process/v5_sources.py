"""Pinned ModelScope archives: bounded downloads and recoverable safe extraction."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests

REPO = "ptyzjr/Jilin1_Aligned_Train"
API = f"https://www.modelscope.cn/api/v1/datasets/{REPO}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def session() -> requests.Session:
    result = requests.Session()
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if token:
        result.headers["Authorization"] = f"Bearer {token}"
    return result


def check_response(response) -> None:
    if response.status_code in {401, 403}:
        raise PermissionError(f"ModelScope authentication failed (HTTP {response.status_code})")
    if response.status_code not in {200, 206}:
        # Do not interpolate URLs, response bodies, or request headers: they may contain secrets.
        raise ConnectionError(f"ModelScope HTTP {response.status_code}")


@dataclass(frozen=True)
class ArchiveSpec:
    archive: str
    bytes: int
    sha256: str
    tiff_count: int


def parse_manifests(archives: str, checksums: str) -> list[ArchiveSpec]:
    hashes = {}
    for line in checksums.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = PurePosixPath(name.lstrip("* ")).name
        if name in hashes:
            raise ValueError("duplicate checksum entry")
        hashes[name] = digest
    specs = []
    seen = set()
    for row in csv.DictReader(io.StringIO(archives), delimiter="\t"):
        name = row["archive"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.tar\.gz", name) or name in seen:
            raise ValueError("unsafe or duplicate archive")
        seen.add(name)
        digest = row["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or hashes.get(name) != digest:
            raise ValueError("manifest checksum disagreement")
        spec = ArchiveSpec(name, int(row["bytes"]), digest, int(row["tiff_count"]))
        if spec.bytes <= 0 or spec.tiff_count <= 0:
            raise ValueError("invalid archive count or size")
        specs.append(spec)
    if not specs or set(hashes) != seen:
        raise ValueError("manifest archive set disagreement")
    return specs


def lock_source(root: Path) -> dict:
    lock_path = root / "manifests/source.lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        for name, digest in lock["manifest_sha256"].items():
            if sha256(root / "manifests" / name) != digest:
                raise ValueError("locked manifest changed")
        return lock
    with session() as client:
        response = client.get(API + "/repo/tree", params={"Revision": "master"}, timeout=60)
        check_response(response)
        payload = response.json()
        if payload.get("Code") != 200:
            raise ValueError("ModelScope tree lookup failed")
        files = payload["Data"]["Files"]
        revision = next(f["Revision"] for f in files if f["Name"] == "ARCHIVES.tsv")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("source revision must be an immutable commit")
        manifests = {}
        for name in ["ARCHIVES.tsv", "ARCHIVE_INDEX.tsv", "SHA256SUMS", "DOWNLOAD.md"]:
            response = client.get(
                API + "/repo", params={"Revision": revision, "FilePath": name}, timeout=60
            )
            check_response(response)
            manifests[name] = response.content
        specs = parse_manifests(
            manifests["ARCHIVES.tsv"].decode(), manifests["SHA256SUMS"].decode()
        )
    directory = root / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in manifests.items():
        (directory / name).write_bytes(content)
    lock = {
        "schema": "xuannv.source-lock.v5",
        "repository": REPO,
        "revision": revision,
        "created_at": now(),
        "archives": [asdict(spec) for spec in specs],
        "manifest_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in manifests.items()
        },
    }
    write_json(lock_path, lock)
    return lock


def download_archive(spec: ArchiveSpec, directory: Path, revision: str, *, session=None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / spec.archive
    started = now()
    if destination.exists():
        if destination.stat().st_size != spec.bytes or sha256(destination) != spec.sha256:
            raise ValueError("existing completed archive failed integrity check")
        return {
            **asdict(spec),
            "status": "complete",
            "actual_bytes": spec.bytes,
            "retries": 0,
            "started_at": started,
            "finished_at": now(),
        }
    partial = destination.with_name(destination.name + ".partial")
    own_session = session is None
    client = globals()["session"]() if own_session else session
    try:
        for attempt in range(4):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset >= spec.bytes:
                    if offset == spec.bytes and sha256(partial) == spec.sha256:
                        partial.replace(destination)
                        break
                    partial.unlink()
                    offset = 0
                response = client.get(
                    API + "/repo",
                    params={"Revision": revision, "FilePath": "packages/" + spec.archive},
                    headers={"Range": f"bytes={offset}-"} if offset else {},
                    timeout=(30, 90),
                    stream=True,
                )
                try:
                    check_response(response)
                    if response.status_code == 206:
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                        )
                        if (
                            match is None
                            or int(match[1]) != offset
                            or int(match[3]) != spec.bytes
                            or int(match[2]) != spec.bytes - 1
                        ):
                            raise ValueError("invalid Content-Range")
                    else:
                        offset = 0
                    with partial.open("ab" if offset else "wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            if output.tell() + len(chunk) > spec.bytes:
                                raise ValueError("response exceeds expected archive size")
                            output.write(chunk)
                finally:
                    response.close()
                if partial.stat().st_size != spec.bytes:
                    raise ConnectionError("incomplete archive response")
                if sha256(partial) != spec.sha256:
                    partial.unlink()
                    raise ConnectionError("archive checksum mismatch; discarded partial")
                partial.replace(destination)
                break
            except (requests.RequestException, ConnectionError):
                if attempt == 3:
                    raise RuntimeError("archive download failed after three retries") from None
                time.sleep(2**attempt)
        return {
            **asdict(spec),
            "status": "complete",
            "actual_bytes": destination.stat().st_size,
            "retries": attempt,
            "started_at": started,
            "finished_at": now(),
        }
    finally:
        if own_session:
            client.close()


def extract_archive(path: Path, destination: Path, spec: ArchiveSpec) -> dict:
    if path.stat().st_size != spec.bytes or sha256(path) != spec.sha256:
        raise ValueError("archive integrity check failed before extraction")
    marker = destination.parent / "manifests" / "extracted" / (spec.archive + ".json")
    if marker.exists():
        result = json.loads(marker.read_text())
        if result["sha256"] != spec.sha256:
            raise ValueError("extraction marker source changed")
        # Full file existence and decoding are checked by the catalog stage.
        return result
    temporary = destination.parent / (".extract-" + spec.archive)
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    tiffs = 0
    files = 0
    seen = set()
    try:
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in member.name
                    or not relative.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("unsafe archive member")
                target = temporary.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if str(relative) in seen:
                    raise ValueError("unsafe duplicate archive member")
                seen.add(str(relative))
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("unreadable archive member")
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if target.stat().st_size != member.size:
                    raise ValueError("truncated archive member")
                files += 1
                tiffs += target.suffix.lower() in {".tif", ".tiff"}
        if tiffs != spec.tiff_count:
            raise ValueError("archive TIFF count disagrees with manifest")
        destination.mkdir(parents=True, exist_ok=True)
        # Publish one complete patch directory at a time; marker is committed only at the end.
        for patch in temporary.iterdir():
            target = destination / patch.name
            if not target.exists():
                patch.replace(target)
                continue
            if target.is_symlink():
                raise ValueError("unsafe destination symlink")
            candidates = patch.rglob("*") if patch.is_dir() else [patch]
            for candidate in candidates:
                if candidate.is_dir():
                    continue
                output = destination / candidate.relative_to(temporary)
                if any(parent.is_symlink() for parent in output.parents):
                    raise ValueError("unsafe destination symlink")
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists():
                    if output.is_symlink() or sha256(output) != sha256(candidate):
                        raise ValueError("non-equivalent existing extracted file")
                else:
                    candidate.replace(output)
        result = {**asdict(spec), "file_count": files, "status": "complete", "finished_at": now()}
        write_json(marker, result)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
