"""Bounded parallel byte ranges within at most two package downloads."""

from __future__ import annotations

import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from xuannv_embedding.data_process import v5_sources
from xuannv_embedding.data_process.v5_sources import (
    API,
    ArchiveSpec,
    check_response,
    download_archive,
    now,
    sha256,
    write_json,
)


def download_chunked(
    spec: ArchiveSpec,
    directory: Path,
    revision: str,
    *,
    workers: int = 8,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> dict:
    if workers < 1 or workers > 8 or chunk_bytes <= 0:
        raise ValueError("invalid chunk transfer limits")
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / spec.archive
    if final.exists():
        return download_archive(spec, directory, revision)
    parts = directory / (spec.archive + ".parts")
    parts.mkdir(exist_ok=True)
    lock = parts / "source.json"
    fingerprint = {"revision": revision, **spec.__dict__, "chunk_bytes": chunk_bytes}
    if lock.exists() and json.loads(lock.read_text()) != fingerprint:
        raise ValueError("partial chunk source fingerprint changed")
    write_json(lock, fingerprint)
    contiguous = final.with_name(final.name + ".partial")
    tasks = [
        (start, min(start + chunk_bytes, spec.bytes) - 1)
        for start in range(0, spec.bytes, chunk_bytes)
    ]
    # Migrate complete ranges from an earlier contiguous download, keeping its original intact.
    if contiguous.exists():
        with contiguous.open("rb") as original:
            for start, end in tasks:
                target = parts / f"{start:012d}.part"
                if end >= contiguous.stat().st_size:
                    break
                if target.exists():
                    continue
                original.seek(start)
                target.write_bytes(original.read(end - start + 1))
                write_json(target.with_suffix(".json"), {"sha256": sha256(target)})
    started = now()

    def fetch(bounds):
        start, end = bounds
        target = parts / f"{start:012d}.part"
        checksum = target.with_suffix(".json")
        if (
            target.exists()
            and checksum.exists()
            and target.stat().st_size == end - start + 1
            and sha256(target) == json.loads(checksum.read_text())["sha256"]
        ):
            return 0
        client = v5_sources.session()
        try:
            for attempt in range(4):
                try:
                    response = client.get(
                        API + "/repo",
                        params={"Revision": revision, "FilePath": "packages/" + spec.archive},
                        headers={"Range": f"bytes={start}-{end}"},
                        stream=True,
                        timeout=(30, 90),
                    )
                    temporary = target.with_suffix(".partial")
                    try:
                        check_response(response)
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                        )
                        if (
                            response.status_code != 206
                            or match is None
                            or tuple(map(int, match.groups())) != (start, end, spec.bytes)
                        ):
                            raise ValueError("chunk Content-Range disagrees with request")
                        with temporary.open("wb") as output:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if output.tell() + len(chunk) > end - start + 1:
                                    raise ValueError("chunk exceeds declared range")
                                output.write(chunk)
                    finally:
                        response.close()
                    if temporary.stat().st_size != end - start + 1:
                        raise ConnectionError("truncated byte range")
                    digest = sha256(temporary)
                    temporary.replace(target)
                    write_json(checksum, {"sha256": digest})
                    return attempt
                except (requests.RequestException, ConnectionError):
                    if attempt == 3:
                        raise RuntimeError("byte range failed after three retries") from None
                    time.sleep(2**attempt)
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        retries = sum(pool.map(fetch, tasks))
    assembled = final.with_name(final.name + ".assembled")
    with assembled.open("wb") as output:
        for start, _ in tasks:
            with (parts / f"{start:012d}.part").open("rb") as part:
                shutil.copyfileobj(part, output, length=1024 * 1024)
    if assembled.stat().st_size != spec.bytes or sha256(assembled) != spec.sha256:
        assembled.unlink()
        raise ValueError("assembled archive failed published SHA256; ranges kept for diagnosis")
    assembled.replace(final)
    if contiguous.exists():
        contiguous.unlink()
    shutil.rmtree(parts)
    return {
        **spec.__dict__,
        "status": "complete",
        "actual_bytes": spec.bytes,
        "retries": retries,
        "started_at": started,
        "finished_at": now(),
        "range_workers": workers,
    }
