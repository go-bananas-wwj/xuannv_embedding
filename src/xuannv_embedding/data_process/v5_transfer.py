"""Bounded parallel byte ranges within at most two package downloads."""

from __future__ import annotations

import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event

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
    route: str = "environment",
) -> dict:
    if route not in {"environment", "direct"}:
        raise ValueError("invalid download route")
    if workers < 1 or workers > 8 or chunk_bytes <= 0:
        raise ValueError("invalid chunk transfer limits")
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / spec.archive
    if final.exists():
        return {**download_archive(spec, directory, revision), "download_route": route}
    parts = directory / (spec.archive + ".parts")
    parts.mkdir(exist_ok=True)
    lock = parts / "source.json"
    fingerprint = {"revision": revision, **spec.__dict__, "chunk_bytes": chunk_bytes}
    if lock.exists() and json.loads(lock.read_text()) != fingerprint:
        raise ValueError("partial chunk source fingerprint changed")
    if not lock.exists() and any(parts.iterdir()):
        raise ValueError("partial chunks have no pinned source fingerprint")
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
    # Explicit request proxies override environment settings, including on CDN redirects.
    # Keep trust_env enabled so configured CA bundles and TLS verification still apply.
    transport = {"proxies": {"http": "", "https": "", "all": ""}} if route == "direct" else {}

    stopped = Event()

    def fetch(bounds):
        start, end = bounds
        target = parts / f"{start:012d}.part"
        temporary = target.with_suffix(".partial")
        checksum = target.with_suffix(".json")
        if stopped.is_set():
            return 0
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
                if stopped.is_set():
                    return 0
                response = None
                received = temporary.stat().st_size if temporary.exists() else 0
                if received > end - start + 1:
                    raise ValueError("partial chunk exceeds pinned range")
                requested_start = start + received
                try:
                    if requested_start <= end:
                        response = client.get(
                            API + "/repo",
                            params={"Revision": revision, "FilePath": "packages/" + spec.archive},
                            headers={"Range": f"bytes={requested_start}-{end}"},
                            stream=True,
                            timeout=(30, 90),
                            **transport,
                        )
                        check_response(response)
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                        )
                        if (
                            response.status_code != 206
                            or match is None
                            or tuple(map(int, match.groups())) != (requested_start, end, spec.bytes)
                        ):
                            raise ValueError("chunk Content-Range disagrees with request")
                        # Append only after the server proves the exact remaining range.
                        with temporary.open("ab") as output:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if output.tell() + len(chunk) > end - start + 1:
                                    raise ValueError("chunk exceeds declared range")
                                output.write(chunk)
                    if temporary.stat().st_size != end - start + 1:
                        raise ConnectionError("truncated byte range")
                    digest = sha256(temporary)
                    temporary.replace(target)
                    write_json(checksum, {"sha256": digest})
                    return attempt
                except Exception as exc:
                    # Never persist exception text, URLs, headers, or response bodies.
                    event = {
                        "run_started_at": started,
                        "at": now(),
                        "attempt": attempt + 1,
                        "range_start": start,
                        "requested_start": requested_start,
                        "range_end": end,
                        "received_bytes": temporary.stat().st_size if temporary.exists() else 0,
                        "http_status": response.status_code if response is not None else None,
                        "error_type": type(exc).__name__,
                    }
                    with target.with_suffix(".events.jsonl").open("a") as log:
                        log.write(json.dumps(event) + "\n")
                    if not isinstance(exc, (requests.RequestException, ConnectionError)):
                        raise
                    if attempt == 3:
                        raise RuntimeError(
                            "byte range failed after three retries; see events"
                        ) from None
                    time.sleep(2**attempt)
                finally:
                    if response is not None:
                        response.close()
        except Exception:
            # Set this in the worker before the executor can start another queued request.
            stopped.set()
            raise
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(fetch, task) for task in tasks]
        try:
            retries = sum(job.result() for job in as_completed(jobs))
        except Exception:
            stopped.set()
            for job in jobs:
                job.cancel()
            raise
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
    events = directory / (spec.archive + ".transfer.jsonl")
    with events.open("a") as output:
        for path in sorted(parts.glob("*.events.jsonl")):
            output.write(path.read_text())
    shutil.rmtree(parts)
    return {
        **spec.__dict__,
        "status": "complete",
        "actual_bytes": spec.bytes,
        "retries": retries,
        "started_at": started,
        "finished_at": now(),
        "range_workers": workers,
        "download_route": route,
    }
