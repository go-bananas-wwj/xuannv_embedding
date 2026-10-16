"""Identity-checked artifact pruning with an append-only, fsynced journal."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path

ROLES = frozenset(
    {
        "derived_embeddings",
        "derived_features",
        "training_cache",
        "model_predictions",
        "temporary_artifact",
        "obsolete_visualization",
        "redundant_log",
        "checkpoint_redundant",
    }
)


def file_identity(path: Path) -> dict:
    path = Path(path).absolute()
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("artifact must be a regular file")
    return {
        "path": str(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "allocated_bytes": info.st_blocks * 512,
    }


def _verify(entry):
    path = Path(entry["path"])
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must be absolute and canonical")
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError("symlink artifacts and ancestors are forbidden")
    try:
        actual = file_identity(path)
    except OSError as exc:
        raise ValueError("registered artifact is missing") from exc
    for key in ["device", "inode", "size", "mtime_ns"]:
        if actual[key] != entry[key]:
            raise ValueError("registered artifact identity changed")
    if "sha256" in entry:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(data)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError("registered artifact content changed")
    return path


def validate_manifest(path: Path, journal: Path) -> dict:
    document = json.loads(Path(path).read_text())
    if document.get("protocol") != "artifact-retention-v1":
        raise ValueError("invalid retention protocol")
    roots = [Path(p).resolve(strict=True) for p in document["allowed_roots"]]
    protected = [Path(p).resolve() for p in document["protected"]]
    protected += [Path(path).resolve(), Path(journal).resolve()]
    requirements = document["requirements"]
    for entry in requirements:
        protected.append(_verify(entry))
    seen = set()
    for entry in document["files"]:
        if entry["role"] not in ROLES:
            raise ValueError("unapproved artifact role")
        artifact = _verify(entry)
        if artifact in seen or not any(artifact.is_relative_to(root) for root in roots):
            raise ValueError("duplicate artifact or path outside approved roots")
        if any(artifact == p or artifact.is_relative_to(p) for p in protected):
            raise ValueError("cannot delete a protected artifact")
        seen.add(artifact)
    return document


def apply_manifest(path: Path, journal: Path) -> dict:
    document = validate_manifest(path, journal)
    journal = Path(journal)
    journal.parent.mkdir(parents=True, exist_ok=True)
    deleted = allocated = logical = 0
    with journal.open("x") as stream:
        for entry in document["files"]:
            artifact = _verify(entry)
            info = artifact.lstat()
            record = {**entry, "state": "pending", "time": time.time()}
            # Write intent durably before the destructive operation.
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            artifact.unlink()
            if info.st_nlink == 1:
                allocated += info.st_blocks * 512
            logical += info.st_size
            deleted += 1
            record.update(state="deleted", time=time.time())
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return {
        "state": "complete",
        "deleted_files": deleted,
        "deleted_logical_bytes": logical,
        "reclaimed_allocated_bytes": allocated,
        "journal": str(journal),
    }
