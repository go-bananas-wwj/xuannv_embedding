"""Storage layout, inventory and safe migration helpers.

The module deliberately separates planning from mutation.  A plan is a small,
reviewable JSON document; ``migrate`` only performs the entries in that plan and
never overwrites an existing destination.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LAYOUT = {
    "raw/china_1pct/s2": "Planetary Computer Sentinel-2 source files",
    "raw/china_1pct/s1": "Planetary Computer Sentinel-1 source files",
    "raw/china_1pct/landsat": "Landsat source files",
    "raw/china_1pct/gaofen": "Gaofen source files",
    "raw/china_1pct/jilin1": "Jilin-1 source packages and manifests",
    "raw/china_1pct/static": "Static labels and auxiliary source files",
    "processed/china_1pct/assets": "Shared materialized and quality assets",
    "processed/china_1pct/versions": "Immutable dataset version manifests",
    "experiments": "Training experiments and runs",
    "models": "Registered model artifacts",
    "products": "Exported embedding products",
    "catalog": "Storage and lineage catalogs",
    "workspace": "Rebuildable temporary work and acquisition state",
    "legacy": "Imported historical layouts",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def initialize_layout(root: Path) -> dict[str, Any]:
    """Create the empty, stable top-level layout and its schema marker."""
    root = root.expanduser().resolve()
    for relative in LAYOUT:
        (root / relative).mkdir(parents=True, exist_ok=True)
    marker = {
        "schema": "xuannv.storage-layout.v1",
        "root": str(root),
        "layout": LAYOUT,
        "created_at": _now(),
    }
    _atomic_json(root / "catalog/storage_layout.json", marker)
    return marker


@dataclass(frozen=True)
class InventoryEntry:
    source: str
    relative_path: str
    kind: str
    size_bytes: int
    mtime_ns: int
    inode: int
    device: int
    sha256: str | None = None
    link_target: str | None = None


def inventory_tree(
    root: Path, *, hash_files: bool = False, max_files: int | None = None
) -> list[InventoryEntry]:
    """Inventory files without following symlinks.

    ``max_files`` is useful for a quick migration rehearsal; a truncated
    inventory is explicitly marked by the caller and must never be used as a
    release lock.
    """
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    entries: list[InventoryEntry] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if not (Path(current) / d).is_symlink())
        for name in sorted(filenames):
            path = Path(current) / name
            stat = path.lstat()
            relative = path.relative_to(root).as_posix()
            link_target = os.readlink(path) if path.is_symlink() else None
            entries.append(
                InventoryEntry(
                    source=str(root),
                    relative_path=relative,
                    kind="symlink" if path.is_symlink() else "file",
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    inode=stat.st_ino,
                    device=stat.st_dev,
                    sha256=None if path.is_symlink() or not hash_files else _sha256(path),
                    link_target=link_target,
                )
            )
            if max_files is not None and len(entries) >= max_files:
                return entries
    return entries


def write_inventory(
    root: Path, output: Path, *, hash_files: bool = False, max_files: int | None = None
) -> dict[str, Any]:
    entries = inventory_tree(root, hash_files=hash_files, max_files=max_files)
    result = {
        "schema": "xuannv.storage-inventory.v1",
        "root": str(root.expanduser().resolve()),
        "hash_files": hash_files,
        "truncated": max_files is not None and len(entries) >= max_files,
        "file_count": len(entries),
        "generated_at": _now(),
        "files": [asdict(entry) for entry in entries],
    }
    _atomic_json(output, result)
    return result


DEFAULT_MAPPINGS = (
    ("/data2/china_xuannv_embedding/data/pc-s2", "raw/china_1pct/s2"),
    ("/data2/china_xuannv_embedding/data/pc-s1", "raw/china_1pct/s1"),
    ("/data2/china_xuannv_embedding/data/pc-ls", "raw/china_1pct/landsat"),
    ("/data2/Gaofen_Aligned_Train", "raw/china_1pct/gaofen"),
    ("/data2/Jilin1_Aligned_Train", "raw/china_1pct/jilin1"),
    ("/data2/china_xuannv_embedding/data/static", "raw/china_1pct/static"),
)


def make_plan(
    root: Path,
    output: Path,
    mappings: tuple[tuple[str, str], ...] = DEFAULT_MAPPINGS,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    entries: list[dict[str, Any]] = []
    for source_text, target_relative in mappings:
        source = Path(source_text)
        target = root / target_relative
        source_stat = source.stat() if source.exists() else None
        entries.append(
            {
                "source": str(source),
                "target": str(target),
                "source_exists": source.exists(),
                "source_is_active_candidate": source.name == "Jilin1_Aligned_Train",
                "target_exists": target.exists(),
                "target_empty": target.is_dir() and not any(target.iterdir()),
                "source_device": source_stat.st_dev if source_stat else None,
                "source_inode": source_stat.st_ino if source_stat else None,
                "source_mtime_ns": source_stat.st_mtime_ns if source_stat else None,
                "operation": (
                    "defer-active-source" if source.name == "Jilin1_Aligned_Train" else "planned"
                ),
            }
        )
    plan = {
        "schema": "xuannv.storage-migration-plan.v1",
        "root": str(root),
        "created_at": _now(),
        "entries": entries,
    }
    _atomic_json(output, plan)
    return plan


def execute_plan(
    plan_path: Path, *, allow_active: bool = False, symlink_compat: bool = True
) -> dict[str, Any]:
    """Execute a reviewed plan using rename; refuse conflicts and active sources."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "xuannv.storage-migration-plan.v1":
        raise ValueError("不支持的迁移计划 schema")
    journal: list[dict[str, Any]] = []

    def save_partial() -> None:
        _atomic_json(
            plan_path.with_name("migration.partial.json"),
            {
                "schema": "xuannv.storage-migration-journal.v1",
                "plan": str(plan_path),
                "updated_at": _now(),
                "complete": False,
                "entries": journal,
            },
        )

    for entry in plan["entries"]:
        source = Path(entry["source"])
        target = Path(entry["target"])
        if not entry.get("source_exists"):
            journal.append({"source": str(source), "status": "missing"})
            save_partial()
            continue
        if entry.get("source_is_active_candidate") and not allow_active:
            journal.append({"source": str(source), "status": "deferred-active-source"})
            save_partial()
            continue
        source_stat = source.stat()
        for key, actual in (
            ("source_device", source_stat.st_dev),
            ("source_inode", source_stat.st_ino),
            ("source_mtime_ns", source_stat.st_mtime_ns),
        ):
            expected = entry.get(key)
            if expected is not None and expected != actual:
                raise RuntimeError(f"迁移前源已变化，拒绝继续: {source} ({key})")
        if source.is_symlink() and target.exists() and source.resolve() == target.resolve():
            journal.append(
                {"source": str(source), "target": str(target), "status": "already-migrated"}
            )
            save_partial()
            continue
        if not source.is_dir():
            raise NotADirectoryError(source)
        if target.is_symlink() or (
            target.exists() and (not target.is_dir() or any(target.iterdir()))
        ):
            raise FileExistsError(f"迁移目标已存在，拒绝覆盖: {target}")
        if target.exists():
            target.rmdir()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_dev != target.parent.stat().st_dev:
            raise OSError(f"源和目标不在同一文件系统，拒绝原子重命名: {source} -> {target}")
        try:
            source.rename(target)
            if symlink_compat:
                source.symlink_to(
                    os.path.relpath(target, source.parent), target_is_directory=target.is_dir()
                )
            if not target.is_dir() or (symlink_compat and not source.is_symlink()):
                raise RuntimeError(f"迁移后校验失败: {source} -> {target}")
        except BaseException as exc:
            journal.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "status": "failed",
                    "error": repr(exc),
                    "target_exists": target.exists(),
                    "source_exists": source.exists(),
                }
            )
            save_partial()
            raise
        journal.append(
            {
                "source": str(source),
                "target": str(target),
                "status": "migrated",
                "compat_symlink": symlink_compat,
            }
        )
        save_partial()
    result = {
        "schema": "xuannv.storage-migration-journal.v1",
        "plan": str(plan_path),
        "finished_at": _now(),
        "complete": True,
        "entries": journal,
    }
    _atomic_json(plan_path.with_name("migration.json"), result)
    return result


def verify_migration(journal_path: Path) -> dict[str, Any]:
    """Verify every migrated pair without reading or modifying payload files."""
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    for entry in journal.get("entries", []):
        if entry.get("status") not in {"migrated", "already-migrated"}:
            continue
        source = Path(entry["source"])
        target = Path(entry["target"])
        source_ok = source.is_symlink() and source.resolve() == target.resolve()
        target_ok = target.is_dir()
        checks.append(
            {
                "source": str(source),
                "target": str(target),
                "source_symlink_ok": source_ok,
                "target_exists": target_ok,
                "passed": source_ok and target_ok,
            }
        )
    return {
        "schema": "xuannv.storage-verification.v1",
        "journal": str(journal_path),
        "checked": len(checks),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "verified_at": _now(),
    }


def rollback_migration(journal_path: Path) -> dict[str, Any]:
    """Restore migrated directories only when the compatibility link is intact."""
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    rolled_back: list[dict[str, Any]] = []
    for entry in reversed(journal.get("entries", [])):
        if entry.get("status") != "migrated":
            continue
        source = Path(entry["source"])
        target = Path(entry["target"])
        if not target.is_dir() or not source.is_symlink() or source.resolve() != target.resolve():
            raise RuntimeError(f"回退前校验失败，拒绝删除或覆盖: {source} -> {target}")
        source.unlink()
        target.rename(source)
        rolled_back.append({"source": str(source), "target": str(target), "status": "rolled-back"})
    result = {
        "schema": "xuannv.storage-rollback.v1",
        "journal": str(journal_path),
        "finished_at": _now(),
        "entries": rolled_back,
    }
    _atomic_json(journal_path.with_name("rollback.json"), result)
    return result


def cleanup_candidates(root: Path, output: Path) -> dict[str, Any]:
    """Write a review-only cleanup list; this function never deletes files."""
    root = root.expanduser().resolve()
    candidates = []
    legacy = root / "legacy"
    if legacy.is_dir():
        for path in sorted(legacy.iterdir()):
            candidates.append(
                {
                    "path": str(path),
                    "reason": "legacy path requires reference and backup review",
                    "deletion_allowed": False,
                }
            )
    result = {
        "schema": "xuannv.storage-cleanup-candidates.v1",
        "root": str(root),
        "generated_at": _now(),
        "candidates": candidates,
        "executed": False,
    }
    _atomic_json(output, result)
    return result


def backup_sources(
    sources: list[Path],
    repository: Path,
    password_file: Path,
    output: Path,
    *,
    restic: str = "restic",
    max_bytes: int = 1024 * 1024**3,
    min_free_bytes: int = 200 * 1024**3,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Back up selected sources with restic and write a redacted receipt.

    A dry run performs the capacity and source checks without contacting the
    repository.  Password material is passed by file and never serialized.
    """
    resolved_sources = [path.expanduser().resolve() for path in sources]
    missing = [str(path) for path in resolved_sources if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    if not password_file.is_file():
        raise FileNotFoundError(password_file)
    repository = repository.expanduser().resolve()
    repository.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(repository.parent)
    if usage.free < min_free_bytes:
        raise OSError(f"备份盘可用空间低于安全下限: {usage.free} < {min_free_bytes}")
    source_bytes = sum(
        item.size_bytes
        for source in resolved_sources
        for item in inventory_tree(source, hash_files=False)
    )
    if source_bytes > max_bytes:
        raise OSError(f"备份批次超过预算: {source_bytes} > {max_bytes}")
    receipt: dict[str, Any] = {
        "schema": "xuannv.backup-receipt.v1",
        "repository": str(repository),
        "sources": [str(path) for path in resolved_sources],
        "source_bytes": source_bytes,
        "max_bytes": max_bytes,
        "min_free_bytes": min_free_bytes,
        "password_file": str(password_file),
        "dry_run": dry_run,
        "started_at": _now(),
    }
    if not dry_run:
        executable = shutil.which(restic) or restic
        command = [executable, "backup", "--json", "--repo", str(repository)]
        command.extend(str(path) for path in resolved_sources)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "RESTIC_PASSWORD_FILE": str(password_file)},
        )
        receipt["exit_code"] = completed.returncode
        receipt["stdout"] = completed.stdout[-4000:]
        receipt["stderr"] = completed.stderr[-4000:]
        receipt["status"] = "completed" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            _atomic_json(output, receipt)
            raise RuntimeError(f"restic backup 失败，退出码 {completed.returncode}")
    else:
        receipt["status"] = "planned"
    receipt["finished_at"] = _now()
    _atomic_json(output, receipt)
    return receipt
