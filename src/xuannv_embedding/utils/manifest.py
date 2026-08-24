from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class ManifestError(ValueError):
    """manifest 内容、sidecar 或完整性校验失败。"""


SourceValue = str | list[str] | None


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"JSON 包含重复字段: {key!r}")
        result[key] = value
    return result


def _json_loads(text: str, context: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_duplicate_safe_object)
    except ManifestError:
        raise
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{context} 不是有效 JSON: {exc}") from exc


def _strict_record(raw: Any, context: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ManifestError(f"{context} 必须是 JSON object")
    allowed = {
        "patch_id",
        "region",
        "sources",
        "source_patch_id",
        "grid",
        "geometry",
        "quality",
        "provenance",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ManifestError(f"{context} 包含未知字段: {', '.join(unknown)}")
    missing = sorted({"patch_id", "region", "sources"} - set(raw))
    if missing:
        raise ManifestError(f"{context} 缺少必填字段: {', '.join(missing)}")
    return raw


def _validate_relative_path(value: str, context: str) -> None:
    if not value or "\\" in value:
        raise ManifestError(f"{context} 必须是非空 POSIX 相对路径")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "://" in value:
        raise ManifestError(f"{context} 必须是数据根目录下的相对路径: {value!r}")


def _validate_sources(value: Any, context: str) -> dict[str, SourceValue]:
    if not isinstance(value, dict):
        raise ManifestError(f"{context} 必须是 source 到相对路径的 mapping")
    result: dict[str, SourceValue] = {}
    for source, paths in value.items():
        if not isinstance(source, str) or not source:
            raise ManifestError(f"{context} 的 source 名称必须是非空字符串")
        if paths is None:
            result[source] = None
        elif isinstance(paths, str):
            _validate_relative_path(paths, f"{context}.{source}")
            result[source] = paths
        elif isinstance(paths, list) and all(isinstance(path, str) for path in paths):
            for index, path in enumerate(paths):
                _validate_relative_path(path, f"{context}.{source}[{index}]")
            result[source] = list(paths)
        else:
            raise ManifestError(f"{context}.{source} 必须是相对路径、相对路径列表或 null")
    return result


def _optional_mapping(value: Any, context: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ManifestError(f"{context} 必须是 object 或 null")
    return value


@dataclass(frozen=True)
class ManifestRecord:
    patch_id: str
    region: str
    sources: dict[str, SourceValue]
    source_patch_id: str | None = None
    grid: dict[str, Any] | None = None
    geometry: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: Any, context: str = "record") -> "ManifestRecord":
        raw = _strict_record(value, context)
        patch_id = raw["patch_id"]
        region = raw["region"]
        if not isinstance(patch_id, str) or not patch_id:
            raise ManifestError(f"{context}.patch_id 必须是非空字符串")
        if not isinstance(region, str) or not region:
            raise ManifestError(f"{context}.region 必须是非空字符串")
        source_patch_id = raw.get("source_patch_id")
        if source_patch_id is not None and not isinstance(source_patch_id, str):
            raise ManifestError(f"{context}.source_patch_id 必须是字符串或 null")
        return cls(
            patch_id=patch_id,
            region=region,
            sources=_validate_sources(raw["sources"], f"{context}.sources"),
            source_patch_id=source_patch_id,
            grid=_optional_mapping(raw.get("grid"), f"{context}.grid"),
            geometry=_optional_mapping(raw.get("geometry"), f"{context}.geometry"),
            quality=_optional_mapping(raw.get("quality"), f"{context}.quality"),
            provenance=_optional_mapping(raw.get("provenance"), f"{context}.provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "patch_id": self.patch_id,
            "region": self.region,
            "sources": self.sources,
        }
        for name in ("source_patch_id", "grid", "geometry", "quality", "provenance"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    def validate(self, context: str = "record") -> None:
        self.from_dict(self.to_dict(), context)


@dataclass(frozen=True)
class ManifestMeta:
    schema_version: str
    months: list[str]
    record_count: int
    generator_version: str
    sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> "ManifestMeta":
        if not isinstance(value, dict):
            raise ManifestError("manifest meta 必须是 JSON object")
        required = {"schema_version", "months", "record_count", "generator_version", "sha256"}
        unknown = sorted(set(value) - required)
        missing = sorted(required - set(value))
        if unknown:
            raise ManifestError(f"manifest meta 包含未知字段: {', '.join(unknown)}")
        if missing:
            raise ManifestError(f"manifest meta 缺少字段: {', '.join(missing)}")
        months = value["months"]
        if not isinstance(months, list) or not all(isinstance(month, str) for month in months):
            raise ManifestError("manifest meta.months 必须是字符串列表")
        count = value["record_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ManifestError("manifest meta.record_count 必须是非负整数")
        sha256 = value["sha256"]
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ManifestError("manifest meta.sha256 必须是 64 位十六进制摘要")
        return cls(
            schema_version=str(value["schema_version"]),
            months=list(months),
            record_count=count,
            generator_version=str(value["generator_version"]),
            sha256=sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "months": self.months,
            "record_count": self.record_count,
            "generator_version": self.generator_version,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ManifestDocument:
    records: list[ManifestRecord]
    meta: ManifestMeta


def manifest_meta_path(path: str | Path) -> Path:
    manifest_path = Path(path)
    return manifest_path.with_suffix(manifest_path.suffix + ".meta.json")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _serialize_records(path: Path, records: list[ManifestRecord]) -> bytes:
    suffix = path.suffix.lower()
    if suffix == ".json":
        text = json.dumps(
            [record.to_dict() for record in records],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return (text + "\n").encode("utf-8")
    if suffix == ".jsonl":
        lines = [
            json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for record in records
        ]
        return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
    raise ManifestError("manifest 文件扩展名必须是 .json 或 .jsonl")


def _validate_unique_record_identities(records: list[ManifestRecord]) -> None:
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        identity = (record.region, record.patch_id)
        if identity in seen:
            raise ManifestError(
                "manifest 包含重复 region/patch_id: "
                f"region={record.region!r}, patch_id={record.patch_id!r}, record[{index}]"
            )
        seen.add(identity)


def write_manifest(
    path: str | Path,
    records: list[ManifestRecord],
    *,
    months: list[str],
    generator_version: str = "xuannv-embedding/1",
) -> ManifestMeta:
    manifest_path = Path(path)
    for index, record in enumerate(records):
        record.validate(f"record[{index}]")
    _validate_unique_record_identities(records)
    payload = _serialize_records(manifest_path, records)
    meta = ManifestMeta(
        schema_version="1",
        months=list(months),
        record_count=len(records),
        generator_version=generator_version,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    _atomic_write(manifest_path, payload)
    meta_payload = (
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(manifest_meta_path(manifest_path), meta_payload)
    return meta


def _parse_records(path: Path, payload: bytes) -> list[ManifestRecord]:
    text = payload.decode("utf-8")
    if path.suffix.lower() == ".json":
        raw = _json_loads(text, str(path))
        if not isinstance(raw, list):
            raise ManifestError("JSON manifest 顶层必须是列表")
        return [
            ManifestRecord.from_dict(item, f"record[{index}]") for index, item in enumerate(raw)
        ]
    if path.suffix.lower() == ".jsonl":
        records: list[ManifestRecord] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            records.append(
                ManifestRecord.from_dict(
                    _json_loads(line, f"{path}:{line_number}"),
                    f"record[{line_number}]",
                )
            )
        return records
    raise ManifestError("manifest 文件扩展名必须是 .json 或 .jsonl")


def load_manifest(
    path: str | Path,
    *,
    verify_sha256: bool = True,
) -> ManifestDocument:
    manifest_path = Path(path)
    meta_path = manifest_meta_path(manifest_path)
    if not meta_path.is_file():
        raise ManifestError(f"manifest v1 缺少 meta sidecar: {meta_path}")
    try:
        payload = manifest_path.read_bytes()
        meta_raw = _json_loads(meta_path.read_text(encoding="utf-8"), str(meta_path))
    except OSError as exc:
        raise ManifestError(f"无法读取 manifest: {exc}") from exc
    meta = ManifestMeta.from_dict(meta_raw)
    if meta.schema_version != "1":
        raise ManifestError(f"不支持的 manifest schema_version: {meta.schema_version!r}")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if verify_sha256 and actual_sha256 != meta.sha256:
        raise ManifestError(
            f"manifest SHA-256 不匹配: sidecar={meta.sha256}, actual={actual_sha256}"
        )
    records = _parse_records(manifest_path, payload)
    _validate_unique_record_identities(records)
    if len(records) != meta.record_count:
        raise ManifestError(
            f"manifest 记录数不匹配: sidecar={meta.record_count}, actual={len(records)}"
        )
    return ManifestDocument(records=records, meta=meta)


def load_legacy_manifest(path: str | Path, *, region: str) -> list[ManifestRecord]:
    """只读适配旧版平铺 JSON；不会写回或伪造 v1 sidecar。"""
    legacy_path = Path(path)
    try:
        raw = _json_loads(legacy_path.read_text(encoding="utf-8"), str(legacy_path))
    except OSError as exc:
        raise ManifestError(f"无法读取 legacy manifest: {exc}") from exc
    if not isinstance(raw, list):
        raise ManifestError("legacy manifest 顶层必须是列表")
    metadata_keys = {
        "patch_id",
        "region",
        "source_patch_id",
        "grid",
        "geometry",
        "quality",
        "provenance",
    }

    legacy_root = legacy_path.parent.parent.resolve()

    def normalize_legacy_path(value: str, context: str) -> str:
        source_path = Path(value)
        resolved = (
            source_path.resolve()
            if source_path.is_absolute()
            else (legacy_path.parent / source_path).resolve()
        )
        try:
            return resolved.relative_to(legacy_root).as_posix()
        except ValueError as exc:
            raise ManifestError(f"{context} 逃逸 legacy 数据根目录: {value!r}") from exc

    def normalize_legacy_value(value: Any, context: str) -> SourceValue:
        if value is None:
            return None
        if isinstance(value, str):
            return normalize_legacy_path(value, context)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [
                normalize_legacy_path(item, f"{context}[{index}]")
                for index, item in enumerate(value)
            ]
        raise ManifestError(f"{context} 必须是路径、路径列表或 null")

    records: list[ManifestRecord] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("patch_id"), str):
            raise ManifestError(f"legacy record[{index}] 缺少 patch_id")
        sources = {
            key: normalize_legacy_value(value, f"legacy record[{index}].{key}")
            for key, value in item.items()
            if key not in metadata_keys
        }
        record_raw = {
            "patch_id": item["patch_id"],
            "region": str(item.get("region") or region),
            "sources": sources,
        }
        for key in metadata_keys - {"patch_id", "region"}:
            if item.get(key) is not None:
                record_raw[key] = item[key]
        records.append(ManifestRecord.from_dict(record_raw, f"legacy record[{index}]"))
    _validate_unique_record_identities(records)
    return records
