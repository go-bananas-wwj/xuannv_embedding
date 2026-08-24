"""实现 ``xuannv data`` 下的区域无关生产命令。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def _jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是有效 JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON object")
            yield value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def grid_main(argv: Sequence[str] | None = None) -> int:
    """从冻结边界和 macrocell inventory 流式构建全国父网格。"""
    parser = argparse.ArgumentParser(prog="xuannv data grid")
    parser.add_argument("--inventory", type=Path, required=True, help="macrocell JSONL")
    parser.add_argument("--boundary", type=Path, required=True, help="冻结的 ADM0 边界")
    parser.add_argument("--sampled-registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zones", type=int, nargs="+", default=list(range(43, 54)))
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args(argv)

    invalid_zones = sorted(set(args.zones) - set(range(43, 54)))
    if invalid_zones:
        parser.error(f"--zones 只允许中国 owner zones 43..53: {invalid_zones}")
    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    if args.output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有父网格目录: {args.output_root}")

    import geopandas as gpd

    from xuannv_embedding.data_process.grid import (
        GridSpec,
        enumerate_macro_patch_records,
        read_sampled_registry_jsonl,
        sampled_registry_key,
        write_zone_records,
    )

    boundary = gpd.read_file(args.boundary).to_crs("EPSG:4326").geometry.union_all()
    sampled_records = read_sampled_registry_jsonl(args.sampled_registry)
    sampled_keys = {sampled_registry_key(record) for record in sampled_records}
    spec = GridSpec()
    inventory = list(_jsonl_records(args.inventory))
    summaries: list[dict[str, Any]] = []
    for zone in sorted(set(args.zones)):
        grid_id = f"utm{zone:02d}n"
        macros = (record for record in inventory if record.get("grid_id") == grid_id)
        records = (
            patch
            for macro in macros
            for patch in enumerate_macro_patch_records(macro, boundary, spec)
        )
        try:
            summary = write_zone_records(
                records,
                sampled_keys,
                args.output_root,
                batch_size=args.batch_size,
            )
        except StopIteration as exc:
            raise ValueError(f"{grid_id} 未生成任何父网格记录") from exc
        summaries.append(
            {
                "grid_id": summary.grid_id,
                "all_count": summary.all_count,
                "sampled_count": summary.sampled_count,
                "unsampled_count": summary.unsampled_count,
                "batch_count": summary.batch_count,
            }
        )
    result = {
        "schema_version": "china_full_1280m_grid_build_v1",
        "inventory": str(args.inventory),
        "boundary": str(args.boundary),
        "sampled_registry": str(args.sampled_registry),
        "zones": sorted(set(args.zones)),
        "summaries": summaries,
    }
    _atomic_json(args.output_root / "grid_build_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def manifest_main(argv: Sequence[str] | None = None) -> int:
    """把只读 legacy manifest 转换为带摘要 sidecar 的 manifest v1。"""
    parser = argparse.ArgumentParser(prog="xuannv data manifest")
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--generator-version", default="xuannv-embedding/1")
    args = parser.parse_args(argv)

    from xuannv_embedding.utils.manifest import load_legacy_manifest, write_manifest

    records = load_legacy_manifest(args.legacy, region=args.region)
    meta = write_manifest(
        args.output,
        records,
        months=args.months,
        generator_version=args.generator_version,
    )
    print(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2))
    return 0


def validate_main(argv: Sequence[str] | None = None) -> int:
    """验证 manifest v1 或完整父网格包的不变量。"""
    parser = argparse.ArgumentParser(prog="xuannv data validate")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--manifest", type=Path)
    target.add_argument("--grid-root", type=Path)
    target.add_argument("--tenfold-root", type=Path)
    parser.add_argument("--sampled-registry", type=Path)
    parser.add_argument("--parent-grid-root", type=Path)
    parser.add_argument("--utm-seam-audit", type=Path)
    parser.add_argument("--package-manifest-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args(argv)

    if args.manifest is not None:
        from xuannv_embedding.utils.manifest import load_manifest

        document = load_manifest(args.manifest)
        report: dict[str, Any] = {
            "kind": "manifest_v1",
            "path": str(args.manifest),
            "record_count": len(document.records),
            "sha256": document.meta.sha256,
            "passed": True,
        }
    elif args.grid_root is not None:
        if args.sampled_registry is None:
            parser.error("--grid-root 需要 --sampled-registry")
        from xuannv_embedding.data_process.grid import (
            audit_grid_package,
            bind_utm_seam_audit_to_package,
            read_sampled_registry_jsonl,
            reconcile_utm_seam_audit,
        )

        report = audit_grid_package(
            args.grid_root,
            read_sampled_registry_jsonl(args.sampled_registry),
            batch_size=args.batch_size,
        )
        if args.utm_seam_audit is not None:
            if args.package_manifest_sha256 is None:
                parser.error("--utm-seam-audit 需要 --package-manifest-sha256 外部信任锚")
            seam_report, binding = bind_utm_seam_audit_to_package(
                args.grid_root,
                args.utm_seam_audit,
                report,
                expected_manifest_sha256=args.package_manifest_sha256,
            )
            report = reconcile_utm_seam_audit(report, seam_report)
            report["package_binding"] = binding
    else:
        if args.utm_seam_audit is not None:
            parser.error("--utm-seam-audit 仅可与 --grid-root 一起使用")
        if args.package_manifest_sha256 is not None:
            parser.error("--package-manifest-sha256 仅可与 --grid-root 一起使用")
        from xuannv_embedding.data_process.partition import audit_tenfold_delivery

        report = audit_tenfold_delivery(
            args.tenfold_root,
            parent_grid_root=args.parent_grid_root,
            batch_size=args.batch_size,
        )
    if args.output is not None:
        _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("passed") is True else 1


def dispatch(command: str, argv: Sequence[str]) -> int:
    """延迟导入可选地理依赖并转发统一命令。"""
    if command == "grid":
        return grid_main(argv)
    if command == "registry":
        from xuannv_embedding.data_process.registry import main

        main(list(argv))
        return 0
    if command == "partition":
        from xuannv_embedding.data_process.partition import main

        main(list(argv))
        return 0
    if command == "materialize":
        from xuannv_embedding.data_process.materialize import main

        main(list(argv))
        return 0
    if command == "preprocess":
        from xuannv_embedding.data_process.preprocess import main

        return main(list(argv))
    if command == "manifest":
        return manifest_main(argv)
    if command == "validate":
        return validate_main(argv)
    raise ValueError(f"未知 data command: {command}")
