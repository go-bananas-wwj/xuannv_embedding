"""Build a reproducible nationwide training-chip registry from an atlas JSONL.

The atlas is deliberately a lightweight, pre-download artifact: each input line
describes one 1280 m candidate chip and the quality/semantic summaries used to
select it.  Raster acquisition happens only after this registry is frozen.

Required atlas fields:
  patch_id, grid_id, grid_row, grid_col, grid_epsg, wgs84_bounds,
  geometry_hash, macro_candidate_count, eligible, eligible_reasons

Recommended fields:
  admin1, ecoregion, strata (list or comma-separated string), and
  stratum_scores (mapping from stratum name to [0, 1] coverage/priority).
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_unit_hash(seed: int, value: str) -> float:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def _parse_strata(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError("strata must be a comma-separated string or a list of strings")


def _read_atlas(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on atlas line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"atlas line {line_number} must be an object")
            yield record


def _validate_record(record: dict[str, Any]) -> None:
    required = (
        "patch_id",
        "grid_id",
        "grid_row",
        "grid_col",
        "grid_epsg",
        "wgs84_bounds",
        "geometry_hash",
        "macro_candidate_count",
        "eligible",
        "eligible_reasons",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"atlas record {record.get('patch_id', '<unknown>')} missing {missing}")
    if not isinstance(record["patch_id"], str) or not record["patch_id"]:
        raise ValueError("patch_id must be a non-empty string")
    if not isinstance(record["grid_id"], str) or not record["grid_id"]:
        raise ValueError(f"{record['patch_id']}: grid_id must be a non-empty string")
    if not isinstance(record["grid_row"], int) or not isinstance(record["grid_col"], int):
        raise ValueError(f"{record['patch_id']}: grid_row and grid_col must be integers")
    if not isinstance(record["grid_epsg"], int) or record["grid_epsg"] <= 0:
        raise ValueError(f"{record['patch_id']}: grid_epsg must be a positive integer")
    bounds = record["wgs84_bounds"]
    if (
        not isinstance(bounds, list)
        or len(bounds) != 4
        or not all(isinstance(v, (int, float)) for v in bounds)
    ):
        raise ValueError(f"{record['patch_id']}: wgs84_bounds must contain four numbers")
    if not isinstance(record["geometry_hash"], str) or not record["geometry_hash"]:
        raise ValueError(f"{record['patch_id']}: geometry_hash must be a non-empty string")
    if (
        not isinstance(record["macro_candidate_count"], int)
        or not 1 <= record["macro_candidate_count"] <= 100
    ):
        raise ValueError(f"{record['patch_id']}: macro_candidate_count must be in [1, 100]")
    if not isinstance(record["eligible"], bool):
        raise ValueError(f"{record['patch_id']}: eligible must be boolean")
    if not isinstance(record["eligible_reasons"], list) or not all(
        isinstance(item, str) for item in record["eligible_reasons"]
    ):
        raise ValueError(f"{record['patch_id']}: eligible_reasons must be a string list")
    _parse_strata(record.get("strata"))


def _macrocell(record: dict[str, Any], macro_side: int) -> tuple[str, int, int]:
    return (
        record["grid_id"],
        record["grid_row"] // macro_side,
        record["grid_col"] // macro_side,
    )


def _stratum_score(record: dict[str, Any], stratum: str, seed: int) -> tuple[float, float]:
    scores = record.get("stratum_scores", {})
    if scores is None:
        scores = {}
    if not isinstance(scores, dict):
        raise ValueError(f"{record['patch_id']}: stratum_scores must be an object")
    raw_score = scores.get(stratum, 1.0)
    if not isinstance(raw_score, (int, float)):
        raise ValueError(f"{record['patch_id']}: score for {stratum} must be numeric")
    # Larger coverage/priority wins; the deterministic hash only breaks ties.
    return float(raw_score), -_stable_unit_hash(seed, f"{stratum}:{record['patch_id']}")


def _parse_quota_overrides(values: list[str]) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for value in values:
        name, separator, count = value.partition("=")
        if not separator or not name or not count.isdigit() or int(count) <= 0:
            raise ValueError(f"invalid quota override {value!r}; use stratum=count")
        quotas[name] = int(count)
    return quotas


def _load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    sampling = policy.get("sampling", {})
    quotas = sampling.get("supplement_quotas", {})
    if not isinstance(quotas, dict) or not all(
        isinstance(name, str) and isinstance(count, int) and count > 0
        for name, count in quotas.items()
    ):
        raise ValueError("policy sampling.supplement_quotas must be a positive integer mapping")
    return policy


def _push_candidate(
    heap: list[tuple[tuple[float, float], int, dict[str, Any]]],
    rank: tuple[float, float],
    counter: int,
    record: dict[str, Any],
    limit: int,
) -> None:
    item = (rank, counter, record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif rank > heap[0][0]:
        heapq.heapreplace(heap, item)


def build_registry(
    atlas_path: Path,
    policy_path: Path,
    macro_side: int | None,
    seed: int | None,
    quota_overrides: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = _load_policy(policy_path)
    sampling = policy["sampling"]
    macro_side = macro_side or int(sampling["macro_side_patches"])
    seed = seed if seed is not None else int(sampling["sampling_seed"])
    if macro_side <= 0:
        raise ValueError("macro_side must be positive")

    quotas = {str(name): int(count) for name, count in sampling["supplement_quotas"].items()}
    quotas.update(quota_overrides)
    reservoir_multiplier = int(sampling.get("supplement_reservoir_multiplier", 5))
    if reservoir_multiplier < 2:
        raise ValueError("supplement_reservoir_multiplier must be at least 2")

    target_total = int(sampling.get("target_total", 0))
    max_total = int(sampling.get("max_total", 0))
    if target_total <= 0 or max_total < target_total:
        raise ValueError("sampling.target_total must be positive and <= sampling.max_total")
    inclusion_probability = float(sampling.get("base_inclusion_probability", 0.01))
    if not 0 < inclusion_probability <= 1:
        raise ValueError("sampling.base_inclusion_probability must be in (0, 1]")
    supplement_max_per_macrocell = int(sampling.get("supplement_max_per_macrocell", 1))
    if supplement_max_per_macrocell < 0:
        raise ValueError("sampling.supplement_max_per_macrocell must be non-negative")
    regional_balance = sampling.get("regional_balance", {})
    if regional_balance is None:
        regional_balance = {}
    if not isinstance(regional_balance, dict):
        raise ValueError("sampling.regional_balance must be an object")
    regional_group_field = str(regional_balance.get("group_field", "regional_group"))
    regional_max_fraction = float(regional_balance.get("max_fraction_per_group_per_stratum", 1.0))
    if not 0 < regional_max_fraction <= 1:
        raise ValueError("regional_balance.max_fraction_per_group_per_stratum must be in (0, 1]")
    require_known_regional_group = bool(
        regional_balance.get("require_known_group_for_supplement", False)
    )

    base_winners: dict[tuple[str, int, int], tuple[float, dict[str, Any]]] = {}
    macro_eligible_counts: Counter[tuple[str, int, int]] = Counter()
    macro_record_counts: Counter[tuple[str, int, int]] = Counter()
    macro_declared_counts: dict[tuple[str, int, int], int] = {}
    reservoirs: dict[str, list[tuple[tuple[float, float], int, dict[str, Any]]]] = {
        name: [] for name in quotas
    }
    seen_patch_ids: set[str] = set()
    total_records = 0
    eligible_records = 0
    counter = 0

    for record in _read_atlas(atlas_path):
        _validate_record(record)
        total_records += 1
        patch_id = record["patch_id"]
        if patch_id in seen_patch_ids:
            raise ValueError(f"duplicate patch_id in atlas: {patch_id}")
        seen_patch_ids.add(patch_id)
        key = _macrocell(record, macro_side)
        macro_record_counts[key] += 1
        declared = record["macro_candidate_count"]
        if key in macro_declared_counts and macro_declared_counts[key] != declared:
            raise ValueError(f"{patch_id}: inconsistent macro_candidate_count in macrocell {key}")
        macro_declared_counts[key] = declared
        if not record["eligible"]:
            continue
        eligible_records += 1
        counter += 1
        macro_eligible_counts[key] += 1
        base_rank = _stable_unit_hash(seed, patch_id)
        incumbent = base_winners.get(key)
        if incumbent is None or base_rank < incumbent[0]:
            base_winners[key] = (base_rank, record)

        strata = set(_parse_strata(record.get("strata")))
        for stratum, quota in quotas.items():
            if stratum not in strata:
                continue
            _push_candidate(
                reservoirs[stratum],
                _stratum_score(record, stratum, seed),
                counter,
                record,
                limit=quota * reservoir_multiplier,
            )

    for key, observed in macro_record_counts.items():
        if observed != macro_declared_counts[key]:
            raise ValueError(
                f"macrocell {key} has {observed} atlas records but declares "
                f"macro_candidate_count={macro_declared_counts[key]}; "
                "atlas must include all candidates"
            )

    selected: dict[str, dict[str, Any]] = {}
    reasons: dict[str, set[str]] = defaultdict(set)
    accepted_base_macrocells = 0
    for key, (_, record) in base_winners.items():
        eligible_count = macro_eligible_counts[key]
        accept_probability = min(1.0, eligible_count * inclusion_probability)
        if _stable_unit_hash(seed, f"base-macrocell:{key}") >= accept_probability:
            continue
        accepted_base_macrocells += 1
        patch_id = record["patch_id"]
        selected[patch_id] = record
        reasons[patch_id].add("base:expected_1pct")

    supplemental_summary: dict[str, dict[str, int]] = {}
    supplemental_per_macrocell: Counter[tuple[str, int, int]] = Counter()
    for stratum, quota in quotas.items():
        ranked = sorted(reservoirs[stratum], key=lambda item: item[0], reverse=True)
        base_coverage = sum(
            stratum in set(_parse_strata(record.get("strata"))) for record in selected.values()
        )
        final_coverage = base_coverage
        regional_cap = math.ceil(quota * regional_max_fraction)
        coverage_by_group = Counter(
            str(record.get(regional_group_field, "unknown"))
            for record in selected.values()
            if stratum in set(_parse_strata(record.get("strata")))
        )
        newly_added = 0
        for _, _, record in ranked:
            if final_coverage >= quota or len(selected) >= max_total:
                break
            patch_id = record["patch_id"]
            if patch_id in selected:
                continue
            key = _macrocell(record, macro_side)
            if supplemental_per_macrocell[key] >= supplement_max_per_macrocell:
                continue
            group = str(record.get(regional_group_field, "unknown"))
            if require_known_regional_group and group == "unknown":
                continue
            if coverage_by_group[group] >= regional_cap:
                continue
            selected[patch_id] = record
            supplemental_per_macrocell[key] += 1
            newly_added += 1
            final_coverage += 1
            coverage_by_group[group] += 1
            reasons[patch_id].add(f"supplement:{stratum}")
        supplemental_summary[stratum] = {
            "requested": quota,
            "base_coverage": base_coverage,
            "final_coverage": final_coverage,
            "unmet": max(0, quota - final_coverage),
            "newly_added": newly_added,
            "reservoir_size": len(reservoirs[stratum]),
            "coverage_by_regional_group": dict(sorted(coverage_by_group.items())),
        }

    registry: list[dict[str, Any]] = []
    by_admin = Counter()
    by_ecoregion = Counter()
    by_reason = Counter()
    for patch_id in sorted(selected):
        record = dict(selected[patch_id])
        record["sampling_reasons"] = sorted(reasons[patch_id])
        record["sampling_seed"] = seed
        record["macro_side_patches"] = macro_side
        registry.append(record)
        by_admin[str(record.get("admin1", "unknown"))] += 1
        by_ecoregion[str(record.get("ecoregion", "unknown"))] += 1
        for reason in reasons[patch_id]:
            by_reason[reason] += 1

    report = {
        "atlas_path": str(atlas_path),
        "atlas_sha256": _sha256(atlas_path),
        "policy_path": str(policy_path),
        "policy_sha256": _sha256(policy_path),
        "total_atlas_records": total_records,
        "eligible_records": eligible_records,
        "eligible_macrocells": len(base_winners),
        "base_selected": accepted_base_macrocells,
        "base_expected": eligible_records * inclusion_probability,
        "base_inclusion_probability": inclusion_probability,
        "total_selected": len(registry),
        "target_total": target_total,
        "max_total": max_total,
        "supplemental": supplemental_summary,
        "selected_by_admin1": dict(sorted(by_admin.items())),
        "selected_by_ecoregion": dict(sorted(by_ecoregion.items())),
        "selected_by_reason": dict(sorted(by_reason.items())),
        "sampling_seed": seed,
        "macro_side_patches": macro_side,
    }
    return registry, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas", required=True, type=Path, help="Quality-qualified candidate atlas JSONL."
    )
    parser.add_argument(
        "--policy", required=True, type=Path, help="Self-contained sampling policy JSON."
    )
    parser.add_argument("--output", required=True, type=Path, help="Selected registry JSONL.")
    parser.add_argument("--report", required=True, type=Path, help="Sampling report JSON.")
    parser.add_argument(
        "--macro-side", type=int, default=None, help="Override policy macrocell side length."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override policy sampling seed.")
    parser.add_argument(
        "--quota",
        action="append",
        default=[],
        help="Override or add one supplemental quota, e.g. worldcover:wetland=3000.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the report without writing."
    )
    args = parser.parse_args(argv)

    registry, report = build_registry(
        atlas_path=args.atlas,
        policy_path=args.policy,
        macro_side=args.macro_side,
        seed=args.seed,
        quota_overrides=_parse_quota_overrides(args.quota),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in registry:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
