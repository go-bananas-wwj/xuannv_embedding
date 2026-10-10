"""Export registered checkpoints from their exact immutable spatial-fold caches."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xuannv_embedding.config import Config
from xuannv_embedding.data.observation_ablation import (
    ablate_inputs,
    mask_audit,
    retain_highres_inputs,
)
from xuannv_embedding.data.raster_dataset import collate_region_batch
from xuannv_embedding.export.embedding import export_embedding_batches
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import _git_sha, _setup_device, build_training_system
from xuannv_embedding.training.experiment import CachedSamples, _json, _sha


def validate_export_identity(run: dict, *, config_sha: str, cache_sha: str) -> None:
    for key, expected in (("config_sha256", config_sha), ("cache_sha256", cache_sha)):
        if run.get(key) != expected:
            raise ValueError(f"export {key} differs from the training registration")


def export_indices(document: dict, splits: list[str] | None) -> list[int]:
    """Keep original grid order while materializing only explicitly requested splits."""
    if splits is None:
        return list(range(len(document["records"])))
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(s not in ("train", "validation", "test", "buffer") for s in splits)
    ):
        raise ValueError("export splits must be distinct canonical partitions")
    selected = [i for s in splits for i in document["split"][s]]
    if (
        not selected
        or any(type(i) is not int or not 0 <= i < len(document["records"]) for i in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("export splits are empty, overlap or contain invalid indices")
    return sorted(selected)


def export_groups(total: int, indices: list[int], batch_size: int):
    """Map selected records back to their original full-export batch positions."""
    if (
        type(total) is not int
        or total < 1
        or type(batch_size) is not int
        or batch_size < 1
        or not indices
        or len(set(indices)) != len(indices)
        or any(type(i) is not int or not 0 <= i < total for i in indices)
    ):
        raise ValueError("invalid export batch geometry")
    selected = set(indices)
    for start in range(0, total, batch_size):
        slots = list(range(start, min(total, start + batch_size)))
        real = [i for i in slots if i in selected]
        if real:
            lookup = {i: j for j, i in enumerate(real)}
            yield real, [lookup.get(i, 0) for i in slots], [i - start for i in real]


def expand_export_batch(value, positions):
    """Fill unused slots from already selected samples, after input masking."""
    if isinstance(value, dict):
        return {k: expand_export_batch(v, positions) for k, v in value.items()}
    if isinstance(value, list):
        return [value[i] for i in positions]
    if isinstance(value, torch.Tensor):
        return value.index_select(0, torch.tensor(positions, device=value.device))
    raise TypeError("unsupported batch value during export slot preservation")


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    config = Config.from_yaml(args.config)
    dropped = getattr(args, "drop_source", [])
    prefix = getattr(args, "prefix_month", None)
    retained = getattr(args, "retain_highres_source", [])
    fraction = getattr(args, "highres_retention", None)
    retention_seed = getattr(args, "retention_seed", 20260926)
    if len(set(dropped)) != len(dropped) or not set(dropped) <= set(config.model.input_sources):
        raise ValueError("ablation sources must be unique registered inputs")
    if prefix is not None and (
        type(prefix) is not int or not 0 <= prefix < len(config.data.months)
    ):
        raise ValueError("invalid prefix month")
    if bool(retained) != (fraction is not None):
        raise ValueError("high-resolution retention requires both sources and fraction")
    if retained and (
        len(set(retained)) != len(retained)
        or set(retained) & set(dropped)
        or any(
            name not in config.model.input_sources
            or config.model.input_sources[name].role != "highres"
            for name in retained
        )
        or not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not 0 <= fraction <= 1
        or type(retention_seed) is not int
        or retention_seed < 0
    ):
        raise ValueError("invalid registered high-resolution retention")
    ablation = bool(dropped) or prefix is not None or bool(retained)
    if ablation and getattr(args, "probe_output", None):
        raise ValueError("ablation requires an explicitly registered downstream evaluation")
    cache_path = args.cache / "cache.json"
    document = json.loads(cache_path.read_text())
    splits = getattr(args, "export_split", None)
    preserve_slots = getattr(args, "preserve_batch_slots", False)
    if preserve_slots and splits is None:
        raise ValueError("batch-slot preservation requires an explicit partial export")
    indices = export_indices(document, splits)
    if splits is not None and getattr(args, "probe_output", None):
        raise ValueError("partial exports require an explicitly registered downstream evaluation")
    training = json.loads((args.checkpoint.parent / "run.json").read_text())
    validate_export_identity(training, config_sha=_sha(args.config), cache_sha=_sha(cache_path))
    if args.output.exists():
        raise FileExistsError("export output exists; use a new directory")
    for i in indices:
        record = document["records"][i]
        if _sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("cached sample checksum mismatch")
    system = build_training_system(config)
    adaptation = training.get("adaptation")
    if adaptation:
        from xuannv_embedding.training.adaptation import initialize_adaptation

        for path_key, digest_key in (
            ("base_checkpoint", "base_checkpoint_sha256"),
            ("base_config", "base_config_sha256"),
        ):
            if _sha(Path(adaptation[path_key])) != adaptation[digest_key]:
                raise ValueError("adaptation base provenance changed")
        initialize_adaptation(
            system,
            config,
            argparse.Namespace(
                initialize=Path(adaptation["base_checkpoint"]),
                base_config=Path(adaptation["base_config"]),
                freeze_base=adaptation["freeze_base"],
                train_semantic_head=adaptation.get("train_semantic_head", False),
                highres_encoding=adaptation["highres_encoding"],
                continue_base=adaptation.get("mode") == "continue_existing_sources",
            ),
            document["split"],
        )
    state = load_training_checkpoint(
        args.checkpoint,
        model=system.model,
        expected_config_sha256=_sha(args.config),
        expected_source_schema={k: asdict(v) for k, v in config.model.input_sources.items()},
        expected_regions=[d.region for d in config.data.datasets],
    )
    if state["git_sha"] != training["git_sha"]:
        raise ValueError("checkpoint training code identity mismatch")
    metadata = {
        "export_git_sha": _git_sha(),
        "training_git_sha": state["git_sha"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha(args.checkpoint),
        "checkpoint_epoch": state["epoch"] + 1,
        "config_sha256": _sha(args.config),
        "cache_sha256": _sha(cache_path),
        "split": document["split"],
        "months": list(config.data.months),
        "dtype": "float32",
        "masking": "none; actual availability retained",
        "selection": (
            "fixed epoch snapshot; downstream selection must be registered separately"
            if args.checkpoint.name.startswith("epoch_")
            else "training validation loss; never test labels"
        ),
        "adaptation": adaptation,
    }
    if splits is not None:
        metadata["exported_indices"] = indices
        metadata["exported_splits"] = sorted(splits)
        metadata["export_batch_size"] = args.batch_size
        metadata["preserve_batch_slots"] = preserve_slots
    if ablation:
        metadata["masking"] = "registered inference input ablation; targets unchanged"
        metadata["input_ablation"] = {
            "dropped_sources": sorted(dropped),
            "last_visible_month_index": prefix,
            "last_visible_month": config.data.months[prefix] if prefix is not None else None,
            "undated_static_inputs": "hidden" if prefix is not None else "unchanged unless dropped",
            "training_time_causality_claim": False,
        }
        if retained:
            metadata["input_ablation"]["highres_retention"] = {
                "sources": sorted(retained),
                "fraction": fraction,
                "seed": retention_seed,
                "rule": "hashed edge, nested raster prefix of originally valid pixels per month",
            }
    del state
    device, distributed, _ = _setup_device(args.device)
    if distributed:
        raise ValueError("export requires an independent single-device process")
    args.output.mkdir(parents=True)
    _json(args.output / "run.json", metadata)
    started = time.monotonic()
    paths = []
    input_audits = []
    try:
        if preserve_slots:

            def selected_batches():
                for selected, padding, keep in export_groups(
                    len(document["records"]), indices, args.batch_size
                ):
                    samples = CachedSamples(document, selected)
                    yield collate_region_batch(
                        [samples[i] for i in range(len(samples))]
                    ), padding, keep

            loader = selected_batches()
        else:
            batches = DataLoader(
                CachedSamples(document, indices),
                batch_size=args.batch_size,
                num_workers=0,
                collate_fn=collate_region_batch,
            )
            loader = ((batch, None, None) for batch in batches)
        for batch, padding, keep in loader:
            if ablation:
                original_audit = mask_audit(batch)
                batch = ablate_inputs(batch, dropped, last_month=prefix)
                if retained:
                    batch = retain_highres_inputs(
                        batch, retained, fraction=fraction, seed=retention_seed
                    )
                changed_audit = mask_audit(batch)
                input_audits.extend(
                    {"original": before, "ablated": after}
                    for before, after in zip(original_audit, changed_audit, strict=True)
                )
            if padding is not None:
                batch = expand_export_batch(batch, padding)
            written = export_embedding_batches(
                system.model,
                [batch],
                args.output / "embeddings",
                device=device,
                output_indices=keep,
            )
            paths.extend(written)
            status = {
                "state": "running",
                "patches": len(paths),
                "total": len(indices),
                "elapsed_seconds": time.monotonic() - started,
            }
            _json(args.output / "status.json", status)
            if len(paths) % 20 == 0:
                print(json.dumps(status), flush=True)
        if ablation:
            _json(args.output / "input_masks.json", input_audits)
            metadata["input_masks_sha256"] = _sha(args.output / "input_masks.json")
        written = dict(zip(indices, paths, strict=True))
        _json(
            args.output / "manifest.json",
            {
                **metadata,
                "records": [
                    {
                        "patch_id": r["patch_id"],
                        "bounds": r["bounds"],
                        "path": str(
                            written.get(i, args.output / "embeddings" / (r["patch_id"] + ".npz"))
                        ),
                    }
                    for i, r in enumerate(document["records"])
                ],
            },
        )
        if getattr(args, "probe_output", None):
            from xuannv_embedding.training.probe_followup import launch_probe

            launch_probe(
                args.output,
                args.cache,
                args.probe_output,
                args.probe_slots or args.probe_output.parent / "cpu_slots",
            )
        _json(args.output / "status.json", {**status, "state": "complete"})
    except BaseException as exc:
        _json(args.output / "status.json", {"state": "failed", "error": repr(exc)})
        raise
