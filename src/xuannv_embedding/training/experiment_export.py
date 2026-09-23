"""Export registered checkpoints from their exact immutable spatial-fold caches."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

from xuannv_embedding.config import Config
from xuannv_embedding.data.observation_ablation import ablate_inputs, mask_audit
from xuannv_embedding.data.raster_dataset import collate_region_batch
from xuannv_embedding.export.embedding import export_embedding_batches
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import _git_sha, _setup_device, build_training_system
from xuannv_embedding.training.experiment import CachedSamples, _json, _sha


def validate_export_identity(run: dict, *, config_sha: str, cache_sha: str) -> None:
    for key, expected in (("config_sha256", config_sha), ("cache_sha256", cache_sha)):
        if run.get(key) != expected:
            raise ValueError(f"export {key} differs from the training registration")


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    config = Config.from_yaml(args.config)
    dropped = getattr(args, "drop_source", [])
    prefix = getattr(args, "prefix_month", None)
    if len(set(dropped)) != len(dropped) or not set(dropped) <= set(config.model.input_sources):
        raise ValueError("ablation sources must be unique registered inputs")
    if prefix is not None and (
        type(prefix) is not int or not 0 <= prefix < len(config.data.months)
    ):
        raise ValueError("invalid prefix month")
    ablation = bool(dropped) or prefix is not None
    if ablation and getattr(args, "probe_output", None):
        raise ValueError("ablation requires an explicitly registered downstream evaluation")
    cache_path = args.cache / "cache.json"
    document = json.loads(cache_path.read_text())
    training = json.loads((args.checkpoint.parent / "run.json").read_text())
    validate_export_identity(training, config_sha=_sha(args.config), cache_sha=_sha(cache_path))
    if args.output.exists():
        raise FileExistsError("export output exists; use a new directory")
    for record in document["records"]:
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
    if ablation:
        metadata["masking"] = "registered inference input ablation; targets unchanged"
        metadata["input_ablation"] = {
            "dropped_sources": sorted(dropped),
            "last_visible_month_index": prefix,
            "last_visible_month": config.data.months[prefix] if prefix is not None else None,
            "undated_static_inputs": "hidden" if prefix is not None else "unchanged unless dropped",
            "training_time_causality_claim": False,
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
        loader = DataLoader(
            CachedSamples(document, list(range(len(document["records"])))),
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=collate_region_batch,
        )
        for batch in loader:
            if ablation:
                original_audit = mask_audit(batch)
                batch = ablate_inputs(batch, dropped, last_month=prefix)
                changed_audit = mask_audit(batch)
                input_audits.extend(
                    {"original": before, "ablated": after}
                    for before, after in zip(original_audit, changed_audit, strict=True)
                )
            written = export_embedding_batches(
                system.model, [batch], args.output / "embeddings", device=device
            )
            paths.extend(written)
            status = {
                "state": "running",
                "patches": len(paths),
                "total": len(document["records"]),
                "elapsed_seconds": time.monotonic() - started,
            }
            _json(args.output / "status.json", status)
            if len(paths) % 20 == 0:
                print(json.dumps(status), flush=True)
        if ablation:
            _json(args.output / "input_masks.json", input_audits)
            metadata["input_masks_sha256"] = _sha(args.output / "input_masks.json")
        _json(
            args.output / "manifest.json",
            {
                **metadata,
                "records": [
                    {"patch_id": r["patch_id"], "bounds": r["bounds"], "path": str(p)}
                    for r, p in zip(document["records"], paths, strict=True)
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
