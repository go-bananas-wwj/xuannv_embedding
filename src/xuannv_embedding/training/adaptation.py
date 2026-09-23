"""Strict initialization of source-adaptation experiments from registered bases."""

import json
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

from xuannv_embedding.config import Config
from xuannv_embedding.models.highres_transformer import HighResTransformerModel
from xuannv_embedding.models.incremental_highres import IncrementalHighResModel
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.experiment import _sha


def initialize_adaptation(system, config, args, split, *, _ancestors=()):
    checkpoint_path = args.initialize.resolve()
    if checkpoint_path in _ancestors or len(_ancestors) >= 16:
        raise ValueError("cyclic or excessively deep adaptation lineage")
    base_config = Config.from_yaml(args.base_config)
    for key, value in asdict(base_config.model).items():
        if (
            key not in {"input_sources", "target_heads", "highres_transformer"}
            and asdict(config.model)[key] != value
        ):
            raise ValueError(f"base architecture changed: {key}")
    for name, source in base_config.model.input_sources.items():
        if config.model.input_sources.get(name) != source:
            raise ValueError("public source schema differs from the base")
    for name, head in base_config.model.target_heads.items():
        if config.model.target_heads.get(name) != head:
            raise ValueError("public reconstruction objective differs from the base")
    if (config.data.months, config.model.embed_dim) != (
        base_config.data.months,
        base_config.model.embed_dim,
    ):
        raise ValueError("time or embedding dimensions differ from the base")
    registration = json.loads((args.initialize.parent / "run.json").read_text())
    if registration["config_sha256"] != _sha(args.base_config):
        raise ValueError("parent registration configuration mismatch")
    for name in ("train", "validation"):
        if registration[name + "_indices"] != split[name]:
            raise ValueError("adaptation spatial split differs from its base")
    base = build_training_system(base_config)
    previous = registration.get("adaptation")
    if previous:
        for path_key, digest_key in (
            ("base_checkpoint", "base_checkpoint_sha256"),
            ("base_config", "base_config_sha256"),
        ):
            if _sha(Path(previous[path_key])) != previous[digest_key]:
                raise ValueError("ancestor adaptation provenance changed")
        initialize_adaptation(
            base,
            base_config,
            Namespace(
                initialize=Path(previous["base_checkpoint"]),
                base_config=Path(previous["base_config"]),
                freeze_base=previous["freeze_base"],
                train_semantic_head=previous.get("train_semantic_head", False),
                highres_encoding=previous["highres_encoding"],
                continue_base=previous.get("mode") == "continue_existing_sources",
            ),
            split,
            _ancestors=(*_ancestors, checkpoint_path),
        )
    elif any(v.role == "highres" for v in base_config.model.input_sources.values()):
        raise ValueError("highres parent requires registered adaptation lineage")
    state = load_training_checkpoint(
        args.initialize,
        model=base.model,
        criterion=base.criterion,
        expected_config_sha256=_sha(args.base_config),
        expected_source_schema={k: asdict(v) for k, v in base_config.model.input_sources.items()},
        expected_regions=[d.region for d in base_config.data.datasets],
    )
    if state["git_sha"] != registration["git_sha"]:
        raise ValueError("base checkpoint code identity mismatch")
    added = set(config.model.input_sources) - set(base_config.model.input_sources)
    sources = {
        s: v.channels
        for s, v in config.model.input_sources.items()
        if s in added and v.role == "highres"
    }
    if set(sources) != added:
        raise ValueError("only additional highres sources are supported")
    heads = {h: v.channels for h, v in config.model.target_heads.items() if v.source in sources}
    continuing = getattr(args, "continue_base", False)
    transformer = args.highres_encoding == "transformer"
    if transformer != (config.model.highres_transformer is not None):
        raise ValueError("transformer encoding and model.highres_transformer must agree")
    if transformer and (previous or not config.data.monthly_highres or continuing):
        raise ValueError("transformer adaptation requires monthly inputs and a public-only base")
    if continuing:
        if added or config.model.target_heads != base_config.model.target_heads:
            raise ValueError("continuation must retain exactly the parent source and target schema")
        if args.freeze_base:
            raise ValueError("continuation control must update existing parameters")
    elif not sources or not heads:
        raise ValueError("adaptation requires highres inputs and reconstruction targets")
    if any(
        v.loss_type != "continuous"
        for v in config.model.target_heads.values()
        if v.source in sources
    ):
        raise ValueError("highres adaptation targets must be continuous")
    train_head = getattr(args, "train_semantic_head", False)
    if train_head and (not args.freeze_base or not transformer or continuing):
        raise ValueError("train-semantic-head requires frozen-base transformer adaptation")
    if train_head and (
        system.criterion.semantic_probe is None or config.training.semantic_probe_weight <= 0
    ):
        raise ValueError("train-semantic-head requires an active semantic objective")
    system.criterion.load_state_dict(state["criterion"], strict=True)
    system.criterion.requires_grad_(not args.freeze_base)
    if train_head:
        system.criterion.semantic_probe.probes.requires_grad_(True)
    if continuing:
        system.model = base.model
        system.model.requires_grad_(True)
        if isinstance(system.model, IncrementalHighResModel):
            system.model.freeze_base = False
            system.model.frozen_sources = set()
    elif transformer:
        system.model = HighResTransformerModel(
            base.model,
            sources,
            heads,
            settings=config.model.highres_transformer,
            freeze_base=args.freeze_base,
        )
    elif isinstance(base.model, IncrementalHighResModel):
        system.model = base.model.extend(
            sources,
            heads,
            native=args.highres_encoding == "native",
            freeze_existing=args.freeze_base,
        )
    else:
        system.model = IncrementalHighResModel(
            base.model,
            sources,
            heads,
            native=args.highres_encoding == "native",
            freeze_base=args.freeze_base,
        )
    result = {
        "base_checkpoint": str(args.initialize),
        "base_checkpoint_sha256": _sha(args.initialize),
        "base_config": str(args.base_config),
        "base_config_sha256": _sha(args.base_config),
        "base_epoch": state["epoch"] + 1,
        "base_git_sha": state["git_sha"],
        "freeze_base": args.freeze_base,
        "highres_encoding": args.highres_encoding,
        "mode": "continue_existing_sources" if continuing else "add_sources",
        "new_sources": list(sources),
        "source_order": list(getattr(system.model, "branches", {})),
    }
    if train_head:
        # Omit the new key for legacy runs: their strict resume identity is unchanged.
        result["train_semantic_head"] = True
        result["semantic_head_parameters"] = [
            name
            for name, parameter in system.criterion.named_parameters()
            if parameter.requires_grad
        ]
    if transformer:
        result["transformer_settings"] = json.loads(
            json.dumps(asdict(config.model.highres_transformer))
        )
        result["source_order"] = list(sources)
    return result
