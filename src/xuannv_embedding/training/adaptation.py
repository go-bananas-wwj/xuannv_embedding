"""Strict initialization of source-adaptation experiments from registered bases."""

import json
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

from xuannv_embedding.config import Config
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
        if key not in {"input_sources", "target_heads"} and asdict(config.model)[key] != value:
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
                highres_encoding=previous["highres_encoding"],
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
    if not sources or not heads:
        raise ValueError("adaptation requires highres inputs and reconstruction targets")
    if any(
        v.loss_type != "continuous"
        for v in config.model.target_heads.values()
        if v.source in sources
    ):
        raise ValueError("highres adaptation targets must be continuous")
    system.criterion.load_state_dict(state["criterion"], strict=True)
    system.criterion.requires_grad_(not args.freeze_base)
    if isinstance(base.model, IncrementalHighResModel):
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
    return {
        "base_checkpoint": str(args.initialize),
        "base_checkpoint_sha256": _sha(args.initialize),
        "base_config": str(args.base_config),
        "base_config_sha256": _sha(args.base_config),
        "base_epoch": state["epoch"] + 1,
        "base_git_sha": state["git_sha"],
        "freeze_base": args.freeze_base,
        "highres_encoding": args.highres_encoding,
        "new_sources": list(sources),
        "source_order": list(system.model.branches),
    }
