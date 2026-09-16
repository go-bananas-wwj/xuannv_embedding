"""Strict initialization of source-adaptation experiments from registered bases."""

import json
from dataclasses import asdict

from xuannv_embedding.config import Config
from xuannv_embedding.models.incremental_highres import IncrementalHighResModel
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.experiment import _sha


def initialize_adaptation(system, config, args, split):
    base_config = Config.from_yaml(args.base_config)
    if any(v.role == "highres" for v in base_config.model.input_sources.values()):
        raise ValueError("initial adaptation requires a public-only base")
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
    for name in ("train", "validation"):
        if registration[name + "_indices"] != split[name]:
            raise ValueError("adaptation spatial split differs from its base")
    base = build_training_system(base_config)
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
    sources = {s: v.channels for s, v in config.model.input_sources.items() if v.role == "highres"}
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
    if args.freeze_base:
        system.criterion.requires_grad_(False)
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
    }
