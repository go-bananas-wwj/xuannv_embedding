"""Training-only gradient attribution before and after frozen-encoder readout warmup."""

from __future__ import annotations

import json
from dataclasses import replace

import torch

from xuannv_embedding.config import Config
from xuannv_embedding.training.runtime import _model_view, _move
from xuannv_embedding.training.target_supervision import TargetSupervision
from xuannv_embedding.training.target_warmup import (
    FORMAT,
    freeze_encoder,
    head_contract,
    load_base,
)


def gradient_parameters(model):
    """Full small highres/fusion/projection groups and one declared STP probe."""
    groups = {name: [] for name in ("highres", "fusion", "vmf", "stp_probe")}
    for name, parameter in model.named_parameters():
        group = None
        if name.startswith(("highres_encoders.", "highres_projections.", "highres_pools.")):
            group = "highres"
        elif name.startswith(("joint.", "gate.")):
            group = "fusion"
        elif name.startswith("bottleneck.") and not name.endswith("log_kappa"):
            group = "vmf"
        elif name.startswith("temporal_encoder.") and not groups["stp_probe"]:
            group = "stp_probe"
        if group:
            parameter.requires_grad_(True)
            groups[group].append((name, parameter))
    if any(not values for values in groups.values()):
        raise ValueError("Missing annual diagnostic parameter group")
    return groups


def gradients(loss, embedding, groups):
    parameters = [p for values in groups.values() for _, p in values]
    tensors = torch.autograd.grad(
        loss, [embedding, *parameters], retain_graph=True, allow_unused=True
    )
    result = {"embedding": tensors[0].detach().float().cpu().flatten()}
    offset = 1
    for group, values in groups.items():
        vectors = []
        for _, parameter in values:
            value = tensors[offset]
            vectors.append(
                torch.zeros(parameter.numel())
                if value is None
                else value.detach().float().cpu().flatten()
            )
            offset += 1
        result[group] = torch.cat(vectors)
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError("Nonfinite diagnostic gradient")
    return result


def gradient_statistics(vectors):
    """No cosine is defined for a zero gradient; retain null rather than inventing 0."""
    names = list(vectors)
    result = {}
    for group in next(iter(vectors.values())):
        matrix = torch.stack([vectors[name][group] for name in names])
        norms = matrix.norm(dim=1)
        gram = matrix @ matrix.T
        denominator = norms[:, None] * norms[None, :]
        cosine = (gram / denominator.clamp(min=1e-30)).clamp(-1, 1)
        result[group] = {
            "norms": dict(zip(names, norms.tolist(), strict=True)),
            "cosines": {
                a: {
                    b: float(cosine[i, j]) if float(denominator[i, j]) > 0 else None
                    for j, b in enumerate(names)
                }
                for i, a in enumerate(names)
            },
            "auxiliary_to_image_norm_ratio": (
                float(norms[names.index("auxiliary_total")] / norms[names.index("image")])
                if "image" in names and float(norms[names.index("image")]) > 0
                else None
            ),
        }
    return result


def single_task_objectives(config, heads, embedding, targets):
    """Reuse the actual production loss and masks without modifying its weights."""
    result = {}
    target = config.training.independent_targets
    for name, weight in target.weights.items():
        module = TargetSupervision(
            config.model.embed_dim,
            {name: weight},
            dem_mean=target.dem_mean,
            dem_std=target.dem_std,
        ).to(embedding.device)
        module.heads[name].load_state_dict(heads.heads[name].state_dict(), strict=True)
        module.eval().requires_grad_(False)
        result[name] = module.loss(module(embedding), targets, ramp=1.0)["total"]
    return result


def diagnose(args):
    from xuannv_embedding.data.annual_dataset import collate_annual_observations
    from xuannv_embedding.data.targets import AnnualTargetReader, collate_targets
    from xuannv_embedding.training.annual import AnnualLoss, prepare_annual_batch
    from xuannv_embedding.training.annual_data import annual_dataset
    from xuannv_embedding.training.annual_experiments import digest, model_state_digest, write_json
    from xuannv_embedding.training.cli import _git_sha, _source_code_sha256

    if args.output.exists():
        raise FileExistsError("Diagnostic output must be new")
    config = Config.from_yaml(args.config)
    spec = config.training.independent_targets
    contract = head_contract(config)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    system, identity = load_base(config, args.initialize_config, args.checkpoint, device)
    model = system.model
    freeze_encoder(model)
    before = model_state_digest(model)
    groups = gradient_parameters(model)
    # Match real training noise and enable checkpointing without constructing an optimizer.
    model.train()
    group_names = {key: [name for name, _ in rows] for key, rows in groups.items()}
    state = torch.load(args.warmup, map_location="cpu", weights_only=True)
    if state["format"] != FORMAT or state["contract"] != contract:
        raise ValueError("Warmup contract differs")
    if state["model_state_sha256"] != before or not state["encoder_unchanged"]:
        raise ValueError("Warmup used a different encoder")
    snapshots = {"initial": state["head_initialization"], "final": state["heads"]}
    if args.midpoint:
        middle = torch.load(args.midpoint, map_location="cpu", weights_only=True)
        if middle["contract"] != contract or middle["model_state_sha256"] != before:
            raise ValueError("Midpoint contract differs")
        snapshots = {
            "initial": snapshots["initial"],
            "midpoint": middle["heads"],
            "final": state["heads"],
        }
    # These identities predate this experiment; no scoring or label-driven re-selection.
    selection = json.loads(args.positions.read_text())
    patches = selection["patch_ids"]
    reader = AnnualTargetReader(
        spec.target_root,
        spec.prepared_root,
        preparation_sha256=spec.preparation_sha256,
        scope=spec.scope,
    )
    if not patches or len(set(patches)) != len(patches) or set(patches) - reader.roles["train"]:
        raise ValueError("Diagnostics require unique frozen TRAIN positions")
    base_config = replace(config, training=replace(config.training, independent_targets=None))
    criterion = AnnualLoss(base_config).to(device).eval()
    if criterion.latent_predictor is not None or criterion.view_predictor is not None:
        raise ValueError("Diagnostics require teacher/EMA objectives disabled")
    results = []
    try:
        for dc in config.data.datasets:
            dataset = annual_dataset(config, dc)
            lookup = {patch: i for i, (_, patch) in enumerate(dataset.records.identities)}
            raw_rows = [dataset[lookup[patch]] for patch in patches]
            raw = collate_annual_observations(raw_rows)
            targets = collate_targets(
                [reader.read(row["patch_id"], row["year"], tasks=spec.weights) for row in raw_rows]
            )
            for context in (False, True):
                cfg = replace(config, model=replace(config.model, annual_context_pooling=context))
                model.context_pooling = context
                batch = prepare_annual_batch(
                    raw, cfg, training=True, generator=torch.Generator().manual_seed(args.mask_seed)
                )
                batch["independent_targets"] = targets
                batch = _move(batch, device)
                captured = {}
                hook = model.bottleneck.register_forward_hook(
                    lambda _module, _inputs, output, cache=captured: cache.update(embedding=output)
                )
                try:
                    devices = [device.index] if device.type == "cuda" else []
                    with torch.random.fork_rng(devices=devices):
                        torch.random.default_generator.manual_seed(args.noise_seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed(args.noise_seed)
                        output = _model_view(model, batch)
                finally:
                    hook.remove()
                embedding = captured["embedding"]
                losses = criterion(
                    output,
                    batch["targets"],
                    batch["target_masks"],
                    highres_targets=batch["highres_targets"],
                )
                common_losses = {
                    "lowres_reconstruction": losses["recon"],
                    "highres_reconstruction": config.training.highres_loss_weight
                    * losses["highres_recon"],
                    "uniformity_epoch0": losses["uniformity_weighted"],
                }
                common = {
                    name: gradients(value, embedding, groups)
                    for name, value in common_losses.items()
                }
                common["image"] = {
                    g: common["lowres_reconstruction"][g] + common["highres_reconstruction"][g]
                    for g in common["lowres_reconstruction"]
                }
                common["uniformity_full"] = {
                    g: value * max(1, config.training.uniformity_warmup_epochs)
                    for g, value in common["uniformity_epoch0"].items()
                }
                item = {
                    "year": raw_rows[0]["year"],
                    "patch_ids": patches,
                    "context_pooling": context,
                    "mask_seed": args.mask_seed,
                    "vmf_noise_seed": args.noise_seed,
                    "common_losses": {
                        name: float(value.detach()) for name, value in common_losses.items()
                    },
                    "snapshots": {},
                }
                for stage, seeds in snapshots.items():
                    item["snapshots"][stage] = {}
                    for seed, values in seeds.items():
                        heads = (
                            TargetSupervision(
                                config.model.embed_dim,
                                spec.weights,
                                dem_mean=spec.dem_mean,
                                dem_std=spec.dem_std,
                            )
                            .to(device)
                            .eval()
                            .requires_grad_(False)
                        )
                        heads.load_state_dict(values, strict=True)
                        objectives = single_task_objectives(
                            config, heads, embedding, batch["independent_targets"]
                        )
                        full = heads.loss(heads(embedding), batch["independent_targets"], ramp=1.0)[
                            "total"
                        ]
                        total = sum(objectives.values())
                        if not torch.allclose(total, full, rtol=1e-5, atol=1e-6):
                            raise AssertionError(
                                "Task-wise objectives differ from production total"
                            )
                        vectors = {
                            **common,
                            **{
                                name: gradients(value, embedding, groups)
                                for name, value in objectives.items()
                            },
                        }
                        vectors["auxiliary_total"] = {
                            g: sum(vectors[name][g] for name in objectives) for g in common["image"]
                        }
                        vectors["combined_epoch0"] = {
                            g: vectors["image"][g]
                            + vectors["uniformity_epoch0"][g]
                            + vectors["auxiliary_total"][g]
                            for g in common["image"]
                        }
                        item["snapshots"][stage][seed] = {
                            "weighted_task_losses": {
                                name: float(value.detach()) for name, value in objectives.items()
                            },
                            "groups": gradient_statistics(vectors),
                        }
                        print(
                            json.dumps(
                                {
                                    "year": item["year"],
                                    "context": context,
                                    "stage": stage,
                                    "seed": seed,
                                }
                            ),
                            flush=True,
                        )
                        del vectors, objectives, full, total, heads
                results.append(item)
                del output, embedding, common, captured, losses, common_losses, batch
        model.context_pooling = config.model.annual_context_pooling
        freeze_encoder(model)
        if model_state_digest(model) != before:
            raise AssertionError("Diagnostic altered encoder state")
    finally:
        reader.close()
    write_json(
        args.output,
        {
            "passed": True,
            "git_sha": _git_sha(),
            "source_code_sha256": _source_code_sha256(),
            "optimizer_updates": 0,
            "encoder_unchanged": True,
            "model_state_sha256": before,
            "initialization_identity": identity,
            "warmup_sha256": digest(args.warmup),
            "positions_sha256": digest(args.positions),
            "parameter_groups": group_names,
            "uniformity_weight_epoch0": config.training.uniformity_weight
            / max(1, config.training.uniformity_warmup_epochs),
            "target_ramp": 1.0,
            "model_mode": "train with fixed vMF noise; no optimizer and no parameter update",
            "samples": results,
            "scope": (
                "FP32 local batches on previously chosen training observations; selected STP "
                "parameter only, not the complete STP gradient. Cosine opposition is a local "
                "mechanism diagnostic, not evidence of downstream causation or accuracy."
            ),
        },
    )
