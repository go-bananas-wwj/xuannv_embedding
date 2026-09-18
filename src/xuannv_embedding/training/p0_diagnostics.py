"""Real-data P0 invariants, without downstream accuracy or model selection claims."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as functional

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import RegionRasterDataset, collate_region_batch
from xuannv_embedding.data_process.highres_catalog import write_json
from xuannv_embedding.data_process.observation_raster import sha256_file
from xuannv_embedding.data_process.p0 import read_rows
from xuannv_embedding.training.checkpoint import load_training_checkpoint
from xuannv_embedding.training.cli import build_training_system
from xuannv_embedding.training.runtime import _move


def output(model, batch: dict):
    return model(
        batch["source_frames"],
        batch["source_masks"],
        batch["timestamps"],
        batch["highres_frames"],
        batch["highres_masks"],
        highres_months=batch["highres_months"],
        output_months=batch["output_months"],
    )


def compare(first: torch.Tensor, second: torch.Tensor) -> dict:
    vectors_first = first.float().movedim(2, -1).reshape(-1, first.shape[2])
    vectors_second = second.float().movedim(2, -1).reshape(-1, second.shape[2])
    return {
        "max_abs_difference": float((first - second).abs().max()),
        "mean_cosine": float(functional.cosine_similarity(vectors_first, vectors_second).mean()),
    }


def diagnose(config_path: Path, checkpoint: Path, device: torch.device) -> dict:
    config = Config.from_yaml(config_path)
    if config.data.highres_mode != "observations" or not config.data.target_months:
        raise ValueError("P0 diagnostics require selected months and observation mode")
    system = build_training_system(config).to(device).eval()
    state = load_training_checkpoint(
        checkpoint,
        model=system.model,
        criterion=system.criterion,
        device=device,
        expected_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        expected_source_schema={
            key: asdict(value) for key, value in config.model.input_sources.items()
        },
        expected_regions=[item.region for item in config.data.datasets],
    )
    root = config.paths.data_root
    example = None
    example_active_count = 0
    split_report = {}
    for split in ("train", "validation", "test"):
        dataset_config = replace(
            config.data.datasets[0], manifest_path=root / f"{split}.manifest.jsonl"
        )
        dataset = RegionRasterDataset(config, dataset_config)
        availability = Counter()
        highres_frames = 0
        finite = True
        for sample in dataset:
            for source, values in sample["source_frames"].items():
                finite &= bool(torch.isfinite(values).all())
                availability[source] += int(sample["source_masks"][source].sum())
            for source, values in sample["highres_frames"].items():
                finite &= bool(torch.isfinite(values).all())
                active = sample["highres_masks"][source].flatten(1).any(dim=1)
                highres_frames += int(active.sum())
                if split == "train" and int(active.sum()) > example_active_count:
                    example = sample
                    example_active_count = int(active.sum())
            for name, mask in sample["target_masks"].items():
                source = config.model.target_heads[name].source
                expected = mask.flatten(1).any(dim=1)
                if not torch.equal(expected, sample["source_masks"][source].bool()):
                    raise ValueError("Target pixel mask and availability mismatch")
        if not finite:
            raise ValueError("Nonfinite real dataset sample")
        split_report[split] = {
            "parents": len(dataset),
            "available_source_months": dict(availability),
            "active_highres_frames": highres_frames,
            "finite": finite,
        }
    if example is None:
        raise ValueError("No high-resolution training example for diagnostics")
    batch = _move(collate_region_batch([example]), device)
    source = next(iter(batch["highres_frames"]))
    with torch.no_grad():
        original = output(system.model, batch).embedding_map
        reordered = copy.deepcopy(batch)
        for key in ("highres_frames", "highres_masks", "highres_months"):
            reordered[key][source] = reordered[key][source].flip(1)
        permutation = compare(original, output(system.model, reordered).embedding_map)
        masked = copy.deepcopy(batch)
        masked["highres_masks"][source].zero_()
        absent = copy.deepcopy(batch)
        absent["highres_frames"] = {}
        absent["highres_masks"] = {}
        absent["highres_months"] = {}
        baseline = output(system.model, absent).embedding_map
        fallback = compare(baseline, output(system.model, masked).embedding_map)
        noisy_masked = copy.deepcopy(masked)
        noisy_masked["highres_frames"][source].fill_(1000)
        masked_values = compare(baseline, output(system.model, noisy_masked).embedding_map)
        assigned = batch["highres_months"][source]
        active = batch["highres_masks"][source].flatten(2).any(dim=-1)
        covered = (
            (assigned[:, :, None] == batch["output_months"][:, None, :]) & active[:, :, None]
        ).any(dim=1)
        uncovered_difference = (
            float((original - baseline)[~covered].abs().max()) if (~covered).any() else 0.0
        )
        duplicate = copy.deepcopy(batch)
        active_index = int(torch.nonzero(active[0])[0])
        for key in ("highres_frames", "highres_masks", "highres_months"):
            duplicate[key][source] = duplicate[key][source][
                :, active_index : active_index + 1
            ].repeat_interleave(2, dim=1)
        duplicated = output(system.model, duplicate).embedding_map
        duplicate_reordered = copy.deepcopy(duplicate)
        for key in ("highres_frames", "highres_masks", "highres_months"):
            duplicate_reordered[key][source] = duplicate_reordered[key][source].flip(1)
        repeated_permutation = compare(
            duplicated, output(system.model, duplicate_reordered).embedding_map
        )
        missing_lowres = copy.deepcopy(batch)
        missing_lowres["source_frames"]["s1"].zero_()
        missing_lowres["source_masks"]["s1"].zero_()
        missing_lowres["target_masks"]["s1_recon"].zero_()
        missing_losses = system(missing_lowres)
        missing_source_recon = float(missing_losses["recon_s1_recon"])
        invalid = copy.deepcopy(masked)
        for name in invalid["source_frames"]:
            invalid["source_frames"][name].zero_()
            invalid["source_masks"][name].zero_()
        for value in invalid["target_masks"].values():
            value.zero_()
        invalid_losses = system(invalid)
        all_invalid_recon = float(invalid_losses["recon"])
        native_difference = compare(baseline, original)
    system.train()
    system.zero_grad(set_to_none=True)
    losses = system(batch)
    losses["total"].backward()
    gradients = {}
    for prefix in ("temporal_stem_bank", "highres_encoders", "highres_fusion"):
        values = [
            parameter.grad
            for name, parameter in system.model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        gradients[prefix] = {
            "tensors": len(values),
            "finite": all(bool(torch.isfinite(value).all()) for value in values),
            "absolute_sum": sum(float(value.abs().sum()) for value in values),
        }
    edge_backward = {}
    for name, current in (
        ("highres_empty", masked),
        ("lowres_source_missing", missing_lowres),
        ("all_invalid", invalid),
    ):
        system.zero_grad(set_to_none=True)
        current_losses = system(current)
        current_losses["total"].backward()
        edge_backward[name] = {
            "loss": float(current_losses["total"].detach()),
            "finite_gradients": all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in system.parameters()
                if parameter.grad is not None
            ),
        }
    system.zero_grad(set_to_none=True)
    data = read_rows(root / "observations.jsonl")
    source_reports = {}
    for name in config.model.input_sources:
        records = [
            row for row in data if row["source_signature"] == name and "materialized_path" in row
        ]
        source_reports[name] = {
            "channels": config.model.input_sources[name].channels,
            "units": "stored_values_unverified",
            "statistics": json.loads((root / "statistics" / f"{name}_stats.json").read_text()),
            "minimum_valid_fraction": min(row["valid_fraction"] for row in records),
            "maximum_bounds_error_m": max(row["bounds_error_m"] for row in records),
        }
    broken_month_config = replace(config, data=replace(config.data, target_months=["2021-02"]))
    broken_month_dataset = RegionRasterDataset(
        broken_month_config, broken_month_config.data.datasets[0]
    )
    original_record = broken_month_dataset.records[0]
    broken_month_dataset.records[0] = replace(
        original_record,
        sources={
            name: value
            for name, value in original_record.sources.items()
            if name in {"s2", "s1", "landsat"}
        },
    )
    broken_sample = broken_month_dataset[0]
    broken_month_missing = int(broken_sample["source_masks"]["s1"].sum()) == 0
    if not (
        permutation["max_abs_difference"] < 1e-5
        and fallback["max_abs_difference"] == 0
        and masked_values["max_abs_difference"] == 0
        and uncovered_difference == 0
        and repeated_permutation["max_abs_difference"] < 1e-5
        and missing_source_recon == 0
        and all_invalid_recon == 0
        and broken_month_missing
        and all(item["finite"] and item["absolute_sum"] > 0 for item in gradients.values())
        and all(item["finite_gradients"] for item in edge_backward.values())
    ):
        raise ValueError("P0 real-data diagnostic invariant failed")
    return {
        "passed": True,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_metrics": state["metrics"],
        "splits": split_report,
        "sources": source_reports,
        "permutation": permutation,
        "repeated_observation_permutation": repeated_permutation,
        "same_weights_empty_fallback": fallback,
        "masked_pixel_values_invariance": masked_values,
        "no_highres_month_max_difference": uncovered_difference,
        "highres_effect": native_difference,
        "missing_source_reconstruction": missing_source_recon,
        "all_invalid_reconstruction": all_invalid_recon,
        "real_s1_202102_missing": broken_month_missing,
        "gradients": gradients,
        "edge_case_backward": edge_backward,
        "diagnostic_parent": example["patch_id"],
        "diagnostic_active_frames": example_active_count,
        "landmark_registration": "not_verified",
        "radiometric_and_cloud_QA": "not_verified",
        "p1_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv diagnose")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    result = diagnose(args.config, args.checkpoint, torch.device(args.device))
    write_json(args.output, result)
    print(json.dumps({"passed": result["passed"], "output": str(args.output)}, ensure_ascii=False))
    return 0
