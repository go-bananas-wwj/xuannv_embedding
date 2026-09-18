"""Join prepared yearly observations to low-resolution manifests for real IO checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
import yaml

from xuannv_embedding.config import Config
from xuannv_embedding.data.raster_dataset import RegionRasterDataset, collate_region_batch
from xuannv_embedding.data_process.pilot_cache import _smoke_config
from xuannv_embedding.data_process.prepare_observations import atomic_json, rows
from xuannv_embedding.utils.manifest import ManifestRecord, load_manifest, write_manifest


def snapshot_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def verify_extracted_hashes(highres: Path) -> int:
    fingerprint_path = highres / "fingerprint.json"
    if not fingerprint_path.is_file():
        return 0
    fingerprint = json.loads(fingerprint_path.read_text())
    catalog = Path(fingerprint["catalog"])
    checked = 0
    for archive in fingerprint["archives"]:
        name = archive["archive"] + ".jsonl"
        original = {
            item["archive_member"]: item.get("sha256")
            for item in rows(catalog / "catalog_parts" / name)
        }
        for item in rows(highres / "parts" / name):
            if item["status"] != "usable":
                continue
            expected = original.get(item["archive_member"])
            if expected is None or expected != item["sha256"]:
                raise ValueError(
                    f"Extracted TIFF differs from archive catalog: {item['archive_member']}"
                )
            checked += 1
    return checked


def pair(highres: Path, lowres: Path, output: Path, max_parents: int = 0) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if lowres.name.endswith(".partial") and not max_parents:
        raise ValueError(
            "Full pairing requires completed lowres output; partial permits probe only"
        )
    summary = json.loads((highres / "summary.json").read_text())
    schemas = json.loads((highres / "sources.json").read_text())
    hashes_checked = verify_extracted_hashes(highres)
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True)
    shared_root = Path(os.path.commonpath([highres.resolve(), lowres.resolve(), output.resolve()]))
    counts, config_paths = {}, []
    for year in sorted(map(int, summary["usable_by_year"])):
        lowres_months = load_manifest(lowres / "train.manifest.jsonl").meta.months
        months = [month for month in lowres_months if month.startswith(f"{year}-")]
        if not months:
            continue
        eligible = {
            name: schema
            for name, schema in schemas.items()
            if schema["product_group"] == "5m"
            and (highres / "statistics" / str(year) / f"{name}_stats.json").is_file()
        }
        if not eligible:
            continue
        statistics = stage / "statistics" / str(year)
        statistics.mkdir(parents=True)
        for name in eligible:
            shutil.copy2(highres / "statistics" / str(year) / f"{name}_stats.json", statistics)
        lowres_sources = [
            name
            for name in ("s2", "s1", "landsat")
            if (lowres / "statistics" / f"{name}_stats.json").is_file()
        ]
        if not lowres_sources:
            raise ValueError("No lowres training statistics")
        for name in lowres_sources:
            shutil.copy2(lowres / "statistics" / f"{name}_stats.json", statistics)
        for split in ("train", "validation", "test"):
            highres_manifest = highres / f"{year}.{split}.manifest.jsonl"
            if not highres_manifest.is_file():
                continue
            extra = {item.patch_id: item for item in load_manifest(highres_manifest).records}
            records = []
            for base in load_manifest(lowres / f"{split}.manifest.jsonl").records:
                current = extra.get(base.patch_id)
                if max_parents and not current:
                    continue
                sources = {}
                for name in lowres_sources:
                    values = base.sources.get(name) or []
                    values = [values] if isinstance(values, str) else values
                    selected = []
                    for value in values:
                        if f"/{year}/" not in value:
                            continue
                        source = lowres / value
                        if max_parents:
                            relative = Path("lowres_snapshot") / value
                            destination = stage / relative
                            snapshot_file(source, destination)
                            mask = source.with_name(source.stem + "_mask.tif")
                            if not mask.is_file():
                                raise FileNotFoundError(mask)
                            snapshot_file(
                                mask, destination.with_name(destination.stem + "_mask.tif")
                            )
                            selected.append((output / relative).relative_to(shared_root).as_posix())
                        else:
                            selected.append(source.relative_to(shared_root).as_posix())
                    sources[name] = selected or None
                for name in eligible:
                    values = current.sources.get(name) if current else None
                    sources[name] = [
                        (highres / value).relative_to(shared_root).as_posix()
                        for value in (values or [])
                    ] or None
                records.append(
                    ManifestRecord(
                        patch_id=base.patch_id,
                        region=f"paired_{year}",
                        sources=sources,
                        grid=base.grid,
                        provenance={"split": split, "year": year},
                        quality={"numeric_QA": True, "cloud_QA": "unverified"},
                    )
                )
                if max_parents and len(records) >= max_parents:
                    break
            if not records:
                continue
            name = f"{year}.{split}.manifest.jsonl"
            write_manifest(
                stage / name, records, months=months, generator_version="paired-observations-v1"
            )
            counts[name] = len(records)
            config = _smoke_config(output, lowres_sources, split)
            config["paths"]["data_root"] = str(shared_root)
            config["model"].update(num_months=len(months), ref_year=year, ref_month=1)
            config["data"].update(
                months=months, highres_mode="observations", highres_max_observations=4
            )
            config["model"]["input_sources"].update(
                {
                    name: {"channels": schema["channels"], "role": "highres"}
                    for name, schema in eligible.items()
                }
            )
            config["model"]["stp"]["highres_fusion_to_embedding"] = True
            dataset = config["data"]["datasets"][0]
            dataset.update(
                region=f"paired_{year}",
                manifest_path=str(output / name),
                statistics_dir=str(output / "statistics" / str(year)),
                patch_grid_path=str(lowres / "selected_parents.jsonl"),
                source_map={source: source for source in config["model"]["input_sources"]},
            )
            config_name = f"{year}.{split}.loader-smoke.yaml"
            (stage / config_name).write_text(yaml.safe_dump(config, sort_keys=False))
            config_paths.append(config_name)
    if not config_paths:
        raise ValueError("No co-temporal lowres/highres records")
    stage.rename(output)
    torch.set_num_threads(2)
    checks = {}
    for name in config_paths:
        config = Config.from_yaml(output / name)
        dataset = RegionRasterDataset(config, config.data.datasets[0], max_records=2)
        count = 0
        for sample in dataset:
            for group in ("source_frames", "highres_frames"):
                if not all(bool(torch.isfinite(value).all()) for value in sample[group].values()):
                    raise ValueError("Nonfinite normalized values")
            if not any(bool(mask.any()) for mask in sample["source_masks"].values()):
                raise ValueError("Probe has no valid lowres")
            count += 1
        checks[name] = count
    report = {
        "manifests": counts,
        "loader_samples_checked": checks,
        "data_interface_ready": True,
        "annual_model_ready": False,
        "scientific_quality_validated": False,
        "probe_only": bool(max_parents),
        "extracted_hashes_matched_to_archive_catalog": hashes_checked,
    }
    atomic_json(output / "summary.json", report)
    return report


def smoke_backward(config_path: Path) -> dict:
    from xuannv_embedding.training.cli import build_training_system

    torch.set_num_threads(2)
    config = Config.from_yaml(config_path)
    dataset = RegionRasterDataset(config, config.data.datasets[0], max_records=2)
    sample = next(
        (
            item
            for item in dataset
            if any(bool(mask.any()) for mask in item["highres_masks"].values())
        ),
        None,
    )
    if sample is None:
        raise ValueError("No valid highres in probe")
    system = build_training_system(config)
    batch = collate_region_batch([sample])
    result = system(batch)
    loss = result["total"]
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite real smoke loss")
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in system.model.named_parameters()
        if name.startswith("highres_encoders.") and parameter.grad is not None
    ]
    if not gradients or not any(bool(gradient.abs().sum() > 0) for gradient in gradients):
        raise ValueError("No highres encoder gradient")
    if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise ValueError("Nonfinite highres gradient")
    return {
        "loss": float(loss.detach()),
        "finite_highres_gradients": True,
        "annual_model": False,
        "device": "cpu",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xuannv data pair-observations")
    parser.add_argument("--highres", type=Path, required=True)
    parser.add_argument("--lowres", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--wait-for-inputs", action="store_true")
    parser.add_argument("--check-backward", action="store_true")
    args = parser.parse_args(argv)
    if args.max_parents < 0:
        parser.error("max-parents must be nonnegative")
    if args.wait_for_inputs:
        while True:
            highres_ready = (args.highres / "summary.json").is_file()
            lowres_ready = (args.lowres / "summary.json").is_file()
            if highres_ready and lowres_ready:
                break
            print(
                json.dumps(
                    {
                        "waiting_for_highres": not highres_ready,
                        "waiting_for_lowres": not lowres_ready,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                ),
                flush=True,
            )
            time.sleep(60)
    print(json.dumps(pair(args.highres, args.lowres, args.output, args.max_parents)), flush=True)
    if args.check_backward:
        config = next(iter(sorted(args.output.glob("*.train.loader-smoke.yaml"))))
        result = smoke_backward(config)
        atomic_json(args.output / "backward_check.json", result)
        print(json.dumps(result), flush=True)
    return 0
