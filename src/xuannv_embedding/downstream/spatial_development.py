"""Fixed readout diagnostics on the pre-existing, pretraining-unseen development role.

Only development labels may be materialized. Macro/component unions and both years
stay together. Ridge strength and pixel budgets are fixed before label access.
This is product-related development evidence, never independent manual accuracy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def spatial_folds(rows, *, folds=5):
    ids = [row["patch_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Development positions must be unique")
    parent = {patch: patch for patch in ids}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    seen = {}
    for row in rows:
        for field in ["macro_id", "spatial_component_id"]:
            key = field, row[field]
            if key in seen:
                parent[find(row["patch_id"])] = find(seen[key])
            else:
                seen[key] = row["patch_id"]
    groups = {}
    for patch in ids:
        groups.setdefault(find(patch), []).append(patch)
    if len(groups) < folds or folds < 2:
        raise ValueError("Too few independent macro/component unions for spatial folds")
    ordered = sorted(
        groups.values(),
        key=lambda g: (-len(g), hashlib.sha256("/".join(sorted(g)).encode()).hexdigest()),
    )
    loads = [0] * folds
    assignments = {}
    for group in ordered:
        fold = min(range(folds), key=lambda i: (loads[i], i))
        name = "group_" + hashlib.sha256("/".join(sorted(group)).encode()).hexdigest()[:16]
        for patch in group:
            assignments[patch] = {"fold": fold, "group": name}
        loads[fold] += len(group)
    return assignments


def sample_indices(valid, key, maximum):
    indices = np.flatnonzero(np.asarray(valid).reshape(-1))
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    if len(indices) > maximum:
        indices = np.random.default_rng(seed).choice(indices, maximum, replace=False)
    return np.sort(indices)


def buffered_folds(rows, assignments, *, folds=5, buffer_m=1290.0):
    import shapely

    from xuannv_embedding.data_process.target_splits import projected_geometries

    result = []
    for fold in range(folds):
        test = [row for row in rows if assignments[row["patch_id"]]["fold"] == fold]
        train = [row for row in rows if assignments[row["patch_id"]]["fold"] != fold]
        distances = {}
        for epsg in {row["grid_epsg"] for row in train}:
            local = [row for row in train if row["grid_epsg"] == epsg]
            test_polygons, _ = projected_geometries(test, epsg)
            train_polygons, _ = projected_geometries(local, epsg)
            values = shapely.distance(train_polygons[:, None], test_polygons[None, :]).min(1)
            distances.update(
                {row["patch_id"]: float(d) for row, d in zip(local, values, strict=True)}
            )
        excluded_groups = {
            assignments[patch]["group"] for patch, d in distances.items() if d < buffer_m
        }
        kept = [
            row["patch_id"]
            for row in train
            if assignments[row["patch_id"]]["group"] not in excluded_groups
        ]
        if not kept:
            raise ValueError("No spatially buffered readout training positions remain")
        result.append(
            {
                "fold": fold,
                "train": sorted(kept),
                "test": sorted(row["patch_id"] for row in test),
                "excluded_training_groups": sorted(excluded_groups),
                "minimum_train_test_boundary_m": min(distances[patch] for patch in kept),
            }
        )
    return result


def fit_readout(features, targets, *, classes=None, regularization=1e-3):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets)
    if x.ndim != 2 or not len(x) or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("Readout training arrays are empty, mismatched or nonfinite")
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-5)
    x = np.column_stack(((x - mean) / std, np.ones(len(x))))
    active = None
    if classes is not None:
        if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= classes).any():
            raise ValueError("Invalid readout training class")
        counts = np.bincount(y, minlength=classes)
        active = counts > 0
        weights = len(y) / (active.sum() * counts[y])
        y = np.eye(classes)[y]
    else:
        y = y.astype(np.float64).reshape(-1, 1)
        weights = np.ones(len(y))
    if not np.isfinite(y).all() or not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Readout targets and regularization must be finite/valid")
    gram = x.T @ (x * weights[:, None]) / weights.sum()
    diagonal = np.ones(x.shape[1]) * regularization
    diagonal[-1] = 0
    coef = np.linalg.solve(gram + np.diag(diagonal), x.T @ (y * weights[:, None]) / weights.sum())
    return {"mean": mean, "std": std, "coef": coef, "active_classes": active}


def predict_readout(readout, features):
    x = (np.asarray(features, dtype=np.float64) - readout["mean"]) / readout["std"]
    prediction = x @ readout["coef"][:-1] + readout["coef"][-1]
    if readout["active_classes"] is not None:
        prediction[:, ~readout["active_classes"]] = -np.inf
    return prediction


def metric_summary(rows):
    if not rows:
        raise ValueError("No development scoring records")
    confusion = np.sum([np.asarray(row["confusion"], dtype=np.int64) for row in rows], axis=0)
    support = confusion.sum(1)
    denominator = support + confusion.sum(0) - confusion.diagonal()
    iou = np.divide(
        confusion.diagonal(), denominator, out=np.zeros(11, dtype=float), where=denominator > 0
    )
    present = support > 0
    non_snow = present.copy()
    non_snow[6] = False
    tp = confusion[7, 7]
    water_denominator = confusion[7].sum() + confusion[:, 7].sum()
    count = sum(row["dem_count"] for row in rows)
    sse = sum(row["dem_sse"] for row in rows)
    sy = sum(row["dem_sum_y"] for row in rows)
    sy2 = sum(row["dem_sum_y2"] for row in rows)
    sst = sy2 - sy * sy / count if count else 0
    return {
        "worldcover_mIoU": float(iou[present].mean()) if present.any() else None,
        "worldcover_non_snow_mIoU": float(iou[non_snow].mean()) if non_snow.any() else None,
        "worldcover_water_F1": float(2 * tp / water_denominator) if water_denominator else None,
        "worldcover_class_support": support.tolist(),
        "worldcover_class_IoU": [float(iou[i]) if present[i] else None for i in range(11)],
        "worldcover_valid_pixels": int(confusion.sum()),
        "dem_valid_cells": count,
        "dem_RMSE_m": float(np.sqrt(sse / count)) if count else None,
        "dem_MAE_m": sum(row["dem_sae"] for row in rows) / count if count else None,
        "dem_R2": 1 - sse / sst if sst > 0 else None,
    }


def prepare(args):
    from xuannv_embedding.data.targets import AnnualTargetReader
    from xuannv_embedding.training.annual_experiments import digest, write_json
    from xuannv_embedding.training.cli import _git_sha, _source_code_sha256
    from xuannv_embedding.utils.manifest import IndexedManifest, write_manifest

    if args.output.exists():
        raise FileExistsError("Development probe output must be new")
    if any(
        args.output.resolve().is_relative_to(path.resolve())
        for path in [args.prepared_root, args.target_root]
    ):
        raise ValueError("Development output must not modify source roots")
    prepared = args.prepared_root
    receipt = json.loads((prepared / "preparation.json").read_text())
    if digest(args.audit_contract) != receipt["initial_audit_contract_sha256"]:
        raise ValueError("Development source audit identity changed")
    if digest(prepared / "experiment_splits.json") != receipt["splits_sha256"]:
        raise ValueError("Development roles changed")
    split = json.loads((prepared / "experiment_splits.json").read_text())
    rows = split["roles"]["development"]
    wanted = {row["patch_id"] for row in rows}
    assignments = spatial_folds(rows)
    folds = buffered_folds(rows, assignments)
    matrix = json.loads(args.matrix_plan.read_text())
    labels = json.loads((Path(matrix["labels"]) / "labels.json").read_text())
    old_ids = {r["patch_id"] for r in labels["records"]} | {
        r[2] for r in json.loads(Path(matrix["cities"]).read_text())
    }
    if wanted & old_ids:
        raise ValueError("New development role overlaps historical candidates")
    audit = json.loads(args.audit_contract.read_text())
    args.output.mkdir(parents=True)
    manifests = []
    found = set()
    for entry in audit["annual_manifests"]:
        source = Path(entry["path"])
        if (
            digest(source) != entry["sha256"]
            or digest(Path(str(source) + ".meta.json")) != entry["sidecar_sha256"]
        ):
            raise ValueError("Audited annual manifest changed")
        records = IndexedManifest(source)
        chosen = [record for record in records if record.patch_id in wanted]
        if ".train." in source.name and chosen:
            raise ValueError("New development role was seen in original pretraining")
        if not chosen:
            continue
        directory = args.output / "data" / source.stem
        directory.mkdir(parents=True)
        for name in ["summary.json", "sources.json", "statistics"]:
            (directory / name).symlink_to(
                (source.parent / name).resolve(), target_is_directory=name == "statistics"
            )
        destination = directory / source.name
        written = write_manifest(
            destination,
            chosen,
            months=records.meta.months,
            generator_version="spatial-development-readout-v1",
        )
        for record in chosen:
            identity = record.patch_id, record.provenance["year"]
            if identity in found:
                raise ValueError("Development observation duplicated across annual splits")
            found.add(identity)
        manifests.append(
            {
                "path": str(destination.resolve()),
                "sha256": written.sha256,
                "region": chosen[0].region,
                "year": chosen[0].provenance["year"],
                "records": len(chosen),
            }
        )
    if found != {(patch, year) for patch in wanted for year in [2020, 2021]}:
        raise ValueError("Frozen development role lacks paired annual observations")
    plan = {
        "schema": "xuannv.spatial-development-readout.v1",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "git_sha": _git_sha(),
        "preparation_source_code_sha256": _source_code_sha256(),
        "role": "development",
        "positions": rows,
        "assignments": assignments,
        "folds": folds,
        "matrix_plan": str(args.matrix_plan.resolve()),
        "matrix_plan_sha256": digest(args.matrix_plan),
        "prepared_root": str(prepared.resolve()),
        "preparation_sha256": digest(prepared / "preparation.json"),
        "target_root": str(args.target_root.resolve()),
        "manifests": manifests,
        "protocol": {
            "folds": 5,
            "train_test_boundary_buffer_m": 1290,
            "grouping": "Macro/component unions; paired years share a parent fold.",
            "regularization": 1e-3,
            "worldcover_train_pixels_per_record": 1024,
            "dem_train_cells_per_record": 256,
            "sampling": "Hash-seeded uniform valid pixels per record.",
            "classification": "Class-balanced TRAIN-fold ridge; fixed argmax.",
            "regression": "Ridge to native40m DSM in meters.",
            "normalization": "Mean/std from TRAIN-fold sampled features only.",
            "feature_grids_m": [10, 40],
            "highres_max_observations": 2,
            "prediction_support": "Common observation validity intersected with target masks.",
            "primary_scope": "Secondary diagnostic; matrix guardrails remain binding.",
            "confirmation_and_validation_opened": False,
            "labels_materialized_after_protocol_freeze": True,
        },
        "limitations": [
            "Same product families as supervision; not manual5m truth.",
            "Both continued-training seeds share one original pretrained model.",
            "New development labels become development evidence after this access.",
        ],
    }
    write_json(args.output / "plan.json", plan)
    label_rows = []
    folder = args.output / "labels"
    folder.mkdir()
    with AnnualTargetReader(
        args.target_root,
        prepared,
        preparation_sha256=plan["preparation_sha256"],
        allow_evaluation=True,
    ) as reader:
        for patch, year in sorted(found):
            targets = reader.read(
                patch, year, role="development", tasks=["worldcover", "dem_elevation"]
            )
            path = folder / f"{patch}.{year}.npz"
            np.savez_compressed(
                path,
                wc=targets["worldcover"]["values"].astype(np.int16),
                wc_valid=targets["worldcover"]["valid"],
                dem=targets["dem_elevation"]["values"],
                dem_valid=targets["dem_elevation"]["valid"],
            )
            label_rows.append(
                {
                    "patch_id": patch,
                    "year": year,
                    "path": str(path.resolve()),
                    "sha256": digest(path),
                }
            )
    write_json(
        args.output / "labels.json",
        {
            "role": "development",
            "records": label_rows,
            "plan_sha256": digest(args.output / "plan.json"),
            "validation_opened": False,
            "confirmation_opened": False,
        },
    )
    print(
        json.dumps(
            {
                "development_positions": len(wanted),
                "annual_records": len(label_rows),
                "groups": len({r["group"] for r in assignments.values()}),
            }
        )
    )


def development_dataset_config(datasets, frozen):
    """Resolve a frozen annual split by its verified year, allowing region aliases."""
    from xuannv_embedding.utils.manifest import IndexedManifest

    manifest = Path(frozen["path"])
    records = IndexedManifest(manifest)
    year = frozen["year"]
    months = [f"{year}-{month:02}" for month in range(1, 13)]
    if records.years != {year} or records.meta.months != months:
        raise ValueError("Development manifest differs from its frozen calendar year")
    candidates = []
    for dataset in datasets:
        source = IndexedManifest(dataset.manifest_path)
        if source.years == {year} and source.meta.months == months:
            candidates.append(dataset)
    named = [dataset for dataset in candidates if dataset.region == frozen["region"]]
    matches = named or candidates
    if len(matches) != 1:
        raise ValueError("Development year/region must identify exactly one configured dataset")
    return replace(
        matches[0], region=frozen["region"], manifest_path=manifest, patch_grid_path=manifest
    )


def export(args):
    import torch
    from torch.utils.data import DataLoader

    from xuannv_embedding.data.annual_dataset import collate_annual_observations
    from xuannv_embedding.models.annual import masked_resample
    from xuannv_embedding.training.annual import prepare_annual_batch
    from xuannv_embedding.training.annual_data import annual_dataset
    from xuannv_embedding.training.annual_experiments import (
        artifact_paths,
        digest,
        load_model,
        write_json,
    )
    from xuannv_embedding.training.cli import _source_code_sha256
    from xuannv_embedding.training.runtime import _model_view, _move

    plan = json.loads(args.plan.read_text())
    if plan["role"] != "development":
        raise ValueError("Only the development role is permitted by this workflow")
    if digest(Path(plan["matrix_plan"])) != plan["matrix_plan_sha256"]:
        raise ValueError("Matrix identity changed")
    matrix = json.loads(Path(plan["matrix_plan"]).read_text())
    cp, checkpoint = artifact_paths(matrix, args.arm)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config, system, identity = load_model(cp, checkpoint, device)
    config = replace(
        config,
        data=replace(
            config.data, highres_max_observations=plan["protocol"]["highres_max_observations"]
        ),
    )
    root = args.plan.parent / "features" / args.arm
    if root.exists():
        raise FileExistsError("Development features must be new")
    root.mkdir(parents=True)
    receipts = []
    with torch.inference_mode():
        for frozen in plan["manifests"]:
            if digest(Path(frozen["path"])) != frozen["sha256"]:
                raise ValueError("Development manifest changed")
            dc = development_dataset_config(config.data.datasets, frozen)
            dataset = annual_dataset(config, dc)
            loader = DataLoader(
                dataset,
                batch_size=2,
                num_workers=args.workers,
                collate_fn=collate_annual_observations,
            )
            for raw in loader:
                batch = _move(prepare_annual_batch(raw, config, training=False), device)
                output = _model_view(system.model, batch, decode=False)
                f10, v10 = masked_resample(
                    output.embedding_map[:, 0], output.validity_mask[:, 0].float(), (128, 128)
                )
                f40, v40 = masked_resample(
                    output.embedding_map[:, 0], output.validity_mask[:, 0].float(), (32, 32)
                )
                for i, meta in enumerate(raw["metadata"]):
                    path = root / f"{meta['patch_id']}.{meta['year']}.npz"
                    features = {
                        "f10": f10[i].half().cpu().numpy(),
                        "f40": f40[i].half().cpu().numpy(),
                        "v10": (v10[i, 0] > 1 - 1e-6).cpu().numpy(),
                        "v40": (v40[i, 0] > 1 - 1e-6).cpu().numpy(),
                    }
                    if not all(np.isfinite(value).all() for value in features.values()):
                        raise ValueError("Nonfinite development features")
                    np.savez_compressed(path, **features)
                    receipts.append(
                        {
                            "patch_id": meta["patch_id"],
                            "year": meta["year"],
                            "path": str(path.resolve()),
                            "sha256": digest(path),
                        }
                    )
    expected = {(p, year) for p in plan["assignments"] for year in [2020, 2021]}
    if len(receipts) != len(expected) or {(r["patch_id"], r["year"]) for r in receipts} != expected:
        raise ValueError("Incomplete or duplicated development features")
    write_json(
        root / "features.json",
        {
            "arm": args.arm,
            "plan_sha256": digest(args.plan),
            "checkpoint_sha256": digest(checkpoint),
            "checkpoint_identity": identity,
            "evaluation_source_code_sha256": _source_code_sha256(),
            "records": receipts,
        },
    )
    print(json.dumps({"arm": args.arm, "development_features": len(receipts)}))


def probe(args):
    from xuannv_embedding.training.annual_experiments import digest, write_json
    from xuannv_embedding.training.cli import _source_code_sha256

    plan = json.loads(args.plan.read_text())
    if plan["role"] != "development":
        raise ValueError("Only the development role is permitted by this workflow")
    labels = json.loads((args.plan.parent / "labels.json").read_text())
    root = args.plan.parent / "features" / args.arm
    index = json.loads((root / "features.json").read_text())
    if labels["plan_sha256"] != digest(args.plan) or index["plan_sha256"] != digest(args.plan):
        raise ValueError("Development plan identity differs")
    if index["arm"] != args.arm or labels["role"] != "development":
        raise ValueError("Development arm or role identity differs")
    protocol = plan["protocol"]
    features = {}
    targets = {}
    baseline = args.plan.parent / "features" / "original_exact"
    for row in index["records"]:
        path = Path(row["path"])
        if digest(path) != row["sha256"]:
            raise ValueError("Development feature digest differs")
        key = row["patch_id"], row["year"]
        with np.load(path) as data:
            features[key] = {name: data[name] for name in data.files}
        with np.load(baseline / path.name) as data:
            if not np.array_equal(data["v10"], features[key]["v10"]) or not np.array_equal(
                data["v40"], features[key]["v40"]
            ):
                raise ValueError("Candidate changed scoring support")
    for row in labels["records"]:
        path = Path(row["path"])
        if digest(path) != row["sha256"]:
            raise ValueError("Development target digest differs")
        with np.load(path) as data:
            targets[row["patch_id"], row["year"]] = {name: data[name] for name in data.files}
    expected = {(patch, year) for patch in plan["assignments"] for year in [2020, 2021]}
    if set(features) != set(targets) or set(features) != expected:
        raise ValueError("Feature/target identity differs")
    output = args.plan.parent / "probes" / args.arm
    if output.exists():
        raise FileExistsError("Development probe output must be new")
    output.mkdir(parents=True)
    rows = {
        key: {"patch_id": key[0], "year": key[1], **plan["assignments"][key[0]]} for key in features
    }
    fits = []
    for fold in range(protocol["folds"]):
        frozen_fold = next(row for row in plan["folds"] if row["fold"] == fold)
        train = [key for key in features if key[0] in frozen_fold["train"]]
        test = [key for key in features if key[0] in frozen_fold["test"]]
        for task, grid, limit, classes in [
            ("wc", "10", protocol["worldcover_train_pixels_per_record"], 11),
            ("dem", "40", protocol["dem_train_cells_per_record"], None),
        ]:
            xs, ys = [], []
            for key in train:
                valid = targets[key][task + "_valid"] & features[key]["v" + grid]
                selected = sample_indices(valid, f"development/{key[0]}/{key[1]}/{task}", limit)
                if not len(selected):
                    continue
                xs.append(features[key]["f" + grid].reshape(64, -1).T[selected])
                ys.append(targets[key][task].reshape(-1)[selected])
            fitted = fit_readout(
                np.concatenate(xs),
                np.concatenate(ys),
                classes=classes,
                regularization=protocol["regularization"],
            )
            fit_path = output / f"fold{fold}_{task}.npz"
            np.savez_compressed(fit_path, **{k: v for k, v in fitted.items() if v is not None})
            fits.append(
                {
                    "fold": fold,
                    "task": task,
                    "training_records": len(train),
                    "training_units": sum(len(y) for y in ys),
                    "test_records": len(test),
                    "path": str(fit_path.resolve()),
                    "sha256": digest(fit_path),
                    "training_parent_ids_sha256": hashlib.sha256(
                        "\n".join(sorted({k[0] for k in train})).encode()
                    ).hexdigest(),
                }
            )
            for key in test:
                valid = targets[key][task + "_valid"] & features[key]["v" + grid]
                selected = np.flatnonzero(valid.reshape(-1))
                prediction = predict_readout(
                    fitted, features[key]["f" + grid].reshape(64, -1).T[selected]
                )
                truth = targets[key][task].reshape(-1)[selected]
                if task == "wc":
                    predicted = prediction.argmax(1)
                    rows[key]["confusion"] = (
                        np.bincount(truth * 11 + predicted, minlength=121).reshape(11, 11).tolist()
                    )
                else:
                    error = prediction[:, 0] - truth
                    rows[key].update(
                        dem_count=len(truth),
                        dem_sse=float(error @ error),
                        dem_sae=float(np.abs(error).sum()),
                        dem_sum_y=float(truth.astype(np.float64).sum()),
                        dem_sum_y2=float(np.square(truth.astype(np.float64)).sum()),
                    )
    result = {
        "arm": args.arm,
        "role": "development",
        "plan_sha256": digest(args.plan),
        "feature_index_sha256": digest(root / "features.json"),
        "label_index_sha256": digest(args.plan.parent / "labels.json"),
        "probe_source_code_sha256": _source_code_sha256(),
        "metrics": metric_summary(list(rows.values())),
        "metrics_by_year": {
            str(year): metric_summary([row for row in rows.values() if row["year"] == year])
            for year in [2020, 2021]
        },
        "per_record": list(rows.values()),
        "fits": fits,
        "validation_opened": False,
        "confirmation_opened": False,
        "independent_manual_accuracy_claim": False,
    }
    write_json(output / "metrics.json", result)
    print(json.dumps({"arm": args.arm, "metrics": result["metrics"]}))
