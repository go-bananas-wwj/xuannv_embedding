"""Paired spatial development probes for frozen annual embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LANDCOVER = [
    "tree",
    "shrub",
    "grass",
    "crop",
    "built",
    "bare",
    "snow",
    "water",
    "wetland",
    "mangrove",
    "moss",
]
ROAD_HALF_WIDTH = {
    "motorway": 6,
    "trunk": 6,
    "primary": 5,
    "secondary": 4,
    "tertiary": 3.5,
    "motorway_link": 3.5,
    "trunk_link": 3.5,
    "primary_link": 3.5,
    "secondary_link": 3,
    "tertiary_link": 3,
    "residential": 2.5,
    "unclassified": 2.5,
    "service": 2,
    "living_street": 2.5,
    "track": 1.5,
    "road": 2.5,
    "pedestrian": 2,
    "footway": 1,
    "path": 1,
    "cycleway": 1,
    "steps": 1,
}

SELECTION_PROTOCOL = {
    "version": 2,
    "reference": "control",
    "product_reference": "also guard the original when declared by plan.product_reference",
    "candidate_set": "trained arms plus explicit plan.selection_baselines",
    "semantic_terms": ["water:F1", "landcover:mIoU"],
    "fine_terms": [
        "road:conv3x3:F1",
        "road:conv3x3:boundaryF1",
        "building:conv3x3:F1",
        "water:conv3x3:F1",
    ],
    "semantic_weight": 0.5,
    "fine_weight": 0.5,
    "maximum_semantic_average_drop": 0.002,
    "maximum_fine_average_drop": 0.002,
    "maximum_water_drop": 0.01,
    "maximum_landcover_drop": 0.005,
    "maximum_road_drop": 0.005,
    "score_tie_tolerance": 0.001,
    "confirmation": (
        "Select once on development evidence, then compare original/control/small/selected "
        "on additional spatial holdout weak labels. Never select again on confirmation."
    ),
}


def select_candidate(plan, root: Path):
    from datetime import datetime, timezone

    from xuannv_embedding.training.annual_experiments import digest, write_json
    from xuannv_embedding.training.cli import _git_sha, _source_code_sha256

    protocol_file = root.parent / "selection_protocol.json"
    locked = json.loads(protocol_file.read_text())
    if locked["protocol"] != SELECTION_PROTOCOL:
        raise ValueError("Selection rule differs from the preregistered protocol")
    entries = {
        **plan["arms"],
        **{tag: plan["baselines"][tag] for tag in plan.get("selection_baselines", [])},
    }
    reference_tags = ["control"]
    if plan.get("product_reference"):
        reference_tags.append(plan["product_reference"])
    evidence, scores = {}, {}
    for tag in dict.fromkeys([*entries, *reference_tags]):
        semantic_path = root / tag / "semantic_probe" / "metrics.json"
        city_path = root / tag / "city_probe" / "metrics.json"
        semantic = json.loads(semantic_path.read_text())["tasks"]
        city = json.loads(city_path.read_text())["tasks"]
        evidence[str(semantic_path.resolve())] = digest(semantic_path)
        evidence[str(city_path.resolve())] = digest(city_path)
        metrics = {
            "water": semantic["water"]["all"]["F1"],
            "landcover": semantic["landcover"]["all"]["mIoU"],
            "road": city["road:conv3x3"]["all"]["F1"],
        }
        semantic_score = (metrics["water"] + metrics["landcover"]) / 2
        fine_score = np.mean(
            [
                city[key]["all"][metric]
                for key, metric in [
                    ("road:conv3x3", "F1"),
                    ("road:conv3x3", "boundaryF1"),
                    ("building:conv3x3", "F1"),
                    ("water:conv3x3", "F1"),
                ]
            ]
        )
        scores[tag] = {
            **metrics,
            "semantic_score": float(semantic_score),
            "fine_score": float(fine_score),
            "joint_score": float((semantic_score + fine_score) / 2),
        }
    for tag, score in scores.items():
        score["eligible"] = all(
            score[key] >= scores[reference][key] - margin
            for reference in reference_tags
            for key, margin in [
                ("semantic_score", 0.002),
                ("fine_score", 0.002),
                ("water", 0.01),
                ("landcover", 0.005),
                ("road", 0.005),
            ]
        )
    excluded = {plan["product_reference"]} if plan.get("product_reference") else {"control"}
    eligible = [
        tag
        for tag in entries
        if tag not in excluded
        and scores[tag]["eligible"]
        and scores[tag]["joint_score"] >= max(scores[ref]["joint_score"] for ref in reference_tags)
    ]
    best = max(eligible, key=lambda tag: scores[tag]["joint_score"]) if eligible else None
    if best is not None:
        tied = [
            tag
            for tag in eligible
            if scores[tag]["joint_score"] >= scores[best]["joint_score"] - 0.001
        ]
        best = min(
            tied,
            key=lambda tag: (
                entries[tag].get("optimizer_steps", 1) != 0,
                len(entries[tag]["changes"]),
                -scores[tag]["joint_score"],
            ),
        )
    diagnostic = max(
        (tag for tag in entries if tag not in excluded),
        key=lambda tag: scores[tag]["joint_score"],
    )
    result = {
        "selected": best,
        "diagnostic_best": diagnostic,
        "scores": scores,
        "references": reference_tags,
        "selection_git_sha": _git_sha(),
        "selection_source_code_sha256": _source_code_sha256(),
        "protocol_sha256": digest(protocol_file),
        "evidence_sha256": evidence,
        "selected_utc": datetime.now(timezone.utc).isoformat(),
        "selected_kind": (
            "preprocessing_only"
            if best is not None and entries[best].get("optimizer_steps") == 0
            else "continued_training" if best is not None else None
        ),
        "interpretation": "Exploratory development selection; confirmation still required.",
    }
    destination = root.parent / "selection.json"
    if destination.exists():
        raise ValueError("Candidate was already selected; do not reselect after confirmation")
    write_json(destination, result)
    return result


def spatial_group(patch_id: str) -> str:
    """Keep patches within a 20.48 km parent-grid block in the same fold."""
    zone, x, y = patch_id.removeprefix("parent_").split(":")
    return f"{zone}:{int(x) // 16}:{int(y) // 16}"


def semantic_groups(plan, ids):
    """Use the frozen coarse holdout units for confirmation uncertainty."""
    if plan.get("confirmation_only") and plan.get("cities"):
        mapping = {row[2]: row[0] for row in json.loads(Path(plan["cities"]).read_text())}
        if mapping:
            if set(ids) - mapping.keys():
                raise ValueError("Confirmation spatial groups do not cover all semantic positions")
            return [mapping[patch_id] for patch_id in ids]
    return [spatial_group(patch_id) for patch_id in ids]


def folds_for(groups: list[str]) -> np.ndarray:
    ordered = sorted(set(groups), key=lambda group: hashlib.sha256(group.encode()).digest())
    if len(ordered) < 5:
        raise ValueError("Spatial development probes require at least five independent groups")
    folds = {group: index % 5 for index, group in enumerate(ordered)}
    return np.asarray([folds[group] for group in groups])


def paired_group_bootstrap(a, b, groups, statistic, *, draws=2000, seed=42):
    """Resample whole cities/spatial groups, preserving pairing and cluster sizes."""
    a, b, groups = np.asarray(a), np.asarray(b), np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 5:
        return {
            "difference": float(statistic(a.sum(0)) - statistic(b.sum(0))),
            "CI95": None,
            "groups": len(unique),
            "patches": len(a),
            "draws": 0,
            "status": "insufficient_spatial_groups_for_interval",
        }
    blocks = [np.flatnonzero(groups == group) for group in unique]
    generator = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        index = np.concatenate(
            [blocks[i] for i in generator.integers(len(blocks), size=len(blocks))]
        )
        differences.append(statistic(a[index].sum(0)) - statistic(b[index].sum(0)))
    return {
        "difference": float(statistic(a.sum(0)) - statistic(b.sum(0))),
        "CI95": np.percentile(differences, [2.5, 97.5]).tolist(),
        "groups": len(unique),
        "patches": len(a),
        "draws": draws,
    }


def semantic_metrics(confusion, task):
    cm = np.asarray(confusion, dtype=np.float64)
    tp, gt, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    denominator = gt + predicted - tp
    if task == "water":
        return {
            "F1": float(2 * tp[1] / max(gt[1] + predicted[1], 1)),
            "IoU": float(tp[1] / max(denominator[1], 1)),
            "OA": float(tp.sum() / max(cm.sum(), 1)),
        }
    present = gt > 0
    return {
        "mIoU": float((tp[present] / denominator[present]).mean()),
        "macroF1": float((2 * tp[present] / (gt + predicted)[present]).mean()),
        "OA": float(tp.sum() / max(cm.sum(), 1)),
        "per_class_IoU": {
            LANDCOVER[i]: float(tp[i] / denominator[i]) for i in np.flatnonzero(present)
        },
    }


def semantic_probe(plan, root: Path, tag: str, device):
    """Fixed linear probes; fit normalization and class weights on training folds only."""
    from xuannv_embedding.training.annual_experiments import digest, write_json
    from xuannv_embedding.training.cli import _git_sha, _source_code_sha256

    labels_root = Path(plan["labels"])
    records = {
        row["patch_id"]: row
        for row in json.loads((labels_root / "labels.json").read_text())["records"]
    }
    ids = sorted(records)
    groups = semantic_groups(plan, ids)
    folds = folds_for(groups)
    features, labels, patches, annual_splits = (
        [],
        {task: [] for task in ["water", "landcover"]},
        [],
        [],
    )
    for index, patch_id in enumerate(ids):
        feature_path = root / tag / "semantic" / f"{patch_id}.npz"
        canonical_path = root / "original" / "semantic" / f"{patch_id}.npz"
        with np.load(feature_path) as candidate, np.load(canonical_path) as original:
            valid = original["valid"] > 0.5
            if not np.all(candidate["valid"][valid] > 0.5):
                raise ValueError("Candidate loses support on canonical scored pixels")
            features.append(
                candidate["emb"].reshape(64, -1).T[valid.reshape(-1)].astype(np.float32)
            )
            annual_splits.append(str(candidate["split"]))
        label_path = labels_root / records[patch_id]["file"]
        if digest(label_path) != records[patch_id]["sha256"]:
            raise ValueError("Label asset digest differs")
        with np.load(label_path) as target:
            for task in labels:
                labels[task].append(target[task].reshape(-1)[valid.reshape(-1)])
        patches.append(np.full(valid.sum(), index, dtype=np.int64))
    features = torch.from_numpy(np.concatenate(features))
    labels = {
        task: torch.from_numpy(np.concatenate(values)).long() for task, values in labels.items()
    }
    patches = torch.from_numpy(np.concatenate(patches))
    confirmation = plan.get("confirmation_only", False)
    if confirmation:
        folds = np.where(np.asarray(annual_splits) == "train", 1, 0)
    fold_pixels = torch.from_numpy(folds)[patches]
    result = {
        "development_only": not confirmation,
        "confirmation_only": confirmation,
        "evaluation_git_sha": _git_sha(),
        "evaluation_source_code_sha256": _source_code_sha256(),
        "label_manifest_sha256": digest(labels_root / "labels.json"),
        "checkpoint_sha256": json.loads((root / tag / "export.json").read_text())[
            "checkpoint_sha256"
        ],
        "fold_unit": (
            "current annual train / spatial holdout" if confirmation else "20.48 km spatial block"
        ),
        "groups": len(set(groups)),
        "ids": ids,
        "folds": folds.tolist(),
        "annual_splits": annual_splits,
        "tasks": {},
    }
    destination = root / tag / "semantic_probe"
    destination.mkdir(exist_ok=True)
    for task, classes in [("water", 2), ("landcover", 11)]:
        y = labels[task]
        keep = y >= 0
        confusion = np.zeros((len(ids), classes, classes), dtype=np.int64)
        for fold in ([0] if confirmation else range(5)):
            torch.manual_seed(100 + fold)
            train = (keep & (fold_pixels != fold)).nonzero().flatten()
            test = (keep & (fold_pixels == fold)).nonzero().flatten()
            generator = torch.Generator().manual_seed(fold)
            train = train[torch.randperm(len(train), generator=generator)[:200_000]]
            mean, std = features[train].mean(0), features[train].std(0).clamp(min=1e-6)
            x_train = ((features[train] - mean) / std).to(device)
            y_train = y[train].to(device)
            counts = torch.bincount(y_train, minlength=classes).float()
            weight = torch.where(
                counts > 0, counts.sum() / (counts.clamp(min=1) * (counts > 0).sum()), 0
            )
            head = nn.Linear(64, classes).to(device)
            optimizer = torch.optim.LBFGS(
                head.parameters(), max_iter=300, history_size=20, line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad()
                loss = (
                    F.cross_entropy(head(x_train), y_train, weight=weight)
                    + 1e-4 * head.weight.square().sum()
                )
                loss.backward()
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                for chunk in torch.split(test, 200_000):
                    prediction = head(((features[chunk] - mean) / std).to(device)).argmax(1).cpu()
                    flat = patches[chunk] * classes * classes + y[chunk] * classes + prediction
                    confusion += (
                        torch.bincount(flat, minlength=len(ids) * classes * classes)
                        .reshape(len(ids), classes, classes)
                        .numpy()
                    )
        np.save(destination / f"{task}.npy", confusion)
        result["tasks"][task] = {"all": semantic_metrics(confusion.sum(0), task)}
        for split_name, select in [
            ("pretraining_seen", np.asarray(annual_splits) == "train"),
            ("pretraining_unseen", np.asarray(annual_splits) != "train"),
        ]:
            if select.any() and (not confirmation or split_name == "pretraining_unseen"):
                result["tasks"][task][split_name] = {
                    **semantic_metrics(confusion[select].sum(0), task),
                    "patches": int(select.sum()),
                }
        print(json.dumps({"semantic": tag, "task": task, **result["tasks"][task]}), flush=True)
    write_json(destination / "metrics.json", result)


def rasterize_osm(path: Path):
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely import wkb
    from shapely.ops import transform

    content = json.loads(path.read_text())
    metadata = content["_meta"]
    x0, _, _, y1 = metadata["bounds"]
    project = Transformer.from_crs(4326, metadata["epsg"], always_xy=True).transform
    output = {}
    for task in ["building", "road", "water"]:
        geometries = []
        for encoded, category in content[task]:
            if task == "road" and category not in ROAD_HALF_WIDTH:
                continue
            geometry = transform(project, wkb.loads(encoded, hex=True))
            if task == "road":
                geometry = geometry.buffer(ROAD_HALF_WIDTH[category], cap_style=2)
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if not geometry.is_empty:
                geometries.append(geometry)
        if task == "building":
            output["n_building"] = len(geometries)
        output[task] = (
            rasterize(
                [(geometry, 1) for geometry in geometries],
                out_shape=(256, 256),
                transform=Affine(5, 0, x0, 0, -5, y1),
                fill=0,
                dtype="uint8",
            )
            if geometries
            else np.zeros((256, 256), np.uint8)
        )
    return output


def cache_osm_labels(path: Path, values: dict) -> None:
    """Parallel probes may share this cache; expose only a complete NPZ archive."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def boundary(mask):
    values = mask.float()[:, None]
    return ((F.max_pool2d(values, 3, 1, 1) + F.max_pool2d(-values, 3, 1, 1)) > 0)[:, 0]


def city_counts(prediction, target):
    truth = target.bool()
    pe, ge = boundary(prediction), boundary(truth)
    pd = F.max_pool2d(pe.float()[:, None], 3, 1, 1)[:, 0] > 0
    gd = F.max_pool2d(ge.float()[:, None], 3, 1, 1)[:, 0] > 0
    masks = [prediction & truth, prediction & ~truth, ~prediction & truth, pe & gd, pe, ge & pd, ge]
    return torch.stack([mask.flatten(1).sum(1) for mask in masks], 1).cpu().numpy()


def city_metrics(counts):
    tp, fp, fn, bp, bpn, br, brn = np.asarray(counts, dtype=np.float64)
    precision, recall = bp / max(bpn, 1), br / max(brn, 1)
    return {
        "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "IoU": float(tp / max(tp + fp + fn, 1)),
        "boundaryF1": float(2 * precision * recall / max(precision + recall, 1e-12)),
    }


def train_city_head(features, labels, train, kind, fold, steps):
    torch.manual_seed(fold)
    device = features.device
    head = (
        nn.Conv2d(64, 1, 1)
        if kind == "linear"
        else nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 3, padding=1)
        )
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    positive = labels[train].float().mean().clamp(1e-4, 1 - 1e-4)
    mean = torch.zeros(64, device=device)
    square = torch.zeros_like(mean)
    for chunk in torch.split(train, 8):
        values = features[chunk].float()
        mean += values.sum((0, 2, 3))
        square += values.square().sum((0, 2, 3))
    count = len(train) * features.shape[-2] * features.shape[-1]
    mean /= count
    std = (square / count - mean.square()).clamp(min=1e-12).sqrt()
    mean, std = mean[None, :, None, None], std[None, :, None, None]
    generator = torch.Generator().manual_seed(fold)
    for _ in range(steps):
        index = train[torch.randint(len(train), (8,), generator=generator).to(device)]
        logits = head((features[index].float() - mean) / std)[:, 0]
        loss = F.binary_cross_entropy_with_logits(
            logits, labels[index].float(), pos_weight=(1 - positive) / positive
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return head.eval(), mean, std


def city_probe(plan, root: Path, tag: str, device):
    from xuannv_embedding.training.annual_experiments import digest, write_json
    from xuannv_embedding.training.cli import _git_sha, _source_code_sha256

    rows = json.loads(Path(plan["cities"]).read_text())
    city = {row[2]: row[0] for row in rows}
    ids = sorted(city)
    folds = folds_for([city[patch_id] for patch_id in ids])
    features, labels, counts, highres, splits = (
        [],
        {task: [] for task in ["building", "road", "water"]},
        [],
        [],
        [],
    )
    cache = root / "osm_labels"
    cache.mkdir(exist_ok=True)
    label_digests = {}
    for patch_id in ids:
        source = Path(plan["osm"]) / f"{patch_id}.json"
        label_digests[patch_id] = digest(source)
        path = cache / f"{patch_id}.npz"
        if not path.exists():
            cache_osm_labels(
                path, {**rasterize_osm(source), "source_sha256": label_digests[patch_id]}
            )
        with np.load(path) as target:
            if str(target["source_sha256"]) != label_digests[patch_id]:
                raise ValueError("Cached OSM label provenance changed")
            counts.append(int(target["n_building"]))
            for task in labels:
                labels[task].append(target[task].copy())
        with np.load(root / tag / "city5m" / f"{patch_id}.npz") as feature:
            if not np.all(feature["valid"] > 0.5):
                raise ValueError("City protocol requires complete support on every scored patch")
            features.append(feature["emb"].copy())
            highres.append(int(feature["highres_obs"]))
            splits.append(str(feature["split"]))
    features = torch.from_numpy(np.stack(features)).to(device)
    feature_mode = plan.get("probe_feature_mode", "native5m")
    if feature_mode == "pooled10m":
        features = F.interpolate(
            F.avg_pool2d(features.float(), 2),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(features.dtype)
    elif feature_mode != "native5m":
        raise ValueError("Unknown city probe feature mode")
    labels = {
        task: torch.from_numpy(np.stack(values)).to(device) for task, values in labels.items()
    }
    scope = {
        "building": np.asarray(counts) >= 20,
        **{task: labels[task].flatten(1).sum(1).cpu().numpy() >= 200 for task in ["road", "water"]},
    }
    confirmation = plan.get("confirmation_only", False)
    if confirmation:
        folds = np.where(np.asarray(splits) == "train", 1, 0)
    scored_scope = {
        task: select & (np.asarray(splits) != "train") if confirmation else select
        for task, select in scope.items()
    }
    seed_offset = int(plan.get("probe_seed_offset", 0))
    save_heads = plan.get("save_probe_heads", False)
    suffix = f"_seed{seed_offset}" if seed_offset or save_heads else ""
    if feature_mode != "native5m":
        suffix = f"_{feature_mode}{suffix}"
    destination = root / tag / f"city_probe{suffix}"
    destination.mkdir(exist_ok=True)
    result = {
        "development_only": not confirmation,
        "evaluation_git_sha": _git_sha(),
        "evaluation_source_code_sha256": _source_code_sha256(),
        "fold_unit": "2 degree spatial block holdout" if confirmation else "city",
        "confirmation_only": confirmation,
        "ids": ids,
        "cities": [city[patch_id] for patch_id in ids],
        "folds": folds.tolist(),
        "highres": highres,
        "annual_splits": splits,
        "probe_steps": plan["probe_steps"],
        "probe_seed_offset": seed_offset,
        "feature_mode": feature_mode,
        "threshold_free_metrics": plan.get("probe_threshold_free", False),
        "label_sha256": label_digests,
        "checkpoint_sha256": json.loads((root / tag / "export.json").read_text())[
            "checkpoint_sha256"
        ],
        "scope": {task: select.tolist() for task, select in scored_scope.items()},
        "tasks": {},
    }
    for task, target in labels.items():
        for kind in ["linear", "conv3x3"]:
            stats = np.zeros((len(ids), 7), dtype=np.int64)
            scores = (
                np.full((len(ids), *target.shape[-2:]), np.nan, dtype=np.float32)
                if plan.get("probe_threshold_free")
                else None
            )
            for fold in ([0] if confirmation else range(5)):
                train = torch.from_numpy(np.flatnonzero(scope[task] & (folds != fold))).to(device)
                test = torch.from_numpy(np.flatnonzero(scope[task] & (folds == fold))).to(device)
                if not len(test):
                    continue
                if not len(train):
                    raise ValueError("City fold has no eligible training patches")
                head, mean, std = train_city_head(
                    features, target, train, kind, fold + seed_offset, plan["probe_steps"]
                )
                with torch.no_grad():
                    for chunk in torch.split(test, 8):
                        logits = head((features[chunk].float() - mean) / std)[:, 0]
                        prediction = logits > 0
                        stats[chunk.cpu().numpy()] = city_counts(prediction, target[chunk])
                        if scores is not None:
                            scores[chunk.cpu().numpy()] = logits.cpu().numpy()
                if save_heads:
                    from xuannv_embedding.training.checkpoint import _atomic_torch_save

                    _atomic_torch_save(
                        {
                            "head": {
                                key: value.detach().cpu()
                                for key, value in head.state_dict().items()
                            },
                            "mean": mean.cpu(),
                            "std": std.cpu(),
                            "task": task,
                            "kind": kind,
                            "fold": fold,
                            "seed": fold + seed_offset,
                            "checkpoint_sha256": result["checkpoint_sha256"],
                        },
                        destination / f"{task}_{kind}_fold{fold}.pt",
                    )
                del head
            np.save(destination / f"{task}_{kind}.npy", stats)
            groups = {
                "all": scored_scope[task],
                "with_highres": scored_scope[task] & (np.asarray(highres) > 0),
                "no_highres": scored_scope[task] & (np.asarray(highres) == 0),
                "pretraining_unseen": scored_scope[task] & (np.asarray(splits) != "train"),
            }
            key = f"{task}:{kind}"
            result["tasks"][key] = {
                group: {**city_metrics(stats[select].sum(0)), "patches": int(select.sum())}
                for group, select in groups.items()
                if select.any()
            }
            if scores is not None:
                from sklearn.metrics import average_precision_score, roc_auc_score

                for group, select in groups.items():
                    if not select.any():
                        continue
                    index = torch.from_numpy(np.flatnonzero(select)).to(device)
                    truth = target[index].cpu().numpy().reshape(-1)
                    confidence = scores[select].reshape(-1)
                    if not np.isfinite(confidence).all():
                        raise ValueError("Missing scored probabilities in probe evaluation")
                    metrics = result["tasks"][key][group]
                    metrics["AP"] = (
                        float(average_precision_score(truth, confidence)) if truth.any() else None
                    )
                    metrics["ROC_AUC"] = (
                        float(roc_auc_score(truth, confidence))
                        if 0 < truth.sum() < len(truth)
                        else None
                    )
            print(json.dumps({"city": tag, "task": key, **result["tasks"][key]}), flush=True)
    write_json(destination / "metrics.json", result)


def compare(plan, root: Path, arms: list[str], reference="original"):
    from xuannv_embedding.training.annual_experiments import write_json
    from xuannv_embedding.training.cli import _git_sha, _source_code_sha256

    result = {
        "reference": reference,
        "development_only": not plan.get("confirmation_only", False),
        "confirmation_only": plan.get("confirmation_only", False),
        "comparisons": {},
        "comparison_git_sha": _git_sha(),
        "comparison_source_code_sha256": _source_code_sha256(),
    }
    for tag in arms:
        if tag == reference:
            continue
        result["comparisons"][tag] = {}
        for suite in ["semantic_probe", "city_probe"]:
            base = root / reference / suite
            candidate = root / tag / suite
            if not (base / "metrics.json").exists() or not (candidate / "metrics.json").exists():
                continue
            metadata = json.loads((base / "metrics.json").read_text())
            other = json.loads((candidate / "metrics.json").read_text())
            if metadata["ids"] != other["ids"] or metadata["folds"] != other["folds"]:
                raise ValueError("Compared development sets or folds differ")
            if suite == "semantic_probe":
                if metadata["label_manifest_sha256"] != other["label_manifest_sha256"]:
                    raise ValueError("Semantic labels differ across compared arms")
                groups = semantic_groups(plan, metadata["ids"])
                tasks = [("water", "F1"), ("landcover", "mIoU")]
                for task, metric in tasks:
                    a, b = np.load(candidate / f"{task}.npy"), np.load(base / f"{task}.npy")
                    scored = (a.sum((1, 2)) > 0) & (b.sum((1, 2)) > 0)
                    result["comparisons"][tag][f"semantic:{task}:{metric}"] = (
                        paired_group_bootstrap(
                            a[scored],
                            b[scored],
                            np.asarray(groups)[scored],
                            lambda cm: semantic_metrics(cm, task)[metric],
                        )
                    )
            else:
                if (
                    metadata["label_sha256"] != other["label_sha256"]
                    or metadata["scope"] != other["scope"]
                ):
                    raise ValueError("City labels or evaluation scope differ")
                for task in metadata["tasks"]:
                    a = np.load(candidate / f"{task.replace(':', '_')}.npy")
                    b = np.load(base / f"{task.replace(':', '_')}.npy")
                    supported = np.asarray(metadata["scope"][task.split(":")[0]], dtype=bool)
                    for name, select in [
                        ("all", supported),
                        ("with_highres", supported & (np.asarray(metadata["highres"]) > 0)),
                        ("no_highres", supported & (np.asarray(metadata["highres"]) == 0)),
                    ]:
                        if not select.any():
                            continue
                        for metric in ["F1", "boundaryF1"]:
                            result["comparisons"][tag][f"city:{task}:{name}:{metric}"] = (
                                paired_group_bootstrap(
                                    a[select],
                                    b[select],
                                    np.asarray(metadata["cities"])[select],
                                    lambda values: city_metrics(values)[metric],
                                )
                            )
    write_json(root / f"comparison_{reference}.json", result)
    return result
